"""
PRobe submission inference entrypoint.

Reads ``API_BASE_URL``, ``MODEL_NAME``, and ``HF_TOKEN`` from the environment,
calls the configured OpenAI-compatible API via the official ``openai`` client,
runs the PRobe ``ProbeEnvironment`` for each requested task, and prints
structured lines for automated evaluation:

  [START] {...}
  [STEP]  {...}
  [END]   {...}

Each JSON object uses **sorted keys** and **compact separators** (stable ordering).

Smoke test (no network, no API keys)::

    python inference.py --smoke

Full run (requires env vars)::

    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py --tasks 0 1 2 --episodes-per-task 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Repository root on sys.path (this file lives at project root)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.baseline import TASKS, run_episode  # noqa: E402
from environment.probe_environment import ProbeEnvironment  # noqa: E402

_SCHEMA = "probe-inference-1"
_DEFAULT_WALL_S = 1140  # stay under 20 min with margin


def _log(tag: str, payload: dict[str, Any]) -> None:
    """Emit one evaluation line with sorted keys."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    print(f"[{tag}] {body}", flush=True)


def _host_only(api_base_url: str) -> str:
    p = urlparse(api_base_url.strip())
    return p.netloc or p.path.split("/")[0] or "invalid-url"


def _reward_01(raw: float) -> float:
    r = float(raw)
    r = min(1.0, max(-1.0, r))
    return round((r + 1.0) / 2.0, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="PRobe inference.py (submission entrypoint)")
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Task ids to run (default: first three graders)",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Episodes per task (default 1 for fast validation)",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=int,
        default=_DEFAULT_WALL_S,
        help="Stop early after this many wall seconds (default 1140)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run without LLM or API keys; one episode on task 0 (deterministic submit path)",
    )
    args = parser.parse_args()

    run_id = str(uuid.uuid4())
    t0 = time.monotonic()

    if args.smoke:
        task_ids = [0]
        episodes_per_task = 1
        client = None
        model_name = None
        api_base_host = "smoke"
        _log(
            "START",
            {
                "api_base_host": api_base_host,
                "model_name": "none",
                "run_id": run_id,
                "schema_version": _SCHEMA,
                "smoke": True,
                "task_ids": task_ids,
            },
        )
    else:
        api_base = os.environ.get("API_BASE_URL", "").strip()
        model_name = os.environ.get("MODEL_NAME", "").strip()
        token = os.environ.get("HF_TOKEN", "").strip()
        missing = [k for k, v in [("API_BASE_URL", api_base), ("MODEL_NAME", model_name), ("HF_TOKEN", token)] if not v]
        if missing:
            _log(
                "END",
                {
                    "episodes_completed": 0,
                    "mean_cumulative_reward": None,
                    "mean_cumulative_reward_01": None,
                    "message": f"Missing required environment variables: {', '.join(missing)}",
                    "status": "error_config",
                },
            )
            return 1
        try:
            from openai import OpenAI
        except ImportError:
            _log(
                "END",
                {
                    "episodes_completed": 0,
                    "mean_cumulative_reward": None,
                    "mean_cumulative_reward_01": None,
                    "message": "openai package not installed",
                    "status": "error_import",
                },
            )
            return 1
        base_url = api_base.rstrip("/")
        if not base_url.endswith("v1"):
            base_url = f"{base_url}/v1"
        client = OpenAI(base_url=base_url, api_key=token)
        api_base_host = _host_only(api_base)
        task_ids = args.tasks
        episodes_per_task = args.episodes_per_task
        _log(
            "START",
            {
                "api_base_host": api_base_host,
                "model_name": model_name,
                "run_id": run_id,
                "schema_version": _SCHEMA,
                "smoke": False,
                "task_ids": task_ids,
            },
        )

    env = ProbeEnvironment()
    results: list[dict[str, Any]] = []

    def on_step(row: dict[str, Any]) -> None:
        _log("STEP", row)

    for task_id in task_ids:
        if task_id < 0 or task_id >= len(TASKS):
            _log(
                "END",
                {
                    "episodes_completed": len(results),
                    "mean_cumulative_reward": None,
                    "mean_cumulative_reward_01": None,
                    "message": f"Invalid task_id {task_id}",
                    "status": "error_task",
                },
            )
            return 1
        for ep in range(episodes_per_task):
            if time.monotonic() - t0 > args.max_wall_seconds:
                rewards = [float(r["cumulative_reward"]) for r in results]
                mean_r = sum(rewards) / len(rewards) if rewards else 0.0
                _log(
                    "END",
                    {
                        "episodes_completed": len(results),
                        "mean_cumulative_reward": round(mean_r, 6),
                        "mean_cumulative_reward_01": round(_reward_01(mean_r), 6),
                        "message": "max_wall_seconds exceeded",
                        "status": "timeout_partial",
                    },
                )
                return 0
            env._reset_count = task_id  # align with training/baseline.py episode selection
            result = run_episode(
                env,
                client,
                task_id,
                model_name=model_name,
                on_step=on_step,
            )
            results.append(result)

    rewards = [float(r["cumulative_reward"]) for r in results]
    mean_r = sum(rewards) / len(rewards) if rewards else 0.0
    _log(
        "END",
        {
            "episodes_completed": len(results),
            "mean_cumulative_reward": round(mean_r, 6),
            "mean_cumulative_reward_01": round(_reward_01(mean_r), 6),
            "status": "ok_smoke" if args.smoke else "ok",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
