"""
Baseline inference script for the CodeReviewAgent OpenEnv environment.

Runs an OpenAI model against all 3 tasks (easy / medium / hard) and
produces a reproducible baseline score.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

Optional env vars:
    OPENAI_MODEL   model name (default: gpt-4o-mini)
    OPENAI_BASE_URL  override endpoint (e.g. Azure)
"""


from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv                                                                                                          
load_dotenv()


# Allow importing the package without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CodeReviewAgent"))

from openai import OpenAI

from CodeReviewAgent.server.CodeReviewAgent_environment import CodereviewagentEnvironment
from CodeReviewAgent.models import (
    ActionType,
    CodereviewagentAction,
    IssueCategory,
    Severity,
)

SYSTEM_PROMPT = """You are a senior software engineer performing a pull-request code review.

You interact with the review environment by emitting a single JSON action per turn.

Available actions:

1. ADD_COMMENT — annotate a specific line:
   {"action_type": "add_comment", "line_number": <int>, "comment": "<text>",
    "severity": "<info|warning|error|critical>", "category": "<bug|security|performance|style|design>"}

2. REQUEST_CHANGES — signal the PR needs work (use after adding all comments):
   {"action_type": "request_changes", "comment": "<brief summary>"}

3. APPROVE — approve the PR (only when you find no significant issues):
   {"action_type": "approve"}

4. SUBMIT_REVIEW — finalise and submit the review (ends the episode):
   {"action_type": "submit_review"}

Strategy:
- Read the code carefully.
- Add one ADD_COMMENT for every issue you find (line number + severity + category).
- Decide REQUEST_CHANGES if issues exist, APPROVE if the code is clean.
- Always end with SUBMIT_REVIEW.

Reply with ONLY a valid JSON object — no markdown fences, no explanation."""


def _parse_action(text: str) -> CodereviewagentAction:
    """Extract JSON from the model response and parse it into an Action."""
    text = text.strip()
    # Strip optional markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
    data = json.loads(text)
    return CodereviewagentAction(**data)


def _format_prompt(obs, step: int) -> str:
    history_lines = ""
    if obs.review_history:
        recent = obs.review_history[-6:]
        history_lines = "\n\nYour review so far:\n" + "\n".join(
            f"  [{e.get('type')}] line={e.get('line')} — {str(e.get('text', ''))[:90]}"
            for e in recent
        )

    return (
        f"File: {obs.file_name}  |  Task: {obs.task_difficulty.upper()}\n"
        f"Objective: {obs.task_description}\n"
        f"Progress: step {step}, issues found {obs.issues_found_count}/{obs.total_issues}\n\n"
        f"```python\n{obs.code_snippet}```"
        f"{history_lines}\n\n"
        "What is your next action?"
    )


def run_task(client: OpenAI, model: str, task_id: int) -> dict:
    """Run one episode on the given task and return a result dict."""
    # Force the environment to start on the right task
    env = CodereviewagentEnvironment()
    # cycle to the correct task_id (reset increments _reset_count)
    for _ in range(task_id):
        env.reset()

    obs = env.reset()
    task_name = obs.task_difficulty
    total_issues = obs.total_issues

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    cumulative_reward = 0.0
    step = 0
    log = []

    print(f"\n{'─'*60}")
    print(f"Task {task_id} [{obs.task_difficulty}]: {obs.file_name}  ({total_issues} issues)")
    print(f"{'─'*60}")

    while True:
        messages.append({"role": "user", "content": _format_prompt(obs, step)})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            assistant_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

            action = _parse_action(assistant_text)
            obs = env.step(action)
            step += 1
            reward = obs.reward or 0.0
            cumulative_reward += reward
            done = obs.done

            log.append({"step": step, "action": action.action_type.value, "reward": reward})
            print(f"  {step:2d}. {action.action_type.value:<20} reward={reward:+.4f}  issues={obs.issues_found_count}/{total_issues}")

            if done:
                break

        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse error: {exc} — forcing submit")
            obs = env.step(CodereviewagentAction(action_type=ActionType.SUBMIT_REVIEW))
            cumulative_reward += obs.reward or 0.0
            break
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            break

    coverage = obs.issues_found_count / total_issues if total_issues else 0.0
    result = {
        "task_id": task_id,
        "difficulty": obs.task_difficulty,
        "file_name": obs.file_name,
        "cumulative_reward": round(cumulative_reward, 4),
        "steps": step,
        "issues_found": obs.issues_found_count,
        "total_issues": total_issues,
        "coverage": round(coverage, 3),
    }
    print(f"\n  → cumulative_reward={result['cumulative_reward']:+.4f}  coverage={coverage:.0%}")
    return result


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY environment variable is not set.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    print(f"CodeReviewAgent Baseline  |  model={model}")

    results = [run_task(client, model, task_id) for task_id in range(3)]

    avg_reward = sum(r["cumulative_reward"] for r in results) / len(results)
    avg_coverage = sum(r["coverage"] for r in results) / len(results)

    print(f"\n{'═'*60}")
    print("BASELINE SUMMARY")
    print(f"{'═'*60}")
    header = f"  {'Task':<6} {'Difficulty':<10} {'Reward':>8} {'Coverage':>10} {'Steps':>6}"
    print(header)
    print(f"  {'─'*50}")
    for r in results:
        print(
            f"  {r['task_id']:<6} {r['difficulty']:<10} "
            f"{r['cumulative_reward']:>+8.4f} {r['coverage']:>9.0%}  {r['steps']:>5}"
        )
    print(f"  {'─'*50}")
    print(f"  {'avg':<6} {'':10} {avg_reward:>+8.4f} {avg_coverage:>9.0%}")

    output = {
        "model": model,
        "results": results,
        "summary": {"avg_cumulative_reward": avg_reward, "avg_coverage": avg_coverage},
    }
    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
