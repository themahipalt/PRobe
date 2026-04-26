---
title: PRobe Environment
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /ui/
tags:
  - openenv
  - code-review
  - rl-training
  - grpo
  - probe
---

# PRobe — an AI code reviewer that can spot backdoors

## Submission links (judge quick access)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/FILL_COLAB_LINK)

> Replace each placeholder below with a real URL before submission.

| Resource | URL |
|---|---|
| 🤗 HuggingFace Space (live environment) | https://huggingface.co/spaces/mahithakur/PRobe |
| 📓 Training notebook (Colab / Kaggle) | [View in Colab](https://colab.research.google.com/drive/FILL_COLAB_LINK) |
| 📝 Mini-blog / writeup (HuggingFace) | [Published on OpenEnv Hackers](https://huggingface.co/spaces/open-env/open-env-hackers) |
| 📊 Training results (Dataset) | https://huggingface.co/datasets/mahithakur/PRobe-training-results |
| 🎥 Demo video (YouTube, < 2 min) | [Create and share](https://www.youtube.com/) |
| 📈 JUDGE_REPORT | [View report](./reports/JUDGE_REPORT.md) |

## TL;DR

PRobe is a training environment where an AI learns to **review Python code like a careful security engineer**:

- Find real bugs and security issues (with correct line numbers)
- Tell the difference between an honest mistake vs. a deliberate backdoor
- Decide whether to **approve**, **request changes**, or **escalate to security**

Unlike many demos, PRobe uses a **deterministic reward** (no “LLM judge”). Keyword-spam on random lines gets penalized; careful, accurate findings score high.

## Try it in 60 seconds

```bash
uv sync
uv run python run.py
```

Then open `http://localhost:8000/ui/` and click **New Episode**.

## Why it exists (simple version)

Real supply-chain attacks (like XZ Utils / SolarWinds) often look like normal code changes. A useful AI reviewer must do more than “scan” — it must **investigate intent** and know when to escalate.

## What’s novel (in plain English)

- **No LLM judge**: reward is deterministic and reproducible.
- **Anti-gaming**: keyword spam on random lines gets penalized.
- **Backdoor escalation**: some tasks require choosing “escalate to security”, not just listing bugs.

## What’s inside (high level)

- **10 tasks** that simulate real review situations (bugs + adversarial backdoors)
- A **mutator** that changes variable names/line numbers so the model can’t memorize answers
- A **grader** that scores outputs based on “right issue + right place + good explanation”
- A lightweight **web UI** so anyone can try an episode in the browser

If you want the full technical design, see `docs/design.md`.

## Training (GRPO)

The training entrypoint is `training/train_grpo.py`.

### Install training dependencies

```bash
pip install -e ".[training]"
```

**Colab:** for actual training, switch to **GPU**: Runtime → Change runtime type → Hardware accelerator → GPU. CPU training is extremely slow, and some TRL configs may error if bf16/fp16 is enabled on CPU.

### Smoke test (no GPU, no model download)

```bash
python training/train_grpo.py --test
```

### Train (example)

```bash
python training/train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 200 \
  --group-size 2 \
  --batch-size 2 \
  --grad-accum 1 \
  --max-seq-len 1024 \
  --max-completion-len 128 \
  --save-steps 50
```

### Resume from a checkpoint

```bash
python training/train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 200 \
  --resume-from outputs/checkpoint-100
```

### Reproduce our run (copy/paste template)

Fill these before submission:

- **Hardware**: (T4 / A100 / …)
- **Steps**: (100 / 200)
- **Runtime**: (~__ minutes)

Example command (200 steps, checkpoints every 50 steps):

```bash
python training/train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 200 \
  --group-size 2 \
  --batch-size 2 \
  --grad-accum 1 \
  --max-seq-len 1024 \
  --max-completion-len 128 \
  --save-steps 50 \
  --output-dir outputs
```

## Outputs

Training writes artifacts under `outputs/` (or your `--output-dir`), including:

- Checkpoints: `checkpoint-*`
- Curves: `training_curves.png`, `per_task_reward.png`
- Demo traces (adversarial tasks): `demo/before_task*.json`, `demo/after_task*.json`

## Before vs. after training (images)

### Latest measured run (Google Colab, partial log captured)

These numbers were extracted from `outputs/training.jsonl` in Colab:

```
==================================================
  COLAB 5-RECORD RUN SUMMARY (from outputs/training.jsonl)
==================================================
  Total records   : 5
  Avg reward      : 0.164
  Best reward     : 0.250
  First 25% avg   : 0.100
  Last 25% avg    : 0.185
  Improvement     : +0.085
```

Quick scan for judges:

- **Mean reward (logged)**: **0.164**
- **Best reward (logged)**: **0.250**
- **First 25% vs last 25% (logged)**: **0.100 → 0.185** (**+0.085**)

> Note: this is **not a full 100-step summary** yet — the `training.jsonl` currently contains only 5 logged records. To report “after 100 steps”, make sure the run actually logs ~100 records (and save/zip `outputs/` before interrupting).

**How to extract the “after N steps” numbers (from `training.jsonl`):** even if you interrupt training, you can compute the same judge-friendly summary from `outputs/training.jsonl` as long as it contains records.

```python
# Colab cell: summarize outputs/training.jsonl and print a README-ready block
import json, pathlib

path = pathlib.Path("outputs/training.jsonl")
recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
assert recs, "training.jsonl is empty"

def get_reward(r):
    # Supports both older keys ("reward") and current key ("reward_total")
    return float(r.get("reward_total", r.get("reward")))

rewards = [get_reward(r) for r in recs]
n = len(rewards)
first_q = rewards[: max(1, n // 4)]
last_q = rewards[3 * n // 4 :] if n >= 4 else rewards

summary = f"""==================================================
  COLAB {n}-RECORD RUN SUMMARY (from outputs/training.jsonl)
==================================================
  Total records   : {n}
  Avg reward      : {sum(rewards)/n:.3f}
  Best reward     : {max(rewards):.3f}
  First 25% avg   : {sum(first_q)/len(first_q):.3f}
  Last 25% avg    : {sum(last_q)/len(last_q):.3f}
  Improvement     : {sum(last_q)/len(last_q) - sum(first_q)/len(first_q):+.3f}
"""
print(summary)
```

After training, these images are written to `outputs/` and help show improvement:

- `outputs/training_curves.png` (reward / loss over steps)
- `outputs/per_task_reward.png` (per-task reward before vs after)

![Training Curves](outputs/training_curves.png)

![Per-task Reward](outputs/per_task_reward.png)

If the images above do not render on GitHub, commit the PNGs into `outputs/` (they are generated by `training/train_grpo.py` after a full run completes).

---

## Repo Structure

```
.
├── agent/
│   ├── client.py               # HTTP client for interacting with the environment server
│   ├── models.py               # Pydantic models: ProbeAction, ProbeObservation, RewardType
│   └── __init__.py
├── environment/
│   ├── app.py                  # FastAPI server (HTTP + WebSocket + static frontend at /ui/)
│   ├── Dockerfile              # Container definition for HuggingFace Spaces
│   ├── episode_memory.py       # Cross-episode JSON memory (injects prior-finding hints)
│   ├── graders.py              # Deterministic reward grader (keyword+line+length verifier)
│   ├── mutator.py              # Code mutation engine (rename / shift / nudge)
│   ├── probe_environment.py    # Core environment: reset / step / state / action handlers
│   ├── requirements.txt        # Server-side Python dependencies
│   ├── scanner.py              # Simulated static-analysis tool (70% recall, FP injection)
│   ├── tasks.py                # 10 task definitions with ground-truth issue lists
│   ├── _import_compat.py       # Import shim for package / script / test contexts
│   └── __init__.py
├── frontend/
│   ├── index.html              # Three-column dashboard layout
│   ├── style.css               # Dark IDE theme (no build step required)
│   └── app.js                  # WebSocket client, code viewer, reward ring, history feed
├── training/
│   ├── baseline.py             # Zero-shot GPT-4o-mini baseline agent + plotting
│   ├── scripted_baseline.py    # Deterministic oracle and spammer stress-tests
│   ├── train_grpo.py           # GRPO training script (TRL + optional Unsloth, 5-phase curriculum)
│   └── __init__.py
├── tests/
│   ├── test_dynamic_world.py   # Tests for mutation engine and scanner noise model
│   ├── test_grader.py          # Tests for reward grader correctness
│   └── __init__.py
├── docs/
│   └── design.md               # Architecture notes
├── outputs/
│   └── scripted_baseline.jsonl # Sample baseline results
├── run.py                      # One-command launcher: starts server + serves frontend
├── openenv.yaml                # OpenEnv manifest (10 tasks, full schema)
├── pyproject.toml              # Project metadata and dependencies
└── pytest.ini                  # Test configuration
```

---

## OpenEnv Compliance Checklist

- [x] Built on `Environment` base class (`ProbeEnvironment(Environment)` in `environment/probe_environment.py`)
- [x] `reset()`, `step()`, `state()` all implemented (async-native via `async_reset` / `async_step` / `async_state`; sync wrappers delegate safely via `asyncio.run`)
- [x] `step()` returns `tuple[ObservationType, RewardType, bool, dict]` (see `async_step` in `probe_environment.py`)
- [x] Dedicated `RewardType` Pydantic v2 model with `model_config = ConfigDict(frozen=True)` (`agent/models.py`)
- [x] Valid `openenv.yaml` manifest (spec_version, name, type, runtime, app, port, 10 tasks, observation schema)
- [x] Client/server separation enforced (`agent/` = client models + HTTP client; `environment/` = server logic)
- [x] No reserved MCP tool names used
- [ ] Hosted on HuggingFace Spaces ([FILL: deploy and add URL to links table above])

