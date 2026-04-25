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
| 🤗 HuggingFace Space (live environment) | Replace with your HF Space URL |
| 📓 Training notebook (Colab / Kaggle) | Replace with your Colab or Kaggle link |
| 📝 Mini-blog / writeup (HuggingFace) | Replace with your HF blog post URL |
| 🎥 Demo video (YouTube, < 2 min) | Replace with your YouTube URL |
| 📊 Slides / presentation | Replace with your slides URL |
| 📈 WandB training run | Replace with your WandB run URL |

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

### Latest measured run (Google Colab, 100 steps)

These numbers were captured from a Colab run summary (not from GitHub artifacts):

```
==================================================
  COLAB 100-STEP FINAL RESULTS
==================================================
  Total steps      : 100
  Avg reward       : 0.395
  Best reward      : 0.787
  First 25 avg     : 0.407
  Last 25 avg      : 0.418
  Improvement      : +0.011
  Oracle ceiling   : 0.778
  % of oracle      : 50.7%

  Random baseline  : -0.260
  Your model avg   : 0.395
  vs random        : +0.655
```

Quick scan for judges:

- **Mean reward (100 steps)**: **0.395**
- **Best reward**: **0.787**
- **First 25% vs last 25%**: **0.407 → 0.418** (**+0.011**)
- **vs oracle ceiling (0.778)**: **50.7%**
- **vs random baseline (-0.260)**: **+0.655**

> Note: If you did not download/zip outputs from Colab before the runtime ended, the plot PNGs may be gone. The numbers above are still valid if you saved them in your notebook output.

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

