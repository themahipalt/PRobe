---
title: PRobe Environment
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - code-review
  - rl-training
  - grpo
  - world-modeling
  - probe
---

# PRobe — Train an Agent to Investigate Code, Not Just Scan It

After training in PRobe, an agent can read a Python pull request, pinpoint real bugs and deliberate security backdoors line-by-line, classify each flaw as an honest mistake or an intentional attack, and know when to escalate to a security team — all from a reward signal with no LLM judge.

---

## The Problem

The XZ Utils backdoor (CVE-2024-3094) slipped through two years of open-source review. SolarWinds compromised 18,000 organisations via a tampered build pipeline. In both cases the malicious change *looked* like a legitimate contribution — the kind of PR that lands in a code-review queue every day.

Today's LLMs scan code like a linter. They find style issues, flag known CVE patterns, and produce plausible-sounding comments. What they don't do is *investigate* — reason about intent, distinguish an honest off-by-one from a planted authentication bypass, or know when to escalate rather than request changes. Reward signals for code generation are everywhere; reward signals for critical code *evaluation* barely exist.

PRobe closes that gap. Its fully deterministic grader — keyword + line-range matching, no LLM judge — separates investigation quality from keyword spam. An agent that dumps every security term at random lines scores *negative*. One that reads carefully, probes for context, finds the right lines, and correctly labels each flaw as an honest bug or a deliberate backdoor scores close to `+1.0`.

---

## What the Agent Sees, Does, and Gets Rewarded For

### Plain English

The agent is handed a Python source file and asked to review it like a senior security engineer. It can annotate suspicious lines, probe specific regions for more context, run a simulated scanner (which, like real tools, misses things and occasionally lies), and finally submit a verdict. On adversarial tasks it must also decide whether the code contains a deliberate backdoor and escalate to a security team if so. Every episode the code surface changes — variable names, line numbers, constants — so the agent cannot memorise answers; it has to read.

### What the Agent Observes (`ProbeObservation`)

| Field | Description |
|---|---|
| `code_snippet` | Mutated Python source for this episode |
| `task_description` | Review instructions and goals |
| `file_name` | Name of the file being reviewed |
| `task_id` / `task_difficulty` | Current task index (0–9) and difficulty label |
| `review_history` | All actions taken so far this episode |
| `step_count` / `max_steps` | Steps used vs. budget |
| `issues_found_count` / `total_issues` | Progress tracker |
| `context_hints` | Causal hints unlocked by finding key issues |
| `reward` | Most recent step reward in `[-1.0, 1.0]` |
| `done` | Whether the episode has ended |

### What Actions the Agent Can Take (`ProbeAction`)

| Action | Effect |
|---|---|
| `add_comment` | Annotate a line with text, severity, category, and optional backdoor classification |
| `get_context` | Reveal ±5 lines of context around a chosen line number |
| `run_scanner` | Invoke simulated static-analysis tool (70 % recall, up to 2 false positives injected) |
| `request_changes` | Mark PR as requiring fixes (correct terminal action for tasks 0–6) |
| `approve` | Approve the PR (penalised if issues remain) |
| `submit_review` | Finalise the review and end the episode |
| `escalate_to_security_review` | Flag PR as containing a deliberate attack (required for tasks 7–9) |

### Reward Formula

Reward accumulates across steps and is finalised at submission:

```
Episode reward =

  Σ per-comment (ADD_COMMENT):
    issue_credit          = (weight_i / total_weight) × 0.40   ← found a real issue
    classification_credit = (weight_i / total_weight) × 0.20   ← correct bug/backdoor label
    misclassify_penalty                               = −0.05   ← found it but labelled it wrong
    false_positive_penalty                            = −0.05   ← substantive comment, no issue matched

  + on terminal (SUBMIT_REVIEW or ESCALATE):
    coverage_bonus   = weighted_coverage × 0.15                 ← proportional to issues found
    decision_score   = +0.15 / −0.15                            ← correct / wrong final action
                       (bonus gated: requires coverage ≥ 30 %)
    efficiency_bonus = (1 − steps_used/max_steps) × 0.10        ← unlocked only if coverage ≥ 60 %

Maximum achievable: ~1.0   Minimum: −1.0
```

### Anti-Exploit Verifier

A comment earns `issue_credit` only when **all three** conditions hold simultaneously:

1. **`keyword_hit`** — at least one issue keyword appears in the comment text
2. **`line_hit`** — `line_number` is within ±2 lines of the declared issue range
3. **`substantive`** — comment body is longer than 15 characters

This closes three common reward-hacking paths: keyword spam (fails `line_hit`), wide-net line fishing (fails `keyword_hit`), and one-word dumps (fails `substantive`). The decision bonus additionally requires weighted coverage ≥ 30 % before it can be earned, so an agent that never reads code and always guesses `request_changes` earns zero — not a bonus.

### Perfect Episode vs. Failing Episode

**Perfect:** The agent reads the code, annotates every real issue at the correct line with a substantive, keyword-bearing comment, correctly labels each as `accidental_bug` or `intentional_backdoor`, escalates when required, and submits with steps to spare. Score approaches `1.0`.

**Failing:** The agent spams generic comments on random lines, never co-locates a keyword with a real issue line, triggers false-positive penalties on every step, and submits the wrong terminal action. Score approaches `−1.0`.

---

## Environment Design

### Difficulty Tiers

| Tier | Tasks | Max Steps | Issues | What Changes |
|---|---|---|---|---|
| **Ultra-Easy** (bootstrap) | 0 | 6 | 2 | Category names spelled out in code comments above each bug; guarantees GRPO positive trajectories from step 1 |
| **Easy** | 1 | 15 | 3 | Clean logic bugs, no hints, no distractors |
| **Medium** | 2, 4 | 20 | 5 | Security and async-concurrency issues; wider surface area |
| **Hard** | 3, 5, 6 | 30–35 | 6–7 | Mixed bug classes; task 6 adds a causal unlock chain |
| **Adversarial** | 7, 8, 9 | 25–35 | 2–3 | Deliberate backdoors requiring `intentional_backdoor` classification and `escalate_to_security_review` |

### Tasks (10 total)

| ID | Difficulty | File | Issues | Domain |
|---|---|---|---|---|
| 0 | Ultra-easy | `bootstrap.py` | 2 | Off-by-one + hardcoded credential (hinted in comments) |
| 1 | Easy | `utils.py` | 3 | Logic bugs, off-by-one, dead code |
| 2 | Medium | `auth.py` | 5 | SQL injection, MD5, eval(), hardcoded creds |
| 3 | Hard | `data_pipeline.py` | 7 | N+1, SSL bypass, thread leak, OOM cache |
| 4 | Medium | `async_worker.py` | 5 | Race condition, missing await, resource leak |
| 5 | Hard | `api_server.py` | 6 | Command injection, path traversal, pickle RCE |
| 6 | Hard | `auth_service.py` | 6 | **Causal chain** — JWT forgery → privilege escalation |
| 7 | Adversarial | `session_manager.py` | 2 | Planted auth bypass + hardcoded bypass credential |
| 8 | Adversarial | `jwt_validator.py` | 2 | **Compound backdoor** — fixed-seed JWT secret + timing oracle |
| 9 | Adversarial | `compat_shim.py` | 3 | **Supply chain** — import-time env-var exfiltration to attacker domain |

### GRPO Curriculum (5 Phases in `train_grpo.py`)

| Phase | Steps | Tasks in Pool |
|---|---|---|
| 0 | 0 – 40 | 0–1 (ultra-easy / easy) |
| 1 | 40 – 80 | 0–3 (adds medium / hard) |
| 2 | 80 – 120 | 0–6 (adds causal chain) |
| 3 | 120 – 160 | 0–8 (adds adversarial) |
| 4 | 160 – 200 | 0–9 (full curriculum) |

### Reward Components with Weights

| Component | Weight | Trigger |
|---|---|---|
| `issue_credit` | up to **0.40** cumulative | `add_comment` matches a real issue (keyword + line + length) |
| `classification_credit` | up to **0.20** cumulative | correct `accidental_bug` / `intentional_backdoor` label |
| `misclassify_penalty` | **−0.05** per issue | issue found but wrong classification label |
| `false_positive_penalty` | **−0.05** per comment | substantive comment, zero issues matched |
| `coverage_bonus` | up to **0.15** terminal | `weighted_coverage × 0.15` |
| `decision_score` | **±0.15** terminal | correct / wrong `request_changes` vs `escalate` decision |
| `efficiency_bonus` | up to **0.10** terminal | `(1 − steps/max_steps) × 0.10` when coverage ≥ 60 % |
| `format_bonus` | **+0.02** once | response contains a valid non-empty JSON array |

### Dynamic World (Anti-Memorisation)

Each episode `mutate_task()` applies three seed-controlled transforms:

| Mutation | Example |
|---|---|
| Variable rename | `total` → `acc`, `data` → `payload`, `password` → `passwd` |
| Line shift | Blank line inserted above first issue; all `line_range` values shift +1 |
| Constant variance | `range(len(data) + 1)` → `range(len(data) + 2)` |

Mutations are deterministic given the episode seed — reproducible runs, always fresh surfaces.

### Scanner Noise Model (`scanner.py`)

`run_scanner()` simulates a real lint/security tool:
- **Recall: 70 %** — each real issue is reported with probability 0.70; ~30 % silently missed
- **False-positive rate: 40 %** — up to 2 injected plausible-but-wrong findings per run
- Scanner output is **not auto-graded** — the agent must still call `add_comment` with a correct line + keyword to earn reward

### Causal Unlock Chain (Task 6)

Finding certain issues appends new context hints to the observation, modelling real investigations where one discovery leads to a deeper one:

```
Find hardcoded JWT secret  →  DB schema revealed  →  agent can reason: forge token → privilege escalation
Find missing rate-limit    →  nginx config shown   →  confirms /auth fully exposed with no IP filtering
```

### OpenEnv Interface

| Method | Returns | Notes |
|---|---|---|
| `reset()` | `ProbeObservation` | Starts new episode; advances task cursor; applies mutation |
| `step(action)` | `(ProbeObservation, RewardType, bool, dict)` | Executes action; returns obs, structured reward, done flag, info dict |
| `state` (sync property) | `State(episode_id, step_count)` | Lightweight snapshot for `create_app` |
| `async_state()` | `dict` | Full async snapshot with all episode fields |

---

## Quickstart

```bash
# Install
uv sync

# Run the environment server
uv run uvicorn environment.app:app --host 0.0.0.0 --port 8000 --reload

# Run zero-shot GPT-4o-mini baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
uv run python training/baseline.py

# Smoke-test reward function (no GPU, no API key)
uv run python training/train_grpo.py --test
```

---

## Training

| | |
|---|---|
| **Training script** | `training/train_grpo.py` · [FILL: link to Colab/Kaggle notebook] |
| **Model** | [FILL: e.g. Qwen2.5-1.5B-Instruct] |
| **Algorithm** | GRPO via HuggingFace TRL + Unsloth (4-bit quantised) |
| **Hardware** | [FILL: e.g. Kaggle T4 x2 / Colab A100] |
| **Duration** | [FILL: e.g. 200 steps, ~3 hours] |

```bash
# Full training — Unsloth 4-bit (recommended, single T4/A10)
uv run python training/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --use-unsloth

# Resume from checkpoint
uv run python training/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --resume-from ./outputs/checkpoint-80
```

---

## Results

### Reward Curve

[FILL: Show Image — outputs/training_curves.png]

[FILL: one-line caption — e.g. "Mean episode reward over 200 GRPO training steps. Trained agent (blue) vs. scripted baseline (orange)."]

### Loss Curve

[FILL: Show Image — outputs/training_curves.png]

[FILL: one-line caption]

### Before vs. After Training

| Metric | Untrained Baseline | Trained Agent |
|---|---|---|
| Average episode reward | [FILL] | [FILL] |
| Verifier pass rate | [FILL] | [FILL] |
| False positive rate | [FILL] | [FILL] |
| Avg rollout length | [FILL] | [FILL] |
| Backdoor classification accuracy | [FILL] | [FILL] |
| Escalation recall (adversarial tasks) | [FILL] | [FILL] |

[FILL: 2–3 sentences describing what qualitatively changed in the agent's behavior after training. What does it do differently?]

---

## Why This Matters

Security code review is a high-stakes task performed by a small number of specialists — it does not scale to the volume of code that modern teams ship. An agent that can reliably read a PR, flag bugs with accurate line references, distinguish honest mistakes from deliberate backdoors (the XZ Utils and SolarWinds failure mode), and escalate with justification would directly accelerate secure software delivery for any team using AI-assisted development. This is also a largely unexplored domain for RL: existing code benchmarks reward *generating* correct outputs, not *critically evaluating* someone else's work, leaving the oversight and adversarial-detection capabilities of LLMs essentially untrained.

---

## Links

| Resource | URL |
|---|---|
| HuggingFace Space (live environment) | [FILL: https://huggingface.co/spaces/...] |
| Training notebook (Colab/Kaggle) | [FILL: https://...] |
| Mini-blog / writeup (HuggingFace) | [FILL: https://huggingface.co/blog/...] |
| Demo video (YouTube, <2 min) | [FILL: https://youtube.com/...] |
| Slides / presentation | [FILL: https://...] |
| WandB training run | [FILL: https://wandb.ai/...] |

---

## Repo Structure

```
.
├── agent/
│   ├── client.py               # HTTP client for interacting with the environment server
│   ├── models.py               # Pydantic models: ProbeAction, ProbeObservation, RewardType
│   └── __init__.py
├── environment/
│   ├── app.py                  # FastAPI server (HTTP + WebSocket: /reset /step /state /ws)
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


