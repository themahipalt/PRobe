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

# PRobe — Teaching Machines to Think Like Security Engineers

## Quick Links for Judges

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1624TxcO3kJXLyDTyhENUH22w81Wa2XIb#scrollTo=krnbsm0fq3dH)

| Resource | Link |
|---|---|
| 🤗 **Live Demo** (try it now) | https://huggingface.co/spaces/mahithakur/PRobe |
| 📓 **Training Code** (Colab) | [Open Notebook](https://colab.research.google.com/drive/1624TxcO3kJXLyDTyhENUH22w81Wa2XIb#scrollTo=krnbsm0fq3dH) |
| 📝 **Blog Post** | [Read on Discussions](https://huggingface.co/spaces/mahithakur/PRobe/discussions/1) |
| 📊 **Results & Data** | [Datasets Hub](https://huggingface.co/datasets/mahithakur/PRobe-training-results) |
| 📈 **Full Report** | [Evaluation Metrics](./reports/JUDGE_REPORT.md) |

---

## What This Is (In Plain English)

Imagine teaching a student to review code like a security expert — not just to find obvious bugs, but to **understand the intent behind the code**. That's what PRobe does. It's a training ground where AI systems learn to review Python code the way a careful, skeptical engineer would.

The key difference from other benchmarks? **PRobe doesn't use an AI judge.** Instead, it uses simple, transparent rules. If you find the right bug on the right line with a clear explanation, you get rewarded. If you spam random keywords or miss the actual problem, you lose points. It's honest, reproducible, and fair.

---

## Why This Matters: The Problem We're Solving

Think about recent security disasters. The **XZ Utils backdoor** and **SolarWinds supply chain attack** had something in common: the malicious code *looked like normal changes*. To anyone scanning for obvious syntax errors or known vulnerabilities, everything seemed fine.

Here's the uncomfortable truth: **Most code review tools are pattern matchers.** They say "here's a potential bug" based on keywords and patterns they've learned. But a deliberate backdoor isn't a pattern. It's an *intention*. It's someone carefully hiding malice inside what looks like a legitimate improvement.

Modern AI systems are better than pattern matchers, but they still struggle with this. They find bugs, sure. But can they spot something that was deliberately hidden? Can they tell the difference between "I made a mistake" and "I embedded a backdoor"? And most importantly, can they **know when to escalate** to a human expert?

PRobe asks these questions directly.

---

## The Approach: Learning Through Feedback

Here's the philosophy behind how PRobe works:

**1. It teaches through real scenarios.** Not abstract examples. You get 10 tasks that simulate actual code review situations. Some are simple bugs. Some are security issues. Some are deliberately hidden backdoors designed to look innocent.

**2. It rewards clarity and precision.** Finding a bug is good. Finding the right bug on the right line with a clear explanation is better. Vague hand-waving gets penalized. This teaches the AI to think carefully, not just make guesses.

**3. It prevents gaming.** Traditional benchmarks often get broken by clever prompt engineering. PRobe uses deterministic grading — the score is based on facts (line number, keywords found, explanation length), not opinion. You can't trick it.

**4. It teaches judgment.** Some code doesn't have bugs — it has danger signs. Maybe the intent is unclear. Maybe the code is suspicious. In these cases, the right answer isn't "approve" or "request changes." It's "escalate to security." PRobe explicitly teaches this.

---

## How It Works: The 60-Second Tour

```bash
# 1. Clone and set up
uv sync
uv run python run.py

# 2. Open in your browser
# Visit http://localhost:8000/ui/ and click "New Episode"
```

You'll see:
- A Python file with 1-3 hidden bugs or security issues
- A rubric explaining what a good review should find
- A space for the model to write its findings
- A score based on accuracy and clarity

Try it yourself first. You'll understand what we're teaching.

---

## What Makes This Different: Three Core Ideas

**Idea 1: Determinism Is a Feature, Not a Limitation**

Most benchmarks use an LLM as a judge. It's flexible, but it's also black-box and expensive. We went the opposite direction: simple rules, full transparency. Your score isn't a mystery. You can look at the grader code and understand exactly why you got that score.

**Idea 2: Prevention Matters More Than Detection**

We don't just test "can you find bugs?" We also test "can you avoid false alarms?" If you claim to find 10 issues but only 3 are real, you don't get full credit. This teaches systems to be *careful*, not just confident.

**Idea 3: Intent Matters**

Code can be wrong by accident or wrong by design. These are different problems. PRobe explicitly teaches the difference and rewards systems that can tell them apart.

---

## The Technical Foundation

### What You're Optimizing

- **Speed:** Find issues quickly
- **Accuracy:** Right issue + right line number
- **Confidence:** Clear, well-reasoned explanations
- **Judgment:** Know when to escalate

### How Learning Happens

We use a technique called **GRPO** (Group Relative Policy Optimization). It's a method where the system learns by comparing its own attempts. "This attempt was better than that one, so let's learn from the difference." It's efficient and works with modest compute.

In our tests, a system trained on 50 code review episodes improved from 60% accuracy to 78% — an 18-point gain. Not perfect, but real progress.

### What's Inside the Box

- **10 carefully designed tasks** — from simple bugs to subtle backdoors
- **A mutation engine** — changes variable names and line numbers so nothing can be memorized
- **Honest grading** — deterministic, transparent scoring
- **A learning loop** — reinforcement learning that rewards careful thinking

---

## Try Training Yourself

You can run the training in Google Colab (free GPU, no setup required):

```bash
# Install
pip install -e ".[training]"

# Quick test (no GPU needed)
python training/train_grpo.py --test

# Full training (uses GPU)
python training/train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 200 \
  --group-size 2 \
  --batch-size 2
```

Results are saved to `outputs/` and visualized in graphs.

---

## What You Get Out

After training, you'll see:
- **Learning curves** — how reward improves over time
- **Per-task improvement** — which types of issues the system learned to spot
- **Concrete examples** — before and after responses to actual code

---

## The Big Picture: Why This Matters

We're at an interesting moment in AI. Systems can now read and reason about code. But reasoning isn't just pattern matching. It's asking "what is the author trying to do?" and "is there something hidden here?"

PRobe is a small experiment in teaching machines to ask these questions. Not perfectly. Not completely. But honestly and transparently.

If this kind of thinking becomes part of code review — human and AI together — then maybe we can catch the next XZ Utils before it ships.

---

## Technical Details

Full architecture in `docs/design.md`

- **Environment:** FastAPI server + WebSocket UI
- **Grader:** Deterministic reward algorithm
- **Trainer:** GRPO using Hugging Face TRL
- **Frontend:** Simple, no build step required

---

## The Structure (If You're Curious)

```
.
├── environment/          # The core: tasks, grader, server
├── agent/               # Client code and models
├── training/            # Learning scripts (GRPO)
├── frontend/            # UI (HTML + JavaScript, no build)
├── tests/               # 88 tests, all passing
├── outputs/             # Training results
├── reports/             # Evaluation metrics
└── run.py              # One-command launcher
```

---

## A Final Thought

Code review is fundamentally about *judgment*. Not just finding errors, but understanding context, questioning intent, and knowing when to ask for help. 

We built PRobe because we think machines should be trained to make better judgments. Not because they'll replace humans, but because they might become better partners to humans who care about security.

Try it. See what you think. The code is open, the grading is transparent, and the results are reproducible.

---

**Questions?** Open a discussion in the Space, or check out the [full blog post](https://huggingface.co/spaces/mahithakur/PRobe/discussions/1).

Made for the OpenEnv Hackathon.
