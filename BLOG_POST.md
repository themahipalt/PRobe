# PRobe: Training an AI Code Reviewer That Spots Backdoors

*An interactive RL environment where models learn to review code like security engineers, not linters.*

## The Problem: Supply Chain Attacks Look Like Normal Code

Recent attacks like [XZ Utils](https://en.wikipedia.org/wiki/XZ_Utils_backdoor) and [SolarWinds](https://www.microsoft.com/en-us/security/blog/2020/12/18/analyzing-solorigate-samples-using-microsoft-defender-for-endpoint/) remind us that **malicious code can hide in plain sight**. A backdoor often looks like a legitimate refactor or bug fix—it's indistinguishable from normal changes without understanding *intent*.

Current LLM code reviewers excel at spotting obvious bugs (undefined variables, off-by-one errors) but struggle with intentional sabotage. They're pattern matchers, not investigators.

## Our Solution: PRobe — A Deterministic Training Environment

PRobe is a **Python code review environment** where AI agents learn to:

1. **Find real bugs and security issues** — with accurate line numbers
2. **Distinguish honest mistakes from deliberate backdoors** — and decide when to escalate
3. **Explain findings precisely** — vague answers get penalized

Unlike many "LLM judge" benchmarks, PRobe uses **deterministic, reproducible rewards**. No expensive API calls to grade submissions. No gaming via keyword spam.

### What Makes PRobe Different

| Feature | Traditional Benchmarks | PRobe |
|---------|----------------------|-------|
| **Reward** | LLM evaluator (variable, slow, expensive) | Deterministic algorithm (fast, reproducible) |
| **Feedback** | "Good job!" or "Try again" | Specific penalties for imprecision |
| **Anti-gaming** | None | Keyword spam on wrong lines penalized |
| **Tasks** | Curated examples | 10 procedurally generated scenarios |
| **Escalation** | Find bugs OR approve | Find bugs OR request changes OR escalate to security |

## How It Works (60 seconds)

```bash
# 1. Clone and setup
git clone https://huggingface.co/spaces/themahipalthakur/PRobe
cd PRobe
uv sync

# 2. Run locally
uv run python run.py

# 3. Open browser
# Visit http://localhost:8000/ui/ and click "New Episode"
```

The UI shows:
- A **Python file** with 1-3 injected bugs or backdoors
- A grading rubric explaining what the model should find
- The model's response (approval, change request, or escalation)
- A **score** based on accuracy and confidence

## Technical Architecture

### The Environment (`environment.py`)

- **10 tasks** with progressively harder scenarios:
  - Tasks 0-2: Simple bugs (off-by-one, undefined var, logic error)
  - Tasks 3-5: Security issues (hardcoded API keys, disabled SSL, SQL injection)
  - Tasks 6-9: Adversarial backdoors (hiding intent via indirection, obfuscation)

- **Mutation system**: Variable names and line numbers shift each episode, preventing memorization
- **Causal unlocks**: Context hints appear only after the model finds related issues
- **Episode memory**: Models can reference previous findings within an episode

### The Grader (`grader.py`)

Scores are deterministic:

```
Base reward = % of issues found + % correct line numbers + quality bonus
Penalties for:
  - Vague explanations (< 10 words per issue)
  - False positives (claiming issues that don't exist)
  - Lazy submissions (category-only, no detail)
```

### Training with GRPO

We use **Group Relative Policy Optimization** (GRPO, from DeepSeek-R1) to train code reviewers:

```python
# Training on 50 episodes per task
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
)
trainer.train()
```

**Why GRPO?**
- Doesn't require a reference model (unlike PPO)
- Groups examples for relative reward comparison
- Works well with small compute budgets

## Results: Before & After Training

Our baseline (GPT-4o-mini) finds ~60% of issues. After GRPO training on 50 episodes:

- **Final 10 episodes avg**: 78% accuracy
- **Improvement**: +18 percentage points
- **Training time**: ~2 hours on single GPU

See full metrics in `reports/JUDGE_REPORT.md`.

## Try It Now

- **Live Space**: https://huggingface.co/spaces/themahipalthakur/PRobe
- **Training Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/FILL_COLAB_LINK)
- **Code**: https://github.com/themahipalthakur/PRobe

## Why This Matters

Security reviews are expensive and rare. PRobe demonstrates that AI can be trained to **reason about intent**, not just pattern-match. The deterministic grader ensures:

- ✅ Reproducible results
- ✅ No gaming via prompt engineering
- ✅ Transparent scoring (anyone can verify the grade)

This is a step toward AI systems that integrate into real security workflows—not because they're perfect, but because they're *verifiable*.

---

**Made for the [OpenEnv Hackathon](https://huggingface.co/spaces/open-env/open-env-hackers)**

Questions? Open an issue on [GitHub](https://github.com/themahipalthakur/PRobe).
