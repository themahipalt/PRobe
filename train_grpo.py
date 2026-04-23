"""
GRPO Training Script for CodeReviewAgent
=========================================
Trains a causal LM to review Python code using Group Relative Policy Optimization (GRPO).

The model is given a code snippet and must output a JSON array of issues.
The reward function compares the model's output to ground-truth issues using
the same keyword + line-range matching as the OpenEnv environment grader.

Usage:
    # Install dependencies first:
    pip install trl>=0.12 transformers datasets accelerate unsloth

    # Train:
    python train_grpo.py

    # Override model or output dir:
    MODEL_ID=Qwen/Qwen2.5-3B-Instruct OUTPUT_DIR=./my_run python train_grpo.py

Environment variables:
    MODEL_ID     HuggingFace model to fine-tune (default: Qwen/Qwen2.5-1.5B-Instruct)
    OUTPUT_DIR   Where to save checkpoints and final model (default: ./grpo_output)
    NUM_EPOCHS   Training epochs (default: 3)
    USE_UNSLOTH  Set to "1" to use Unsloth for faster training (default: 0)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

# ── allow running from project root without installing the package ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CodeReviewAgent"))

from CodeReviewAgent.server.tasks import TASKS

# ── config ────────────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./grpo_output")
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
USE_UNSLOTH = os.environ.get("USE_UNSLOTH", "0") == "1"
NUM_GENERATIONS = 4   # completions sampled per prompt during GRPO
MAX_NEW_TOKENS = 600  # max tokens in model output
REPEAT_DATASET = 8    # repeat tasks N times to get more training steps

SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the provided Python code carefully and \
identify ALL issues including bugs, security vulnerabilities, performance problems, \
and design issues.

Output your review as a JSON array. Each element must be an object with these fields:
  "line"     : integer — the primary line number of the issue
  "category" : one of "bug", "security", "performance", "style", "design"
  "severity" : one of "info", "warning", "error", "critical"
  "comment"  : string — clear description of the problem and how to fix it

Output ONLY valid JSON (no markdown fences, no extra text).\
"""


# ── dataset helpers ───────────────────────────────────────────────────────────

def make_user_message(task: dict[str, Any]) -> str:
    return (
        f"File: {task['file_name']}\n"
        f"Task: {task['description']}\n\n"
        f"```python\n{task['code']}\n```\n\n"
        "Provide your review as a JSON array of issues:"
    )


def build_dataset() -> "datasets.Dataset":
    from datasets import Dataset

    rows: list[dict] = []
    for task in TASKS:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(task)},
        ]
        rows.append({"prompt": prompt, "task_id": task["id"]})

    rows = rows * REPEAT_DATASET
    return Dataset.from_list(rows)


# ── reward function ───────────────────────────────────────────────────────────

def _parse_json_output(text: str) -> list[dict]:
    """Extract JSON array from model output, stripping optional markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array")
    return parsed


def score_review(completion: str, task: dict[str, Any]) -> float:
    """
    Score a single model completion against ground-truth issues.

    Reward breakdown (matches OpenEnv grader logic):
      +weight/total_weight * 0.70   per correctly identified issue (max 0.70)
      -0.02                         per false-positive comment (>15 chars, no match)
      max reward clamped to [−0.20, 1.0]
    """
    try:
        issues_found = _parse_json_output(completion)
    except (json.JSONDecodeError, ValueError):
        return -0.10  # malformed JSON

    total_weight: float = sum(iss["weight"] for iss in task["issues"])
    found_ids: set[str] = set()
    false_positives = 0

    for found in issues_found:
        if not isinstance(found, dict):
            continue
        comment = str(found.get("comment", "")).lower()
        line = found.get("line")
        category = found.get("category")

        matched = False
        for gt in task["issues"]:
            if gt["id"] in found_ids:
                continue
            kw_hit = any(kw.lower() in comment for kw in gt["keywords"])
            start, end = gt["line_range"]
            line_hit = line is not None and (start - 3) <= int(line) <= (end + 3)
            cat_hit = category == gt["category"]

            if kw_hit and (line_hit or cat_hit):
                found_ids.add(gt["id"])
                matched = True
                break

        if not matched and len(comment) > 15:
            false_positives += 1

    found_weight = sum(
        gt["weight"] for gt in task["issues"] if gt["id"] in found_ids
    )
    coverage = found_weight / total_weight if total_weight > 0 else 0.0
    reward = coverage * 0.70 - false_positives * 0.02
    return float(max(-0.20, min(1.0, reward)))


def reward_fn(completions: list[str], task_id: list[int], **_kwargs) -> list[float]:
    """
    GRPO reward function.

    Receives a batch of completions and the corresponding task_id column
    from the dataset. Returns a float reward for each completion.
    """
    rewards = []
    for completion, tid in zip(completions, task_id):
        task = TASKS[tid % len(TASKS)]
        rewards.append(score_review(completion, task))
    return rewards


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str):
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel  # type: ignore[import]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Model      : {MODEL_ID}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Epochs     : {NUM_EPOCHS}")
    print(f"Unsloth    : {USE_UNSLOTH}")
    print(f"Tasks      : {len(TASKS)}")
    print()

    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    dataset = build_dataset()
    print(f"Dataset size: {len(dataset)} rows ({len(TASKS)} tasks × {REPEAT_DATASET} repeats)")

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        max_prompt_length=1024,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",   # change to "wandb" if you want W&B logging
        bf16=True,
        temperature=0.9,
        top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    print("Starting GRPO training …")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")


# ── quick smoke-test (no GPU needed) ─────────────────────────────────────────

def _run_reward_smoke_test() -> None:
    """Verify reward_fn works correctly before training."""
    print("Running reward smoke test …")

    # Perfect review for task 0 (easy)
    task0 = TASKS[0]
    perfect_review = json.dumps([
        {"line": 4, "category": "bug", "severity": "error",
         "comment": "Off-by-one error: range(len(numbers) + 1) causes IndexError on the last iteration."},
        {"line": 7, "category": "style", "severity": "info",
         "comment": "unused_result is assigned but never used anywhere."},
        {"line": 17, "category": "bug", "severity": "error",
         "comment": "max_val == item uses == (comparison) instead of = (assignment); max is never updated."},
    ])
    r = score_review(perfect_review, task0)
    assert r > 0.60, f"Perfect review should score > 0.6, got {r}"
    print(f"  Perfect review reward  : {r:.4f}  ✓")

    # Malformed JSON
    r_bad = score_review("this is not json", task0)
    assert r_bad < 0, f"Bad JSON should give negative reward, got {r_bad}"
    print(f"  Malformed JSON reward  : {r_bad:.4f}  ✓")

    # Empty array (missed all issues)
    r_empty = score_review("[]", task0)
    assert r_empty == 0.0, f"Empty array should score 0.0, got {r_empty}"
    print(f"  Empty array reward     : {r_empty:.4f}  ✓")

    print("Smoke test passed.\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_reward_smoke_test()
    else:
        main()
