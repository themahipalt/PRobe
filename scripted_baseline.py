"""
Scripted (rule-based) baseline for CodeReviewAgent.

Uses Python AST + regex patterns to flag issues deterministically — no LLM.
Purpose: stress-test the grader (verify it catches correct flags and ignores
spurious ones) before running expensive LLM rollouts.

Usage:
    python scripted_baseline.py                  # all tasks
    python scripted_baseline.py --task-id 2      # single task
    python scripted_baseline.py --output logs/scripted.jsonl
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CodeReviewAgent"))

from CodeReviewAgent.server.CodeReviewAgent_environment import CodereviewagentEnvironment
from CodeReviewAgent.models import ActionType, CodereviewagentAction, IssueCategory, Severity


# ── Issue detection rules ─────────────────────────────────────────────────────

@dataclass
class DetectedIssue:
    line: int
    comment: str
    severity: str
    category: str


def _detect_issues(code: str) -> list[DetectedIssue]:
    """
    Apply all rules to `code` and return a deduplicated list of detected issues.
    Each rule is independent; duplicates on the same line are merged.
    """
    issues: list[DetectedIssue] = []
    seen_lines: set[int] = set()

    def _add(line: int, comment: str, severity: str, category: str) -> None:
        if line in seen_lines:
            return
        seen_lines.add(line)
        issues.append(DetectedIssue(line=line, comment=comment,
                                     severity=severity, category=category))

    lines = code.splitlines()

    # ── Rule 1: off-by-one — range(len(x) + 1) ───────────────────────────
    pat_obo = re.compile(r"range\s*\(\s*len\s*\([^)]+\)\s*\+\s*1\s*\)")
    for i, ln in enumerate(lines, 1):
        if pat_obo.search(ln):
            _add(i, "Off-by-one error: range(len(x) + 1) iterates one index too far, causing IndexError on the last iteration.", "error", "bug")

    # ── Rule 2: hardcoded credentials ─────────────────────────────────────
    # Match: CRED_VAR = "some-string"  where CRED_VAR contains password/secret/key/token
    pat_cred = re.compile(
        r'(?:password|passwd|secret|api_key|apikey|token|credential|db_pass)\s*=\s*["\'][^"\']{4,}["\']',
        re.IGNORECASE,
    )
    for i, ln in enumerate(lines, 1):
        if pat_cred.search(ln):
            _add(i, "Hardcoded credential detected — move to environment variable (os.environ).", "critical", "security")

    # ── Rule 3: eval() call ───────────────────────────────────────────────
    pat_eval = re.compile(r'\beval\s*\(')
    for i, ln in enumerate(lines, 1):
        if pat_eval.search(ln):
            _add(i, "eval() on untrusted input allows arbitrary code execution (RCE).", "critical", "security")

    # ── Rule 4: pickle.loads ──────────────────────────────────────────────
    pat_pickle = re.compile(r'\bpickle\.loads\s*\(')
    for i, ln in enumerate(lines, 1):
        if pat_pickle.search(ln):
            _add(i, "pickle.loads() on untrusted data allows arbitrary code execution (insecure deserialization).", "critical", "security")

    # ── Rule 5: hashlib.md5 ───────────────────────────────────────────────
    pat_md5 = re.compile(r'\bhashlib\.md5\s*\(')
    for i, ln in enumerate(lines, 1):
        if pat_md5.search(ln):
            _add(i, "hashlib.md5 is cryptographically broken for password storage — use bcrypt, argon2, or pbkdf2.", "error", "security")

    # ── Rule 6: SSL verify=False ──────────────────────────────────────────
    pat_ssl = re.compile(r'verify\s*=\s*False')
    for i, ln in enumerate(lines, 1):
        if pat_ssl.search(ln):
            _add(i, "SSL certificate verification disabled (verify=False) — enables MITM attacks.", "error", "security")

    # ── Rule 7: subprocess shell=True ────────────────────────────────────
    pat_shell = re.compile(r'shell\s*=\s*True')
    for i, ln in enumerate(lines, 1):
        if pat_shell.search(ln):
            _add(i, "subprocess with shell=True and unsanitised input allows OS command injection.", "critical", "security")

    # ── Rule 8: path traversal — os.path.join without abspath guard ───────
    pat_path = re.compile(r'os\.path\.join\s*\(')
    pat_abspath = re.compile(r'os\.path\.(?:abspath|realpath)')
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "path"
                        and func.attr == "join"):
                    join_line = node.lineno
                    # Look for an abspath/realpath within 5 lines after
                    window = "\n".join(lines[join_line - 1:join_line + 4])
                    if not pat_abspath.search(window):
                        _add(join_line,
                             "os.path.join without os.path.abspath/realpath guard allows path traversal (../../../etc/passwd).",
                             "critical", "security")
    except SyntaxError:
        # Fall back to regex if code has syntax errors (untested snippets)
        for i, ln in enumerate(lines, 1):
            if pat_path.search(ln):
                ctx = "\n".join(lines[max(0, i - 1):min(len(lines), i + 4)])
                if not pat_abspath.search(ctx):
                    _add(i, "os.path.join without path normalisation — possible path traversal.", "critical", "security")

    # ── Rule 9: assignment operator bug (== instead of =) in loop body ────
    # Heuristic: `identifier == identifier` on its own line (no if/while/assert)
    pat_assign_bug = re.compile(r'^\s{4,}(\w+)\s*==\s*(\w+)\s*$')
    for i, ln in enumerate(lines, 1):
        if pat_assign_bug.match(ln):
            _add(i, f"Possible assignment bug: '{ln.strip()}' uses == (comparison) instead of = (assignment).", "error", "bug")

    # ── Rule 10: missing await — async func calling coroutine without await ─
    pat_async_def = re.compile(r'^\s*async\s+def\s+')
    pat_no_await = re.compile(r'^\s{4,}(\w+)\s*=\s*(\w+)\s*\(')  # assignment without await
    in_async_fn = False
    for i, ln in enumerate(lines, 1):
        if pat_async_def.match(ln):
            in_async_fn = True
        elif re.match(r'^\s*def\s+', ln):
            in_async_fn = False
        if in_async_fn and pat_no_await.match(ln) and "await " not in ln and "async " not in ln:
            # Only flag if RHS looks like a coroutine call (lowercase function name)
            m = pat_no_await.match(ln)
            if m and m.group(2)[0].islower() and m.group(2) not in {"len", "str", "int", "list", "dict"}:
                _add(i, f"Possible missing await: '{ln.strip()}' in async function may be returning an unawaited coroutine.", "critical", "bug")

    # ── Rule 11: thread not joined ────────────────────────────────────────
    pat_thread_start = re.compile(r'\.start\s*\(\s*\)')
    pat_thread_join = re.compile(r'\.join\s*\(\s*\)')
    thread_start_lines: list[int] = []
    for i, ln in enumerate(lines, 1):
        if pat_thread_start.search(ln):
            thread_start_lines.append(i)
    full_code = "\n".join(lines)
    if thread_start_lines and not pat_thread_join.search(full_code):
        for ln_no in thread_start_lines:
            _add(ln_no, "Thread started but never joined or tracked — resource leak; exceptions in thread are silently swallowed.", "error", "bug")

    # ── Rule 12: SQL injection — f-string in SQL-like query ───────────────
    pat_sql = re.compile(r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE|WHERE).*\{', re.IGNORECASE)
    for i, ln in enumerate(lines, 1):
        if pat_sql.search(ln):
            _add(i, "SQL injection: f-string interpolation in SQL query — use parameterized queries with placeholders.", "critical", "security")

    # ── Rule 13: unbounded in-memory cache ───────────────────────────────
    pat_cache_assign = re.compile(r'self\.\w*cache\w*\s*=\s*\{\}')
    for i, ln in enumerate(lines, 1):
        if pat_cache_assign.search(ln):
            _add(i, "Unbounded in-memory cache (self.cache = {}) will grow indefinitely — use functools.lru_cache or set a max size.", "warning", "design")

    # Sort by line number
    issues.sort(key=lambda x: x.line)
    return issues


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_scripted_episode(
    task_id: int,
    episode_idx: int,
    jsonl_file,
) -> dict:
    env = CodereviewagentEnvironment()

    for _ in range(task_id + 1):
        obs = await env.async_reset()

    total_issues = obs.total_issues
    episode_id = obs.metadata.get("episode_id", "?")

    print(f"\n{'─'*60}")
    print(f"Task {task_id} [{obs.task_difficulty}]  {obs.file_name}  ({total_issues} issues) [scripted]")
    print(f"{'─'*60}")

    detected = _detect_issues(obs.code_snippet)
    print(f"  Detector found {len(detected)} candidate issue(s)")

    cumulative_reward = 0.0
    step = 0

    # ADD_COMMENT for each detected issue
    for issue in detected:
        action = CodereviewagentAction(
            action_type=ActionType.ADD_COMMENT,
            line_number=issue.line,
            comment=issue.comment,
            severity=Severity(issue.severity),
            category=IssueCategory(issue.category),
        )
        obs, reward_obj, done, info = await env.async_step(action)
        step += 1
        cumulative_reward += reward_obj.total

        step_record = {
            "agent": "scripted",
            "episode_id": episode_id,
            "episode_idx": episode_idx,
            "task_id": task_id,
            "task_difficulty": obs.task_difficulty,
            "step": step,
            "action_type": "add_comment",
            "line_number": issue.line,
            "comment": issue.comment[:120],
            "reward_total": reward_obj.total,
            "reward_components": reward_obj.components,
            "reward_passed": reward_obj.passed,
            "cumulative_reward": round(cumulative_reward, 4),
            "issues_found": obs.issues_found_count,
            "total_issues": total_issues,
            "done": done,
        }
        if jsonl_file:
            jsonl_file.write(json.dumps(step_record) + "\n")
            jsonl_file.flush()

        print(
            f"  {step:2d}. add_comment line={issue.line:<4}"
            f" reward={reward_obj.total:+.4f}"
            f"  cum={cumulative_reward:+.4f}"
            f"  {'HIT' if reward_obj.passed else '---'}"
        )
        if done:
            break

    if not done:
        # REQUEST_CHANGES if issues found, else APPROVE
        if obs.issues_found_count > 0:
            action = CodereviewagentAction(
                action_type=ActionType.REQUEST_CHANGES,
                comment="Scripted agent: found issues requiring changes.",
            )
        else:
            action = CodereviewagentAction(action_type=ActionType.APPROVE)
        obs, reward_obj, done, _ = await env.async_step(action)
        step += 1
        cumulative_reward += reward_obj.total
        print(f"  {step:2d}. {action.action_type.value:<20} reward={reward_obj.total:+.4f}")

    if not done:
        action = CodereviewagentAction(action_type=ActionType.SUBMIT_REVIEW)
        obs, reward_obj, done, _ = await env.async_step(action)
        step += 1
        cumulative_reward += reward_obj.total
        print(f"  {step:2d}. submit_review              reward={reward_obj.total:+.4f}  [terminal]")

    coverage = obs.issues_found_count / total_issues if total_issues else 0.0
    print(f"\n  → cumulative={cumulative_reward:+.4f}  coverage={coverage:.0%}  steps={step}")

    return {
        "agent": "scripted",
        "episode_idx": episode_idx,
        "task_id": task_id,
        "difficulty": obs.task_difficulty,
        "file_name": obs.file_name,
        "cumulative_reward": round(cumulative_reward, 4),
        "steps": step,
        "issues_found": obs.issues_found_count,
        "total_issues": total_issues,
        "coverage": round(coverage, 3),
        "detector_candidates": len(detected),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    print("CodeReviewAgent Scripted Baseline (AST + regex, no LLM)")

    from CodeReviewAgent.server.tasks import TASKS
    task_ids = [args.task_id] if args.task_id is not None else list(range(len(TASKS)))

    jsonl_file = open(args.output, "a") if args.output else None  # noqa: SIM115
    all_results: list[dict] = []

    try:
        for task_id in task_ids:
            result = await run_scripted_episode(task_id, 0, jsonl_file)
            all_results.append(result)
    finally:
        if jsonl_file:
            jsonl_file.close()

    avg_reward = sum(r["cumulative_reward"] for r in all_results) / len(all_results)
    avg_coverage = sum(r["coverage"] for r in all_results) / len(all_results)

    print(f"\n{'═'*65}")
    print("SCRIPTED BASELINE SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Task':<6} {'Diff':<12} {'Reward':>8} {'Coverage':>10} {'Cands':>6}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(
            f"  {r['task_id']:<6} {r['difficulty']:<12}"
            f" {r['cumulative_reward']:>+8.4f} {r['coverage']:>9.0%}"
            f"  {r['detector_candidates']:>5}"
        )
    print(f"  {'─'*50}")
    print(f"  {'avg':<19} {avg_reward:>+8.4f} {avg_coverage:>9.0%}")

    if args.output:
        print(f"\nStep log → {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scripted (rule-based) baseline for CodeReviewAgent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-id", type=int, default=None, dest="task_id",
                        help="Run a single task (default: all)")
    parser.add_argument("--output", default=None, help="JSONL path for per-step logs")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(_parse_args()))
