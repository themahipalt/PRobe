"""
Programmatic grader for CodeReviewAgent tasks.

Scoring design
--------------
During the episode (ADD_COMMENT actions):
  +weight/total_weight * 0.6   per newly found issue  (max 0.6 across all issues)
  -0.02                        per false-positive comment (substantive but wrong)

Final score (SUBMIT_REVIEW):
  +coverage * 0.20             coverage bonus (0–0.20)
  +/-0.10                      correct vs incorrect final decision
  +efficiency * 0.10           step-efficiency bonus when coverage ≥ 60 %

Maximum achievable total ≈ 1.0  (0.6 + 0.20 + 0.10 + 0.10)
"""

from typing import Any, Dict, List, Tuple


class CodeReviewGrader:
    def __init__(self, task: Dict[str, Any]) -> None:
        self.task = task
        self.total_weight: float = sum(iss["weight"] for iss in task["issues"])

    # ── Per-comment scoring ───────────────────────────────────────────────

    def score_comment(
        self,
        line_number: int | None,
        comment: str | None,
        category: str | None,
        already_found: List[str],
    ) -> Tuple[float, List[str]]:
        """
        Score an ADD_COMMENT action.

        Returns (reward_delta, list_of_newly_found_issue_ids).
        A comment "finds" an issue when:
          - at least one issue keyword appears in the comment text, AND
          - the line number is within ±3 of the issue's line range
            OR the category matches the issue's category.
        """
        if not comment:
            return 0.0, []

        comment_lower = comment.lower()
        newly_found: List[str] = []
        total_reward = 0.0

        for issue in self.task["issues"]:
            if issue["id"] in already_found:
                continue

            keyword_hit = any(kw.lower() in comment_lower for kw in issue["keywords"])
            line_hit = self._line_in_range(line_number, issue["line_range"])
            category_hit = category is not None and category == issue["category"]

            if keyword_hit and line_hit:
                reward = (issue["weight"] / self.total_weight) * 0.6
                newly_found.append(issue["id"])
                total_reward += reward

        return round(total_reward, 4), newly_found

    # ── Terminal scoring ──────────────────────────────────────────────────

    def final_score(
        self,
        issues_found: List[str],
        review_decision: str | None,
        step_count: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Compute the terminal reward component on SUBMIT_REVIEW.

        Returns a dict with keys: value, breakdown, explanation.
        """
        unique_found = list(set(issues_found))
        found_weight = sum(
            iss["weight"]
            for iss in self.task["issues"]
            if iss["id"] in unique_found
        )
        coverage = found_weight / self.total_weight if self.total_weight > 0 else 0.0

        correct_decision = self.task.get("correct_decision", "request_changes")
        decision_score = 0.10 if review_decision == correct_decision else -0.10

        efficiency = max(0.0, 1.0 - step_count / max_steps)
        efficiency_bonus = round(0.10 * efficiency, 4) if coverage >= 0.60 else 0.0

        coverage_bonus = round(coverage * 0.20, 4)
        total = round(
            max(-1.0, min(1.0, coverage_bonus + decision_score + efficiency_bonus)), 4
        )

        breakdown = {
            "coverage_bonus": coverage_bonus,
            "decision_score": round(decision_score, 4),
            "efficiency_bonus": efficiency_bonus,
        }
        explanation = (
            f"Found {len(unique_found)}/{len(self.task['issues'])} issues "
            f"(weighted coverage {coverage:.0%}). "
            f"Decision '{review_decision}' was "
            f"{'correct' if review_decision == correct_decision else 'incorrect'}. "
            f"Used {step_count}/{max_steps} steps."
        )
        return {"value": total, "breakdown": breakdown, "explanation": explanation}

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _line_in_range(
        line_number: int | None,
        line_range: Tuple[int, int],
        tolerance: int = 3,
    ) -> bool:
        if line_number is None:
            return False
        start, end = line_range
        return (start - tolerance) <= line_number <= (end + tolerance)
