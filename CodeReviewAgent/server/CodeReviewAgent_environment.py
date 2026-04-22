"""
CodeReviewAgent Environment Implementation.

Simulates a real-world pull-request code review workflow:
  1. Agent reads code and task description (Observation).
  2. Agent calls ADD_COMMENT for each issue it finds (partial rewards).
  3. Agent sets a decision with REQUEST_CHANGES or APPROVE.
  4. Agent ends the episode with SUBMIT_REVIEW (terminal reward).

Tasks cycle automatically on each reset: easy → medium → hard → easy …
"""

from typing import Any, Dict, List
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionType,
        CodereviewagentAction,
        CodereviewagentObservation,
        IssueCategory,
    )
    from .grader import CodeReviewGrader
    from .tasks import TASKS
except ImportError:
    from models import (  # type: ignore[no-redef]
        ActionType,
        CodereviewagentAction,
        CodereviewagentObservation,
        IssueCategory,
    )
    from server.grader import CodeReviewGrader  # type: ignore[no-redef]
    from server.tasks import TASKS  # type: ignore[no-redef]


class CodereviewagentEnvironment(Environment):
    """
    OpenEnv-compliant code review environment.

    Three tasks of increasing difficulty are cycled through on each reset.
    Reward is provided both during the episode (per found issue) and at
    the terminal step (coverage + decision quality + efficiency).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._reset_count = 0
        # Pre-populate _ep with task 0 so step() is safe before an explicit reset()
        task = TASKS[0]
        self._grader = CodeReviewGrader(task)
        self._ep: Dict[str, Any] = {
            "task": task,
            "review_comments": [],
            "issues_found": [],
            "review_decision": None,
            "review_submitted": False,
            "cumulative_reward": 0.0,
        }

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(self) -> CodereviewagentObservation:
        task_id = self._reset_count % len(TASKS)
        self._reset_count += 1
        self._episode_id = str(uuid4())
        self._step_count = 0

        task = TASKS[task_id]
        self._grader = CodeReviewGrader(task)
        self._ep = {
            "task": task,
            "review_comments": [],
            "issues_found": [],
            "review_decision": None,
            "review_submitted": False,
            "cumulative_reward": 0.0,
        }
        return self._make_obs(reward=0.0, done=False)

    def step(self, action: CodereviewagentAction) -> CodereviewagentObservation:  # type: ignore[override]
        self._step_count += 1
        task = self._ep["task"]
        reward = 0.0
        done = False
        explanation = ""

        if action.action_type == ActionType.ADD_COMMENT:
            reward, explanation = self._handle_add_comment(action)

        elif action.action_type == ActionType.REQUEST_CHANGES:
            reward, explanation = self._handle_request_changes(action)

        elif action.action_type == ActionType.APPROVE:
            reward, explanation = self._handle_approve()

        elif action.action_type == ActionType.SUBMIT_REVIEW:
            reward, explanation, done = self._handle_submit_review()

        # Step-budget exhaustion
        if not done and self._step_count >= task["max_steps"]:
            reward -= 0.05
            explanation += " [Step limit reached.]"
            done = True

        self._ep["cumulative_reward"] = round(
            self._ep["cumulative_reward"] + reward, 4
        )
        return self._make_obs(reward=round(max(-1.0, min(1.0, reward)), 4), done=done)

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self._step_count)

    # ── Action handlers ───────────────────────────────────────────────────

    def _handle_add_comment(
        self, action: CodereviewagentAction
    ) -> tuple[float, str]:
        entry = {
            "type": "comment",
            "line": action.line_number,
            "text": action.comment,
            "severity": action.severity.value if action.severity else None,
            "category": action.category.value if action.category else None,
        }
        self._ep["review_comments"].append(entry)

        score, new_finds = self._grader.score_comment(  # type: ignore[union-attr]
            line_number=action.line_number,
            comment=action.comment,
            category=action.category.value if action.category else None,
            already_found=self._ep["issues_found"],
        )
        self._ep["issues_found"].extend(new_finds)

        if score > 0:
            return score, f"Issue(s) identified: {new_finds}"
        # Penalise substantive but incorrect comments (false positives)
        if action.comment and len(action.comment.strip()) > 15:
            return -0.02, "Comment did not match any known issue."
        return 0.0, "Comment recorded."

    def _handle_request_changes(
        self, action: CodereviewagentAction
    ) -> tuple[float, str]:
        self._ep["review_decision"] = "request_changes"
        self._ep["review_comments"].append(
            {"type": "request_changes", "text": action.comment}
        )
        if self._ep["issues_found"]:
            return 0.05, "Request-changes decision recorded after finding issues."
        return -0.05, "Request-changes with no issues identified yet."

    def _handle_approve(self) -> tuple[float, str]:
        self._ep["review_decision"] = "approve"
        total = len(self._ep["task"]["issues"])
        found = len(set(self._ep["issues_found"]))
        if total > 0 and found < total * 0.5:
            return -0.15, "Approved despite significant unreviewed issues."
        return 0.02, "Approval recorded."

    def _handle_submit_review(self) -> tuple[float, str, bool]:
        if self._ep.get("review_submitted"):
            return -0.05, "Review already submitted.", False
        self._ep["review_submitted"] = True
        task = self._ep["task"]
        result = self._grader.final_score(  # type: ignore[union-attr]
            issues_found=list(set(self._ep["issues_found"])),
            review_decision=self._ep.get("review_decision"),
            step_count=self._step_count,
            max_steps=task["max_steps"],
        )
        return result["value"], result["explanation"], True

    # ── Observation builder ───────────────────────────────────────────────

    def _make_obs(self, reward: float, done: bool) -> CodereviewagentObservation:
        task = self._ep["task"]
        return CodereviewagentObservation(
            code_snippet=task["code"],
            task_description=task["description"],
            file_name=task["file_name"],
            task_id=task["id"],
            task_difficulty=task["difficulty"],
            review_history=list(self._ep.get("review_comments", [])),
            step_count=self._step_count,
            max_steps=task["max_steps"],
            issues_found_count=len(set(self._ep.get("issues_found", []))),
            total_issues=len(task["issues"]),
            done=done,
            reward=reward,
            metadata={
                "cumulative_reward": self._ep.get("cumulative_reward", 0.0),
                "review_decision": self._ep.get("review_decision"),
            },
        )
