"""
Data models for the CodeReviewAgent Environment.

An agent reviews Python source files, identifies bugs, security issues,
and design problems, then submits a structured review.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ActionType(str, Enum):
    ADD_COMMENT = "add_comment"
    REQUEST_CHANGES = "request_changes"
    APPROVE = "approve"
    SUBMIT_REVIEW = "submit_review"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DESIGN = "design"


class CodereviewagentAction(Action):
    """
    Action for the CodeReviewAgent environment.

    - ADD_COMMENT: annotate a specific line with a review comment
    - REQUEST_CHANGES: mark the PR as needing changes
    - APPROVE: approve the PR (only when no significant issues remain)
    - SUBMIT_REVIEW: finalize and submit the review (ends the episode)
    """

    action_type: ActionType = Field(..., description="Type of review action")
    line_number: Optional[int] = Field(None, description="Source line being commented on")
    comment: Optional[str] = Field(None, description="Review comment text")
    severity: Optional[Severity] = Field(None, description="Issue severity level")
    category: Optional[IssueCategory] = Field(None, description="Issue category")


class CodereviewagentObservation(Observation):
    """
    Observation from the CodeReviewAgent environment.

    Contains the code to review, task instructions, and the running
    review history so the agent can track what it has already flagged.
    """

    code_snippet: str = Field(default="", description="Python source code to review")
    task_description: str = Field(default="", description="Review instructions and goals")
    file_name: str = Field(default="", description="Name of the file being reviewed")
    task_id: int = Field(default=0, description="Current task index (0=easy, 1=medium, 2=hard)")
    task_difficulty: str = Field(default="easy", description="Task difficulty label")
    review_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of actions taken so far this episode",
    )
    step_count: int = Field(default=0, description="Steps taken in current episode")
    max_steps: int = Field(default=20, description="Step budget for this task")
    issues_found_count: int = Field(default=0, description="Number of issues identified so far")
    total_issues: int = Field(default=0, description="Total issues in this task")
