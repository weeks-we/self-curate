"""Pluggable task execution and deterministic capability reports."""

from .contracts import (
    TaskEvaluation,
    TaskRunRequest,
    TaskRunResult,
    TaskRunner,
    TaskSuite,
)
from .service import TaskValidationFailure, TaskValidator

__all__ = [
    "TaskEvaluation",
    "TaskRunRequest",
    "TaskRunResult",
    "TaskRunner",
    "TaskSuite",
    "TaskValidationFailure",
    "TaskValidator",
]
