"""Evidence-based Leval feedback and bounded Lopt proposal orchestration."""

from .leval import (
    ValidationProvider,
    check_external_result,
    result_from_evaluation,
    build_feedback,
)
from .lopt import SharedModelClient, ProposalAttempt, program_diff, propose_program
from .service import (
    FeedbackStep,
    run_feedback_step,
    feedback_step_to_directory,
    request_from_preparation,
)

__all__ = [
    "ValidationProvider",
    "check_external_result",
    "result_from_evaluation",
    "build_feedback",
    "SharedModelClient",
    "ProposalAttempt",
    "program_diff",
    "propose_program",
    "FeedbackStep",
    "run_feedback_step",
    "feedback_step_to_directory",
    "request_from_preparation",
]

from .prompts import PromptTemplate, default_prompts, load_prompts, ROLES
from .history import HistoryEvent
from .runtime import RoleRuntime, RoleReply
from .roles import (
    ExecutionAgent,
    EvaluationAgent,
    OptimizationAgent,
    RoleEvaluation,
    RoleCycle,
    RoleTeam,
)

__all__ += [
    "PromptTemplate",
    "default_prompts",
    "load_prompts",
    "ROLES",
    "HistoryEvent",
    "RoleRuntime",
    "RoleReply",
    "ExecutionAgent",
    "EvaluationAgent",
    "OptimizationAgent",
    "RoleEvaluation",
    "RoleCycle",
    "RoleTeam",
]

from .scope import RoleScope, parse_role_scope, request_scope
from .handoff import publish_role_cycle, load_role_cycle

__all__ += [
    "RoleScope",
    "parse_role_scope",
    "request_scope",
    "publish_role_cycle",
    "load_role_cycle",
]
