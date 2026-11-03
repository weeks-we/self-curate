"""Imported verification -> grounded feedback -> program proposal contracts."""

from dataclasses import dataclass

from .errors import ContractError, ErrorCode
from .program import ProgramSpec, selection_identity
from .selection import BatchManifest
from .signals import ModelRef
from .types import ArtifactRef, integer, require, sha256, text
from .workflow import (
    WorkflowRecord,
    ObjectiveSpec,
    SkillValue,
    ConstraintResult,
    ResourceUsage,
    header,
    typed,
    unique,
    optional_number,
    optional_text,
    finite,
)


@dataclass(frozen=True)
class ValidationRequest(WorkflowRecord):
    request_id: str
    model_ref: ModelRef
    program: ProgramSpec
    batch: BatchManifest
    objective: ObjectiveSpec
    selection_context_hash: str
    measure: str
    protocol: ArtifactRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.request_id, "request_id")
        require(
            isinstance(self.model_ref, ModelRef)
            and isinstance(self.program, ProgramSpec)
            and isinstance(self.batch, BatchManifest)
            and isinstance(self.objective, ObjectiveSpec),
            "Invalid validation request inputs",
        )
        sha256(self.selection_context_hash, "selection_context_hash")
        require(
            self.measure in ("predicted_gain", "reward_delta"),
            "Declare predicted_gain or reward_delta explicitly",
        )
        require(
            isinstance(self.protocol, ArtifactRef),
            "Validation protocol needs immutable identity",
        )
        identity = selection_identity(
            self.program,
            self.batch.policy_hash,
            self.selection_context_hash,
            self.batch.snapshot_hash,
        )
        require(
            identity == (self.batch.program_hash, self.batch.execution_hash)
            and self.batch.model_hash == self.model_ref.model_hash,
            "Validation request model/program/batch mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            self.backend != "real"
            or (
                self.model_ref.tokenizer.backend != "fixture"
                and self.batch.synthetic is not True
                and not self.protocol.uri.strip()
                .lower()
                .startswith(("mock://", "fixture://"))
                and all(
                    not ref.uri.strip().lower().startswith(("mock://", "fixture://"))
                    for ref in (
                        self.objective.estimator,
                        self.objective.acceptance_policy,
                    )
                )
            ),
            "Fixture validation request cannot be real",
        )

    @property
    def request_hash(self):
        return self.record_hash


@dataclass(frozen=True)
class ValidationResult(WorkflowRecord):
    result_id: str
    request_hash: str
    status: str
    skill_values: tuple[SkillValue, ...]
    total_value: float | None
    constraints: tuple[ConstraintResult, ...]
    reason: str | None
    evidence: ArtifactRef
    usage: ResourceUsage
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.result_id, "result_id")
        sha256(self.request_hash, "request_hash")
        require(
            self.status in ("valid", "invalid", "pending"),
            "Invalid verification result status",
        )
        typed(self.skill_values, SkillValue, "validation skill values")
        unique(tuple(v.skill for v in self.skill_values), "validation skills")
        typed(self.constraints, ConstraintResult, "validation constraints")
        unique(tuple(v.name for v in self.constraints), "validation constraint names")
        require(
            isinstance(self.evidence, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Validation needs evidence and declared/unknown costs",
        )
        optional_number(self.total_value, "validation total")
        optional_text(self.reason, "validation reason")
        if self.status == "valid":
            require(
                bool(self.skill_values)
                and all(v.value is not None for v in self.skill_values)
                and self.total_value is not None
                and all(v.passed is not None for v in self.constraints),
                "Valid external result needs complete numeric observations",
            )
        else:
            require(
                self.total_value is None,
                "Unverified result cannot carry a rankable total",
            )
            text(self.reason, "validation reason")


@dataclass(frozen=True)
class FeedbackFact(WorkflowRecord):
    code: str
    skill: str | None
    value: float | None
    bound: float | None
    message: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.code, "fact.code")
        text(self.message, "fact.message")
        optional_text(self.skill, "fact.skill")
        optional_number(self.value, "fact.value")
        optional_number(self.bound, "fact.bound")


@dataclass(frozen=True)
class EditHint(WorkflowRecord):
    skill: str
    direction: str
    reason: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "hint.skill")
        text(self.reason, "hint.reason")
        require(
            self.direction in ("increase", "decrease"),
            "Invalid exposure hint direction",
        )


@dataclass(frozen=True)
class LevalFeedback(WorkflowRecord):
    feedback_id: str
    request_hash: str
    result_hash: str
    model_ref: ModelRef
    objective_hash: str
    batch_id: str
    measure: str
    protocol_hash: str
    status: str
    total_value: float | None
    facts: tuple[FeedbackFact, ...]
    hypotheses: tuple[EditHint, ...]
    summary: str
    memory_namespace: str
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.feedback_id, "feedback_id")
        text(self.summary, "feedback summary")
        text(self.memory_namespace, "leval memory namespace")
        require(isinstance(self.model_ref, ModelRef), "Invalid feedback ModelRef")
        for name in (
            "request_hash",
            "result_hash",
            "objective_hash",
            "batch_id",
            "protocol_hash",
        ):
            sha256(getattr(self, name), name)
        require(
            self.measure in ("predicted_gain", "reward_delta")
            and self.status in ("valid", "invalid", "pending"),
            "Invalid feedback measure/status",
        )
        optional_number(self.total_value, "feedback total")
        typed(self.facts, FeedbackFact, "feedback facts")
        typed(self.hypotheses, EditHint, "feedback hypotheses")
        require(
            (self.status == "valid" and self.total_value is not None)
            or (
                self.status != "valid"
                and self.total_value is None
                and not self.hypotheses
            ),
            "Missing/invalid evidence cannot drive optimization",
        )


@dataclass(frozen=True)
class EditPolicy(WorkflowRecord):
    share_step: float
    minimum_share: float
    maximum_share: float
    evidence_token_step: int
    max_changed_nodes: int
    max_nodes: int
    max_model_attempts: int
    model_timeout_seconds: float
    allow_seed_change: bool = False
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in (
            "share_step",
            "minimum_share",
            "maximum_share",
            "model_timeout_seconds",
        ):
            finite(getattr(self, name), name, 0)
        require(
            0 < self.share_step < 1
            and 0 < self.minimum_share < self.maximum_share < 1
            and self.model_timeout_seconds > 0,
            "Invalid edit bounds/timeout",
        )
        for name in (
            "evidence_token_step",
            "max_changed_nodes",
            "max_nodes",
            "max_model_attempts",
        ):
            integer(getattr(self, name), name, 1)
        require(
            self.max_nodes <= 256
            and self.max_model_attempts <= 3
            and type(self.allow_seed_change) is bool,
            "Invalid proposal budget",
        )


@dataclass(frozen=True)
class ProgramChange(WorkflowRecord):
    added: tuple[str, ...]
    removed: tuple[str, ...]
    modified: tuple[str, ...]
    output_changed: bool
    seed_changed: bool
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in ("added", "removed", "modified"):
            unique(getattr(self, name), name)
        require(
            type(self.output_changed) is bool and type(self.seed_changed) is bool,
            "Invalid change flags",
        )


@dataclass(frozen=True)
class LoptProposal(WorkflowRecord):
    proposal_id: str
    model_ref: ModelRef
    parent_recipe_hash: str
    feedback_hash: str
    program: ProgramSpec | None
    changes: ProgramChange
    rationale: str
    hypothesis: str
    status: str
    method: str
    memory_namespace: str
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.proposal_id, "proposal_id")
        text(self.rationale, "rationale")
        text(self.hypothesis, "hypothesis")
        text(self.memory_namespace, "lopt memory namespace")
        require(
            isinstance(self.model_ref, ModelRef)
            and isinstance(self.changes, ProgramChange),
            "Invalid proposal identity/change",
        )
        sha256(self.parent_recipe_hash, "parent_recipe_hash")
        sha256(self.feedback_hash, "feedback_hash")
        require(
            self.status in ("proposed", "blocked") and self.method in ("rule", "model"),
            "Invalid proposal status/method",
        )
        require(
            (self.program is not None) == (self.status == "proposed"),
            "Only proposed candidates contain a program",
        )
        require(
            self.program is None or isinstance(self.program, ProgramSpec),
            "Invalid candidate program",
        )


FEEDBACK_TYPES = {
    cls.__name__: cls
    for cls in (
        ValidationRequest,
        ValidationResult,
        FeedbackFact,
        EditHint,
        LevalFeedback,
        EditPolicy,
        ProgramChange,
        LoptProposal,
    )
}


def parse_feedback_contract(kind, value):
    from .workflow_parsing import CONTRACT_TYPES, _decode

    require(
        isinstance(kind, str) and kind in FEEDBACK_TYPES, "Unknown feedback contract"
    )
    try:
        return _decode(
            FEEDBACK_TYPES[kind],
            value,
            kind,
            registry={**CONTRACT_TYPES, **FEEDBACK_TYPES},
        )
    except ContractError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD,
            "Malformed feedback contract",
            kind=kind,
            reason=type(exc).__name__,
        ) from exc
