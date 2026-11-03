"""P0 workflow records. No model calls, gradient math, training or state mutation."""

from dataclasses import dataclass, field, fields
import math
from types import MappingProxyType

from .errors import ContractError, ErrorCode
from .features import ComputeCost, GradientFeature, GradientSpec
from .program import ProgramSpec
from .selection import BatchManifest
from .signals import ModelRef
from .types import ArtifactRef, Record, digest, integer, require, sha256, text


def finite(value, name, minimum=None):
    require(type(value) in (int, float), f"{name} must be finite numeric data")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid and (minimum is None or value >= minimum), f"Invalid {name}")


def unique(values, name, *, nonempty=False, item_type=str):
    require(
        type(values) is tuple and (bool(values) or not nonempty),
        f"Invalid {name} tuple",
    )
    require(all(isinstance(v, item_type) for v in values), f"Invalid {name} item")
    if item_type is str:
        for value in values:
            text(value, name)
    require(len(set(values)) == len(values), f"Duplicate {name}")


def typed(values, cls, name):
    require(
        type(values) is tuple and all(isinstance(v, cls) for v in values),
        f"Invalid {name}",
    )


def header(record):
    require(record.schema_version == "1", f"Unsupported {type(record).__name__} schema")
    if hasattr(record, "backend"):
        require(record.backend in ("mock", "real"), "backend must be mock or real")


def artifact(value, name):
    require(isinstance(value, ArtifactRef), f"Invalid {name} artifact")


def model(value):
    require(isinstance(value, ModelRef), "Expected ModelRef")


def optional_text(value, name):
    if value is not None:
        text(value, name)


def optional_number(value, name, minimum=None):
    if value is not None:
        finite(value, name, minimum)


def json_value(value):
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, tuple):
        return [json_value(v) for v in value]
    return value


class WorkflowRecord(Record):
    def to_dict(self):
        return {
            f.name: json_value(getattr(self, f.name)) for f in fields(self) if f.init
        }

    @property
    def record_hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class TaskBudget(WorkflowRecord):
    max_steps: int
    max_output_tokens: int
    max_context_tokens: int
    max_wall_seconds: float
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in ("max_steps", "max_output_tokens", "max_context_tokens"):
            integer(getattr(self, name), name, 1)
        finite(self.max_wall_seconds, "max_wall_seconds", 0)
        require(self.max_wall_seconds > 0, "Task wall budget must be positive")


@dataclass(frozen=True)
class TaskRecord(WorkflowRecord):
    task_id: str
    version: str
    skills: tuple[str, ...]
    payload: ArtifactRef
    environment: ArtifactRef
    scorer: ArtifactRef
    split: str
    budget: TaskBudget
    backend: str
    family_ids: tuple[str, ...] = ()
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.task_id, "task_id")
        text(self.version, "task.version")
        unique(self.skills, "task.skills", nonempty=True)
        unique(self.family_ids, "task.family_ids")
        for name in ("payload", "environment", "scorer"):
            artifact(getattr(self, name), name)
        require(self.split in ("train", "development", "test"), "Invalid task split")
        require(isinstance(self.budget, TaskBudget), "Invalid task budget")

    @property
    def task_hash(self):
        return self.record_hash


@dataclass(frozen=True)
class ResourceUsage(WorkflowRecord):
    compute: ComputeCost = ComputeCost()
    agent_input_tokens: int | None = None
    agent_output_tokens: int | None = None
    scoring_tokens: int | None = None
    new_gradient_samples: int | None = None
    new_gradient_tokens: int | None = None
    cache_hits: int | None = None
    tool_calls: int | None = None
    training_supervised_tokens: int | None = None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        require(isinstance(self.compute, ComputeCost), "Invalid compute cost")
        for f in fields(self):
            if (
                f.name not in ("compute", "schema_version")
                and getattr(self, f.name) is not None
            ):
                integer(getattr(self, f.name), f.name)


@dataclass(frozen=True)
class CostEvent(WorkflowRecord):
    event_id: str
    attempt_id: str
    operation: str
    subject_id: str
    model_hash: str | None
    status: str
    usage: ResourceUsage
    receipt: ArtifactRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in ("event_id", "attempt_id", "subject_id"):
            text(getattr(self, name), name)
        require(
            self.operation
            in (
                "selection",
                "scoring",
                "sample_gradient",
                "target_gradient",
                "rollout",
                "training",
                "evaluation",
                "capability",
                "role_call",
            ),
            "Invalid cost operation",
        )
        require(
            self.status in ("completed", "failed", "cancelled"), "Invalid cost status"
        )
        if self.model_hash is not None:
            sha256(self.model_hash, "cost.model_hash")
        require(
            self.operation == "selection" or self.model_hash is not None,
            "Model operation cost must identify its model",
        )
        require(isinstance(self.usage, ResourceUsage), "Invalid resource usage")
        artifact(self.receipt, "cost receipt")


@dataclass(frozen=True)
class CostRef(WorkflowRecord):
    event_ids: tuple[str, ...]
    events_hash: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        unique(self.event_ids, "cost.event_ids")
        sha256(self.events_hash, "cost.events_hash")


@dataclass(frozen=True)
class CostLedger(WorkflowRecord):
    ledger_id: str
    events: tuple[CostEvent, ...]
    backend: str
    schema_version: str = "1"
    _events_by_id: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        header(self)
        text(self.ledger_id, "ledger_id")
        typed(self.events, CostEvent, "ledger.events")
        unique(tuple(e.event_id for e in self.events), "ledger event IDs")
        require(
            all(e.backend == self.backend for e in self.events), "Mixed cost backends"
        )
        object.__setattr__(
            self,
            "_events_by_id",
            MappingProxyType({e.event_id: e for e in self.events}),
        )

    def reference(self, event_ids):
        require(
            type(event_ids) in (tuple, list), "Cost reference expects an ID tuple/list"
        )
        ids = tuple(event_ids)
        unique(ids, "cost reference IDs")
        by_id = self._events_by_id
        require(
            set(ids).issubset(by_id), "Unknown cost event", ErrorCode.ARTIFACT_MISMATCH
        )
        return CostRef(ids, digest([by_id[i].to_dict() for i in sorted(ids)]))

    def summary(self):
        result = {}
        names = tuple(
            f.name
            for f in fields(ResourceUsage)
            if f.name not in ("compute", "schema_version")
        )
        for name in ("wall_seconds", "gpu_seconds", "processed_tokens", *names):
            values = [
                getattr(
                    (
                        e.usage.compute
                        if name in ("wall_seconds", "gpu_seconds", "processed_tokens")
                        else e.usage
                    ),
                    name,
                )
                for e in self.events
            ]
            known = [v for v in values if v is not None]
            try:
                subtotal = (
                    math.fsum(known)
                    if name in ("wall_seconds", "gpu_seconds")
                    else sum(known)
                )
            except OverflowError as exc:
                raise ContractError(
                    ErrorCode.INVALID_RECORD, "Cost total overflows"
                ) from exc
            result[name] = {
                "known_subtotal": subtotal,
                "missing_events": len(values) - len(known),
                "total": subtotal if len(known) == len(values) else None,
            }
        return result


@dataclass(frozen=True)
class RolloutRecord(WorkflowRecord):
    rollout_id: str
    model_ref: ModelRef
    task_id: str
    task_hash: str
    seed: int
    sampling: ArtifactRef
    actions: ArtifactRef
    observations: ArtifactRef
    tokenization_hash: str
    action_mask_hash: str
    generated_tokens: int
    reward: float | None
    status: str
    stop_reason: str
    scorer: ArtifactRef
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.model_ref)
        for name in ("rollout_id", "task_id", "stop_reason"):
            text(getattr(self, name), name)
        for name in ("task_hash", "tokenization_hash", "action_mask_hash"):
            sha256(getattr(self, name), name)
        for name in ("sampling", "actions", "observations", "scorer"):
            artifact(getattr(self, name), name)
        integer(self.seed, "rollout.seed")
        integer(self.generated_tokens, "generated_tokens")
        require(
            isinstance(self.cost, CostRef) and bool(self.cost.event_ids),
            "Executed rollout requires cost events",
        )
        require(
            self.status
            in (
                "completed",
                "task_failed",
                "budget_exhausted",
                "environment_failed",
                "cancelled",
            ),
            "Invalid rollout status",
        )
        optional_number(self.reward, "reward")
        if self.status in ("completed", "task_failed"):
            require(self.reward is not None, "Scored task outcome requires reward")
        if self.status in ("environment_failed", "cancelled"):
            require(
                self.reward is None, "Infrastructure failure cannot become zero reward"
            )


@dataclass(frozen=True)
class TargetGradientSpec(WorkflowRecord):
    skill: str
    gradient: GradientSpec
    reward_config: ArtifactRef
    advantage_config: ArtifactRef
    aggregation_config: ArtifactRef
    objective: str = "reward_ascent"
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "target.skill")
        require(
            isinstance(self.gradient, GradientSpec)
            and self.gradient.scope == "assistant",
            "Target coordinates must describe generated assistant actions",
        )
        require(
            self.objective == "reward_ascent",
            "Target objective must declare reward ascent",
        )
        for name in ("reward_config", "advantage_config", "aggregation_config"):
            artifact(getattr(self, name), name)


@dataclass(frozen=True)
class TargetGradientFeature(WorkflowRecord):
    target_id: str
    model_ref: ModelRef
    spec: TargetGradientSpec
    task_set_hash: str
    task_ids: tuple[str, ...]
    rollout_ids: tuple[str, ...]
    rollouts_hash: str
    vector: ArtifactRef | None
    dimension: int
    dtype: str
    norm: float | None
    validity: str
    reason: str | None
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.model_ref)
        text(self.target_id, "target_id")
        require(
            isinstance(self.spec, TargetGradientSpec), "Invalid target specification"
        )
        sha256(self.task_set_hash, "target.task_set_hash")
        sha256(self.rollouts_hash, "target.rollouts_hash")
        unique(self.task_ids, "target.task_ids", nonempty=True)
        unique(self.rollout_ids, "target.rollout_ids")
        integer(self.dimension, "target.dimension", 1)
        g = self.spec.gradient
        require(
            self.dimension
            == (g.parameter_count if g.projection is None else g.projection.dimension)
            and self.dtype == g.storage_dtype,
            "Target vector coordinates/dtype mismatch",
        )
        require(
            self.validity in ("valid", "zero", "invalid", "pending"),
            "Invalid target validity",
        )
        require(isinstance(self.cost, CostRef), "Invalid target cost")
        optional_text(self.reason, "target.reason")
        optional_number(self.norm, "target.norm", 0)
        if self.vector is not None:
            artifact(self.vector, "target vector")
        if self.validity in ("valid", "zero"):
            require(
                self.vector is not None
                and self.norm is not None
                and bool(self.cost.event_ids)
                and bool(self.rollout_ids),
                "Computed target requires rollout, vector, norm and cost",
            )
            require(
                (self.norm > 0) == (self.validity == "valid"),
                "Target norm/validity mismatch",
            )
        else:
            require(self.norm is None, "Missing/invalid target norm must stay missing")
        if self.validity == "pending":
            require(self.vector is None, "Pending target cannot have a computed vector")
        if self.validity != "valid":
            text(self.reason, "target validity reason")

    @property
    def target_version(self):
        data = self.to_dict()
        return digest(
            {k: v for k, v in data.items() if k not in ("target_id", "cost", "reason")}
        )


@dataclass(frozen=True)
class SkillWeight(WorkflowRecord):
    skill: str
    weight: float
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "weight.skill")
        finite(self.weight, "weight", 0)


@dataclass(frozen=True)
class SkillConstraint(WorkflowRecord):
    name: str
    skill: str
    minimum: float | None
    maximum: float | None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.name, "constraint.name")
        text(self.skill, "constraint.skill")
        optional_number(self.minimum, "constraint.minimum")
        optional_number(self.maximum, "constraint.maximum")
        require(
            self.minimum is not None or self.maximum is not None,
            "Constraint needs a bound",
        )
        require(
            self.minimum is None
            or self.maximum is None
            or self.minimum <= self.maximum,
            "Reversed constraint",
        )


@dataclass(frozen=True)
class ObjectiveSpec(WorkflowRecord):
    objective_id: str
    weights: tuple[SkillWeight, ...]
    constraints: tuple[SkillConstraint, ...]
    estimator: ArtifactRef
    acceptance_policy: ArtifactRef
    aggregation: str = "weighted_sum"
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.objective_id, "objective_id")
        typed(self.weights, SkillWeight, "objective weights")
        typed(self.constraints, SkillConstraint, "objective constraints")
        unique(tuple(w.skill for w in self.weights), "weighted skills", nonempty=True)
        require(
            any(w.weight > 0 for w in self.weights), "Objective needs positive weight"
        )
        unique(tuple(c.name for c in self.constraints), "constraint names")
        require(
            all(c.skill in {w.skill for w in self.weights} for c in self.constraints),
            "Constraint skill must be declared",
        )
        artifact(self.estimator, "estimator")
        artifact(self.acceptance_policy, "acceptance policy")
        require(self.aggregation == "weighted_sum", "Unsupported objective aggregation")

    @property
    def objective_hash(self):
        return self.record_hash


@dataclass(frozen=True)
class GradientCoverage(WorkflowRecord):
    required_ids: tuple[str, ...]
    computed_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in ("required_ids", "computed_ids", "missing_ids"):
            unique(getattr(self, name), name)
        require(
            not set(self.computed_ids).intersection(self.missing_ids)
            and set(self.required_ids)
            == set(self.computed_ids) | set(self.missing_ids),
            "Coverage must partition exact required IDs",
        )

    @property
    def complete(self):
        return bool(self.required_ids) and not self.missing_ids


@dataclass(frozen=True)
class TargetRef(WorkflowRecord):
    skill: str
    target_version: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "target reference skill")
        sha256(self.target_version, "target_version")


@dataclass(frozen=True)
class SkillValue(WorkflowRecord):
    skill: str
    value: float | None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "value.skill")
        optional_number(self.value, "skill value")


@dataclass(frozen=True)
class ConstraintResult(WorkflowRecord):
    name: str
    value: float | None
    passed: bool | None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.name, "constraint result name")
        optional_number(self.value, "constraint value")
        require(
            self.passed is None or type(self.passed) is bool,
            "Constraint result must be boolean or missing",
        )
        require(
            (self.value is None) == (self.passed is None),
            "Missing constraint outcome must remain missing",
        )


@dataclass(frozen=True)
class EvaluationReport(WorkflowRecord):
    evaluation_id: str
    model_ref: ModelRef
    batch_id: str
    objective_hash: str
    targets: tuple[TargetRef, ...]
    sample_cache_keys: tuple[str, ...]
    coverage: GradientCoverage
    skill_values: tuple[SkillValue, ...]
    total_value: float | None
    constraints: tuple[ConstraintResult, ...]
    diagnostics: ArtifactRef
    status: str
    reason: str | None
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.model_ref)
        text(self.evaluation_id, "evaluation_id")
        sha256(self.batch_id, "evaluation.batch_id")
        sha256(self.objective_hash, "evaluation.objective_hash")
        typed(self.targets, TargetRef, "evaluation targets")
        unique(tuple(t.skill for t in self.targets), "evaluation target skills")
        unique(self.sample_cache_keys, "sample cache keys")
        for key in self.sample_cache_keys:
            sha256(key, "sample cache key")
        require(isinstance(self.coverage, GradientCoverage), "Invalid coverage")
        require(
            len(self.sample_cache_keys) == len(self.coverage.computed_ids),
            "Computed coverage requires one sample cache reference per ID",
        )
        typed(self.skill_values, SkillValue, "skill values")
        unique(tuple(v.skill for v in self.skill_values), "evaluation skills")
        typed(self.constraints, ConstraintResult, "constraint results")
        unique(tuple(c.name for c in self.constraints), "constraint result names")
        artifact(self.diagnostics, "evaluation diagnostics")
        require(isinstance(self.cost, CostRef), "Invalid evaluation cost")
        require(
            self.status in ("valid", "invalid", "pending"), "Invalid evaluation status"
        )
        optional_text(self.reason, "evaluation.reason")
        optional_number(self.total_value, "total value")
        if self.status == "valid":
            require(
                self.coverage.complete
                and bool(self.targets)
                and {t.skill for t in self.targets}
                == {v.skill for v in self.skill_values}
                and self.total_value is not None
                and bool(self.skill_values)
                and all(v.value is not None for v in self.skill_values)
                and all(c.passed is not None for c in self.constraints)
                and bool(self.cost.event_ids),
                "Valid valuation requires complete gradients and numeric outcomes/cost",
            )
        else:
            require(
                self.total_value is None,
                "Incomplete valuation cannot carry a rankable total",
            )
            text(self.reason, "evaluation validity reason")

    @property
    def feasible(self):
        return self.status == "valid" and all(c.passed for c in self.constraints)


@dataclass(frozen=True)
class CapabilityMetric(WorkflowRecord):
    skill: str
    task_ids: tuple[str, ...]
    rollout_ids: tuple[str, ...]
    sample_count: int
    aggregation: str
    mean_reward: float | None
    uncertainty: float | None
    uncertainty_method: str | None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.skill, "capability skill")
        unique(self.task_ids, "capability task IDs", nonempty=True)
        unique(self.rollout_ids, "capability rollout IDs")
        integer(self.sample_count, "sample_count")
        require(
            self.sample_count == len(self.rollout_ids),
            "Rollout count differs from declared sample count",
        )
        require(
            self.aggregation in ("task_mean", "rollout_mean"),
            "Capability aggregation must be explicit",
        )
        optional_number(self.mean_reward, "mean reward")
        optional_number(self.uncertainty, "uncertainty", 0)
        optional_text(self.uncertainty_method, "uncertainty method")
        require(
            (self.uncertainty is None) == (self.uncertainty_method is None),
            "Uncertainty requires a declared method",
        )


@dataclass(frozen=True)
class CapabilityReport(WorkflowRecord):
    report_id: str
    model_ref: ModelRef
    task_ids: tuple[str, ...]
    task_set_hash: str
    rollout_ids: tuple[str, ...]
    split: str
    metrics: tuple[CapabilityMetric, ...]
    status: str
    reason: str | None
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.model_ref)
        text(self.report_id, "capability.report_id")
        unique(self.task_ids, "report task IDs", nonempty=True)
        unique(self.rollout_ids, "report rollout IDs")
        sha256(self.task_set_hash, "capability task set")
        typed(self.metrics, CapabilityMetric, "capability metrics")
        unique(tuple(m.skill for m in self.metrics), "capability skills", nonempty=True)
        require(
            set().union(*(set(m.task_ids) for m in self.metrics)) == set(self.task_ids),
            "Capability metric tasks must cover exact report tasks",
        )
        require(
            set().union(*(set(m.rollout_ids) for m in self.metrics))
            == set(self.rollout_ids),
            "Capability metric rollouts must cover exact report rollouts",
        )
        require(self.split in ("development", "test"), "Invalid capability split")
        require(
            self.status in ("valid", "invalid", "pending"), "Invalid capability status"
        )
        require(isinstance(self.cost, CostRef), "Invalid capability cost")
        optional_text(self.reason, "capability reason")
        if self.status == "valid":
            require(
                bool(self.rollout_ids)
                and all(
                    m.mean_reward is not None and m.sample_count > 0
                    for m in self.metrics
                )
                and bool(self.cost.event_ids),
                "Valid capability needs observed rewards and costs",
            )
        else:
            text(self.reason, "capability validity reason")
            require(
                all(m.mean_reward is None for m in self.metrics),
                "Invalid capability cannot fabricate a mean reward",
            )


@dataclass(frozen=True)
class TrainingRecord(WorkflowRecord):
    training_id: str
    starting_model: ModelRef
    batch_id: str
    training_data_hash: str
    recipe: ArtifactRef
    status: str
    candidate_model: ModelRef | None
    checkpoint: ArtifactRef | None
    processed_supervised_tokens: int | None
    receipt: ArtifactRef
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.starting_model)
        text(self.training_id, "training_id")
        sha256(self.batch_id, "training.batch_id")
        sha256(self.training_data_hash, "training_data_hash")
        artifact(self.recipe, "training recipe")
        artifact(self.receipt, "training receipt")
        require(isinstance(self.cost, CostRef), "Invalid training cost")
        require(
            self.status in ("pending", "completed", "failed"), "Invalid training status"
        )
        if self.candidate_model is not None:
            model(self.candidate_model)
            require(
                all(
                    getattr(self.candidate_model, k) == getattr(self.starting_model, k)
                    for k in (
                        "base_model_id",
                        "revision",
                        "tokenizer",
                        "parameterization",
                    )
                ),
                "Candidate must retain base/tokenizer parameterization identity",
            )
        if self.checkpoint is not None:
            artifact(self.checkpoint, "candidate checkpoint")
        if self.processed_supervised_tokens is not None:
            integer(self.processed_supervised_tokens, "training processed tokens")
        if self.status == "completed":
            require(
                self.candidate_model is not None
                and self.checkpoint is not None
                and self.processed_supervised_tokens is not None
                and self.processed_supervised_tokens > 0
                and bool(self.cost.event_ids),
                "Completed training needs candidate, checkpoint, tokens and costs",
            )
        if self.status == "failed":
            require(
                bool(self.cost.event_ids), "Failed training must preserve cost events"
            )
        if self.status == "pending":
            require(
                self.candidate_model is None
                and self.checkpoint is None
                and self.processed_supervised_tokens is None,
                "Pending training cannot have execution outputs",
            )


@dataclass(frozen=True)
class RoundRecord(WorkflowRecord):
    round_id: str
    attempt_id: str
    starting_model: ModelRef
    accepted_model: ModelRef
    accepted_version_before: int
    accepted_version_after: int
    objective_hash: str
    decision_policy_hash: str
    selected_batch_id: str | None
    evaluation_id: str | None
    training: TrainingRecord | None
    before_capability_id: str | None
    after_capability_id: str | None
    decision: str
    reason: str | None
    decision_evidence: ArtifactRef | None
    cost: CostRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        model(self.starting_model)
        model(self.accepted_model)
        text(self.round_id, "round_id")
        text(self.attempt_id, "attempt_id")
        for name in ("accepted_version_before", "accepted_version_after"):
            integer(getattr(self, name), name)
        for name in ("objective_hash", "decision_policy_hash"):
            sha256(getattr(self, name), name)
        if self.selected_batch_id is not None:
            sha256(self.selected_batch_id, "selected_batch_id")
        for name in (
            "evaluation_id",
            "before_capability_id",
            "after_capability_id",
            "reason",
        ):
            optional_text(getattr(self, name), name)
        require(isinstance(self.cost, CostRef), "Invalid round cost")
        require(
            self.training is None or isinstance(self.training, TrainingRecord),
            "Invalid training record",
        )
        require(
            self.decision in ("pending", "accept", "reject", "needs_more_evidence"),
            "Invalid round decision",
        )
        if self.decision_evidence is not None:
            artifact(self.decision_evidence, "decision evidence")
        require(
            (self.before_capability_id is None) == (self.after_capability_id is None),
            "Paired capability references must be supplied together",
        )
        if self.evaluation_id is not None or self.training is not None:
            require(
                self.selected_batch_id is not None and self.evaluation_id is not None,
                "Training must reference an evaluated exact batch",
            )
        if self.decision != "pending":
            text(self.reason, "decision reason")
            artifact(self.decision_evidence, "decision evidence")
        if self.decision in ("accept", "needs_more_evidence"):
            require(
                self.training is not None
                and self.training.status == "completed"
                and self.before_capability_id is not None,
                "Decision requires trained candidate and paired capability reports",
            )
        if self.decision == "accept":
            require(
                self.accepted_model == self.training.candidate_model
                and self.accepted_model.model_hash != self.starting_model.model_hash
                and self.accepted_version_after == self.accepted_version_before + 1,
                "Acceptance must advance to the exact candidate identity",
            )
        else:
            require(
                self.accepted_model == self.starting_model
                and self.accepted_version_after == self.accepted_version_before,
                "Non-accepting attempt cannot change accepted model/version",
            )

    @property
    def training_use_decision(self):
        """Explicit M1 label adapter; does not record or start any training."""
        return {
            "accept": "accepted",
            "reject": "rejected",
            "pending": "pending",
            "needs_more_evidence": "pending",
        }[self.decision]


@dataclass(frozen=True)
class WorkflowBundle(WorkflowRecord):
    bundle_id: str
    backend: str
    model_ref: ModelRef
    program: ProgramSpec
    selection_context_hash: str
    batch: BatchManifest
    objective: ObjectiveSpec
    tasks: tuple[TaskRecord, ...]
    rollouts: tuple[RolloutRecord, ...]
    target_gradients: tuple[TargetGradientFeature, ...]
    sample_gradients: tuple[GradientFeature, ...]
    evaluations: tuple[EvaluationReport, ...]
    capabilities: tuple[CapabilityReport, ...]
    ledger: CostLedger
    round_record: RoundRecord | None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        text(self.bundle_id, "bundle_id")
        model(self.model_ref)
        require(
            isinstance(self.program, ProgramSpec)
            and isinstance(self.batch, BatchManifest)
            and isinstance(self.objective, ObjectiveSpec)
            and isinstance(self.ledger, CostLedger),
            "Invalid workflow roots",
        )
        sha256(self.selection_context_hash, "selection context hash")
        for name, cls in (
            ("tasks", TaskRecord),
            ("rollouts", RolloutRecord),
            ("target_gradients", TargetGradientFeature),
            ("sample_gradients", GradientFeature),
            ("evaluations", EvaluationReport),
            ("capabilities", CapabilityReport),
        ):
            typed(getattr(self, name), cls, name)
        require(
            self.round_record is None or isinstance(self.round_record, RoundRecord),
            "Invalid round record",
        )
        from .workflow_validation import validate_workflow

        validate_workflow(self)
