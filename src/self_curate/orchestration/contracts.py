"""Two-stage outer learning and explicit, externally configured decisions."""

from dataclasses import dataclass
from typing import Protocol

from self_curate.contracts import ArtifactRef, BatchManifest, ModelRef
from self_curate.contracts.types import require, sha256, text
from self_curate.contracts.workflow import WorkflowRecord
from self_curate.training import TrainingOutcome
from self_curate.validation import TaskEvaluation


@dataclass(frozen=True)
class OuterHandoff(WorkflowRecord):
    attempt_id: str
    search_hash: str
    best_request_hash: str
    batch: BatchManifest
    starting_model: ModelRef
    backend: str
    history_artifacts: tuple[ArtifactRef, ...] = ()
    schema_version: str = "1"

    def __post_init__(self):
        text(self.attempt_id, "outer attempt")
        sha256(self.search_hash, "stopped search hash")
        sha256(self.best_request_hash, "historical best request")
        require(
            self.schema_version == "1"
            and isinstance(self.batch, BatchManifest)
            and isinstance(self.starting_model, ModelRef),
            "Invalid outer handoff",
        )
        require(
            self.backend in ("mock", "real")
            and self.batch.model_hash == self.starting_model.model_hash,
            "Outer input model differs from inner frozen model",
        )
        require(
            type(self.history_artifacts) is tuple
            and all(isinstance(x, ArtifactRef) for x in self.history_artifacts),
            "Invalid curation history references",
        )
        if self.backend == "real":
            require(
                not any(
                    x.uri.strip().lower().startswith(("fixture://", "mock://"))
                    for x in self.history_artifacts
                ),
                "Mock history cannot be declared real",
            )


@dataclass(frozen=True)
class DecisionContext(WorkflowRecord):
    attempt_id: str
    stage: str
    accepted_model: ModelRef
    training: TrainingOutcome
    evaluation: TaskEvaluation
    baseline: TaskEvaluation | None
    policy: ArtifactRef | None

    def __post_init__(self):
        text(self.attempt_id, "decision attempt")
        require(self.stage in ("subset", "curation"), "Unknown decision stage")
        require(
            isinstance(self.training, TrainingOutcome)
            and isinstance(self.evaluation, TaskEvaluation),
            "Incomplete decision evidence",
        )
        require(
            self.training.request.data.kind == self.stage
            and self.training.request.attempt_id == self.attempt_id
            and self.training.request.starting_model == self.accepted_model
            and self.training.result.candidate_model == self.evaluation.report.model_ref
            and self.training.result.checkpoint == self.evaluation.checkpoint,
            "Decision uses another candidate or parent model",
        )
        require(
            self.evaluation.protocol_verified
            and all(
                r.attempt_id == self.attempt_id
                for r in self.evaluation.execution_requests
            ),
            "Decision needs task execution evidence from the same attempt",
        )
        if self.baseline is not None:
            require(
                isinstance(self.baseline, TaskEvaluation)
                and self.baseline.report.model_ref == self.accepted_model
                and self.baseline.suite == self.evaluation.suite,
                "Baseline model or task/sampling/scoring protocol differs",
            )
            require(
                self.baseline.protocol_verified,
                "Baseline has no verified execution protocol",
            )
        require(
            self.policy is None or isinstance(self.policy, ArtifactRef),
            "Invalid decision policy",
        )


@dataclass(frozen=True)
class AcceptanceDecision(WorkflowRecord):
    context_hash: str
    outcome: str
    reason: str
    evidence: ArtifactRef
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported acceptance decision")
        sha256(self.context_hash, "decision context hash")
        require(
            self.outcome in ("accept", "reject", "needs_more_evidence"),
            "Unknown acceptance decision",
        )
        text(self.reason, "acceptance reason")
        require(isinstance(self.evidence, ArtifactRef), "Decision evidence is required")


class DecisionProvider(Protocol):
    def decide(self, context: DecisionContext) -> AcceptanceDecision:
        """Apply the pinned policy. No numeric acceptance defaults are supplied."""
        ...
