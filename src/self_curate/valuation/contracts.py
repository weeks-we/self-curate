"""Inner validation identities. Algorithms are fixed by protocol, never by Lopt."""

from dataclasses import dataclass

from self_curate.contracts import ArtifactRef, ModelRef, ResourceUsage, TokenizerRef
from self_curate.contracts.types import integer, require, sha256
from self_curate.contracts.workflow import WorkflowRecord, finite
from self_curate.training import TrainingExample
from self_curate.validation import TaskEvaluation


@dataclass(frozen=True)
class NICERecipe(WorkflowRecord):
    parameter_coordinates_hash: str
    parameter_count: int
    adam_epsilon: float
    max_grad_norm: float
    projection_dimension: int | None = None
    projection_seed: int = 0
    zero_epsilon: float = 1e-12
    implementation: str = "nice-single-checkpoint-v1"
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported NICE recipe")
        sha256(self.parameter_coordinates_hash, "parameter coordinates")
        integer(self.parameter_count, "parameter count", 1)
        for name in ("adam_epsilon", "max_grad_norm", "zero_epsilon"):
            finite(getattr(self, name), name, 0)
            require(getattr(self, name) > 0, "NICE thresholds must be positive")
        if self.projection_dimension is not None:
            integer(self.projection_dimension, "projection dimension", 1)
        integer(self.projection_seed, "projection seed")
        require(self.projection_seed < 2**64, "Projection seed exceeds uint64")
        require(self.implementation == "nice-single-checkpoint-v1", "Unknown estimator")


@dataclass(frozen=True)
class ReplayTrajectory(WorkflowRecord):
    """Teacher-forced replay inputs supplied by the task adapter, not generated here.

    Each step retains the actual input prefix and assistant action labels. The
    upstream adapter owns authenticity; the source rollout hash pins its evidence.
    """

    rollout_hash: str
    steps: tuple[TrainingExample, ...]
    evidence: ArtifactRef
    tokenizer: TokenizerRef
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported replay")
        sha256(self.rollout_hash, "source rollout")
        require(
            type(self.steps) is tuple
            and all(isinstance(s, TrainingExample) for s in self.steps),
            "Replay requires tokenized assistant steps",
        )
        require(isinstance(self.evidence, ArtifactRef), "Replay evidence required")
        require(
            isinstance(self.tokenizer, TokenizerRef),
            "Replay tokenizer/template identity required",
        )


@dataclass(frozen=True)
class NICEReference(WorkflowRecord):
    evaluation: TaskEvaluation
    replays: tuple[ReplayTrajectory, ...]
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported NICE reference")
        require(
            isinstance(self.evaluation, TaskEvaluation)
            and self.evaluation.protocol_verified
            and self.evaluation.report.status == "valid",
            "NICE needs complete, protocol-bound development rollouts",
        )
        require(
            type(self.replays) is tuple
            and all(isinstance(r, ReplayTrajectory) for r in self.replays),
            "Typed replays required",
        )
        by_hash = {r.rollout_hash: r for r in self.replays}
        require(
            len(by_hash) == len(self.replays)
            and set(by_hash) == {r.record_hash for r in self.evaluation.rollouts},
            "Replay must cover every exact task/seed, including failed rewards",
        )
        for rollout in self.evaluation.rollouts:
            replay = by_hash[rollout.record_hash]
            require(
                replay.tokenizer == self.model_ref.tokenizer,
                "Replay tokenizer/template differs from evaluated model",
            )
            require(0 <= rollout.reward <= 1, "NICE uses fixed [0,1] rewards")
            require(
                sum(s.supervised_tokens for s in replay.steps)
                == rollout.generated_tokens,
                "Replay assistant labels must cover generated actions exactly once",
            )
            require(
                rollout.reward == 0 or bool(replay.steps),
                "A rewarded rollout needs actual assistant replay tokens",
            )
            if self.evaluation.suite.backend == "real":
                require(
                    not replay.evidence.uri.strip()
                    .lower()
                    .startswith(("fixture://", "mock://")),
                    "Mock replay cannot be real",
                )

    @property
    def model_ref(self) -> ModelRef:
        return self.evaluation.report.model_ref


@dataclass(frozen=True)
class NICEScores(WorkflowRecord):
    request_hash: str
    data_hash: str
    reference_hash: str
    recipe_hash: str
    sample_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    scores: tuple[tuple[float, ...], ...]
    raw_scores: tuple[tuple[float, ...], ...]
    target_norms: tuple[float, ...]
    evidence: ArtifactRef
    usage: ResourceUsage
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported NICE scores")
        for name in ("request_hash", "data_hash", "reference_hash", "recipe_hash"):
            sha256(getattr(self, name), name)
        for ids in (self.sample_ids, self.task_ids):
            require(
                type(ids) is tuple
                and bool(ids)
                and len(ids) == len(set(ids))
                and all(type(v) is str and v.strip() for v in ids),
                "NICE needs unique ordered IDs",
            )
        for matrix in (self.scores, self.raw_scores):
            require(
                type(matrix) is tuple and len(matrix) == len(self.sample_ids),
                "Bad score rows",
            )
            for row in matrix:
                require(
                    type(row) is tuple and len(row) == len(self.task_ids),
                    "Bad score columns",
                )
                for value in row:
                    finite(value, "cosine")
                    require(
                        -1.0000000001 <= value <= 1.0000000001, "Cosine outside [-1,1]"
                    )
        require(
            type(self.target_norms) is tuple
            and len(self.target_norms) == len(self.task_ids),
            "Bad target norms",
        )
        for norm in self.target_norms:
            finite(norm, "target norm", 0)
        for j, norm in enumerate(self.target_norms):
            if norm == 0:
                require(
                    all(row[j] == 0 for row in (*self.scores, *self.raw_scores)),
                    "A zero target cannot have nonzero cosine",
                )
        require(
            isinstance(self.evidence, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "NICE evidence/usage required",
        )
        require(self.backend in ("mock", "real"), "Invalid NICE domain")
