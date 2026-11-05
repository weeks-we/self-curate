"""Typed SFT inputs; data production and model loading remain injectable."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from self_curate.contracts import (
    ArtifactRef,
    ModelRef,
    ResourceUsage,
    TokenizerRef,
    TrainingRecord,
    CostLedger,
)
from self_curate.contracts.types import integer, require, sha256, text
from self_curate.contracts.workflow import WorkflowRecord, finite, unique


@dataclass(frozen=True)
class TrainingExample(WorkflowRecord):
    data_id: str
    content_hash: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported training example schema")
        text(self.data_id, "training data ID")
        sha256(self.content_hash, "training content hash")
        require(
            type(self.input_ids) is tuple and len(self.input_ids) >= 2,
            "SFT needs a complete causal sequence with at least two tokens",
        )
        require(
            type(self.labels) is tuple and len(self.labels) == len(self.input_ids),
            "SFT labels must align exactly with input tokens",
        )
        for token, label in zip(self.input_ids, self.labels):
            integer(token, "input token")
            require(
                type(label) is int and label in (-100, token),
                "Only original payload tokens or ignore labels are supported",
            )
        require(self.labels[0] == -100, "First token has no causal prediction position")
        require(self.supervised_tokens > 0, "SFT needs nonempty assistant supervision")

    @property
    def supervised_tokens(self):
        return sum(label != -100 for label in self.labels)


@dataclass(frozen=True)
class TrainingData(WorkflowRecord):
    kind: str
    source_hash: str
    tokenizer: TokenizerRef
    examples: tuple[TrainingExample, ...]
    provenance: tuple[ArtifactRef, ...]
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported training data schema")
        require(self.kind in ("subset", "curation"), "Unknown SFT dataset kind")
        sha256(self.source_hash, "SFT source")
        require(isinstance(self.tokenizer, TokenizerRef), "Missing tokenizer identity")
        require(
            type(self.examples) is tuple
            and bool(self.examples)
            and all(isinstance(x, TrainingExample) for x in self.examples),
            "SFT needs typed, ordered examples",
        )
        unique(
            tuple(x.data_id for x in self.examples), "SFT example IDs", nonempty=True
        )
        require(
            type(self.provenance) is tuple
            and bool(self.provenance)
            and all(isinstance(x, ArtifactRef) for x in self.provenance),
            "Training inputs need provenance",
        )
        require(self.backend in ("mock", "real"), "Invalid training backend")
        if self.backend == "real":
            require(
                self.tokenizer.backend != "fixture"
                and not any(
                    x.uri.strip().lower().startswith(("fixture://", "mock://"))
                    for x in self.provenance
                ),
                "Known synthetic SFT inputs cannot be real",
            )

    @property
    def supervised_tokens(self):
        return sum(x.supervised_tokens for x in self.examples)


@dataclass(frozen=True)
class SFTRecipe(WorkflowRecord):
    learning_rate: float
    epochs: int
    accumulation_steps: int
    max_sequence_tokens: int
    seed: int
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    loss_reduction: str = "supervised_token_mean"
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported SFT recipe schema")
        finite(self.learning_rate, "learning rate", 0)
        require(self.learning_rate > 0, "Learning rate must be positive")
        for name in ("epochs", "accumulation_steps", "max_sequence_tokens"):
            integer(getattr(self, name), name, 1)
        integer(self.seed, "training seed")
        finite(self.weight_decay, "weight decay", 0)
        if self.max_grad_norm is not None:
            finite(self.max_grad_norm, "max gradient norm", 0)
            require(
                self.max_grad_norm > 0, "Gradient clipping threshold must be positive"
            )
        require(self.optimizer in ("adamw", "sgd"), "Unsupported optimizer")
        require(
            self.loss_reduction == "supervised_token_mean", "Unsupported SFT reduction"
        )


@dataclass(frozen=True)
class TrainingRequest(WorkflowRecord):
    training_id: str
    attempt_id: str
    starting_model: ModelRef
    data: TrainingData
    batch_id: str
    training_data_hash: str
    recipe: SFTRecipe
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        text(self.training_id, "training ID")
        text(self.attempt_id, "attempt ID")
        require(
            self.schema_version == "1"
            and isinstance(self.starting_model, ModelRef)
            and isinstance(self.data, TrainingData)
            and isinstance(self.recipe, SFTRecipe),
            "Invalid training request",
        )
        sha256(self.batch_id, "training batch ID")
        sha256(self.training_data_hash, "training data hash")
        require(self.backend == self.data.backend, "Mixed training domains")
        require(
            self.starting_model.tokenizer == self.data.tokenizer,
            "SFT tokenizer/template differs from starting model",
        )
        require(
            all(
                len(x.input_ids) <= self.recipe.max_sequence_tokens
                for x in self.data.examples
            ),
            "Complete SFT trajectories exceed the recipe; truncation is forbidden",
        )


@dataclass(frozen=True)
class BackendTrainingResult(WorkflowRecord):
    candidate_model: ModelRef
    checkpoint: ArtifactRef
    processed_supervised_tokens: int
    optimizer_steps: int
    mean_loss: float | None
    usage: ResourceUsage

    def __post_init__(self):
        require(
            isinstance(self.candidate_model, ModelRef)
            and isinstance(self.checkpoint, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Invalid backend training result",
        )
        integer(self.processed_supervised_tokens, "processed SFT tokens", 1)
        integer(self.optimizer_steps, "optimizer steps", 1)
        if self.mean_loss is not None:
            finite(self.mean_loss, "training fit diagnostic", 0)
        require(
            self.usage.training_supervised_tokens == self.processed_supervised_tokens,
            "SFT usage differs from processed supervision",
        )


@dataclass(frozen=True)
class TrainingOutcome(WorkflowRecord):
    request: TrainingRequest
    record: TrainingRecord
    ledger: CostLedger
    result: BackendTrainingResult

    def __post_init__(self):
        require(
            isinstance(self.request, TrainingRequest)
            and isinstance(self.record, TrainingRecord)
            and isinstance(self.ledger, CostLedger)
            and isinstance(self.result, BackendTrainingResult),
            "Invalid SFT outcome",
        )
        require(
            self.record.status == "completed"
            and self.record.training_id == self.request.training_id
            and self.record.starting_model == self.request.starting_model
            and self.record.batch_id == self.request.batch_id
            and self.record.training_data_hash == self.request.training_data_hash
            and self.record.recipe.sha256 == self.request.recipe.record_hash
            and self.record.candidate_model == self.result.candidate_model
            and self.record.checkpoint == self.result.checkpoint
            and self.record.processed_supervised_tokens
            == self.result.processed_supervised_tokens
            and self.record.backend == self.request.backend == self.ledger.backend
            and self.ledger.reference(self.record.cost.event_ids) == self.record.cost,
            "SFT outcome changed its request, weights, data or costs",
        )
        require(
            len(self.ledger.events) == 1
            and all(
                e.operation == "training"
                and e.status == "completed"
                and e.subject_id == self.request.training_id
                and e.model_hash == self.request.starting_model.model_hash
                and e.attempt_id == self.request.attempt_id
                and e.usage == self.result.usage
                for e in self.ledger.events
            ),
            "SFT physical operation cost mismatch",
        )
        require(
            self.record.receipt == self.ledger.events[0].receipt,
            "Training receipt differs from its physical operation",
        )
        require(
            self.result.processed_supervised_tokens
            == self.request.data.supervised_tokens * self.request.recipe.epochs
            and self.result.optimizer_steps
            == self.request.recipe.epochs
            * (
                (
                    len(self.request.data.examples)
                    + self.request.recipe.accumulation_steps
                    - 1
                )
                // self.request.recipe.accumulation_steps
            ),
            "Training outcome differs from fixed exposure/optimizer budgets",
        )
        require(
            self.result.candidate_model.model_hash
            != self.request.starting_model.model_hash,
            "Training outcome did not produce a distinct candidate",
        )
        if self.request.backend == "real":
            require(
                not self.result.checkpoint.uri.strip()
                .lower()
                .startswith(("fixture://", "mock://")),
                "Mock checkpoint cannot enter a real training outcome",
            )


class TrainingBackend(Protocol):
    backend: str

    def train(
        self, request: TrainingRequest, output: Path
    ) -> BackendTrainingResult: ...


class CurationDataProvider(Protocol):
    def prepare(self, handoff) -> TrainingData:
        """Supply externally curated/tokenized filtering trajectories for this handoff."""
        ...
