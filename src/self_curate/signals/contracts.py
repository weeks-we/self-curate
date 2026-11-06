"""Forward-only signal production, retaining the existing M1 signal contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from self_curate.contracts import (
    ArtifactRef,
    ModelRef,
    ResourceUsage,
    SignalRecord,
    SignalSpec,
    TrajectoryRecord,
    CostLedger,
)
from self_curate.contracts.types import integer, require, sha256, text
from self_curate.contracts.workflow import WorkflowRecord, finite, unique


@dataclass(frozen=True)
class ForwardRequest(WorkflowRecord):
    scoring_id: str
    data_id: str
    content_hash: str
    tokenization_hash: str
    model_ref: ModelRef
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    computation_hash: str
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported forward request")
        text(self.scoring_id, "scoring ID")
        text(self.data_id, "data ID")
        for name in ("content_hash", "tokenization_hash", "computation_hash"):
            sha256(getattr(self, name), name)
        require(isinstance(self.model_ref, ModelRef), "Scoring requires exact ModelRef")
        require(
            type(self.input_ids) is tuple and len(self.input_ids) >= 2,
            "Complete causal context required",
        )
        for token in self.input_ids:
            integer(token, "input token")
            require(token < 2**63, "Token ID exceeds supported integer range")
        require(
            type(self.positions) is tuple and bool(self.positions),
            "Scoring positions required",
        )
        for p in self.positions:
            integer(p, "target position", 1)
            require(p < len(self.input_ids), "Target position exceeds complete input")
        require(
            self.positions == tuple(sorted(set(self.positions))),
            "Positions must be unique and ordered",
        )
        require(self.backend in ("mock", "real"), "Explicit scoring domain required")
        require(
            self.backend != "real" or self.model_ref.tokenizer.backend != "fixture",
            "Fixture model cannot produce real signals",
        )


@dataclass(frozen=True)
class TokenMetricResult(WorkflowRecord):
    request_hash: str
    positions: tuple[int, ...]
    token_nll: tuple[float, ...]
    token_entropy: tuple[float, ...]
    evidence: ArtifactRef
    usage: ResourceUsage
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported token statistics")
        sha256(self.request_hash, "request hash")
        require(
            type(self.positions) is tuple and bool(self.positions),
            "Token statistics need positions",
        )
        for p in self.positions:
            integer(p, "target position", 1)
        require(
            self.positions == tuple(sorted(set(self.positions))),
            "Duplicated or unordered positions",
        )
        for values in (self.token_nll, self.token_entropy):
            require(
                type(values) is tuple and len(values) == len(self.positions),
                "Misaligned token statistics",
            )
            for value in values:
                finite(value, "token statistic", 0)
        require(
            isinstance(self.evidence, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Scoring evidence/usage required",
        )
        require(self.backend in ("mock", "real"), "Invalid scoring domain")
        require(
            self.backend != "real"
            or not self.evidence.uri.strip()
            .lower()
            .startswith(("fixture://", "mock://")),
            "Mock scoring evidence cannot be real",
        )


@dataclass(frozen=True)
class LogitsOutput:
    """One raw, untempered full-vocabulary logit row per requested TARGET position.

    Producer owns model loading and row extraction: row for target p must come
    from the causal model's output at p-1 with the actual complete input prefix.
    No raw logits are copied into the general workflow JSON contracts.
    """

    request_hash: str
    model_ref: ModelRef
    positions: tuple[int, ...]
    logits: object
    evidence: ArtifactRef
    usage: ResourceUsage

    def __post_init__(self):
        sha256(self.request_hash, "logits request hash")
        require(
            isinstance(self.model_ref, ModelRef), "Logits need exact model identity"
        )
        require(
            type(self.positions) is tuple and bool(self.positions),
            "Logit target positions required",
        )
        require(
            isinstance(self.evidence, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Logits need evidence and incremental usage",
        )


class ScoringDataProvider(Protocol):
    def records(self) -> Iterable[TrajectoryRecord]: ...


class LogitsProvider(Protocol):
    model_ref: ModelRef
    implementation_hash: (
        str  # Include adapter, raw-logit dtype/config and remote code identity.
    )

    def forward(self, request: ForwardRequest) -> LogitsOutput: ...


class TokenMetricBackend(Protocol):
    backend: str
    implementation_hash: str

    def score(self, request: ForwardRequest, output: Path) -> TokenMetricResult: ...


@dataclass(frozen=True)
class SignalOutcome:
    data_id: str
    spec: SignalSpec
    status: str
    signal: SignalRecord | None
    reason: str | None = None

    def __post_init__(self):
        text(self.data_id, "outcome data ID")
        require(isinstance(self.spec, SignalSpec), "Outcome requires SignalSpec")
        require(
            self.status in ("computed", "cached", "missing"), "Invalid signal outcome"
        )
        if self.status == "missing":
            require(
                self.signal is None, "Missing signal cannot carry fabricated values"
            )
            text(self.reason, "missing reason")
        else:
            require(
                isinstance(self.signal, SignalRecord)
                and self.signal.binding.data_id == self.data_id
                and self.signal.spec == self.spec
                and self.signal.nll is not None
                and self.signal.entropy is not None,
                "Computed signals require complete bound NLL and entropy",
            )
            require(self.reason is None, "Computed result cannot have missing reason")

    def to_dict(self):
        return {
            "data_id": self.data_id,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "signal": None if self.signal is None else self.signal.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScoringBatch:
    batch_id: str
    model_ref: ModelRef
    computation_hash: str
    records: tuple[TrajectoryRecord, ...]
    outcomes: tuple[SignalOutcome, ...]
    ledger: CostLedger
    backend: str

    def __post_init__(self):
        text(self.batch_id, "scoring batch ID")
        sha256(self.computation_hash, "scoring computation hash")
        require(
            isinstance(self.model_ref, ModelRef)
            and isinstance(self.ledger, CostLedger),
            "Invalid scoring batch identities",
        )
        require(
            type(self.records) is tuple
            and all(isinstance(r, TrajectoryRecord) for r in self.records),
            "Typed scoring records required",
        )
        unique(tuple(r.data_id for r in self.records), "scoring data IDs")
        require(
            type(self.outcomes) is tuple
            and all(isinstance(o, SignalOutcome) for o in self.outcomes),
            "Typed signal outcomes required",
        )
        by_id = {r.data_id: r for r in self.records}
        unique(
            tuple(o.data_id + ":" + o.spec.spec_hash for o in self.outcomes),
            "signal outcomes",
        )
        require(
            self.backend in ("mock", "real") and self.backend == self.ledger.backend,
            "Mixed scoring domains",
        )
        require(
            self.backend != "real"
            or (
                self.model_ref.tokenizer.backend != "fixture"
                and not any(r.known_mock_artifacts for r in self.records)
            ),
            "Mock records/model cannot be published as real signals",
        )
        require(
            all(
                e.operation == "scoring"
                and e.attempt_id == self.batch_id
                and e.model_hash == self.model_ref.model_hash
                and e.subject_id in by_id
                and e.status == "completed"
                for e in self.ledger.events
            ),
            "Scoring ledger belongs to another batch/model/data",
        )
        require(
            all(
                getattr(e.usage, name) in (None, 0)
                for e in self.ledger.events
                for name in (
                    "new_gradient_samples",
                    "new_gradient_tokens",
                    "training_supervised_tokens",
                )
            ),
            "Forward scoring batch contains reported gradient or training work",
        )
        for item in self.outcomes:
            require(
                item.data_id in by_id
                and item.spec.computation_hash == self.computation_hash,
                "Foreign data/spec in scoring batch",
            )
            if item.signal is not None:
                require(
                    item.signal.model_ref == self.model_ref
                    and item.signal in by_id[item.data_id].signals,
                    "Scoring outcome was not attached to its record",
                )
            else:
                from self_curate.data import signal_binding
                from self_curate.contracts import ContractError, ErrorCode

                try:
                    signal_binding(by_id[item.data_id], item.spec)
                except ContractError as exc:
                    require(
                        exc.code == ErrorCode.MISSING_SIGNAL,
                        "Invalid missing-scope record",
                    )
                else:
                    require(False, "Available scope cannot be reported as missing")

    @property
    def updated_records(self):
        return self.records

    def metric_view(self):
        from self_curate.data import MetricView

        return MetricView(self.records)

    def to_dict(self):
        return {
            "schema_version": "1",
            "batch_id": self.batch_id,
            "model_ref": self.model_ref.to_dict(),
            "computation_hash": self.computation_hash,
            "backend": self.backend,
            "records": [
                {"data_id": r.data_id, "record_hash": r.record_hash}
                for r in self.records
            ],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "ledger": self.ledger.to_dict(),
        }
