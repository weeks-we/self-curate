"""Versioned containers for externally computed signals; no scoring algorithms."""

from dataclasses import dataclass
import math
from typing import Literal

from .errors import ContractError, ErrorCode
from .types import (
    ArtifactRef,
    Record,
    TokenizerRef,
    digest,
    integer,
    require,
    sha256,
    text,
)


@dataclass(frozen=True)
class ModelRef(Record):
    base_model_id: str
    revision: str
    weights_hash: str
    tokenizer: TokenizerRef
    parameterization: str
    adapter_hash: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        text(self.base_model_id, "model.base_model_id")
        text(self.revision, "model.revision")
        text(self.parameterization, "model.parameterization")
        sha256(self.weights_hash, "model.weights_hash")
        require(isinstance(self.tokenizer, TokenizerRef), "Invalid model tokenizer")
        if self.adapter_hash is not None:
            sha256(self.adapter_hash, "model.adapter_hash")
        require(self.schema_version == "1", "Unsupported ModelRef schema")

    @property
    def model_hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class SignalSpec(Record):
    scope: Literal["assistant", "skill_evidence"]
    reduction: Literal["token_mean", "token_sum"]
    log_base: Literal["e", "2"]
    computation_id: str
    computation_hash: str
    skill: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        require(
            isinstance(self.scope, str)
            and self.scope in {"assistant", "skill_evidence"},
            "Invalid signal scope",
        )
        require(
            isinstance(self.reduction, str)
            and self.reduction in {"token_mean", "token_sum"},
            "Invalid signal reduction",
        )
        require(
            isinstance(self.log_base, str) and self.log_base in {"e", "2"},
            "Invalid logarithm base",
        )
        text(self.computation_id, "signal.computation_id")
        sha256(self.computation_hash, "signal.computation_hash")
        require(
            (self.skill is not None) == (self.scope == "skill_evidence"),
            "Only skill_evidence scope requires a skill",
        )
        if self.skill is not None:
            text(self.skill, "signal.skill")
        require(self.schema_version == "1", "Unsupported SignalSpec schema")

    @property
    def spec_hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class SignalBinding(Record):
    data_id: str
    content_hash: str
    tokenization_hash: str
    selection_hash: str
    token_count: int
    schema_version: str = "1"

    def __post_init__(self):
        text(self.data_id, "signal.data_id")
        for name in ("content_hash", "tokenization_hash", "selection_hash"):
            sha256(getattr(self, name), f"signal.{name}")
        integer(self.token_count, "signal.token_count", minimum=1)
        require(self.schema_version == "1", "Unsupported SignalBinding schema")


@dataclass(frozen=True)
class PercentileReference(Record):
    reference_group: str
    reference_ids_hash: str
    cdf_version: str
    artifact: ArtifactRef
    model_hash: str
    spec_hash: str
    schema_version: str = "1"

    def __post_init__(self):
        text(self.reference_group, "reference_group")
        text(self.cdf_version, "cdf_version")
        for name in ("reference_ids_hash", "model_hash", "spec_hash"):
            sha256(getattr(self, name), name)
        require(
            isinstance(self.artifact, ArtifactRef),
            "CDF must reference a content-hashed artifact",
        )
        require(self.schema_version == "1", "Unsupported percentile reference schema")


@dataclass(frozen=True)
class SignalRecord(Record):
    binding: SignalBinding
    model_ref: ModelRef
    spec: SignalSpec
    provenance: tuple[ArtifactRef, ...]
    nll: float | None = None
    entropy: float | None = None
    nll_percentile: float | None = None
    entropy_percentile: float | None = None
    percentile_reference: PercentileReference | None = None
    schema_version: str = "1"

    def __post_init__(self):
        require(isinstance(self.binding, SignalBinding), "Invalid signal binding")
        require(isinstance(self.model_ref, ModelRef), "Invalid signal model")
        require(isinstance(self.spec, SignalSpec), "Invalid signal specification")
        require(
            type(self.provenance) is tuple
            and bool(self.provenance)
            and all(isinstance(p, ArtifactRef) for p in self.provenance),
            "Computed signals require provenance references",
        )
        require(
            len(set(self.provenance)) == len(self.provenance),
            "Duplicate signal provenance",
        )
        require(
            self.nll is not None or self.entropy is not None,
            "An empty signal is missing, not a computed record",
        )
        for name in ("nll", "entropy", "nll_percentile", "entropy_percentile"):
            value = getattr(self, name)
            if value is None:
                continue
            require(type(value) in {int, float}, f"{name} must be a number or null")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ContractError(
                    ErrorCode.INVALID_RECORD, f"{name} exceeds finite float range"
                ) from exc
            require(
                math.isfinite(value) and value >= 0,
                f"{name} must be finite and nonnegative",
            )
            if name.endswith("percentile"):
                require(value <= 1, f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        ref = self.percentile_reference
        require(
            ref is None or isinstance(ref, PercentileReference),
            "Invalid percentile reference",
        )
        if ref is not None:
            require(
                ref.model_hash == self.model_ref.model_hash
                and ref.spec_hash == self.spec.spec_hash,
                "CDF belongs to another model or scoring specification",
                ErrorCode.STALE_CACHE,
            )
        for metric in ("nll", "entropy"):
            if getattr(self, f"{metric}_percentile") is not None:
                require(
                    ref is not None and getattr(self, metric) is not None,
                    "Percentile needs its raw value and explicit CDF reference",
                )
        require(self.schema_version == "1", "Unsupported SignalRecord schema")

    @property
    def raw_key(self):
        return digest(
            {
                "binding": self.binding.to_dict(),
                "model": self.model_ref.to_dict(),
                "spec": self.spec.to_dict(),
            }
        )

    @property
    def cache_key(self):
        return digest(
            {
                "raw_key": self.raw_key,
                "reference": (
                    None
                    if self.percentile_reference is None
                    else self.percentile_reference.to_dict()
                ),
            }
        )

    def require_value(
        self,
        metric: Literal["nll", "entropy"],
        scale: Literal["raw", "percentile"] = "raw",
    ) -> float:
        require(
            metric in {"nll", "entropy"} and scale in {"raw", "percentile"},
            "Unknown signal metric or scale",
        )
        name = metric if scale == "raw" else f"{metric}_percentile"
        value = getattr(self, name)
        if value is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "Requested signal has not been supplied",
                metric=metric,
                scale=scale,
                cache_key=self.cache_key,
            )
        return value
