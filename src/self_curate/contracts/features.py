"""Data-bound labels and versioned sample-gradient references; no computation."""

from dataclasses import dataclass
import math

from .signals import ModelRef, SignalBinding
from .types import (
    AnnotationSet,
    ArtifactRef,
    Record,
    digest,
    integer,
    require,
    sha256,
    text,
)


@dataclass(frozen=True)
class DataTag(Record):
    name: str
    value: str

    def __post_init__(self):
        text(self.name, "tag.name")
        text(self.value, "tag.value")


@dataclass(frozen=True)
class DataLabels(Record):
    task_type: str | None = None
    repository: str | None = None
    duplicate_groups: tuple[str, ...] = ()
    tags: tuple[DataTag, ...] = ()
    schema_version: str = "1"

    def __post_init__(self):
        for name in ("task_type", "repository"):
            if getattr(self, name) is not None:
                text(getattr(self, name), name)
        require(type(self.duplicate_groups) is tuple, "duplicate_groups must be tuple")
        for group in self.duplicate_groups:
            text(group, "duplicate_group")
        require(
            len(set(self.duplicate_groups)) == len(self.duplicate_groups),
            "Duplicate group label",
        )
        require(
            type(self.tags) is tuple and all(isinstance(t, DataTag) for t in self.tags),
            "Invalid data tags",
        )
        require(len(set(self.tags)) == len(self.tags), "Duplicate data tag")
        require(self.schema_version == "1", "Unsupported DataLabels schema")


def _provenance(refs):
    require(
        type(refs) is tuple
        and bool(refs)
        and all(isinstance(r, ArtifactRef) for r in refs),
        "External results require provenance references",
    )
    require(len(set(refs)) == len(refs), "Duplicate provenance")


def _finite_nonnegative(value, name):
    require(type(value) in (int, float), f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError:
        require(False, f"{name} exceeds finite float range")
    require(
        math.isfinite(result) and result >= 0, f"{name} must be finite and nonnegative"
    )
    return result


@dataclass(frozen=True)
class AnnotationRecord(Record):
    binding: SignalBinding
    source_hash: str
    annotations: AnnotationSet | None
    provenance: tuple[ArtifactRef, ...]
    schema_version: str = "1"

    def __post_init__(self):
        require(
            isinstance(self.binding, SignalBinding), "Invalid annotation data binding"
        )
        sha256(self.source_hash, "annotation.source_hash")
        require(
            self.annotations is None or isinstance(self.annotations, AnnotationSet),
            "Invalid annotations",
        )
        _provenance(self.provenance)
        require(self.schema_version == "1", "Unsupported AnnotationRecord schema")


@dataclass(frozen=True)
class MetadataRecord(Record):
    binding: SignalBinding
    source_hash: str
    labels: DataLabels | None
    provenance: tuple[ArtifactRef, ...]
    schema_version: str = "1"

    def __post_init__(self):
        require(
            isinstance(self.binding, SignalBinding), "Invalid metadata data binding"
        )
        sha256(self.source_hash, "metadata.source_hash")
        require(
            self.labels is None or isinstance(self.labels, DataLabels),
            "Invalid metadata",
        )
        _provenance(self.provenance)
        require(self.schema_version == "1", "Unsupported MetadataRecord schema")


@dataclass(frozen=True)
class ProjectionSpec(Record):
    method: str
    dimension: int
    seed: int
    config_hash: str
    schema_version: str = "1"

    def __post_init__(self):
        text(self.method, "projection.method")
        integer(self.dimension, "projection.dimension", 1)
        integer(self.seed, "projection.seed")
        sha256(self.config_hash, "projection.config_hash")
        require(self.schema_version == "1", "Unsupported ProjectionSpec schema")


@dataclass(frozen=True)
class GradientSpec(Record):
    parameter_coordinates_hash: str
    parameter_count: int
    loss_id: str
    reduction: str
    normalization_id: str
    normalization_hash: str
    compute_dtype: str
    transform_id: str
    transform_hash: str
    implementation_id: str
    implementation_hash: str
    computation_hash: str
    storage_dtype: str = "float32"
    projection: ProjectionSpec | None = None
    optimizer_state_hash: str | None = None
    scope: str = "assistant"
    skill: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        for name in (
            "parameter_coordinates_hash",
            "normalization_hash",
            "transform_hash",
            "implementation_hash",
            "computation_hash",
        ):
            sha256(getattr(self, name), name)
        integer(self.parameter_count, "parameter_count", 1)
        for name in (
            "loss_id",
            "normalization_id",
            "transform_id",
            "implementation_id",
        ):
            text(getattr(self, name), name)
        require(
            self.reduction in ("token_mean", "token_sum"), "Invalid gradient reduction"
        )
        require(
            self.compute_dtype in ("float16", "bfloat16", "float32", "float64"),
            "Invalid compute dtype",
        )
        require(
            self.storage_dtype in ("float16", "bfloat16", "float32", "float64"),
            "Invalid storage dtype",
        )
        require(self.scope in ("assistant", "skill_evidence"), "Invalid gradient scope")
        require(
            (self.skill is not None) == (self.scope == "skill_evidence"),
            "Skill scope requires skill",
        )
        if self.skill is not None:
            text(self.skill, "gradient.skill")
        require(
            self.projection is None or isinstance(self.projection, ProjectionSpec),
            "Invalid projection",
        )
        if self.optimizer_state_hash is not None:
            sha256(self.optimizer_state_hash, "optimizer_state_hash")
        require(self.schema_version == "1", "Unsupported GradientSpec schema")

    @property
    def spec_hash(self):
        return digest(self.to_dict())


@dataclass(frozen=True)
class ComputeCost(Record):
    wall_seconds: float | None = None
    gpu_seconds: float | None = None
    processed_tokens: int | None = None
    schema_version: str = "1"

    def __post_init__(self):
        for name in ("wall_seconds", "gpu_seconds"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_nonnegative(value, name))
        if self.processed_tokens is not None:
            integer(self.processed_tokens, "processed_tokens")
        require(self.schema_version == "1", "Unsupported ComputeCost schema")


@dataclass(frozen=True)
class GradientFeature(Record):
    binding: SignalBinding
    model_ref: ModelRef
    spec: GradientSpec
    vector: ArtifactRef
    dimension: int
    dtype: str
    norm: float
    provenance: tuple[ArtifactRef, ...]
    cost: ComputeCost = ComputeCost()
    schema_version: str = "1"

    def __post_init__(self):
        require(
            isinstance(self.binding, SignalBinding), "Invalid gradient data binding"
        )
        require(
            isinstance(self.model_ref, ModelRef)
            and isinstance(self.spec, GradientSpec),
            "Invalid gradient context",
        )
        require(
            isinstance(self.vector, ArtifactRef),
            "Gradient vector needs a content-hashed artifact",
        )
        integer(self.dimension, "gradient.dimension", 1)
        expected = (
            self.spec.parameter_count
            if self.spec.projection is None
            else self.spec.projection.dimension
        )
        require(
            self.dimension == expected,
            "Gradient dimension does not match coordinate/projection specification",
        )
        require(
            self.dtype in ("float16", "bfloat16", "float32", "float64"),
            "Invalid vector dtype",
        )
        require(
            self.dtype == self.spec.storage_dtype,
            "Vector dtype differs from gradient storage specification",
        )
        object.__setattr__(
            self, "norm", _finite_nonnegative(self.norm, "gradient.norm")
        )
        require(isinstance(self.cost, ComputeCost), "Invalid gradient cost")
        _provenance(self.provenance)
        require(self.schema_version == "1", "Unsupported GradientFeature schema")

    @property
    def validity(self):
        return "zero" if self.norm == 0 else "valid"

    @property
    def cache_key(self):
        return digest(
            {
                "binding": self.binding.to_dict(),
                "model": self.model_ref.to_dict(),
                "spec": self.spec.to_dict(),
            }
        )
