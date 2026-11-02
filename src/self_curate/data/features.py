"""Validated attachment of externally supplied labels and gradient artifacts."""

from dataclasses import replace
from typing import Protocol

from self_curate.contracts import (
    AnnotationRecord,
    ContractError,
    DataLabels,
    ErrorCode,
    GradientFeature,
    GradientSpec,
    MetadataRecord,
    ModelRef,
)
from self_curate.contracts.types import digest, require
from .enrichment import attach_annotations, data_binding


class GradientProvider(Protocol):
    def compute(
        self, record, model_ref: ModelRef, spec: GradientSpec
    ) -> GradientFeature: ...


def make_annotation_record(record, annotations, *, provenance):
    return AnnotationRecord(
        data_binding(record),
        digest(record.source.to_dict()),
        annotations,
        tuple(provenance),
    )


def attach_annotation_record(record, envelope: AnnotationRecord, **stale_policy):
    require(isinstance(envelope, AnnotationRecord), "Expected AnnotationRecord")
    require(
        envelope.binding == data_binding(record)
        and envelope.source_hash == digest(record.source.to_dict()),
        "Annotation labels belong to another data/version",
        ErrorCode.STALE_CACHE,
    )
    updated = attach_annotations(record, envelope.annotations, **stale_policy)
    return replace(updated, annotation_record=envelope, normalization_version="m1-v3")


def make_metadata_record(record, labels: DataLabels | None, *, provenance):
    return MetadataRecord(
        data_binding(record), digest(record.source.to_dict()), labels, tuple(provenance)
    )


def attach_metadata_record(record, envelope: MetadataRecord):
    require(isinstance(envelope, MetadataRecord), "Expected MetadataRecord")
    require(
        envelope.binding == data_binding(record)
        and envelope.source_hash == digest(record.source.to_dict()),
        "Metadata labels belong to another data/version",
        ErrorCode.STALE_CACHE,
    )
    return replace(
        record,
        metadata=envelope.labels,
        metadata_record=envelope,
        normalization_version="m1-v3",
    )


def gradient_binding(record, spec: GradientSpec):
    require(isinstance(spec, GradientSpec), "Expected GradientSpec")
    return data_binding(record, spec.scope, spec.skill)


def validate_gradient(record, feature: GradientFeature):
    require(isinstance(feature, GradientFeature), "Expected GradientFeature")
    require(
        feature.model_ref.tokenizer == record.tokenization.identity,
        "Gradient tokenizer/template changed",
        ErrorCode.STALE_CACHE,
    )
    try:
        expected = gradient_binding(record, feature.spec)
    except ContractError as exc:
        if exc.code == ErrorCode.MISSING_SIGNAL:
            raise ContractError(
                ErrorCode.STALE_CACHE, "Gradient scope no longer exists"
            ) from exc
        raise
    require(
        feature.binding == expected,
        "Gradient data, tokens, mask or selected positions changed",
        ErrorCode.STALE_CACHE,
    )


def make_gradient(
    record,
    model_ref: ModelRef,
    spec: GradientSpec,
    *,
    vector,
    dimension,
    dtype,
    norm,
    provenance,
    cost=None,
):
    from self_curate.contracts import ComputeCost

    feature = GradientFeature(
        gradient_binding(record, spec),
        model_ref,
        spec,
        vector,
        dimension,
        dtype,
        norm,
        tuple(provenance),
        ComputeCost() if cost is None else cost,
    )
    validate_gradient(record, feature)
    return feature


def attach_gradients(record, features):
    features = tuple(features)
    if not features:
        return record
    merged = {}
    for feature in (*record.gradients, *features):
        validate_gradient(record, feature)
        old = merged.get(feature.cache_key)
        if old is not None:
            require(
                (old.vector, old.dimension, old.dtype, old.norm)
                == (feature.vector, feature.dimension, feature.dtype, feature.norm),
                "Conflicting gradient under the same cache identity",
                ErrorCode.GRADIENT_CONFLICT,
            )
            from self_curate.contracts import ComputeCost

            cost = {}
            for name in ("wall_seconds", "gpu_seconds", "processed_tokens"):
                a, b = getattr(old.cost, name), getattr(feature.cost, name)
                require(
                    a is None or b is None or a == b,
                    "Conflicting cost declarations for one gradient; keep separate attempt ledger",
                    ErrorCode.GRADIENT_CONFLICT,
                )
                cost[name] = a if a is not None else b
            feature = replace(
                old,
                cost=ComputeCost(**cost),
                provenance=tuple(
                    sorted(
                        set(old.provenance + feature.provenance),
                        key=lambda r: (r.uri, r.sha256),
                    )
                ),
            )
        merged[feature.cache_key] = feature
    refs = set(record.usage.gradient_extraction)
    for feature in merged.values():
        refs.update(feature.provenance)
    return replace(
        record,
        gradients=tuple(merged[k] for k in sorted(merged)),
        usage=replace(
            record.usage,
            gradient_extraction=tuple(sorted(refs, key=lambda r: (r.uri, r.sha256))),
        ),
        normalization_version="m1-v3",
    )


def get_gradient(record, model_ref, spec):
    for feature in record.gradients:
        if feature.model_ref == model_ref and feature.spec == spec:
            validate_gradient(record, feature)
            return feature
    raise ContractError(
        ErrorCode.MISSING_SIGNAL,
        "No gradient in requested model/specification",
        model_hash=model_ref.model_hash,
        spec_hash=spec.spec_hash,
    )
