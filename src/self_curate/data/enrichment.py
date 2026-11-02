"""Attach external annotations/signals, without invoking their producers."""

from dataclasses import replace
from typing import Iterable, Protocol

from self_curate.contracts import (
    AnnotationRecord,
    AnnotationSet,
    ContractError,
    ErrorCode,
    ModelRef,
    PercentileReference,
    SignalBinding,
    SignalRecord,
    SignalSpec,
    TrajectoryRecord,
    digest,
)
from self_curate.contracts.types import require, text
from .normalize import evidence_metrics


class AnnotationProvider(Protocol):
    def annotate(self, record: TrajectoryRecord) -> "AnnotationRecord | None": ...


class SignalProvider(Protocol):
    def compute(
        self, record: TrajectoryRecord, model_ref: ModelRef, spec: SignalSpec
    ) -> SignalRecord: ...


def data_binding(
    record: TrajectoryRecord, scope="assistant", skill=None
) -> SignalBinding:
    require(
        isinstance(record, TrajectoryRecord),
        "Expected trajectory",
    )
    require(
        scope in ("assistant", "skill_evidence")
        and (skill is not None) == (scope == "skill_evidence"),
        "Invalid feature scope",
    )
    mask = record.tokenization.assistant_mask
    if scope == "skill_evidence":
        annotation = (
            next((a for a in record.annotations.skills if a.skill == skill), None)
            if record.annotations is not None
            else None
        )
        if annotation is None or annotation.spans is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "Skill evidence has not been annotated",
                skill=skill,
            )
        selection = [False] * len(mask)
        for span in annotation.spans:
            require(
                span.message_index < len(record.tokenization.message_spans),
                "Invalid evidence message index",
                ErrorCode.INVALID_EVIDENCE,
            )
            payload = record.tokenization.message_spans[span.message_index]
            require(
                payload.start <= span.start < span.end <= payload.end,
                "Invalid evidence token span",
                ErrorCode.INVALID_EVIDENCE,
            )
            for i in range(span.start, span.end):
                selection[i] = mask[i]
        mask = tuple(selection)
    require(
        any(mask),
        "Requested signal scope has no supervised tokens",
        ErrorCode.MISSING_SIGNAL,
        scope=scope,
        skill=skill,
    )

    return SignalBinding(
        record.data_id,
        record.content_hash,
        digest(record.tokenization.to_dict()),
        digest(list(mask)),
        sum(mask),
    )


def signal_binding(record: TrajectoryRecord, spec: SignalSpec) -> SignalBinding:
    require(isinstance(spec, SignalSpec), "Expected SignalSpec")
    return data_binding(record, spec.scope, spec.skill)


def validate_signal(record: TrajectoryRecord, signal: SignalRecord) -> None:
    require(isinstance(signal, SignalRecord), "Expected SignalRecord")
    require(
        signal.model_ref.tokenizer == record.tokenization.identity,
        "Signal tokenizer/template differs from trajectory",
        ErrorCode.STALE_CACHE,
    )
    try:
        expected = signal_binding(record, signal.spec)
    except ContractError as exc:
        if exc.code == ErrorCode.MISSING_SIGNAL:
            raise ContractError(
                ErrorCode.STALE_CACHE,
                "Signal scope no longer has compatible annotations",
            ) from exc
        raise
    require(
        signal.binding == expected,
        "Signal data, tokenization, selected positions or token count changed",
        ErrorCode.STALE_CACHE,
    )


def make_signal(
    record: TrajectoryRecord,
    model_ref: ModelRef,
    spec: SignalSpec,
    *,
    provenance,
    nll=None,
    entropy=None,
    nll_percentile=None,
    entropy_percentile=None,
    percentile_reference: PercentileReference | None = None,
) -> SignalRecord:
    """Package already-computed values. This function performs no model work."""
    signal = SignalRecord(
        signal_binding(record, spec),
        model_ref,
        spec,
        tuple(provenance),
        nll,
        entropy,
        nll_percentile,
        entropy_percentile,
        percentile_reference,
    )
    validate_signal(record, signal)
    return signal


def _merge(left, right):
    values = {}
    for name in ("nll", "entropy", "nll_percentile", "entropy_percentile"):
        a, b = getattr(left, name), getattr(right, name)
        require(
            a is None or b is None or a == b,
            "Conflicting values under the same signal identity",
            ErrorCode.SIGNAL_CONFLICT,
            metric=name,
        )
        values[name] = a if a is not None else b
    refs = tuple(
        sorted(set(left.provenance + right.provenance), key=lambda r: (r.uri, r.sha256))
    )
    return replace(left, **values, provenance=refs)


def attach_signals(
    record: TrajectoryRecord, signals: Iterable[SignalRecord]
) -> TrajectoryRecord:
    """Return an enriched copy; merge complementary fields, reject conflicts."""
    supplied = tuple(signals)
    if not supplied:
        return record
    merged = {}
    for signal in (*record.signals, *supplied):
        validate_signal(record, signal)
        # CDF views must agree about their common underlying raw computation.
        for previous in merged.values():
            if previous.raw_key == signal.raw_key:
                for name in ("nll", "entropy"):
                    a, b = getattr(previous, name), getattr(signal, name)
                    require(
                        a is None or b is None or a == b,
                        "CDF views disagree about raw signal",
                        ErrorCode.SIGNAL_CONFLICT,
                        metric=name,
                    )
        merged[signal.cache_key] = (
            _merge(merged[signal.cache_key], signal)
            if signal.cache_key in merged
            else signal
        )
    forward = set(record.usage.forward_scoring)
    for signal in merged.values():
        forward.update(signal.provenance)
    usage = replace(
        record.usage,
        forward_scoring=tuple(sorted(forward, key=lambda r: (r.uri, r.sha256))),
    )
    return replace(
        record,
        signals=tuple(merged[k] for k in sorted(merged)),
        usage=usage,
        normalization_version="m1-v3",
    )


def get_signal(
    record: TrajectoryRecord,
    model_ref: ModelRef,
    spec: SignalSpec,
    *,
    percentile_reference: PercentileReference | None = None,
) -> SignalRecord:
    """Exact-context lookup. Never substitute another checkpoint/scope/CDF."""
    require(isinstance(model_ref, ModelRef), "Expected ModelRef")
    require(
        percentile_reference is None
        or isinstance(percentile_reference, PercentileReference),
        "Expected PercentileReference or missing",
    )
    expected = signal_binding(record, spec)
    for signal in record.signals:
        if (
            signal.model_ref == model_ref
            and signal.spec == spec
            and signal.percentile_reference == percentile_reference
        ):
            validate_signal(record, signal)
            require(
                signal.binding == expected,
                "Signal binding changed",
                ErrorCode.STALE_CACHE,
            )
            return signal
    raise ContractError(
        ErrorCode.MISSING_SIGNAL,
        "No signal in the requested context",
        model_hash=model_ref.model_hash,
        spec_hash=spec.spec_hash,
    )


def attach_annotations(
    record: TrajectoryRecord,
    annotations: AnnotationSet | None,
    *,
    declared_skills: tuple[str, ...] | None = None,
    drop_stale_signals: bool = False,
    drop_stale_gradients: bool = False,
) -> TrajectoryRecord:
    """Attach external labels, recompute evidence metrics, protect skill signals.

    Stale skill-scoped signals raise by default. Explicit dropping removes them
    from the new record only; the original record and scoring history remain.
    """
    require(
        annotations is None or isinstance(annotations, AnnotationSet),
        "Expected AnnotationSet or missing",
    )
    require(type(drop_stale_signals) is bool, "drop_stale_signals must be boolean")
    require(type(drop_stale_gradients) is bool, "drop_stale_gradients must be boolean")
    skills = (
        tuple(m.skill for m in record.evidence_metrics)
        if declared_skills is None
        else declared_skills
    )
    require(type(skills) is tuple, "declared_skills must be tuple")
    for skill in skills:
        text(skill, "declared_skill")
    require(len(set(skills)) == len(skills), "Duplicate declared skill")
    if declared_skills and annotations is not None:
        require(
            {a.skill for a in annotations.skills}.issubset(skills),
            "Undeclared annotation skill",
        )
    metrics, union = evidence_metrics(record.tokenization, annotations, skills)
    updated = replace(
        record,
        annotations=annotations,
        evidence_metrics=metrics,
        stats=replace(record.stats, evidence_union_tokens=union),
        signals=(),
        gradients=(),
        annotation_record=None,
        normalization_version="m1-v3",
    )
    valid = []
    for signal in record.signals:
        try:
            validate_signal(updated, signal)
        except ContractError as exc:
            if not drop_stale_signals or exc.code != ErrorCode.STALE_CACHE:
                raise
        else:
            valid.append(signal)
    updated = attach_signals(updated, valid)
    from .features import attach_gradients, validate_gradient

    gradients = []
    for feature in record.gradients:
        try:
            validate_gradient(updated, feature)
        except ContractError as exc:
            if not drop_stale_gradients or exc.code != ErrorCode.STALE_CACHE:
                raise
        else:
            gradients.append(feature)
    return attach_gradients(updated, gradients)
