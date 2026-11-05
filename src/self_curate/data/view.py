"""In-memory, ID-aligned metric access for operators. No persistence or filtering."""

from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType

from self_curate.contracts import (
    ContractError,
    ErrorCode,
    GradientSpec,
    ModelRef,
    PercentileReference,
    SignalSpec,
    TrajectoryRecord,
    digest,
)
from self_curate.contracts.types import require, text
from .enrichment import data_binding
from .normalize import evidence_metrics, validate_tool_links
from .tokenization import validate_alignment


def _reference_key(reference):
    require(
        reference is None or isinstance(reference, PercentileReference),
        "Expected PercentileReference or missing",
    )
    return None if reference is None else digest(reference.to_dict())


def _signal_context(model_ref, spec, reference):
    require(
        isinstance(model_ref, ModelRef) and isinstance(spec, SignalSpec),
        "Expected ModelRef and SignalSpec",
    )
    ref_key = _reference_key(reference)
    if reference is not None:
        require(
            reference.model_hash == model_ref.model_hash
            and reference.spec_hash == spec.spec_hash,
            "CDF version does not match query context",
            ErrorCode.STALE_CACHE,
        )
    return (model_ref.model_hash, spec.spec_hash, ref_key)


@dataclass(frozen=True)
class MetricBatch:
    data_ids: tuple[str, ...]
    content_hashes: tuple[str, ...]
    columns: object
    missing: tuple[tuple[str, str], ...]
    context: object

    def __post_init__(self):
        require(
            type(self.data_ids) is tuple
            and len(set(self.data_ids)) == len(self.data_ids),
            "Metric batch IDs must be unique",
        )
        for data_id in self.data_ids:
            text(data_id, "data_id")
        require(
            type(self.content_hashes) is tuple
            and len(self.content_hashes) == len(self.data_ids),
            "Content versions must align with data IDs",
        )
        require(
            all(
                type(v) is tuple and len(v) == len(self.data_ids)
                for v in self.columns.values()
            ),
            "Metric columns must align with data_ids",
        )
        expected = tuple(
            (data_id, name)
            for i, data_id in enumerate(self.data_ids)
            for name, values in self.columns.items()
            if values[i] is None
        )
        require(
            self.missing == expected, "Missing-cell labels must match column contents"
        )
        object.__setattr__(self, "columns", MappingProxyType(dict(self.columns)))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def column(self, name):
        require(name in self.columns, "Unknown requested column", field=name)
        return self.columns[name]

    def by_id(self):
        return {
            data_id: {name: values[i] for name, values in self.columns.items()}
            for i, data_id in enumerate(self.data_ids)
        }


class MetricView:
    """Build once for an immutable record snapshot; subsequent lookups use dicts.

    Expensive token/position validation happens at construction, once per
    data_id/scope/skill. Batch reads preserve requested ID order, including
    missing cells. Rebuild explicitly after enriching or updating usage.
    """

    LENGTH_FIELDS = (
        "assistant_turns",
        "tool_calls",
        "tool_observations",
        "context_tokens",
        "supervised_tokens",
    )
    SIGNAL_FIELDS = ("nll", "entropy", "nll_percentile", "entropy_percentile")
    EVIDENCE_FIELDS = (
        "share",
        "density",
        "supervised_evidence_tokens",
        "complete_events",
        "behaviors",
        "status",
    )

    def __init__(self, records):
        by_id, static, evidence, signals, gradients, bindings, raw_values = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        signal_keys, gradient_keys = {}, {}
        for record in records:
            require(
                isinstance(record, TrajectoryRecord),
                "MetricView requires TrajectoryRecord",
            )
            # A caller may construct/replace a record without passing through
            # read_pool(). Verify cross-field semantics once at this boundary.
            validate_tool_links(record.messages)
            validate_alignment(record.tokenization, record.messages)
            skills = tuple(m.skill for m in record.evidence_metrics)
            require(
                len(set(skills)) == len(skills),
                "Duplicate evidence skill rows",
                ErrorCode.INVALID_EVIDENCE,
            )
            expected, union = evidence_metrics(
                record.tokenization, record.annotations, skills
            )
            require(
                {m.skill: m for m in expected}
                == {m.skill: m for m in record.evidence_metrics}
                and union == record.stats.evidence_union_tokens,
                "Evidence metrics disagree with actual annotations/token mask",
                ErrorCode.INVALID_EVIDENCE,
                data_id=record.data_id,
            )
            require(
                record.data_id not in by_id,
                "Duplicate data_id in metric view",
                ErrorCode.DUPLICATE_ID,
                data_id=record.data_id,
            )
            data_id = record.data_id
            by_id[data_id] = record
            usage = record.usage
            labels = record.metadata
            static[data_id] = {
                **{name: getattr(record.stats, name) for name in self.LENGTH_FIELDS},
                "quality_status": record.quality.status,
                "skills": (
                    None
                    if record.annotations is None
                    else tuple(sorted(a.skill for a in record.annotations.skills))
                ),
                "source_id": record.source.source_id,
                "family_ids": record.source.family_ids,
                "task_type": None if labels is None else labels.task_type,
                "repository": None if labels is None else labels.repository,
                "duplicate_groups": None if labels is None else labels.duplicate_groups,
                "tags": (
                    None
                    if labels is None
                    else tuple((t.name, t.value) for t in labels.tags)
                ),
                "annotation_version": (
                    None
                    if record.annotations is None
                    else digest(
                        record.annotation_record.to_dict()
                        if record.annotation_record is not None
                        else record.annotations.to_dict()
                    )
                ),
                "metadata_version": (
                    None
                    if labels is None
                    else digest(
                        record.metadata_record.to_dict()
                        if record.metadata_record is not None
                        else labels.to_dict()
                    )
                ),
                "source_version": digest(record.source.to_dict()),
                "usage_version": digest(usage.to_dict()),
                "used_forward_scoring": bool(usage.forward_scoring),
                "used_gradient_extraction": bool(usage.gradient_extraction),
                "used_candidate_sft": bool(usage.candidate_sft),
                "used_accepted_sft": bool(usage.accepted_sft),
                "used_rejected_sft": any(
                    u.decision == "rejected" for u in usage.candidate_sft
                ),
                "pending_sft": any(
                    u.decision == "pending" for u in usage.candidate_sft
                ),
            }
            for metric in record.evidence_metrics:
                evidence[data_id, metric.skill] = metric
            for feature, target in chain(
                ((s, signals) for s in record.signals),
                ((g, gradients) for g in record.gradients),
            ):
                binding_key = (data_id, feature.spec.scope, feature.spec.skill)
                if binding_key not in bindings:
                    bindings[binding_key] = data_binding(
                        record, feature.spec.scope, feature.spec.skill
                    )
                require(
                    feature.binding == bindings[binding_key]
                    and feature.model_ref.tokenizer == record.tokenization.identity,
                    "Stale feature cannot enter a metric view",
                    ErrorCode.STALE_CACHE,
                )
                key = (data_id, feature.model_ref.model_hash, feature.spec.spec_hash)
                if target is signals:
                    previous = raw_values.get(key, (None, None))
                    current = (feature.nll, feature.entropy)
                    require(
                        all(
                            a is None or b is None or a == b
                            for a, b in zip(previous, current)
                        ),
                        "Conflicting raw values across CDF views",
                        ErrorCode.SIGNAL_CONFLICT,
                    )
                    raw_values[key] = tuple(
                        a if a is not None else b for a, b in zip(previous, current)
                    )
                    key += (_reference_key(feature.percentile_reference),)
                require(
                    key not in target,
                    "Ambiguous metric context for data_id",
                    ErrorCode.INVALID_RECORD,
                    data_id=data_id,
                )
                target[key] = feature
                (signal_keys if target is signals else gradient_keys)[
                    key
                ] = feature.cache_key
        self._records = MappingProxyType(by_id)
        self._static = MappingProxyType(
            {k: MappingProxyType(v) for k, v in static.items()}
        )
        self._evidence = MappingProxyType(evidence)
        self._signals = MappingProxyType(signals)
        self._gradients = MappingProxyType(gradients)
        self._signal_keys = MappingProxyType(signal_keys)
        self._gradient_keys = MappingProxyType(gradient_keys)
        self._data_ids = tuple(by_id)

    @property
    def data_ids(self):
        """Read-only membership of the snapshot used by execution identities."""
        return self._data_ids

    def record(self, data_id):
        text(data_id, "data_id")
        if data_id not in self._records:
            raise ContractError(
                ErrorCode.UNKNOWN_DATA_ID, "Unknown data_id", data_id=data_id
            )
        return self._records[data_id]

    def _ids(self, data_ids):
        require(
            not isinstance(data_ids, str), "Pass an iterable of IDs, not one string"
        )
        ids = tuple(data_ids)
        for data_id in ids:
            text(data_id, "data_id")
            self.record(data_id)
        require(
            len(set(ids)) == len(ids),
            "Repeated IDs would duplicate metric rows",
            ErrorCode.DUPLICATE_ID,
        )
        return ids

    def _batch(self, ids, fields, getter, missing, context):
        require(missing in ("error", "none"), "Missing policy must be error or none")
        require(
            type(fields) is tuple and bool(fields) and len(set(fields)) == len(fields),
            "Requested fields must be a nonempty unique tuple",
        )
        columns = {name: [] for name in fields}
        absent = []
        for data_id in ids:
            for name in fields:
                value = getter(data_id, name)
                if value is None:
                    absent.append((data_id, name))
                columns[name].append(value)
        if absent and missing == "error":
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "Requested cells have not been supplied",
                missing=absent,
            )
        return MetricBatch(
            ids,
            tuple(self._records[d].content_hash for d in ids),
            {name: tuple(values) for name, values in columns.items()},
            tuple(absent),
            context,
        )

    def static_batch(self, data_ids, fields, *, missing="none"):
        ids = self._ids(data_ids)
        allowed = set(self.LENGTH_FIELDS) | {
            "quality_status",
            "skills",
            "source_id",
            "family_ids",
            "task_type",
            "repository",
            "duplicate_groups",
            "tags",
            "annotation_version",
            "metadata_version",
            "source_version",
            "usage_version",
            "used_forward_scoring",
            "used_gradient_extraction",
            "used_candidate_sft",
            "used_accepted_sft",
            "used_rejected_sft",
            "pending_sft",
        }
        require(all(f in allowed for f in fields), "Unknown static field")
        return self._batch(
            ids, fields, lambda d, f: self._static[d][f], missing, {"kind": "static"}
        )

    def signal(self, data_id, model_ref, spec, *, percentile_reference=None):
        self.record(data_id)
        key = (data_id, *_signal_context(model_ref, spec, percentile_reference))
        if key not in self._signals:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "No signal in requested context",
                data_id=data_id,
            )
        return self._signals[key]

    def signal_batch(
        self,
        data_ids,
        model_ref,
        spec,
        *,
        fields=("nll", "entropy"),
        percentile_reference=None,
        missing="error",
    ):
        ids = self._ids(data_ids)
        require(all(f in self.SIGNAL_FIELDS for f in fields), "Unknown signal field")
        context = _signal_context(model_ref, spec, percentile_reference)

        def getter(data_id, name):
            signal = self._signals.get((data_id, *context))
            return None if signal is None else getattr(signal, name)

        keys = tuple(self._signal_keys.get((d, *context)) for d in ids)
        return self._batch(
            ids,
            fields,
            getter,
            missing,
            {
                "kind": "signal",
                "model_hash": context[0],
                "spec_hash": context[1],
                "reference_hash": context[2],
                "feature_keys": keys,
            },
        )

    def evidence_batch(
        self,
        data_ids,
        skill,
        *,
        fields=("share", "density", "supervised_evidence_tokens"),
        missing="none",
    ):
        ids = self._ids(data_ids)
        text(skill, "skill")
        require(
            all(f in self.EVIDENCE_FIELDS for f in fields), "Unknown evidence field"
        )

        def getter(data_id, name):
            metric = self._evidence.get((data_id, skill))
            return None if metric is None else getattr(metric, name)

        return self._batch(
            ids,
            fields,
            getter,
            missing,
            {
                "kind": "evidence",
                "skill": skill,
                "annotation_versions": tuple(
                    self._static[d]["annotation_version"] for d in ids
                ),
            },
        )

    def gradient(self, data_id, model_ref, spec):
        require(
            isinstance(model_ref, ModelRef) and isinstance(spec, GradientSpec),
            "Expected ModelRef and GradientSpec",
        )
        self.record(data_id)
        feature = self._gradients.get((data_id, model_ref.model_hash, spec.spec_hash))
        if feature is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "No gradient in requested context",
                data_id=data_id,
            )
        return feature

    def gradient_batch(self, data_ids, model_ref, spec, *, missing="error"):
        require(
            isinstance(model_ref, ModelRef) and isinstance(spec, GradientSpec),
            "Expected ModelRef and GradientSpec",
        )
        ids = self._ids(data_ids)
        context = (model_ref.model_hash, spec.spec_hash)
        keys = tuple(self._gradient_keys.get((d, *context)) for d in ids)
        return self._batch(
            ids,
            ("gradient",),
            lambda d, _: self._gradients.get((d, *context)),
            missing,
            {
                "kind": "gradient",
                "model_hash": context[0],
                "spec_hash": context[1],
                "feature_keys": keys,
            },
        )
