"""M1 normalization and evidence metrics. No model, scoring or training calls."""

from self_curate.contracts import (
    AnnotationSet,
    EvidenceMetrics,
    TokenizedConversation,
    TrajectoryRecord,
    TrajectoryStats,
    digest,
)
from self_curate.contracts.errors import ErrorCode
from self_curate.contracts.parsing import (
    annotations,
    fields,
    message,
    quality_check,
    sequence,
    source_ref,
    signal_record,
    usage_history,
    annotation_record,
    metadata_record,
    data_labels,
    gradient_feature,
)
from self_curate.contracts.types import require, text
from .tokenization import TokenizationBackend, validate_alignment


def validate_tool_links(messages):
    seen, pending = set(), set()
    for index, msg in enumerate(messages):
        if msg.role != "tool":
            require(
                not pending,
                "Unanswered tool calls before next non-tool message",
                message_index=index,
            )
        if msg.role == "assistant":
            for call in msg.tool_calls:
                require(
                    call.call_id not in seen,
                    "Duplicate tool call ID",
                    call_id=call.call_id,
                )
                seen.add(call.call_id)
                pending.add(call.call_id)
        elif msg.role == "tool":
            require(
                msg.tool_call_id in pending,
                "Orphan or repeated tool response",
                message_index=index,
            )
            pending.remove(msg.tool_call_id)
    require(
        not pending,
        "Trajectory ends with unanswered tool calls",
        pending=sorted(pending),
    )


def _union(spans):
    result = []
    for start, end in sorted(spans):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def evidence_metrics(encoded, annotation_set: AnnotationSet | None, declared_skills=()):
    """Intersect evidence with supervision, union overlaps before counting.

    Missing annotation != annotated empty span set. Shares use the union across
    every skill as denominator, so overlapping skills can sum to more than one.
    """
    prefix = [0]
    for supervise in encoded.assistant_mask:
        prefix.append(prefix[-1] + int(supervise))
    rows = {} if annotation_set is None else {a.skill: a for a in annotation_set.skills}
    if annotation_set is not None:
        require(
            annotation_set.tokenizer_hash == encoded.identity.tokenizer_hash
            and annotation_set.template_hash == encoded.identity.template_hash,
            "Evidence offsets belong to another tokenizer/template",
            ErrorCode.TOKENIZATION_MISMATCH,
        )
    all_spans = []
    for annotation in rows.values():
        for span in annotation.spans or ():
            require(
                span.message_index < len(encoded.message_spans),
                "Evidence message index out of range",
                ErrorCode.INVALID_EVIDENCE,
            )
            message_span = encoded.message_spans[span.message_index]
            require(
                message_span.start <= span.start < span.end <= message_span.end,
                "Evidence must be inside the referenced message payload",
                ErrorCode.INVALID_EVIDENCE,
            )
            all_spans.append((span.start, span.end))

    def count(spans):
        return sum(prefix[end] - prefix[start] for start, end in _union(spans))

    total = count(all_spans)
    metrics = []
    for skill in sorted(set(declared_skills) | rows.keys()):
        row = rows.get(skill)
        missing = row is None or row.spans is None
        n = None if missing else count((span.start, span.end) for span in row.spans)
        metrics.append(
            EvidenceMetrics(
                skill=skill,
                status="missing" if missing else "annotated",
                supervised_evidence_tokens=n,
                share=None if missing or total == 0 else n / total,
                density=None if missing or prefix[-1] == 0 else n / prefix[-1],
                complete_events=None if row is None else row.complete_events,
                behaviors=None if row is None else row.behaviors,
            )
        )
    return tuple(metrics), total


def normalize_trajectory(
    raw: dict, backend: TokenizationBackend, declared_skills: tuple[str, ...] = ()
) -> TrajectoryRecord:
    require(
        type(declared_skills) is tuple, "declared_skills must be an immutable tuple"
    )
    for skill in declared_skills:
        text(skill, "declared_skill")
    require(
        len(set(declared_skills)) == len(declared_skills), "Duplicate declared skill"
    )
    fields(
        raw,
        ("schema_version", "data_id", "source", "messages", "quality"),
        (
            "annotations",
            "usage",
            "tokenization",
            "signals",
            "metadata",
            "gradients",
            "annotation_record",
            "metadata_record",
        ),
    )
    require(raw["schema_version"] == "1", "Unsupported trajectory schema")
    text(raw["data_id"], "data_id")
    messages = tuple(message(m) for m in sequence(raw["messages"], "messages"))
    require(bool(messages), "Trajectory must contain messages")
    validate_tool_links(messages)
    source = source_ref(raw["source"])
    quality = quality_check(raw["quality"])
    annotation_set = annotations(raw.get("annotations"))
    if declared_skills and annotation_set is not None:
        unknown = {a.skill for a in annotation_set.skills} - set(declared_skills)
        require(not unknown, "Annotation has undeclared skills", skills=sorted(unknown))
    usage = usage_history(raw.get("usage", {}))
    encoded = backend.encode(messages, raw.get("tokenization"))
    require(
        isinstance(encoded, TokenizedConversation),
        "Backend must return TokenizedConversation",
        ErrorCode.TOKENIZATION_MISMATCH,
    )
    require(
        encoded.identity == backend.identity,
        "Backend returned a different identity",
        ErrorCode.TOKENIZATION_MISMATCH,
    )
    validate_alignment(encoded, messages)
    metrics, union = evidence_metrics(encoded, annotation_set, declared_skills)
    record = TrajectoryRecord(
        data_id=raw["data_id"],
        content_hash=digest({"messages": [m.to_dict() for m in messages]}),
        source=source,
        messages=messages,
        quality=quality,
        tokenization=encoded,
        annotations=annotation_set,
        evidence_metrics=metrics,
        stats=TrajectoryStats(
            assistant_turns=sum(m.role == "assistant" for m in messages),
            tool_calls=sum(len(m.tool_calls) for m in messages),
            tool_observations=sum(m.role == "tool" for m in messages),
            context_tokens=len(encoded.input_ids),
            supervised_tokens=sum(encoded.assistant_mask),
            evidence_union_tokens=union,
        ),
        usage=usage,
        metadata=data_labels(raw.get("metadata")),
    )
    from .enrichment import attach_signals
    from .features import (
        attach_annotation_record,
        attach_metadata_record,
        attach_gradients,
    )

    envelope = annotation_record(raw.get("annotation_record"))
    if envelope is not None:
        if "annotations" in raw:
            require(
                record.annotations == envelope.annotations,
                "Inline annotations and envelope disagree",
            )
        record = attach_annotation_record(
            record, envelope, declared_skills=declared_skills
        )
    envelope = metadata_record(raw.get("metadata_record"))
    if envelope is not None:
        if "metadata" in raw:
            require(
                record.metadata == envelope.labels,
                "Inline metadata and envelope disagree",
            )
        record = attach_metadata_record(record, envelope)

    record = attach_signals(
        record,
        tuple(signal_record(s) for s in sequence(raw.get("signals", []), "signals")),
    )
    return attach_gradients(
        record,
        tuple(
            gradient_feature(g) for g in sequence(raw.get("gradients", []), "gradients")
        ),
    )
