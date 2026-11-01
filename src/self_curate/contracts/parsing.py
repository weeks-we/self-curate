"""Strict JSON boundary. Unknown and duplicate fields are errors, never discarded."""

import json
import math

from .errors import ContractError, ErrorCode
from .types import (
    AnnotationSet,
    ArtifactRef,
    EvidenceSpan,
    Message,
    QualityCheck,
    SkillAnnotation,
    SourceRef,
    TokenizerRef,
    TokenizedConversation,
    TokenSpan,
    ToolCall,
    TrainingUse,
    UsageHistory,
    canonical_json,
    require,
)


def load_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key", key=key)
            result[key] = value
        return result

    def constant(value):
        raise ContractError(
            ErrorCode.INVALID_RECORD, "Nonfinite JSON number", value=value
        )

    def number(value):
        result = float(value)
        require(math.isfinite(result), "JSON number overflows finite float range")
        return result

    try:
        return json.loads(
            text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number
        )
    except json.JSONDecodeError as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD, "Invalid JSON", line=exc.lineno, column=exc.colno
        ) from exc
    except ContractError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD, "JSON exceeds supported numeric or nesting limits"
        ) from exc


def fields(value, required, optional=()):
    require(type(value) is dict, "Expected a JSON object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    require(
        not missing and not unknown,
        "Schema fields mismatch",
        missing=sorted(missing),
        unknown=sorted(unknown),
    )
    return value


def sequence(value, field):
    require(type(value) is list, f"{field} must be a JSON array")
    return value


def artifact(value):
    return ArtifactRef(**fields(value, ("uri", "sha256"), ("schema_version",)))


def tokenizer_ref(value):
    return TokenizerRef(
        **fields(
            value,
            (
                "tokenizer_id",
                "revision",
                "tokenizer_hash",
                "template_id",
                "template_hash",
                "backend",
            ),
            ("schema_version",),
        )
    )


def source_ref(value):
    row = fields(
        value,
        ("source_id", "revision", "record_id", "artifact"),
        ("family_ids", "schema_version"),
    )
    return SourceRef(
        **{
            **row,
            "artifact": artifact(row["artifact"]),
            "family_ids": tuple(sequence(row.get("family_ids", []), "family_ids")),
        }
    )


def message(value):
    row = fields(value, ("role", "content"), ("tool_calls", "tool_call_id"))
    calls = []
    for call in sequence(row.get("tool_calls", []), "tool_calls"):
        fields(call, ("call_id", "name", "arguments"))
        arguments = (
            load_json(call["arguments"])
            if isinstance(call["arguments"], str)
            else call["arguments"]
        )
        require(type(arguments) is dict, "Tool call arguments must be a JSON object")
        calls.append(ToolCall(call["call_id"], call["name"], canonical_json(arguments)))
    return Message(**{**row, "tool_calls": tuple(calls)})


def quality_check(value):
    row = fields(
        value, ("status", "method"), ("verifier_id", "evidence", "schema_version")
    )
    return QualityCheck(
        **{
            **row,
            "evidence": (
                artifact(row["evidence"]) if row.get("evidence") is not None else None
            ),
        }
    )


def annotations(value):
    if value is None:
        return None
    row = fields(
        value, ("tokenizer_hash", "template_hash", "skills"), ("schema_version",)
    )
    skills = []
    for item in sequence(row["skills"], "skills"):
        fields(
            item,
            ("skill", "annotator_id"),
            ("spans", "complete_events", "behaviors", "confidence"),
        )
        spans = (
            None
            if item.get("spans") is None
            else tuple(
                EvidenceSpan(**fields(s, ("message_index", "start", "end")))
                for s in sequence(item["spans"], "spans")
            )
        )
        behaviors = (
            None
            if item.get("behaviors") is None
            else tuple(sequence(item["behaviors"], "behaviors"))
        )
        skills.append(
            SkillAnnotation(**{**item, "spans": spans, "behaviors": behaviors})
        )
    return AnnotationSet(**{**row, "skills": tuple(skills)})


def usage_history(value):
    row = fields(
        value,
        (),
        (
            "forward_scoring",
            "gradient_extraction",
            "candidate_sft",
            "accepted_sft",
            "schema_version",
        ),
    )
    candidates = []
    for use in sequence(row.get("candidate_sft", []), "candidate_sft"):
        fields(use, ("attempt_id", "model_id", "receipt", "decision"))
        candidates.append(TrainingUse(**{**use, "receipt": artifact(use["receipt"])}))
    return UsageHistory(
        forward_scoring=tuple(
            artifact(v)
            for v in sequence(row.get("forward_scoring", []), "forward_scoring")
        ),
        gradient_extraction=tuple(
            artifact(v)
            for v in sequence(row.get("gradient_extraction", []), "gradient_extraction")
        ),
        candidate_sft=tuple(candidates),
        accepted_sft=tuple(sequence(row.get("accepted_sft", []), "accepted_sft")),
        schema_version=row.get("schema_version", "1"),
    )


def tokenization(value):
    row = fields(
        value,
        ("identity", "input_ids", "assistant_mask", "message_spans"),
        ("schema_version",),
    )
    return TokenizedConversation(
        identity=tokenizer_ref(row["identity"]),
        input_ids=tuple(sequence(row["input_ids"], "input_ids")),
        assistant_mask=tuple(sequence(row["assistant_mask"], "assistant_mask")),
        message_spans=tuple(
            TokenSpan(**fields(s, ("start", "end")))
            for s in sequence(row["message_spans"], "message_spans")
        ),
        schema_version=row.get("schema_version", "1"),
    )
