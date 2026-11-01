"""Immutable M1 records. Token offsets are global, half-open [start, end)."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Literal

from .errors import ContractError, ErrorCode

SCHEMA_VERSION = "1"


def canonical_json(value) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD, "Value is not finite JSON-serializable data"
        ) from exc


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require(condition, message, code=ErrorCode.INVALID_RECORD, **details):
    if not condition:
        raise ContractError(code, message, **details)


def text(value, field):
    require(
        isinstance(value, str) and bool(value.strip()), f"{field} must be nonempty text"
    )


def integer(value, field, minimum=0):
    require(
        type(value) is int and value >= minimum,
        f"{field} must be an integer >= {minimum}",
    )


def sha256(value, field):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{field} must be a lowercase SHA256",
    )


class Record:
    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRef(Record):
    uri: str
    sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        text(self.uri, "artifact.uri")
        sha256(self.sha256, "artifact.sha256")
        require(self.schema_version == SCHEMA_VERSION, "Unsupported artifact schema")


@dataclass(frozen=True)
class TokenizerRef(Record):
    tokenizer_id: str
    revision: str
    tokenizer_hash: str
    template_id: str
    template_hash: str
    backend: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for name in ("tokenizer_id", "revision", "template_id", "backend"):
            text(getattr(self, name), f"tokenizer.{name}")
        sha256(self.tokenizer_hash, "tokenizer.tokenizer_hash")
        sha256(self.template_hash, "tokenizer.template_hash")
        require(self.schema_version == SCHEMA_VERSION, "Unsupported tokenizer schema")


@dataclass(frozen=True)
class SourceRef(Record):
    source_id: str
    revision: str
    record_id: str
    artifact: ArtifactRef
    family_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for name in ("source_id", "revision", "record_id"):
            text(getattr(self, name), f"source.{name}")
        require(
            isinstance(self.artifact, ArtifactRef),
            "source.artifact must be ArtifactRef",
        )
        require(type(self.family_ids) is tuple, "family_ids must be an immutable tuple")
        for family in self.family_ids:
            text(family, "family_id")
        require(
            len(set(self.family_ids)) == len(self.family_ids), "Duplicate family ID"
        )
        require(self.schema_version == SCHEMA_VERSION, "Unsupported source schema")


@dataclass(frozen=True)
class ToolCall(Record):
    call_id: str
    name: str
    arguments: str

    def __post_init__(self):
        text(self.call_id, "tool_call.call_id")
        text(self.name, "tool_call.name")
        try:
            parsed = json.loads(self.arguments, parse_constant=lambda _: None)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                ErrorCode.INVALID_RECORD, "Tool arguments must be JSON object text"
            ) from exc
        require(isinstance(parsed, dict), "Tool arguments must be a JSON object")
        require(
            canonical_json(parsed) == self.arguments,
            "Tool arguments must be canonical JSON",
        )


@dataclass(frozen=True)
class Message(Record):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self):
        require(
            isinstance(self.role, str)
            and self.role in {"system", "user", "assistant", "tool"},
            "Unsupported message role",
        )
        require(
            isinstance(self.content, str),
            "Only text message content is supported; multimodal input is rejected",
        )
        require(
            type(self.tool_calls) is tuple
            and all(isinstance(c, ToolCall) for c in self.tool_calls),
            "Invalid tool_calls",
        )
        require(
            not self.tool_calls or self.role == "assistant",
            "Only assistant can call tools",
        )
        require(
            (self.tool_call_id is not None) == (self.role == "tool"),
            "Only tool messages require tool_call_id",
        )
        if self.tool_call_id is not None:
            text(self.tool_call_id, "tool_call_id")
        if self.role == "assistant":
            require(
                bool(self.content) or bool(self.tool_calls),
                "Assistant message must contain text or tool calls",
            )


@dataclass(frozen=True)
class QualityCheck(Record):
    status: Literal["pass", "fail", "unknown"]
    method: Literal["executable", "manual", "teacher", "unverified"]
    verifier_id: str | None = None
    evidence: ArtifactRef | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        require(
            isinstance(self.status, str) and self.status in {"pass", "fail", "unknown"},
            "Invalid quality status",
        )
        require(
            isinstance(self.method, str)
            and self.method in {"executable", "manual", "teacher", "unverified"},
            "Invalid quality method",
        )
        require(self.schema_version == SCHEMA_VERSION, "Unsupported quality schema")
        if self.verifier_id is not None:
            text(self.verifier_id, "quality.verifier_id")
        require(
            self.evidence is None or isinstance(self.evidence, ArtifactRef),
            "Invalid quality evidence",
        )
        if self.status == "pass":
            require(
                self.method in {"executable", "manual"}
                and self.verifier_id is not None
                and self.evidence is not None,
                "Quality pass requires executable/manual provenance; teacher judgment alone is insufficient",
            )


@dataclass(frozen=True)
class TokenSpan(Record):
    start: int
    end: int

    def __post_init__(self):
        integer(self.start, "span.start")
        integer(self.end, "span.end")
        require(self.start <= self.end, "Reversed token span")


@dataclass(frozen=True)
class EvidenceSpan(Record):
    message_index: int
    start: int
    end: int

    def __post_init__(self):
        integer(self.message_index, "evidence.message_index")
        integer(self.start, "evidence.start")
        integer(self.end, "evidence.end")
        require(
            self.start < self.end,
            "Evidence spans must be nonempty",
            ErrorCode.INVALID_EVIDENCE,
        )


@dataclass(frozen=True)
class SkillAnnotation(Record):
    skill: str
    spans: tuple[EvidenceSpan, ...] | None
    annotator_id: str
    complete_events: int | None = None
    behaviors: tuple[str, ...] | None = None
    confidence: float | None = None

    def __post_init__(self):
        text(self.skill, "annotation.skill")
        text(self.annotator_id, "annotation.annotator_id")
        require(
            self.spans is None
            or (
                type(self.spans) is tuple
                and all(isinstance(s, EvidenceSpan) for s in self.spans)
            ),
            "Invalid annotation spans",
        )
        if self.complete_events is not None:
            integer(self.complete_events, "complete_events")
        if self.behaviors is not None:
            require(type(self.behaviors) is tuple, "behaviors must be tuple or missing")
            for behavior in self.behaviors:
                text(behavior, "behavior")
            require(
                len(set(self.behaviors)) == len(self.behaviors), "Duplicate behavior"
            )
        if self.confidence is not None:
            require(
                type(self.confidence) in {int, float}
                and math.isfinite(self.confidence)
                and 0 <= self.confidence <= 1,
                "confidence must be finite and between 0 and 1",
            )


@dataclass(frozen=True)
class AnnotationSet(Record):
    tokenizer_hash: str
    template_hash: str
    skills: tuple[SkillAnnotation, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        sha256(self.tokenizer_hash, "annotations.tokenizer_hash")
        sha256(self.template_hash, "annotations.template_hash")
        require(
            type(self.skills) is tuple
            and all(isinstance(a, SkillAnnotation) for a in self.skills),
            "Invalid annotations",
        )
        require(
            len({a.skill for a in self.skills}) == len(self.skills),
            "Duplicate skill annotation",
        )
        require(self.schema_version == SCHEMA_VERSION, "Unsupported annotation schema")


@dataclass(frozen=True)
class TrainingUse(Record):
    attempt_id: str
    model_id: str
    receipt: ArtifactRef
    decision: Literal["pending", "rejected", "accepted"]

    def __post_init__(self):
        text(self.attempt_id, "attempt_id")
        text(self.model_id, "model_id")
        require(
            isinstance(self.receipt, ArtifactRef),
            "Training receipt must be ArtifactRef",
        )
        require(
            isinstance(self.decision, str)
            and self.decision in {"pending", "rejected", "accepted"},
            "Invalid training decision",
        )


@dataclass(frozen=True)
class UsageHistory(Record):
    forward_scoring: tuple[ArtifactRef, ...] = ()
    gradient_extraction: tuple[ArtifactRef, ...] = ()
    candidate_sft: tuple[TrainingUse, ...] = ()
    accepted_sft: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        for refs in (self.forward_scoring, self.gradient_extraction):
            require(
                type(refs) is tuple and all(isinstance(r, ArtifactRef) for r in refs),
                "Invalid usage artifact references",
            )
        require(
            type(self.candidate_sft) is tuple
            and all(isinstance(r, TrainingUse) for r in self.candidate_sft),
            "Invalid candidate history",
        )
        require(type(self.accepted_sft) is tuple, "accepted_sft must be tuple")
        for attempt in self.accepted_sft:
            text(attempt, "accepted_sft.attempt_id")
        candidates = {r.attempt_id: r for r in self.candidate_sft}
        require(
            len(candidates) == len(self.candidate_sft), "Duplicate training attempt"
        )
        require(
            len(set(self.accepted_sft)) == len(self.accepted_sft),
            "Duplicate accepted attempt",
        )
        require(
            set(self.accepted_sft)
            == {a for a, r in candidates.items() if r.decision == "accepted"},
            "Accepted history must exactly match accepted candidate attempts",
        )
        require(self.schema_version == SCHEMA_VERSION, "Unsupported usage schema")


@dataclass(frozen=True)
class TokenizedConversation(Record):
    identity: TokenizerRef
    input_ids: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    message_spans: tuple[TokenSpan, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        require(isinstance(self.identity, TokenizerRef), "Invalid tokenizer identity")
        require(
            type(self.input_ids) is tuple and bool(self.input_ids),
            "input_ids must be a nonempty tuple",
        )
        for token in self.input_ids:
            integer(token, "input_id")
        require(
            type(self.assistant_mask) is tuple
            and len(self.assistant_mask) == len(self.input_ids)
            and all(type(v) is bool for v in self.assistant_mask),
            "assistant_mask must have one boolean per token",
        )
        require(
            type(self.message_spans) is tuple
            and all(isinstance(s, TokenSpan) for s in self.message_spans),
            "Invalid message spans",
        )
        end = 0
        for span in self.message_spans:
            require(
                end <= span.start <= span.end <= len(self.input_ids),
                "Message payload spans overlap or exceed token sequence",
            )
            end = span.end
        require(
            self.schema_version == SCHEMA_VERSION, "Unsupported tokenization schema"
        )


@dataclass(frozen=True)
class EvidenceMetrics(Record):
    skill: str
    status: Literal["annotated", "missing"]
    supervised_evidence_tokens: int | None
    share: float | None
    density: float | None
    complete_events: int | None
    behaviors: tuple[str, ...] | None

    def __post_init__(self):
        text(self.skill, "metric.skill")
        require(
            self.status in {"annotated", "missing"}, "Invalid evidence metric status"
        )
        if self.status == "missing":
            require(
                self.supervised_evidence_tokens is None
                and self.share is None
                and self.density is None,
                "Missing evidence cannot have numerical metrics",
            )
        else:
            integer(self.supervised_evidence_tokens, "supervised_evidence_tokens")
        for name in ("share", "density"):
            value = getattr(self, name)
            require(
                value is None
                or (
                    type(value) in {int, float}
                    and math.isfinite(value)
                    and 0 <= value <= 1
                ),
                "Invalid evidence ratio",
            )


@dataclass(frozen=True)
class TrajectoryStats(Record):
    assistant_turns: int
    tool_calls: int
    tool_observations: int
    context_tokens: int
    supervised_tokens: int
    evidence_union_tokens: int

    def __post_init__(self):
        for name in (
            "assistant_turns",
            "tool_calls",
            "tool_observations",
            "context_tokens",
            "supervised_tokens",
            "evidence_union_tokens",
        ):
            integer(getattr(self, name), name)
        require(
            0 < self.supervised_tokens <= self.context_tokens,
            "Invalid supervision/context counts",
        )
        require(
            self.evidence_union_tokens <= self.supervised_tokens,
            "Evidence exceeds supervision",
        )


@dataclass(frozen=True)
class TrajectoryRecord(Record):
    data_id: str
    content_hash: str
    source: SourceRef
    messages: tuple[Message, ...]
    quality: QualityCheck
    tokenization: TokenizedConversation
    annotations: AnnotationSet | None
    evidence_metrics: tuple[EvidenceMetrics, ...]
    stats: TrajectoryStats
    usage: UsageHistory
    normalization_version: str = "m1-v1"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        text(self.data_id, "data_id")
        sha256(self.content_hash, "content_hash")
        require(
            type(self.messages) is tuple
            and all(isinstance(m, Message) for m in self.messages),
            "Invalid messages",
        )
        for value, expected in (
            (self.source, SourceRef),
            (self.quality, QualityCheck),
            (self.tokenization, TokenizedConversation),
            (self.stats, TrajectoryStats),
            (self.usage, UsageHistory),
        ):
            require(isinstance(value, expected), "Invalid nested trajectory contract")
        require(
            self.annotations is None or isinstance(self.annotations, AnnotationSet),
            "Invalid annotation set",
        )
        require(
            type(self.evidence_metrics) is tuple
            and all(isinstance(m, EvidenceMetrics) for m in self.evidence_metrics),
            "Invalid evidence metrics",
        )
        require(
            self.content_hash
            == digest({"messages": [m.to_dict() for m in self.messages]}),
            "Trajectory content hash mismatch",
        )
        require(
            self.schema_version == SCHEMA_VERSION
            and self.normalization_version == "m1-v1",
            "Unsupported trajectory version",
        )

    @property
    def record_hash(self):
        return digest(self.to_dict())

    @property
    def labels(self):
        return tuple(
            token if supervise else -100
            for token, supervise in zip(
                self.tokenization.input_ids, self.tokenization.assistant_mask
            )
        )
