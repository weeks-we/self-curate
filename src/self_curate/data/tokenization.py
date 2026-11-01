"""Model-independent adapter boundary; fixture token IDs are explicitly synthetic."""

from typing import Protocol

from self_curate.contracts import (
    ContractError,
    ErrorCode,
    Message,
    TokenizerRef,
    TokenizedConversation,
    TokenSpan,
    canonical_json,
    digest,
)
from self_curate.contracts.parsing import tokenization
from self_curate.contracts.types import require


class TokenizationBackend(Protocol):
    @property
    def identity(self) -> TokenizerRef: ...

    def encode(
        self, messages: tuple[Message, ...], provided: dict | None = None
    ) -> TokenizedConversation: ...


class FixtureTokenizer:
    """One Unicode codepoint per payload token. Not a pretrained tokenizer."""

    identity = TokenizerRef(
        tokenizer_id="fixture-unicode",
        revision="1",
        tokenizer_hash=digest("fixture-unicode-v1"),
        template_id="fixture-role-header-v1",
        template_hash=digest("fixture-role-header-v1"),
        backend="fixture",
    )

    def encode(self, messages, provided=None):
        require(
            provided is None, "Fixture backend will not discard supplied tokenization"
        )
        ids, mask, spans = [], [], []
        for msg in messages:
            ids.append(
                1_200_000 + ("system", "user", "assistant", "tool").index(msg.role)
            )
            mask.append(False)
            payload = msg.content
            if msg.tool_calls:
                payload += ("\n" if payload else "") + canonical_json(
                    [c.to_dict() for c in msg.tool_calls]
                )
            start = len(ids)
            ids.extend(ord(char) + 1 for char in payload)
            mask.extend([msg.role == "assistant"] * len(payload))
            spans.append(TokenSpan(start, len(ids)))
            ids.append(1_200_010)
            mask.append(False)
        return TokenizedConversation(
            self.identity, tuple(ids), tuple(mask), tuple(spans)
        )


class PrecomputedTokenizer:
    """Ingest actual externally produced IDs/alignment, without loading a model.

    Identity/alignment are validated. Authenticity of the declared tokenizer is
    an upstream provenance obligation, not something token IDs alone prove.
    """

    def __init__(self, identity: TokenizerRef):
        require(
            identity.backend != "fixture",
            "Precomputed backend cannot disguise fixture tokenization",
        )
        self.identity = identity

    def encode(self, messages, provided=None):
        if provided is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "Precomputed backend requires supplied tokenization",
            )
        result = tokenization(provided)
        require(
            result.identity == self.identity,
            "Tokenization identity differs from configured identity",
            ErrorCode.TOKENIZATION_MISMATCH,
        )
        return result


def validate_alignment(
    encoded: TokenizedConversation, messages: tuple[Message, ...]
) -> None:
    require(
        len(encoded.message_spans) == len(messages),
        "Need one payload span per message",
        ErrorCode.TOKENIZATION_MISMATCH,
    )
    expected = [False] * len(encoded.input_ids)
    for msg, span in zip(messages, encoded.message_spans):
        require(
            span.end > span.start or (not msg.content and not msg.tool_calls),
            "Nonempty payload has empty token span",
            ErrorCode.TOKENIZATION_MISMATCH,
        )
        if msg.role == "assistant":
            expected[span.start : span.end] = [True] * (span.end - span.start)
    require(
        tuple(expected) == encoded.assistant_mask,
        "Mask must supervise all assistant payload positions and no header/user/system/tool positions",
        ErrorCode.TOKENIZATION_MISMATCH,
    )
    require(
        any(expected),
        "Trajectory has no assistant supervision",
        ErrorCode.ZERO_SUPERVISION,
    )
