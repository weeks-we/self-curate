"""Actual chat-template payload alignment. Never infer supervision from role tokens."""

from copy import deepcopy
import json
from self_curate.contracts import (
    TokenizedConversation,
    TokenSpan,
    digest,
)
from self_curate.contracts.types import require


def canonical_messages(rows):
    result = []
    for row in rows:
        require(
            set(row) <= {"role", "content", "tool_calls", "tool_call_id"},
            "Unsupported source message fields; preserve raw and quarantine",
        )
        value = {"role": row["role"], "content": row.get("content") or ""}
        require(
            row.get("content") is None or isinstance(row["content"], str),
            "Multimodal messages require a separate adapter",
        )
        if row.get("tool_calls"):
            calls = []
            for call in row["tool_calls"]:
                if "function" in call:
                    require(
                        set(call) <= {"id", "type", "function"}
                        and call.get("type", "function") == "function",
                        "Unsupported tool call",
                    )
                    args = call["function"]["arguments"]
                    calls.append(
                        {
                            "call_id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": (
                                json.loads(args) if isinstance(args, str) else args
                            ),
                        }
                    )
                else:
                    calls.append(call)
            value["tool_calls"] = calls
        if row.get("tool_call_id") is not None:
            value["tool_call_id"] = row["tool_call_id"]
        result.append(value)
    return result


def hf_messages(messages):
    result = []
    for msg in messages:
        row = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id is not None:
            row["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            row["tool_calls"] = [
                {
                    "id": c.call_id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.loads(c.arguments)},
                }
                for c in msg.tool_calls
            ]
        result.append(row)
    return result


class ChatTemplateTokenizer:
    def __init__(self, tokenizer, identity):
        require(
            identity.backend != "fixture" and getattr(tokenizer, "is_fast", False),
            "Actual fast tokenizer with offsets required",
        )
        self.tokenizer, self.identity = tokenizer, identity

    def align(self, messages):
        rows = hf_messages(messages)
        # Tool serialization is part of the payload; native templates may split
        # calls across control tokens. Reject unsupported layouts, never guess.
        nonce = digest(rows)[:20]
        marked = deepcopy(rows)
        for i, row in enumerate(marked):
            begin = f"[[SC_{nonce}_{i}_BEGIN]]"
            end = f"[[SC_{nonce}_{i}_END]]"
            if row.get("tool_calls") and not row["content"]:
                row["tool_calls"][0]["function"]["name"] = (
                    begin + end + row["tool_calls"][0]["function"]["name"]
                )
            else:
                row["content"] = begin + row["content"] + end
        render = lambda value: self.tokenizer.apply_chat_template(
            value, tokenize=False, add_generation_prompt=False
        )
        actual = render(rows)
        instrumented = render(marked)
        clean = ""
        chars = []
        cursor = 0
        for i, row in enumerate(rows):
            begin = f"[[SC_{nonce}_{i}_BEGIN]]"
            end = f"[[SC_{nonce}_{i}_END]]"
            require(
                instrumented.count(begin) == instrumented.count(end) == 1,
                "Template transformed or repeated a payload marker",
            )
            left = instrumented.index(begin, cursor)
            right = instrumented.index(end, left + len(begin))
            clean += instrumented[cursor:left]
            start = len(clean)
            payload = instrumented[left + len(begin) : right]
            require(
                payload == row["content"],
                "Template transformed message payload; explicit adapter required",
            )
            clean += payload
            chars.append((start, len(clean)))
            cursor = right + len(end)
        clean += instrumented[cursor:]
        require(clean == actual, "Marker rendering changed template control flow")
        for i, row in enumerate(rows):
            if not row.get("tool_calls"):
                continue
            # Pinned Qwen XML tool serialization: keep all native call markup
            # inside the assistant payload, but exclude the role/EOT framing.
            left, right = chars[i]
            limit = chars[i + 1][0] if i + 1 < len(chars) else len(actual)
            stop = actual.rfind("<|im_end|>", right, limit)
            require(
                stop >= right,
                "Unsupported native tool payload frame; cannot infer the mask",
            )
            if not row["content"]:
                header = actual.rfind("<|im_start|>assistant\n", 0, left)
                require(header >= 0, "Unsupported assistant role frame")
                start = actual.rfind("<tool_call>", header, left)
                require(start >= header, "Unsupported native Qwen tool serialization")
                left = start
            chars[i] = (left, stop)
        encoded = self.tokenizer(
            actual, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = list(encoded["input_ids"])
        offsets = [tuple(x) for x in encoded["offset_mapping"]]
        expected = self.tokenizer.apply_chat_template(
            rows, tokenize=True, add_generation_prompt=False
        )
        require(
            ids == expected, "Text encoding differs from actual chat-template token IDs"
        )
        spans = []
        mask = [False] * len(ids)
        for msg, (left, right) in zip(messages, chars):
            indices = [
                j
                for j, (a, b) in enumerate(offsets)
                if a < b and a < right and b > left
            ]
            if left == right:
                indices = []
            require(
                all(left <= offsets[j][0] < offsets[j][1] <= right for j in indices),
                "Token crosses payload boundary; refusing an approximate mask",
            )
            require(
                not (msg.content or msg.tool_calls) or bool(indices),
                "Nonempty payload has no token alignment",
            )
            start = (
                indices[0]
                if indices
                else next(
                    (j for j, (a, b) in enumerate(offsets) if a >= left), len(ids)
                )
            )
            stop = indices[-1] + 1 if indices else start
            require(indices == list(range(start, stop)), "Noncontiguous payload tokens")
            spans.append(TokenSpan(start, stop))
            if msg.role == "assistant":
                mask[start:stop] = [True] * (stop - start)
        return (
            TokenizedConversation(self.identity, tuple(ids), tuple(mask), tuple(spans)),
            actual,
            tuple(offsets),
            tuple(chars),
        )

    def encode(self, messages, provided=None):
        require(
            provided is None,
            "Actual tokenizer does not consume unauthenticated token IDs",
        )
        return self.align(messages)[0]
