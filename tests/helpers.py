from self_curate.contracts import ArtifactRef, TokenizerRef, canonical_json, digest
from self_curate.data import FixtureTokenizer


def ref(name):
    return ArtifactRef(f"fixture://{name}", digest(name)).to_dict()


def record(data_id="sample-1", *, content="fix then test", tool_output="ok"):
    return {
        "schema_version": "1",
        "data_id": data_id,
        "source": {
            "source_id": "synthetic-coding",
            "revision": "1",
            "record_id": data_id,
            "artifact": ref(data_id),
            "family_ids": ["fixture-family"],
        },
        "messages": [
            {"role": "system", "content": "Use tools to fix the bug."},
            {"role": "user", "content": "Fix the failing test."},
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "call_id": "call-1",
                        "name": "shell",
                        "arguments": {"command": "pytest"},
                    }
                ],
            },
            {"role": "tool", "content": tool_output, "tool_call_id": "call-1"},
            {"role": "assistant", "content": "The fixture test passed."},
        ],
        "quality": {"status": "unknown", "method": "unverified"},
    }


def golden_record():
    """Manually specified alignment, independent of fixture tokenization code.

    Assistant payloads: [4,9) and [14,16), 7 supervised tokens.
    A: [4,8), B: [6,9); union 5, overlap 2. A also labels
    two user tokens, which must not enter evidence numerators.
    """
    row = record()
    row["messages"] = [
        {"role": r, "content": c}
        for r, c in (
            ("user", "prompt"),
            ("assistant", "first"),
            ("user", "again"),
            ("assistant", "ok"),
        )
    ]
    identity = TokenizerRef(
        "golden-test-tokenizer",
        "1",
        digest("golden-tokenizer"),
        "golden-test-template",
        digest("golden-template"),
        "external",
    )
    row["tokenization"] = {
        "identity": identity.to_dict(),
        "input_ids": list(range(17)),
        "assistant_mask": [i in {4, 5, 6, 7, 8, 14, 15} for i in range(17)],
        "message_spans": [
            {"start": a, "end": b} for a, b in ((1, 3), (4, 9), (10, 13), (14, 16))
        ],
    }
    row["annotations"] = {
        "schema_version": "1",
        "tokenizer_hash": identity.tokenizer_hash,
        "template_hash": identity.template_hash,
        "skills": [
            {
                "skill": "recovery",
                "annotator_id": "manual-fixture",
                "spans": [
                    {"message_index": 1, "start": 4, "end": 8},
                    {"message_index": 1, "start": 6, "end": 8},
                    {"message_index": 0, "start": 1, "end": 3},
                ],
                "complete_events": 1,
                "behaviors": ["test_after_final_edit"],
            },
            {
                "skill": "testing",
                "annotator_id": "manual-fixture",
                "spans": [{"message_index": 1, "start": 6, "end": 9}],
            },
            {"skill": "empty", "annotator_id": "manual-fixture", "spans": []},
            {"skill": "missing", "annotator_id": "manual-fixture"},
        ],
    }
    return row, identity


def fixture_annotation(row, spans=None):
    identity = FixtureTokenizer.identity
    row["annotations"] = {
        "tokenizer_hash": identity.tokenizer_hash,
        "template_hash": identity.template_hash,
        "skills": [
            {"skill": "recovery", "annotator_id": "fixture-author", "spans": spans}
        ],
    }
    return row


def write_input(path, rows):
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path
