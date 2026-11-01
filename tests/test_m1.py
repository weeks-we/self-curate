from dataclasses import replace
import copy
import json
import os
import random
from pathlib import Path
import subprocess
import sys

import pytest

from helpers import golden_record, record, ref, write_input
from self_curate.contracts import ContractError, ErrorCode, canonical_json, digest
from self_curate.contracts.parsing import load_json
from self_curate.data import (
    FixtureTokenizer,
    ImportConfig,
    PrecomputedTokenizer,
    build_pool,
    load_config,
    normalize_trajectory,
    read_pool,
)
from self_curate.data.pool import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def config(path, **overrides):
    values = dict(
        input_path=path,
        backend="fixture",
        skills=("recovery",),
        max_context_tokens=10000,
        oversize_policy="reject",
        rejected_history_policy="exclude",
    )
    values.update(overrides)
    return ImportConfig(**values)


def test_independent_golden_evidence_overlap_mask_and_missing():
    raw, identity = golden_record()
    result = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    metrics = {m.skill: m for m in result.evidence_metrics}
    assert result.stats.supervised_tokens == 7
    assert result.stats.context_tokens == 17
    assert result.stats.evidence_union_tokens == 5
    assert metrics["recovery"].supervised_evidence_tokens == 4
    assert metrics["recovery"].share == pytest.approx(4 / 5)
    assert metrics["testing"].share == pytest.approx(3 / 5)
    assert metrics["recovery"].density == pytest.approx(4 / 7)
    assert metrics["recovery"].share + metrics["testing"].share > 1
    assert metrics["empty"].supervised_evidence_tokens == 0
    assert metrics["empty"].share == 0
    assert metrics["missing"].supervised_evidence_tokens is None
    assert metrics["missing"].share is metrics["missing"].density is None
    assert metrics["recovery"].complete_events == 1
    assert metrics["recovery"].behaviors == ("test_after_final_edit",)
    assert all(
        label == (token if mask else -100)
        for label, token, mask in zip(
            result.labels,
            result.tokenization.input_ids,
            result.tokenization.assistant_mask,
        )
    )


def test_zero_evidence_denominator_is_missing_share_not_zero():
    raw, identity = golden_record()
    raw["annotations"]["skills"] = [
        {
            "skill": "testing",
            "annotator_id": "fixture",
            "spans": [{"message_index": 0, "start": 1, "end": 3}],
        }
    ]
    result = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    assert result.stats.evidence_union_tokens == 0
    assert result.evidence_metrics[0].supervised_evidence_tokens == 0
    assert result.evidence_metrics[0].share is None
    assert result.evidence_metrics[0].density == 0


def test_tool_long_output_does_not_increase_supervision():
    short = normalize_trajectory(
        record(tool_output="x"), FixtureTokenizer(), ("recovery",)
    )
    long = normalize_trajectory(
        record(tool_output="x" * 20000), FixtureTokenizer(), ("recovery",)
    )
    assert long.stats.context_tokens - short.stats.context_tokens == 19999
    assert long.stats.supervised_tokens == short.stats.supervised_tokens
    assert (
        short.stats.assistant_turns == 2
        and short.stats.tool_calls == 1
        and short.stats.tool_observations == 1
    )
    for msg, span in zip(long.messages, long.tokenization.message_spans):
        if msg.role != "assistant":
            assert not any(long.tokenization.assistant_mask[span.start : span.end])


@pytest.mark.parametrize(
    "mutation", ["tokenizer", "template", "mask_tool", "mask_partial", "span", "index"]
)
def test_evidence_and_alignment_mismatch_rejected(mutation):
    raw, identity = golden_record()
    if mutation == "tokenizer":
        raw["annotations"]["tokenizer_hash"] = "0" * 64
    elif mutation == "template":
        raw["annotations"]["template_hash"] = "0" * 64
    elif mutation == "mask_tool":
        raw["tokenization"]["assistant_mask"][1] = True
    elif mutation == "mask_partial":
        raw["tokenization"]["assistant_mask"][4] = False
    elif mutation == "span":
        raw["annotations"]["skills"][0]["spans"][0]["end"] = 10
    else:
        raw["annotations"]["skills"][0]["spans"][0]["message_index"] = 99
    with pytest.raises(ContractError) as error:
        normalize_trajectory(raw, PrecomputedTokenizer(identity))
    assert error.value.code in {
        ErrorCode.TOKENIZATION_MISMATCH,
        ErrorCode.INVALID_EVIDENCE,
    }


def test_no_supervision_rejected():
    raw = record()
    raw["messages"] = [{"role": "user", "content": "no reply"}]
    with pytest.raises(ContractError) as error:
        normalize_trajectory(raw, FixtureTokenizer())
    assert error.value.code == ErrorCode.ZERO_SUPERVISION


@pytest.mark.parametrize(
    "kind", ["orphan", "unanswered", "duplicate", "double_response"]
)
def test_tool_links_rejected(kind):
    raw = record()
    if kind == "orphan":
        raw["messages"][3]["tool_call_id"] = "wrong-id"
    elif kind == "unanswered":
        raw["messages"].pop(3)
    elif kind == "duplicate":
        raw["messages"][2]["tool_calls"] *= 2
    else:
        raw["messages"].insert(4, copy.deepcopy(raw["messages"][3]))
    with pytest.raises(ContractError):
        normalize_trajectory(raw, FixtureTokenizer())


def test_quality_cannot_be_teacher_only_pass():
    raw = record()
    raw["quality"] = {
        "status": "pass",
        "method": "teacher",
        "verifier_id": "teacher",
        "evidence": ref("judgment"),
    }
    with pytest.raises(ContractError):
        normalize_trajectory(raw, FixtureTokenizer())
    raw["quality"]["method"] = "executable"
    assert normalize_trajectory(raw, FixtureTokenizer()).quality.status == "pass"


@pytest.mark.parametrize(
    "kind", ["extra", "multimodal", "role", "schema", "bool_token"]
)
def test_invalid_data_not_silently_dropped(kind):
    raw, identity = golden_record()
    if kind == "extra":
        raw["surprise"] = 1
    elif kind == "multimodal":
        raw["messages"][0]["content"] = [{"type": "image"}]
    elif kind == "role":
        raw["messages"][0]["role"] = "alien"
    elif kind == "schema":
        raw["schema_version"] = "2"
    else:
        raw["tokenization"]["input_ids"][0] = True
    with pytest.raises(ContractError):
        normalize_trajectory(raw, PrecomputedTokenizer(identity))


@pytest.mark.parametrize(
    "value", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"x":']
)
def test_bad_json_is_structured(value):
    with pytest.raises(ContractError):
        load_json(value)


def test_content_identity_separate_from_annotations_and_usage():
    raw = record()
    first = normalize_trajectory(raw, FixtureTokenizer())
    raw["usage"] = {"forward_scoring": [ref("forward-score")]}
    second = normalize_trajectory(raw, FixtureTokenizer())
    assert first.content_hash == second.content_hash
    assert first.record_hash != second.record_hash
    assert second.usage.forward_scoring and not second.usage.gradient_extraction
    raw["source"]["record_id"] = "another-origin"
    other_origin = normalize_trajectory(raw, FixtureTokenizer())
    assert other_origin.content_hash == first.content_hash
    assert other_origin.record_hash != second.record_hash
    raw["messages"][4]["content"] += " changed"
    assert (
        normalize_trajectory(raw, FixtureTokenizer()).content_hash != first.content_hash
    )


def test_pool_reproducible_roundtrip_no_overwrite(tmp_path):
    path = write_input(tmp_path / "raw.jsonl", [record("b"), record("a")])
    cfg = config(path)
    first = build_pool(cfg, tmp_path / "one")
    second = build_pool(cfg, tmp_path / "two")
    assert first == second
    pool = read_pool(tmp_path / "one")
    assert [r.data_id for r in pool.records] == ["a", "b"]
    assert pool.manifest["synthetic"] is True
    assert pool.records[0].quality.status == "unknown"  # M2 QualityGate has not run.
    assert pool.records[0].evidence_metrics[0].share is None
    audit = load_json((tmp_path / "one" / "audit.json").read_text(encoding="utf-8"))
    assert audit["duplicate_content_records"] == 1
    assert file_sha256(path) == first["input_sha256"]
    with pytest.raises(ContractError) as error:
        build_pool(cfg, tmp_path / "one")
    assert error.value.code == ErrorCode.OUTPUT_EXISTS


def test_duplicate_even_if_excluded_and_atomic_failure(tmp_path):
    source = write_input(tmp_path / "raw.jsonl", [record("same"), record("same")])
    with pytest.raises(ContractError) as error:
        build_pool(config(source), tmp_path / "out")
    assert (
        error.value.code == ErrorCode.DUPLICATE_ID and error.value.details["line"] == 2
    )
    assert not (tmp_path / "out").exists()
    assert list(tmp_path.glob(".out.*.staging/failure.json"))


def test_oversize_reject_is_explicit_and_never_truncates(tmp_path):
    rows = [record("short"), record("long", tool_output="x" * 20000)]
    path = write_input(tmp_path / "raw.jsonl", rows)
    original = path.read_bytes()
    build_pool(config(path, max_context_tokens=1000), tmp_path / "out")
    pool = read_pool(tmp_path / "out")
    assert [r.data_id for r in pool.records] == ["short"]
    assert pool.excluded[0]["reason"] == "CONTEXT_LIMIT"
    assert path.read_bytes() == original
    with pytest.raises(ContractError) as error:
        build_pool(
            config(path, max_context_tokens=1000, oversize_policy="error"),
            tmp_path / "error",
        )
    assert error.value.code == ErrorCode.BUDGET_EXHAUSTED


def test_usage_histories_policies_remain_distinct(tmp_path):
    rows = [
        record("fresh"),
        record("scored"),
        record("rejected"),
        record("accepted"),
        record("pending"),
    ]
    rows[1]["usage"] = {
        "forward_scoring": [ref("score")],
        "gradient_extraction": [ref("gradient")],
    }
    for row, decision in zip(rows[2:], ("rejected", "accepted", "pending")):
        row["usage"] = {
            "candidate_sft": [
                {
                    "attempt_id": "try-1",
                    "model_id": "candidate-1",
                    "receipt": ref(decision),
                    "decision": decision,
                }
            ],
            "accepted_sft": ["try-1"] if decision == "accepted" else [],
        }
    path = write_input(tmp_path / "raw.jsonl", rows)
    build_pool(config(path), tmp_path / "strict")
    strict = read_pool(tmp_path / "strict")
    assert {r.data_id for r in strict.records} == {"fresh", "scored"}
    assert {r["reason"] for r in strict.excluded} == {
        "REJECTED_TRAINING_HISTORY",
        "ACCEPTED_TRAINING_HISTORY",
        "PENDING_TRAINING_ATTEMPT",
    }
    build_pool(config(path, rejected_history_policy="allow"), tmp_path / "allow")
    pool = read_pool(tmp_path / "allow")
    assert {r.data_id for r in pool.records} == {"fresh", "scored", "rejected"}
    rejected = next(r for r in pool.records if r.data_id == "rejected")
    assert rejected.usage.candidate_sft and not rejected.usage.accepted_sft


def test_corrupt_accepted_history_rejected():
    row = record()
    row["usage"] = {"accepted_sft": ["never-trained"]}
    with pytest.raises(ContractError):
        normalize_trajectory(row, FixtureTokenizer())


def test_modified_artifact_and_derived_fields_cannot_pass_read(tmp_path):
    path = write_input(tmp_path / "raw.jsonl", [record()])
    build_pool(config(path), tmp_path / "out")
    data = tmp_path / "out" / "trajectories.jsonl"
    row = load_json(data.read_text(encoding="utf-8"))
    row["stats"]["supervised_tokens"] += 1
    data.write_text(canonical_json(row) + "\n", encoding="utf-8")
    with pytest.raises(ContractError) as error:
        read_pool(tmp_path / "out")
    assert error.value.code == ErrorCode.ARTIFACT_MISMATCH
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = load_json(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["trajectories.jsonl"]["sha256"] = file_sha256(data)
    manifest["pool_id"] = digest({k: v for k, v in manifest.items() if k != "pool_id"})
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    with pytest.raises(ContractError):
        read_pool(tmp_path / "out")


@pytest.mark.parametrize(
    "key,value",
    [
        ("backend", "gpu"),
        ("max_context_tokens", True),
        ("skills", []),
        ("schema_version", "99"),
        ("unexpected", 1),
    ],
)
def test_config_validation(tmp_path, key, value):
    row = {
        "schema_version": "1",
        "input_path": "raw.jsonl",
        "backend": "fixture",
        "skills": ["recovery"],
        "max_context_tokens": 1000,
        "oversize_policy": "reject",
        "rejected_history_policy": "exclude",
    }
    row[key] = value
    path = tmp_path / "config.json"
    path.write_text(canonical_json(row), encoding="utf-8")
    with pytest.raises(ContractError) as error:
        load_config(path)
    assert error.value.code == ErrorCode.INVALID_CONFIG


def test_precomputed_requires_real_supplied_alignment_and_identity():
    raw, identity = golden_record()
    backend = PrecomputedTokenizer(identity)
    missing = copy.deepcopy(raw)
    missing.pop("tokenization")
    with pytest.raises(ContractError) as error:
        normalize_trajectory(missing, backend)
    assert error.value.code == ErrorCode.MISSING_SIGNAL
    with pytest.raises(ContractError):
        normalize_trajectory(raw, PrecomputedTokenizer(replace(identity, revision="2")))
    with pytest.raises(ContractError):
        normalize_trajectory(raw, FixtureTokenizer())


def test_cli_end_to_end_and_clean_import(tmp_path):
    source = write_input(tmp_path / "raw.jsonl", [record()])
    cfg = tmp_path / "config.json"
    cfg.write_text(
        canonical_json({**config(source).to_dict(), "tokenizer": None}),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "self_curate", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    assert run("--help").returncode == 0
    check = run("validate-config", "--config", str(cfg))
    assert check.returncode == 0, check.stderr
    result = run("normalize", "--config", str(cfg), "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    inspect = run("inspect-pool", "--pool", str(tmp_path / "out"))
    assert inspect.returncode == 0 and json.loads(inspect.stdout)["records"] == 1
    repeated = run("normalize", "--config", str(cfg), "--output", str(tmp_path / "out"))
    assert (
        repeated.returncode == 2
        and json.loads(repeated.stderr)["error"]["code"] == "OUTPUT_EXISTS"
    )
    bare = subprocess.run(
        [
            sys.executable,
            "-c",
            "import self_curate.data,sys; print('torch' in sys.modules, 'transformers' in sys.modules)",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert bare.stdout.strip() == "False False"


def test_interval_metrics_against_brute_force_token_sets():
    rng = random.Random(11)
    for _ in range(100):
        raw, identity = golden_record()
        span_sets = []
        for name in ("a", "b", "c"):
            spans = []
            for __ in range(rng.randrange(8)):
                msg = rng.randrange(4)
                payload = raw["tokenization"]["message_spans"][msg]
                start = rng.randrange(payload["start"], payload["end"])
                end = rng.randrange(start + 1, payload["end"] + 1)
                spans.append({"message_index": msg, "start": start, "end": end})
            span_sets.append(
                {"skill": name, "annotator_id": "random-fixture", "spans": spans}
            )
        raw["annotations"]["skills"] = span_sets
        result = normalize_trajectory(raw, PrecomputedTokenizer(identity))
        supervised = {
            i for i, value in enumerate(raw["tokenization"]["assistant_mask"]) if value
        }
        expected = {
            r["skill"]: set().union(
                *(set(range(s["start"], s["end"])) for s in r["spans"])
            )
            & supervised
            for r in span_sets
        }
        union = set().union(*expected.values())
        assert result.stats.evidence_union_tokens == len(union)
        for metric in result.evidence_metrics:
            assert metric.supervised_evidence_tokens == len(expected[metric.skill])
            assert metric.density == pytest.approx(
                len(expected[metric.skill]) / len(supervised)
            )
            assert metric.share == (
                pytest.approx(len(expected[metric.skill]) / len(union))
                if union
                else None
            )


def test_precomputed_pool_does_not_claim_authenticity(tmp_path):
    raw, identity = golden_record()
    path = write_input(tmp_path / "external.jsonl", [raw])
    cfg = config(
        path,
        backend="precomputed",
        tokenizer=identity,
        skills=("recovery", "testing", "empty", "missing"),
    )
    manifest = build_pool(cfg, tmp_path / "out")
    assert manifest["synthetic"] is None
    pool = read_pool(tmp_path / "out")
    assert pool.records[0].stats.supervised_tokens == 7
    audit = load_json((tmp_path / "out" / "audit.json").read_text(encoding="utf-8"))
    assert audit["source_authenticity_verified"] is False


def test_publish_has_bounded_retry_and_preserves_failure(tmp_path, monkeypatch):
    from self_curate.data import pool as implementation

    original = Path.rename
    attempts = []

    def flaky(path, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("simulated transient lock")
        return original(path, target)

    path = write_input(tmp_path / "raw.jsonl", [record()])
    monkeypatch.setattr(Path, "rename", flaky)
    monkeypatch.setattr(implementation.time, "sleep", lambda _: None)
    build_pool(config(path), tmp_path / "ok")
    assert len(attempts) == 3
    assert read_pool(tmp_path / "ok").records
    attempts.clear()

    def denied(path, target):
        attempts.append(1)
        raise PermissionError("persistent lock")

    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError):
        build_pool(config(path), tmp_path / "failed")
    assert len(attempts) == 3 and not (tmp_path / "failed").exists()
    assert list(tmp_path.glob(".failed.*.staging/failure.json"))


def test_unknown_pool_schema_and_missing_manifest_are_structured(tmp_path):
    with pytest.raises(ContractError) as error:
        read_pool(tmp_path / "missing")
    assert error.value.code == ErrorCode.ARTIFACT_MISMATCH
    path = write_input(tmp_path / "raw.jsonl", [record()])
    build_pool(config(path), tmp_path / "out")
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = load_json(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = []
    manifest["pool_id"] = digest({k: v for k, v in manifest.items() if k != "pool_id"})
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    with pytest.raises(ContractError) as error:
        read_pool(tmp_path / "out")
    assert error.value.code == ErrorCode.ARTIFACT_MISMATCH


def test_real_fixture_file_has_all_required_acceptance_cases(tmp_path):
    cfg = load_config(ROOT / "configs" / "m1.fixture.json")
    build_pool(cfg, tmp_path / "fixture")
    pool = read_pool(tmp_path / "fixture")
    assert len(pool.records) == 3
    assert {r.quality.status for r in pool.records} == {"pass", "fail", "unknown"}
    overlap = next(r for r in pool.records if r.data_id == "overlapping-evidence")
    assert sum(m.share for m in overlap.evidence_metrics) > 1
    assert any(all(m.share is None for m in r.evidence_metrics) for r in pool.records)
    long = next(r for r in pool.records if r.data_id == "long-tool-output")
    assert long.stats.context_tokens > 4000 and long.stats.supervised_tokens < 500
