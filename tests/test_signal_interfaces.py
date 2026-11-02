"""External-value transport tests. No language model or scoring implementation."""

from dataclasses import replace
import copy
import json

import pytest

from helpers import golden_record, record, ref, write_input
from test_m1 import config
from self_curate.contracts import (
    ArtifactRef,
    ContractError,
    ErrorCode,
    ModelRef,
    PercentileReference,
    SignalSpec,
    canonical_json,
    digest,
    parse_annotations,
    parse_signal_record,
)
from self_curate.contracts.parsing import load_json
from self_curate.data import (
    FixtureTokenizer,
    PrecomputedTokenizer,
    attach_annotations,
    attach_signals,
    build_pool,
    get_signal,
    make_signal,
    normalize_trajectory,
    read_pool,
    signal_binding,
)
from self_curate.data.pool import file_sha256


def sample():
    return normalize_trajectory(record(), FixtureTokenizer(), ("recovery", "testing"))


def model(row, version="initial"):
    return ModelRef(
        "fixture-model", version, digest(version), row.tokenization.identity, "fixture"
    )


def spec(**changes):
    return replace(
        SignalSpec(
            "assistant",
            "token_mean",
            "e",
            "external-fixture-forward-v1",
            digest("forward-config"),
        ),
        **changes,
    )


def provenance(name="forward"):
    return (ArtifactRef(**ref(name)),)


def cdf(m, s, group="coding"):
    return PercentileReference(
        group,
        digest(["reference-a", "reference-b"]),
        "1",
        ArtifactRef(**ref(f"cdf-{group}")),
        m.model_hash,
        s.spec_hash,
    )


def test_reserved_slots_are_missing_and_do_not_run_a_provider():
    row = sample()
    assert row.signals == () and row.to_dict()["signals"] == ()
    assert row.annotations is None
    assert all(m.status == "missing" for m in row.evidence_metrics)
    with pytest.raises(ContractError) as error:
        get_signal(row, model(row), spec())
    assert error.value.code == ErrorCode.MISSING_SIGNAL


def test_attach_partial_values_and_exact_checkpoint_lookup():
    row = sample()
    before = row.to_dict()
    initial, updated = model(row), model(row, "accepted-1")
    first = make_signal(row, initial, spec(), provenance=provenance(), nll=0)
    second = make_signal(
        row, updated, spec(), provenance=provenance("next"), nll=1.5, entropy=2.5
    )
    enriched = attach_signals(row, [second, first])
    assert row.to_dict() == before
    assert (
        enriched.content_hash == row.content_hash
        and enriched.record_hash != row.record_hash
    )
    assert len(enriched.signals) == 2
    assert get_signal(enriched, initial, spec()).require_value("nll") == 0
    assert get_signal(enriched, updated, spec()).require_value("entropy") == 2.5
    with pytest.raises(ContractError) as error:
        get_signal(enriched, initial, spec()).require_value("entropy")
    assert error.value.code == ErrorCode.MISSING_SIGNAL
    with pytest.raises(ContractError):
        get_signal(enriched, model(row, "uncomputed"), spec())
    assert len(enriched.usage.forward_scoring) == 2
    assert not enriched.usage.gradient_extraction and not enriched.usage.candidate_sft


def test_complementary_values_merge_and_conflicts_fail_without_mutation():
    row, s = sample(), spec()
    m = model(row)
    nll = make_signal(row, m, s, provenance=provenance("nll"), nll=1.25)
    ent = make_signal(row, m, s, provenance=provenance("entropy"), entropy=2.75)
    enriched = attach_signals(row, [nll, ent])
    assert enriched == attach_signals(row, [ent, nll])
    assert len(enriched.signals) == 1 and len(enriched.signals[0].provenance) == 2
    assert enriched.signals[0].nll == 1.25 and enriched.signals[0].entropy == 2.75
    unchanged = enriched.record_hash
    with pytest.raises(ContractError) as error:
        attach_signals(enriched, [replace(nll, nll=9)])
    assert error.value.code == ErrorCode.SIGNAL_CONFLICT
    assert enriched.record_hash == unchanged


@pytest.mark.parametrize(
    "field,value",
    [
        ("nll", -1),
        ("entropy", float("nan")),
        ("entropy", float("inf")),
        ("nll", True),
        ("entropy", "2"),
    ],
)
def test_bad_external_values_are_rejected(field, value):
    row = sample()
    with pytest.raises(ContractError):
        make_signal(row, model(row), spec(), provenance=provenance(), **{field: value})


def test_empty_signal_is_absence_not_a_zero_record():
    row = sample()
    with pytest.raises(ContractError):
        make_signal(row, model(row), spec(), provenance=provenance())


@pytest.mark.parametrize("mutation", ["data_id", "content", "tokens", "tokenizer"])
def test_bindings_reject_unrelated_or_stale_data(mutation):
    row = sample()
    signal = make_signal(row, model(row), spec(), provenance=provenance(), nll=1)
    other = row
    if mutation == "data_id":
        other = replace(row, data_id="other-id")
    elif mutation == "content":
        raw = record(content="different assistant payload")
        other = normalize_trajectory(raw, FixtureTokenizer())
    elif mutation == "tokens":
        ids = list(row.tokenization.input_ids)
        ids[0] += 1
        other = replace(
            row, tokenization=replace(row.tokenization, input_ids=tuple(ids))
        )
    else:
        signal = replace(
            signal,
            model_ref=replace(
                signal.model_ref,
                tokenizer=replace(signal.model_ref.tokenizer, template_hash="0" * 64),
            ),
        )
    with pytest.raises(ContractError) as error:
        attach_signals(other, [signal])
    assert error.value.code == ErrorCode.STALE_CACHE


@pytest.mark.parametrize(
    "changes",
    [{"reduction": "token_sum"}, {"log_base": "2"}, {"computation_hash": "0" * 64}],
)
def test_lookup_never_substitutes_a_different_computation(changes):
    row = sample()
    enriched = attach_signals(
        row, [make_signal(row, model(row), spec(), provenance=provenance(), nll=1)]
    )
    with pytest.raises(ContractError) as error:
        get_signal(enriched, model(row), spec(**changes))
    assert error.value.code == ErrorCode.MISSING_SIGNAL


def test_external_labels_have_a_direct_attachment_interface():
    raw, identity = golden_record()
    supplied = raw.pop("annotations")
    row = normalize_trajectory(
        raw, PrecomputedTokenizer(identity), ("recovery", "testing", "empty", "missing")
    )
    before = row.record_hash
    labels = parse_annotations(supplied)
    enriched = attach_annotations(row, labels)
    assert row.annotations is None and row.record_hash == before
    assert enriched.stats.evidence_union_tokens == 5
    assert enriched.content_hash == row.content_hash
    assert enriched.evidence_metrics[2].status == "annotated"  # recovery, alphabetical
    with pytest.raises(ContractError):
        attach_annotations(row, replace(labels, tokenizer_hash="0" * 64))


def test_skill_scoring_scope_is_exactly_the_supervised_evidence_union():
    raw, identity = golden_record()
    row = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    skill = spec(scope="skill_evidence", skill="recovery")
    assert signal_binding(row, spec()).token_count == 7
    assert signal_binding(row, skill).token_count == 4
    assert (
        signal_binding(row, skill).selection_hash
        != signal_binding(row, spec()).selection_hash
    )
    for name in ("empty", "missing", "unknown"):
        with pytest.raises(ContractError) as error:
            signal_binding(row, spec(scope="skill_evidence", skill=name))
        assert error.value.code == ErrorCode.MISSING_SIGNAL


def test_annotation_changes_preserve_full_scope_but_invalidate_changed_skill_scope():
    raw, identity = golden_record()
    row = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    full, skill = spec(), spec(scope="skill_evidence", skill="recovery")
    enriched = attach_signals(
        row,
        [
            make_signal(row, model(row), s, provenance=provenance(s.scope), nll=1)
            for s in (full, skill)
        ],
    )
    labels = copy.deepcopy(raw["annotations"])
    labels["skills"][0]["spans"] = [{"message_index": 1, "start": 4, "end": 5}]
    labels = parse_annotations(labels)
    with pytest.raises(ContractError) as error:
        attach_annotations(enriched, labels)
    assert error.value.code == ErrorCode.STALE_CACHE
    updated = attach_annotations(enriched, labels, drop_stale_signals=True)
    assert len(updated.signals) == 1 and get_signal(updated, model(row), full).nll == 1
    assert len(enriched.signals) == 2
    assert len(updated.usage.forward_scoring) == 2  # Prior computation really occurred.


def test_percentiles_are_bound_to_model_reference_group_and_cdf():
    row, s = sample(), spec()
    m = model(row)
    reference = cdf(m, s)
    signal = make_signal(
        row,
        m,
        s,
        provenance=provenance(),
        nll=1.2,
        entropy=2.3,
        nll_percentile=0.4,
        entropy_percentile=0.8,
        percentile_reference=reference,
    )
    parsed = parse_signal_record(json.loads(canonical_json(signal.to_dict())))
    assert parsed == signal
    enriched = attach_signals(row, [signal])
    assert (
        get_signal(enriched, m, s, percentile_reference=reference).require_value(
            "nll", "percentile"
        )
        == 0.4
    )
    with pytest.raises(ContractError):
        get_signal(enriched, m, s, percentile_reference=cdf(m, s, "another-group"))
    with pytest.raises(ContractError):
        replace(signal, percentile_reference=replace(reference, model_hash="0" * 64))
    with pytest.raises(ContractError):
        replace(signal, percentile_reference=replace(reference, spec_hash="0" * 64))
    with pytest.raises(ContractError):
        replace(signal, percentile_reference=None)
    with pytest.raises(ContractError):
        replace(signal, nll_percentile=1.01)
    other_view = replace(signal, percentile_reference=cdf(m, s, "other"))
    assert (
        other_view.raw_key == signal.raw_key
        and other_view.cache_key != signal.cache_key
    )
    with pytest.raises(ContractError):
        attach_signals(enriched, [replace(other_view, nll=9)])


def test_signal_input_and_pool_output_roundtrip(tmp_path):
    raw = record()
    row = normalize_trajectory(raw, FixtureTokenizer(), ("recovery",))
    signal = make_signal(
        row, model(row), spec(), provenance=provenance(), nll=1.25, entropy=2.5
    )
    raw["signals"] = json.loads(canonical_json([signal.to_dict()]))
    source = write_input(tmp_path / "external-values.jsonl", [raw])
    build_pool(config(source), tmp_path / "out")
    pool = read_pool(tmp_path / "out")
    assert get_signal(pool.records[0], model(row), spec()).nll == 1.25
    assert pool.records[0].signals[0].entropy == 2.5
    audit = load_json((tmp_path / "out/audit.json").read_text(encoding="utf-8"))
    assert audit["signal_records"] == audit["records_with_signals"] == 1
    assert len(pool.records[0].usage.forward_scoring) == 1


def test_old_m1_pool_is_readable_without_rewriting_evidence(tmp_path):
    source = write_input(tmp_path / "source.jsonl", [record()])
    build_pool(config(source), tmp_path / "legacy")
    root = tmp_path / "legacy"
    data = root / "trajectories.jsonl"
    row = load_json(data.read_text(encoding="utf-8"))
    row.pop("signals")
    for name in ("metadata", "gradients", "annotation_record", "metadata_record"):
        row.pop(name)
    row["normalization_version"] = "m1-v1"
    data.write_text(canonical_json(row) + "\n", encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["record_hash"] = digest(row)
    manifest["artifacts"]["trajectories.jsonl"]["sha256"] = file_sha256(data)
    manifest["pool_id"] = digest({k: v for k, v in manifest.items() if k != "pool_id"})
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    before = {p.name: file_sha256(p) for p in root.iterdir()}
    loaded = read_pool(root).records[0]
    assert loaded.normalization_version == "m1-v1" and loaded.signals == ()
    assert loaded.record_hash == manifest["records"][0]["record_hash"]
    assert "signals" not in loaded.to_dict()
    assert before == {p.name: file_sha256(p) for p in root.iterdir()}
    enriched = attach_signals(
        loaded,
        [make_signal(loaded, model(loaded), spec(), provenance=provenance(), nll=1)],
    )
    assert enriched.normalization_version == "m1-v3" and enriched.signals


def test_signal_parser_rejects_unknown_fields():
    row = sample()
    value = json.loads(
        canonical_json(
            make_signal(
                row, model(row), spec(), provenance=provenance(), nll=1
            ).to_dict()
        )
    )
    value["unexpected"] = 1
    with pytest.raises(ContractError):
        parse_signal_record(value)
