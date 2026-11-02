"""M1 alignment review: data matching, versions, usage and operator-facing reads."""

from dataclasses import replace
import json

import pytest

from helpers import golden_record, record, ref, write_input
from test_m1 import config
from test_signal_interfaces import model, provenance, spec
from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ContractError,
    DataLabels,
    DataTag,
    ErrorCode,
    ProjectionSpec,
    GradientSpec,
    TrainingUse,
    UsageHistory,
    canonical_json,
    digest,
    parse_gradient_feature,
)
from self_curate.data import (
    FixtureTokenizer,
    MetricView,
    PrecomputedTokenizer,
    attach_annotation_record,
    attach_annotations,
    attach_gradients,
    attach_metadata_record,
    attach_signals,
    attach_usage,
    build_pool,
    get_gradient,
    make_annotation_record,
    make_gradient,
    make_metadata_record,
    make_signal,
    normalize_trajectory,
    read_pool,
    record_candidate_training,
    resolve_training,
)


def gradient_spec(**changes):
    base = GradientSpec(
        digest("coordinates-v1"),
        8,
        "assistant-ce-v1",
        "token_mean",
        "none-v1",
        digest("normalization-v1"),
        "float32",
        "raw-v1",
        digest("raw-transform-v1"),
        "fixture-backward-v1",
        digest("backward-v1"),
        digest("grad-config-v1"),
    )
    return replace(base, **changes)


def sample(data_id="a"):
    return normalize_trajectory(
        record(data_id), FixtureTokenizer(), ("recovery", "testing")
    )


def feature(row, m=None, s=None, **changes):
    m, s = m or model(row), s or gradient_spec()
    args = dict(
        vector=ArtifactRef(**ref("vector")),
        dimension=s.parameter_count if s.projection is None else s.projection.dimension,
        dtype=s.storage_dtype,
        norm=2.0,
        provenance=provenance("backward"),
        cost=ComputeCost(wall_seconds=0.5, processed_tokens=row.stats.context_tokens),
    )
    args.update(changes)
    return make_gradient(row, m, s, **args)


def test_bound_external_labels_reject_another_id_even_with_identical_content():
    raw, identity = golden_record()
    a = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    b = replace(a, data_id="another-id")
    envelope = make_annotation_record(a, a.annotations, provenance=provenance("labels"))
    updated = attach_annotation_record(a, envelope)
    assert (
        updated.annotation_record == envelope and updated.annotations == a.annotations
    )
    with pytest.raises(ContractError) as error:
        attach_annotation_record(b, envelope)
    assert error.value.code == ErrorCode.STALE_CACHE
    labels = make_metadata_record(
        a,
        DataLabels("bugfix", "repo/demo", ("dup-1",), (DataTag("language", "python"),)),
        provenance=provenance("metadata"),
    )
    assert attach_metadata_record(a, labels).metadata.repository == "repo/demo"
    with pytest.raises(ContractError):
        attach_metadata_record(b, labels)
    moved_source = replace(
        a, source=replace(a.source, revision="different-source-version")
    )
    with pytest.raises(ContractError):
        attach_annotation_record(moved_source, envelope)
    with pytest.raises(ContractError):
        attach_metadata_record(moved_source, labels)


def test_view_does_not_accept_conflicting_raw_values_in_separate_cdf_views():
    from test_signal_interfaces import cdf

    row = sample()
    m, s = model(row), spec()
    first = make_signal(
        row,
        m,
        s,
        provenance=provenance(),
        nll=1,
        nll_percentile=0.5,
        percentile_reference=cdf(m, s, "a"),
    )
    second = make_signal(
        row,
        m,
        s,
        provenance=provenance(),
        nll=2,
        nll_percentile=0.8,
        percentile_reference=cdf(m, s, "b"),
    )
    malformed = replace(row, signals=(first, second))
    with pytest.raises(ContractError) as error:
        MetricView((malformed,))
    assert error.value.code == ErrorCode.SIGNAL_CONFLICT


def test_metadata_is_distinct_from_training_labels_and_survives_pool(tmp_path):
    raw = record()
    row = sample(raw["data_id"])
    envelope = make_metadata_record(
        row, DataLabels("bugfix", "repo/demo", ("d1",)), provenance=provenance("meta")
    )
    enriched = attach_metadata_record(row, envelope)
    assert enriched.content_hash == row.content_hash and enriched.labels == row.labels
    raw["metadata_record"] = json.loads(canonical_json(envelope.to_dict()))
    path = write_input(tmp_path / "input.jsonl", [raw])
    build_pool(config(path, skills=("recovery", "testing")), tmp_path / "pool")
    loaded = read_pool(tmp_path / "pool").records[0]
    assert loaded.metadata == enriched.metadata and loaded.metadata_record == envelope


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_coordinates_hash": "0" * 64},
        {"loss_id": "another-loss"},
        {"reduction": "token_sum"},
        {"normalization_hash": "0" * 64},
        {"compute_dtype": "float64"},
        {"storage_dtype": "float16"},
        {"transform_hash": "0" * 64},
        {"optimizer_state_hash": "0" * 64},
        {"implementation_hash": "0" * 64},
        {"computation_hash": "0" * 64},
        {"projection": ProjectionSpec("count-sketch-v1", 4, 11, digest("projection"))},
        {"projection": ProjectionSpec("count-sketch-v1", 4, 22, digest("projection"))},
    ],
)
def test_gradient_versions_never_reuse_another_spec(changes):
    row = sample()
    old = feature(row)
    new_spec = gradient_spec(**changes)
    newer = feature(row, s=new_spec)
    assert old.cache_key != newer.cache_key
    enriched = attach_gradients(row, [old, newer])
    assert len(enriched.gradients) == 2
    assert get_gradient(enriched, model(row), new_spec).spec == new_spec
    with pytest.raises(ContractError):
        get_gradient(attach_gradients(row, [old]), model(row), new_spec)


def test_gradient_checkpoint_and_data_mismatch_are_separate():
    row = sample()
    old, new = feature(row), feature(row, m=model(row, "accepted-1"))
    enriched = attach_gradients(row, [old, new])
    assert get_gradient(enriched, model(row), gradient_spec()) == old
    assert get_gradient(enriched, model(row, "accepted-1"), gradient_spec()) == new
    with pytest.raises(ContractError) as error:
        attach_gradients(sample("b"), [old])
    assert error.value.code == ErrorCode.STALE_CACHE
    with pytest.raises(ContractError) as error:
        attach_gradients(enriched, [replace(old, norm=5)])
    assert error.value.code == ErrorCode.GRADIENT_CONFLICT


def test_gradient_missing_zero_and_invalid_are_distinct():
    row = sample()
    with pytest.raises(ContractError):
        get_gradient(row, model(row), gradient_spec())
    zero = feature(row, norm=0)
    assert zero.validity == "zero"
    assert (
        get_gradient(attach_gradients(row, [zero]), model(row), gradient_spec()).norm
        == 0
    )
    for changes in (
        {"norm": -1},
        {"norm": float("nan")},
        {"dimension": 7},
        {"dtype": "float16"},
    ):
        with pytest.raises(ContractError):
            feature(row, **changes)
    assert not row.usage.gradient_extraction
    updated = attach_gradients(row, [zero])
    assert updated.usage.gradient_extraction and not updated.usage.candidate_sft


def test_gradient_json_roundtrip_and_pool_storage(tmp_path):
    raw = record()
    row = sample(raw["data_id"])
    supplied = feature(row)
    payload = json.loads(canonical_json(supplied.to_dict()))
    assert parse_gradient_feature(payload) == supplied
    raw["gradients"] = [payload]
    path = write_input(tmp_path / "raw.jsonl", [raw])
    build_pool(config(path, skills=("recovery", "testing")), tmp_path / "pool")
    pool = read_pool(tmp_path / "pool")
    assert pool.records[0].gradients == (supplied,)
    assert (
        pool.metric_view().gradient(row.data_id, model(row), gradient_spec())
        == supplied
    )


def test_relabeling_invalidates_only_affected_skill_gradients():
    raw, identity = golden_record()
    row = normalize_trajectory(raw, PrecomputedTokenizer(identity))
    full, skill = gradient_spec(), gradient_spec(
        scope="skill_evidence", skill="recovery"
    )
    row = attach_gradients(row, [feature(row, s=full), feature(row, s=skill)])
    old = row
    annotation = row.annotations
    first = replace(
        annotation.skills[0], spans=(replace(annotation.skills[0].spans[0], end=5),)
    )
    annotation = replace(annotation, skills=(first, *annotation.skills[1:]))
    with pytest.raises(ContractError) as error:
        attach_annotations(row, annotation)
    assert error.value.code == ErrorCode.STALE_CACHE
    updated = attach_annotations(row, annotation, drop_stale_gradients=True)
    assert len(updated.gradients) == 1 and get_gradient(updated, model(row), full)
    assert len(old.gradients) == 2
    assert updated.usage.gradient_extraction == old.usage.gradient_extraction


def test_usage_transitions_preserve_exposure_and_reject_rewrites():
    row = sample()
    pending = TrainingUse(
        "attempt-1", "candidate-1", ArtifactRef(**ref("training")), "pending"
    )
    trained = record_candidate_training(row, pending)
    assert (
        not row.usage.candidate_sft
        and trained.usage.candidate_sft[0].decision == "pending"
    )
    rejected = resolve_training(trained, "attempt-1", "rejected")
    assert rejected.usage.candidate_sft and not rejected.usage.accepted_sft
    with pytest.raises(ContractError):
        resolve_training(rejected, "attempt-1", "accepted")
    accepted = resolve_training(trained, "attempt-1", "accepted")
    assert accepted.usage.accepted_sft == ("attempt-1",)
    assert (
        record_candidate_training(accepted, replace(pending, decision="accepted"))
        == accepted
    )
    with pytest.raises(ContractError):
        attach_usage(accepted, UsageHistory())
    with pytest.raises(ContractError):
        record_candidate_training(trained, replace(pending, model_id="other-model"))
    with pytest.raises(ContractError):
        resolve_training(row, "never-trained", "rejected")


def test_operator_batches_preserve_id_order_missing_cells_and_versions():
    a, b, c = sample("a"), sample("b"), sample("c")
    m, s = model(a), spec()
    a = attach_signals(a, [make_signal(a, m, s, provenance=provenance(), nll=0.0)])
    b = attach_signals(
        b, [make_signal(b, m, s, provenance=provenance(), nll=2.0, entropy=3.0)]
    )
    view = MetricView((a, b, c))
    batch = view.signal_batch(("b", "c", "a"), m, s, missing="none")
    assert batch.data_ids == ("b", "c", "a")
    assert batch.column("nll") == (2.0, None, 0.0)
    assert batch.column("entropy") == (3.0, None, None)
    assert batch.content_hashes == (b.content_hash, c.content_hash, a.content_hash)
    assert batch.context["model_hash"] == m.model_hash
    assert batch.context["feature_keys"] == (
        b.signals[0].cache_key,
        None,
        a.signals[0].cache_key,
    )
    assert batch.by_id()["a"]["nll"] == 0.0
    assert set(batch.missing) == {("c", "nll"), ("c", "entropy"), ("a", "entropy")}
    with pytest.raises(ContractError):
        view.signal_batch(("b", "a"), m, s)
    with pytest.raises(ContractError):
        view.signal_batch(("b",), model(a, "other-checkpoint"), s)
    assert view.static_batch(("b", "a"), ("supervised_tokens", "skills")).column(
        "skills"
    ) == (None, None)
    for ids, code in (
        (("a", "a"), ErrorCode.DUPLICATE_ID),
        (("unknown",), ErrorCode.UNKNOWN_DATA_ID),
    ):
        with pytest.raises(ContractError) as error:
            view.static_batch(ids, ("supervised_tokens",))
        assert error.value.code == code
    with pytest.raises(ContractError):
        MetricView((a, a))


def test_declared_skills_are_not_positive_labels_and_snapshot_does_not_mutate():
    raw, identity = golden_record()
    row = normalize_trajectory(
        raw,
        PrecomputedTokenizer(identity),
        ("empty", "missing", "recovery", "testing", "unlabeled"),
    )
    view = MetricView((row,))
    assert (
        "unlabeled"
        not in view.static_batch((row.data_id,), ("skills",)).column("skills")[0]
    )
    assert view.evidence_batch((row.data_id,), "missing").column("share") == (None,)
    use = TrainingUse("a", "candidate", ArtifactRef(**ref("receipt")), "accepted")
    updated = record_candidate_training(row, use)
    assert view.static_batch((row.data_id,), ("used_accepted_sft",)).column(
        "used_accepted_sft"
    ) == (False,)
    assert MetricView((updated,)).static_batch(
        (row.data_id,), ("used_accepted_sft",)
    ).column("used_accepted_sft") == (True,)


def test_reads_do_not_revalidate_large_payloads_or_scan_features(monkeypatch):
    import self_curate.data.view as implementation

    row = sample()
    row = attach_signals(
        row, [make_signal(row, model(row), spec(), provenance=provenance(), nll=1)]
    )
    row = attach_gradients(row, [feature(row)])
    view = MetricView((row,))

    def forbidden(*args, **kwargs):
        raise AssertionError("Lookup unexpectedly revalidated token payload")

    monkeypatch.setattr(implementation, "data_binding", forbidden)
    for _ in range(100):
        assert view.record(row.data_id) is row
        assert view.signal(row.data_id, model(row), spec()).nll == 1
        assert view.gradient(row.data_id, model(row), gradient_spec()).norm == 2
        assert (
            view.static_batch((row.data_id,), ("supervised_tokens",)).column(
                "supervised_tokens"
            )[0]
            > 0
        )


def test_joint_metric_alignment_against_independent_id_oracle():
    import random

    rows, oracle = [], {}
    for i in range(30):
        row = sample(f"data-{i:02d}")
        initial, next_model = model(row), model(row, "next")
        a = make_signal(
            row,
            initial,
            spec(),
            provenance=provenance(),
            nll=float(i),
            entropy=None if i % 3 == 0 else float(100 - i),
        )
        b = make_signal(
            row,
            next_model,
            spec(),
            provenance=provenance("next"),
            nll=float(i + 1000),
            entropy=float(2 * i),
        )
        row = attach_signals(row, [a, b])
        if i % 5 == 0:
            row = attach_gradients(row, [feature(row)])
        if i % 4:
            decision = ("pending", "rejected", "accepted")[i % 4 - 1]
            row = record_candidate_training(
                row,
                TrainingUse("attempt", "model", ArtifactRef(**ref("train")), decision),
            )
        rows.append(row)
        oracle[row.data_id] = (
            float(i),
            None if i % 3 == 0 else float(100 - i),
            float(i + 1000),
            float(2 * i),
            i % 5 == 0,
            i % 4 == 3,
        )
    random.Random(11).shuffle(rows)
    view = MetricView(rows)
    ids = list(oracle)
    random.Random(22).shuffle(ids)
    for subset in (tuple(ids), tuple(ids[::2]), tuple(ids[::-3])):
        old = view.signal_batch(subset, model(rows[0]), spec(), missing="none")
        new = view.signal_batch(subset, model(rows[0], "next"), spec())
        grad = view.gradient_batch(
            subset, model(rows[0]), gradient_spec(), missing="none"
        )
        used = view.static_batch(subset, ("used_accepted_sft",))
        assert old.data_ids == new.data_ids == grad.data_ids == used.data_ids == subset
        assert old.column("nll") == tuple(oracle[d][0] for d in subset)
        assert old.column("entropy") == tuple(oracle[d][1] for d in subset)
        assert new.column("nll") == tuple(oracle[d][2] for d in subset)
        assert new.column("entropy") == tuple(oracle[d][3] for d in subset)
        assert tuple(v is not None for v in grad.column("gradient")) == tuple(
            oracle[d][4] for d in subset
        )
        assert used.column("used_accepted_sft") == tuple(oracle[d][5] for d in subset)


def test_length_labels_cannot_disagree_with_underlying_record():
    row = sample()
    for field in (
        "context_tokens",
        "supervised_tokens",
        "assistant_turns",
        "tool_calls",
        "tool_observations",
    ):
        with pytest.raises(ContractError):
            replace(
                row, stats=replace(row.stats, **{field: getattr(row.stats, field) + 1})
            )


def test_conflicting_gradient_cost_is_not_silently_discarded():
    row = sample()
    old = feature(row)
    with pytest.raises(ContractError) as error:
        attach_gradients(row, [old, replace(old, cost=ComputeCost(wall_seconds=2))])
    assert error.value.code == ErrorCode.GRADIENT_CONFLICT


@pytest.mark.parametrize("field", ("norm", "cost", "confidence"))
def test_extreme_numbers_are_structured_errors(field):
    enormous = 10**1000
    with pytest.raises(ContractError):
        if field == "norm":
            feature(sample(), norm=enormous)
        elif field == "cost":
            ComputeCost(wall_seconds=enormous)
        else:
            from self_curate.contracts import SkillAnnotation

            SkillAnnotation("skill", (), "annotator", confidence=enormous)
