from dataclasses import replace
import copy
import json
import random

import pytest

from helpers import ref
from helpers_m2 import MODEL, SCORE, context, fixture_program, node, program, records
from self_curate.contracts import (
    ArtifactRef,
    ContractError,
    ErrorCode,
    TrainingUse,
    canonical_json,
    parse_batch_manifest,
    parse_program,
    verify_batch,
)
from self_curate.data import MetricView, record_candidate_training
from self_curate.operators import (
    BudgetPolicy,
    QualityPolicy,
    SimilarityPair,
    SimilarityTable,
    execute_program,
)


def trace(result, node_id="x"):
    return next(t for t in result.traces if t.node_id == node_id)


def single(rows, op, params, **kwargs):
    return execute_program(program([node("x", op, params)]), context(rows, **kwargs))


def test_complete_eight_operator_pipeline_has_exact_handoff_and_unchanged_inputs():
    rows = records()
    before = tuple(r.record_hash for r in rows)
    ctx = context(rows, target=64, tolerance=0, batch=8, share_tolerance=0)
    p = fixture_program()
    result = execute_program(p, ctx)
    assert result.status == "feasible"
    batch = result.batch
    assert len(batch.members) == 8 and batch.supervised_tokens == 64
    assert batch.actual_branch_tokens == {"a": 32, "b": 32}
    assert batch.actual_branch_shares == {"a": 0.5, "b": 0.5}
    exposure = trace(result, "final_batch").details["semantic_skill_exposure"]
    assert exposure["missing_label_samples"] == 0
    assert exposure["skills"]["recovery"] == {
        "sample_count": 4,
        "trajectory_supervised_tokens": 32,
        "known_evidence_supervised_tokens": 16,
        "evidence_missing_samples": 0,
    }
    assert trace(result, "final_batch").details["branch_share_residuals"] == {
        "a": 0,
        "b": 0,
    }
    assert batch.data_ids == tuple(sorted(batch.data_ids))
    assert (
        set(batch.data_ids) == set(batch.training_order) == set(batch.selection_order)
    )
    assert verify_batch(batch, ctx.view) == batch.training_order
    parsed = parse_batch_manifest(json.loads(canonical_json(batch.to_dict())))
    assert parsed == batch
    assert {t.operator for t in result.traces} >= {
        "QualityGate",
        "BudgetPack",
        "SkillSelect",
        "EvidenceSelect",
        "NLLSelect",
        "EntropySelect",
        "LengthSelect",
        "DedupSelect",
        "SampleSelect",
        "MixAllocate",
    }
    assert tuple(r.record_hash for r in rows) == before
    assert execute_program(p, ctx) == result
    assert execute_program(replace(p, nodes=tuple(reversed(p.nodes))), ctx) == result
    shuffled = list(rows)
    random.Random(33).shuffle(shuffled)
    assert (
        execute_program(
            p, context(shuffled, target=64, tolerance=0, batch=8, share_tolerance=0)
        ).batch
        == batch
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_op",
        "unknown_field",
        "unknown_input",
        "cycle",
        "duplicate",
        "unreachable",
        "quality_node",
        "budget_override",
        "bad_arity",
        "reserved",
        "bad_input_type",
        "bad_output_type",
    ],
)
def test_program_schema_rejects_bad_or_unsafe_graphs(mutation):
    raw = {
        "schema_version": "1",
        "program_id": "p",
        "seed": 11,
        "nodes": [node("x", "LengthSelect", {"supervised_token_range": [1, None]})],
        "output": "x",
    }
    if mutation == "unknown_op":
        raw["nodes"][0]["op"] = "exec"
    elif mutation == "unknown_field":
        raw["nodes"][0]["params"]["python"] = "print(1)"
    elif mutation == "unknown_input":
        raw["nodes"][0]["inputs"] = ["missing"]
    elif mutation == "cycle":
        raw["nodes"][0]["inputs"] = ["x"]
    elif mutation == "duplicate":
        raw["nodes"].append(copy.deepcopy(raw["nodes"][0]))
    elif mutation == "unreachable":
        raw["nodes"].append(
            node("unused", "LengthSelect", {"context_token_range": [1, None]})
        )
    elif mutation == "quality_node":
        raw["nodes"][0]["op"] = "QualityGate"
    elif mutation == "budget_override":
        raw["budget_pack"] = {"token_target": 1}
    elif mutation == "bad_arity":
        raw["nodes"][0]["inputs"] = ["eligible_pool", "x"]
    elif mutation == "reserved":
        raw["nodes"][0]["id"] = "final_batch"
        raw["output"] = "final_batch"
    elif mutation == "bad_input_type":
        raw["nodes"][0]["inputs"] = [[]]
    else:
        raw["output"] = []
    with pytest.raises(ContractError) as error:
        parse_program(raw)
    assert error.value.code == ErrorCode.INVALID_PROGRAM


@pytest.mark.parametrize(
    "op,params",
    [
        ("SkillSelect", {"skills": []}),
        ("SkillSelect", {"skills": ["x"], "match": "maybe"}),
        ("EvidenceSelect", {"target_skill": "x", "min_share": 1.1}),
        ("EvidenceSelect", {"target_skill": "x", "min_events": True}),
        ("NLLSelect", {"interval": [3, 1]}),
        ("NLLSelect", {"keep_fraction": 0}),
        (
            "EntropySelect",
            {"scale": "percentile", "interval": [0, 2], "reference_group": "coding"},
        ),
        ("EntropySelect", {"scope": "skill_evidence", "interval": [0, 1]}),
        ("LengthSelect", {"tool_call_range": [1.5, 3]}),
        ("DedupSelect", {"mode": "semantic"}),
        ("SampleSelect", {"method": "uniform", "size": True}),
        ("SampleSelect", {"method": "uniform", "size": 2, "token_target": 10}),
        ("MixAllocate", {"branch_token_shares": {"a": 0.8, "b": 0.8}}),
    ],
)
def test_operator_parameters_are_strict(op, params):
    with pytest.raises(ContractError) as error:
        program([node("x", op, params)])
    assert error.value.code == ErrorCode.INVALID_PROGRAM


def test_quality_gate_cannot_be_bypassed_and_records_each_reason():
    rows = list(records(8))
    rows[0] = replace(rows[0], quality=replace(rows[0].quality, status="fail"))
    rows[1] = replace(rows[1], quality=replace(rows[1].quality, status="unknown"))
    receipt = ArtifactRef(**ref("train"))
    for i, decision in ((2, "pending"), (3, "accepted"), (4, "rejected")):
        rows[i] = record_candidate_training(
            rows[i], TrainingUse(f"a{i}", "m", receipt, decision)
        )
    policy = QualityPolicy(excluded_ids=(rows[5].data_id,))
    result = single(
        rows, "LengthSelect", {"supervised_token_range": [1, None]}, quality=policy
    )
    assert trace(result, "eligible_pool").output_ids == ("data-06", "data-07")
    assert {r for _, r, _ in trace(result, "eligible_pool").exclusions} == {
        "QUALITY_FAIL",
        "QUALITY_UNKNOWN",
        "PENDING_TRAINING",
        "ACCEPTED_TRAINING_HISTORY",
        "REJECTED_TRAINING_HISTORY",
        "SYSTEM_EXCLUSION",
    }
    assert set(result.batch.data_ids) == {"data-06", "data-07"}
    more = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        quality=QualityPolicy(
            allow_unknown=True, exclude_accepted=False, rejected_policy="allow"
        ),
    )
    assert "data-00" not in more.batch.data_ids and "data-02" not in more.batch.data_ids
    assert {"data-01", "data-03", "data-04"}.issubset(more.batch.data_ids)


def test_skills_any_all_missing_and_registry_are_distinct():
    rows = list(records(4))
    assert trace(single(rows, "SkillSelect", {"skills": ["recovery"]})).output_ids == (
        "data-00",
        "data-02",
    )
    assert (
        trace(
            single(
                rows,
                "SkillSelect",
                {"skills": ["recovery", "regression"], "match": "all"},
            )
        ).output_ids
        == ()
    )
    rows[0] = replace(rows[0], annotations=None)
    assert (
        "data-00"
        not in trace(single(rows, "SkillSelect", {"skills": ["recovery"]})).output_ids
    )
    assert (
        "data-00"
        in trace(
            single(
                rows,
                "SkillSelect",
                {"skills": ["recovery"], "missing_policy": "include"},
            )
        ).output_ids
    )
    with pytest.raises(ContractError):
        single(rows, "SkillSelect", {"skills": ["recovery"], "missing_policy": "error"})


def test_evidence_conditions_use_m1_metrics_and_do_not_fill_missing():
    rows = list(records(4))
    params = {
        "target_skill": "recovery",
        "min_share": 1,
        "min_density": 0.5,
        "min_tokens": 4,
        "min_events": 1,
        "required_behaviors": ["test_after_final_edit"],
    }
    assert trace(single(rows, "EvidenceSelect", params)).output_ids == (
        "data-00",
        "data-02",
    )
    assert (
        trace(single(rows, "EvidenceSelect", {**params, "min_tokens": 5})).output_ids
        == ()
    )
    annotation = rows[0].annotations
    from self_curate.data import attach_annotations

    rows[0] = attach_annotations(
        rows[0],
        replace(
            annotation, skills=(replace(annotation.skills[0], complete_events=None),)
        ),
    )
    assert "data-00" not in trace(single(rows, "EvidenceSelect", params)).output_ids
    with pytest.raises(ContractError):
        single(rows, "EvidenceSelect", {**params, "missing_policy": "error"})


def test_score_intervals_ranking_ties_fraction_and_percentile_identity():
    rows = records(8)
    assert trace(single(rows, "NLLSelect", {"interval": [2, 5]})).output_ids == (
        "data-02",
        "data-03",
        "data-04",
        "data-05",
    )
    assert trace(
        single(
            rows,
            "NLLSelect",
            {"interval": [2, 5], "keep_fraction": 0.5, "rank_direction": "descending"},
        )
    ).output_ids == ("data-05", "data-04")
    assert trace(single(rows, "EntropySelect", {"interval": [2, 4]})).output_ids == (
        "data-04",
        "data-05",
        "data-06",
    )
    percentile = single(
        rows,
        "NLLSelect",
        {"scale": "percentile", "reference_group": "coding", "interval": [0, 0.5]},
    )
    assert trace(percentile).output_ids == ("data-00", "data-01", "data-02", "data-03")
    assert trace(percentile).details["reference_hash"] is not None
    with pytest.raises(ContractError):
        single(
            rows,
            "NLLSelect",
            {
                "scale": "percentile",
                "reference_group": "unregistered",
                "interval": [0, 1],
            },
        )
    with pytest.raises(ContractError):
        single(
            rows,
            "NLLSelect",
            {"interval": [0, 100]},
            model=replace(MODEL, revision="new"),
        )


def test_missing_scores_exclude_or_return_exact_external_request():
    rows = list(records(4))
    rows[0] = replace(rows[0], signals=())
    result = single(
        rows, "NLLSelect", {"interval": [0, 10], "missing_policy": "exclude"}
    )
    assert trace(result).output_ids == ("data-01", "data-02", "data-03")
    with pytest.raises(ContractError) as error:
        single(rows, "NLLSelect", {"interval": [0, 10], "missing_policy": "request"})
    assert error.value.code == ErrorCode.MISSING_SIGNAL
    request = error.value.details["requests"][0]
    assert request["data_ids"] == ("data-00",) and request["metric"] == "nll"
    assert (
        request["model_ref"] == MODEL.to_dict() and request["spec"] == SCORE.to_dict()
    )


def test_fixed_filter_reordering_is_commutative_without_implicit_ranking():
    rows = records(8)
    a = node("a", "SkillSelect", {"skills": ["recovery"]})
    b = node("b", "NLLSelect", {"interval": [2, 6]}, ("a",))
    first = execute_program(program([a, b]), context(rows))
    second = execute_program(
        program(
            [
                node("b", "NLLSelect", {"interval": [2, 6]}),
                node("a", "SkillSelect", {"skills": ["recovery"]}, ("b",)),
            ]
        ),
        context(rows),
    )
    assert (
        first.batch.data_ids
        == second.batch.data_ids
        == ("data-02", "data-04", "data-06")
    )


def test_exact_group_and_semantic_dedup_are_different_and_versioned():
    rows = list(records(4))
    duplicate = replace(rows[0], data_id="copy", signals=(), annotation_record=None)
    result = single(
        (*rows, duplicate),
        "DedupSelect",
        {"mode": "exact", "representative_rule": "lowest_id"},
    )
    assert (
        "copy" in trace(result).output_ids and "data-00" not in trace(result).output_ids
    )
    groups = (("g1",), ("g1", "g2"), ("g2",), ())
    grouped = [
        replace(row, metadata=replace(row.metadata, duplicate_groups=group))
        for row, group in zip(rows, groups)
    ]
    assert trace(
        single(grouped, "DedupSelect", {"mode": "annotated_group"})
    ).output_ids == ("data-00", "data-03")
    pairs = tuple(
        SimilarityPair(
            a.data_id,
            b.data_id,
            a.content_hash,
            b.content_hash,
            0.95 if {a.data_id, b.data_id} == {"data-00", "data-01"} else 0.1,
        )
        for i, a in enumerate(rows)
        for b in rows[i + 1 :]
    )
    table = SimilarityTable(ArtifactRef(**ref("semantic-v1")), pairs)
    semantic = single(
        rows, "DedupSelect", {"mode": "semantic", "threshold": 0.9}, similarities=table
    )
    assert trace(semantic).output_ids == ("data-00", "data-02", "data-03")
    assert trace(semantic).details["comparisons"] > 0
    with pytest.raises(ContractError):
        single(rows, "DedupSelect", {"mode": "semantic", "threshold": 0.9})
    bad_pair = replace(pairs[0], left_content_hash="0" * 64)
    with pytest.raises(ContractError) as error:
        single(
            rows,
            "DedupSelect",
            {"mode": "semantic", "threshold": 0.9},
            similarities=replace(table, pairs=(bad_pair, *pairs[1:])),
        )
    assert error.value.code == ErrorCode.STALE_CACHE


@pytest.mark.parametrize("method", ["uniform", "ranked", "stratified"])
def test_sampling_by_count_and_tokens_is_complete_unique_and_repeatable(method):
    rows = records(12)
    params = {"method": method, "size": 4, "seed": 19}
    if method == "stratified":
        params["strata"] = ["repository"]
    before = random.getstate()
    result = single(rows, "SampleSelect", params, target=32, tolerance=0)
    assert result.status == "feasible" and len(result.batch.members) == 4
    assert result == single(rows, "SampleSelect", params, target=32, tolerance=0)
    assert random.getstate() == before
    tokens = {**params, "token_target": 32}
    tokens.pop("size")
    token_result = single(rows, "SampleSelect", tokens, target=32, tolerance=0)
    assert token_result.batch.supervised_tokens == 32
    short = single(rows, "SampleSelect", {**params, "size": 100})
    assert short.status == "infeasible" and short.batch is None
    if method == "stratified":
        repos = [
            row.metadata.repository
            for row in rows
            if row.data_id in result.batch.data_ids
        ]
        assert repos.count("repo-0") == repos.count("repo-1") == 2


def test_mix_overlap_has_unique_ownership_and_budget_checks_real_shares():
    rows = list(records(8))
    from self_curate.data import attach_annotations
    from self_curate.contracts import SkillAnnotation

    for i, row in enumerate(rows):
        annotations = []
        for name, matches in (("a", i <= 4), ("b", i >= 3)):
            if matches:
                annotations.append(SkillAnnotation(name, (), "fixture"))
        rows[i] = attach_annotations(
            row, replace(row.annotations, skills=tuple(annotations))
        )
    p = program(
        [
            node("a", "SkillSelect", {"skills": ["a"]}),
            node("b", "SkillSelect", {"skills": ["b"]}),
            node(
                "mix",
                "MixAllocate",
                {"branch_token_shares": {"a": 0.5, "b": 0.5}},
                ("a", "b"),
            ),
        ]
    )
    result = execute_program(
        p, context(rows, target=48, tolerance=0, batch=2, share_tolerance=0)
    )
    assert result.batch.data_ids == (
        "data-00",
        "data-01",
        "data-02",
        "data-05",
        "data-06",
        "data-07",
    )
    assert result.batch.actual_branch_tokens == {"a": 24, "b": 24}
    assert len(dict((m.data_id, m.branch) for m in result.batch.members)) == 6
    assert trace(result, "mix").details["overlaps"]
    exposed = trace(result, "final_batch").details["semantic_skill_exposure"]["skills"]
    assert exposed["a"]["sample_count"] == exposed["b"]["sample_count"] == 3
    assert exposed["a"]["known_evidence_supervised_tokens"] == 0
    impossible = execute_program(
        p, context(rows[:5], target=48, tolerance=0, share_tolerance=0)
    )
    assert (
        impossible.status == "infeasible"
        and impossible.reason == "BRANCH_QUOTA_SHORTFALL"
    )


def test_budget_pack_enforces_batch_tokens_and_never_truncates_trajectories():
    rows = records(7)
    before = tuple(r.tokenization for r in rows)
    result = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        target=48,
        tolerance=0,
        batch=2,
    )
    assert result.batch.supervised_tokens == 48 and len(result.batch.members) == 6
    assert tuple(r.tokenization for r in rows) == before
    bad = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        target=50,
        tolerance=0,
        batch=2,
    )
    assert bad.status == "infeasible" and bad.batch is None
    assert json.loads(bad.diagnostics_json)["global_infeasibility_proven"] is False
    with pytest.raises(ContractError):
        BudgetPolicy(10, 1, 1)
    with pytest.raises(ContractError):
        BudgetPolicy(10, 0, True)


def test_training_receipt_detects_changed_rows_but_allows_feature_enrichment():
    rows = records(4)
    result = single(rows, "LengthSelect", {"supervised_token_range": [1, None]})
    manifest = result.batch
    mutated = json.loads(canonical_json(manifest.to_dict()))
    mutated["supervised_tokens"] += 1
    with pytest.raises(ContractError):
        parse_batch_manifest(mutated)
    from self_curate.data import attach_gradients
    from test_m1_review import feature

    enriched = [attach_gradients(row, [feature(row, m=MODEL)]) for row in rows]
    assert verify_batch(manifest, MetricView(enriched)) == manifest.training_order
    changed = list(rows)
    ids = list(changed[0].tokenization.input_ids)
    ids[0] += 1
    changed[0] = replace(
        changed[0],
        signals=(),
        tokenization=replace(changed[0].tokenization, input_ids=tuple(ids)),
    )
    with pytest.raises(ContractError):
        verify_batch(manifest, MetricView(changed))


def test_skill_exposure_keeps_overlapping_labels_and_missing_evidence_distinct():
    from self_curate.contracts import SkillAnnotation
    from self_curate.data import attach_annotations

    rows = list(records(2))
    rows[0] = attach_annotations(
        rows[0],
        replace(
            rows[0].annotations,
            skills=(
                SkillAnnotation("a", (), "fixture"),
                SkillAnnotation("b", None, "fixture"),
            ),
        ),
    )
    rows[1] = replace(rows[1], annotations=None)
    result = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        target=16,
        tolerance=0,
    )
    exposure = trace(result, "final_batch").details["semantic_skill_exposure"]
    assert exposure["missing_label_samples"] == 1
    assert exposure["skills"]["a"]["known_evidence_supervised_tokens"] == 0
    assert exposure["skills"]["b"]["known_evidence_supervised_tokens"] is None
    assert exposure["skills"]["b"]["evidence_missing_samples"] == 1
    assert (
        sum(s["trajectory_supervised_tokens"] for s in exposure["skills"].values())
        == 16
    )
    assert result.batch.supervised_tokens == 16


def test_variable_length_packing_skips_whole_rows_and_admits_greedy_limitations():
    from helpers_m2 import raw_rows
    from self_curate.data import FixtureTokenizer, normalize_trajectory

    def rows_with_lengths(lengths):
        raw = raw_rows(len(lengths))
        for row, length in zip(raw, lengths):
            row.pop("signals")
            row.pop("annotations")
            row["messages"][1]["content"] = "x" * length
        return tuple(
            normalize_trajectory(row, FixtureTokenizer(), ("recovery", "regression"))
            for row in raw
        )

    rows = rows_with_lengths([13, 7, 5, 3])
    before = tuple(r.tokenization for r in rows)
    result = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        target=10,
        tolerance=0,
        batch=2,
        signals=(),
    )
    assert result.batch.data_ids == ("data-01", "data-03")
    assert result.batch.supervised_tokens == 10
    assert tuple(r.tokenization for r in rows) == before
    # A feasible combination exists (6+4), but the frozen ordered greedy picks 9.
    rows = rows_with_lengths([9, 6, 4])
    result = single(
        rows,
        "LengthSelect",
        {"supervised_token_range": [1, None]},
        target=10,
        tolerance=0,
        signals=(),
    )
    assert result.status == "infeasible" and result.batch is None
    assert trace(result, "final_batch").details["global_infeasibility_proven"] is False
