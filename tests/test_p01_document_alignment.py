"""Regression checks derived from the blueprint's exact-manifest and validity rules."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from helpers_m2 import MODEL, context, records
from helpers_p0 import fixture_bundle
from self_curate import contracts as c
from self_curate.cli import main
from self_curate.data import read_pool
from self_curate.data.pool import build_pool, load_config, write_json
from self_curate.operators import prepare_to_directory, verify_preparation

ROOT = Path(__file__).resolve().parents[1]


def resealed_order_change(bundle):
    batch = replace(
        bundle.batch, training_order=tuple(reversed(bundle.batch.training_order))
    )
    evaluation = replace(bundle.evaluations[0], batch_id=batch.batch_id)
    training = replace(
        bundle.round_record.training,
        batch_id=batch.batch_id,
        training_data_hash=batch.training_data_hash,
    )
    round_record = replace(
        bundle.round_record, training=training, selected_batch_id=batch.batch_id
    )
    return replace(
        bundle, batch=batch, evaluations=(evaluation,), round_record=round_record
    )


def test_external_exact_manifest_anchor_rejects_fully_resealed_order_change(tmp_path):
    original = fixture_bundle()
    altered = resealed_order_change(original)
    view = context(records()).view
    anchors = dict(
        expected_model=MODEL,
        expected_objective=original.objective,
        expected_selection_context_hash=original.selection_context_hash,
    )
    # A coherent alternative manifest is still a different historical choice.
    assert (
        c.validate_workflow(altered, view, **anchors)["expected_batch_verified"]
        is False
    )
    assert altered.batch.data_ids == original.batch.data_ids
    assert altered.batch.training_data_hash != original.batch.training_data_hash
    with pytest.raises(c.ContractError, match="exact selected batch"):
        c.validate_workflow(
            altered, view, expected_batch_id=original.batch.batch_id, **anchors
        )
    assert c.validate_workflow(
        original, view, expected_batch_id=original.batch.batch_id, **anchors
    )["expected_batch_verified"]
    path = tmp_path / "resealed.json"
    write_json(path, altered.to_dict())
    with pytest.raises(c.ContractError, match="exact selected batch"):
        c.load_workflow(
            path, view=view, expected_batch_id=original.batch.batch_id, **anchors
        )
    with pytest.raises(c.ContractError):
        c.validate_workflow(original, expected_batch_id="display-name")


@pytest.mark.parametrize(
    "mutation",
    ["no_targets", "no_sample_refs", "partial_sample_refs", "missing_skill_ref"],
)
def test_standalone_valid_valuation_requires_its_declared_references(mutation):
    report = fixture_bundle().evaluations[0]
    raw = json.loads(c.canonical_json(report.to_dict()))
    if mutation == "no_targets":
        raw["targets"] = []
    elif mutation == "no_sample_refs":
        raw["sample_cache_keys"] = []
    elif mutation == "partial_sample_refs":
        raw["sample_cache_keys"].pop()
    else:
        raw["targets"].pop()
    with pytest.raises(c.ContractError):
        c.parse_evaluation_report(raw)


@pytest.mark.parametrize(
    "mutation", ["extra_task", "missing_task", "extra_rollout", "missing_rollout"]
)
def test_standalone_capability_declares_exact_metric_member_union(mutation):
    raw = json.loads(c.canonical_json(fixture_bundle().capabilities[0].to_dict()))
    if mutation == "extra_task":
        raw["metrics"][0]["task_ids"].append("undeclared-task")
    elif mutation == "missing_task":
        raw["task_ids"].append("not-covered-by-any-metric")
    elif mutation == "extra_rollout":
        raw["metrics"][0]["rollout_ids"].append("undeclared-run")
        raw["metrics"][0]["sample_count"] += 1
    else:
        raw["rollout_ids"].append("not-covered-by-any-metric")
    with pytest.raises(c.ContractError):
        c.parse_capability_report(raw)


def test_partial_valuation_preserves_missing_references_without_rankable_total():
    b = fixture_bundle()
    e = b.evaluations[0]
    pending = replace(
        e,
        status="pending",
        reason="not executed",
        targets=(),
        sample_cache_keys=(),
        coverage=c.GradientCoverage(b.batch.data_ids, (), b.batch.data_ids),
        skill_values=(),
        constraints=(),
        total_value=None,
    )
    decoded = c.parse_evaluation_report(json.loads(c.canonical_json(pending.to_dict())))
    assert decoded.total_value is None and not decoded.feasible
    wrong = decoded.to_dict()
    wrong["coverage"]["computed_ids"] = [b.batch.data_ids[0]]
    wrong["coverage"]["missing_ids"] = list(b.batch.data_ids[1:])
    with pytest.raises(c.ContractError):
        c.parse_evaluation_report(wrong)


def test_p1_and_cli_check_expected_best_manifest(tmp_path, capsys):
    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    out = tmp_path / "prepared"
    prepared = prepare_to_directory(
        pool,
        ROOT / "configs/programs/m2.fixture.json",
        ROOT / "configs/m2.fixture.json",
        ROOT / "configs/p1.objective.fixture.json",
        out,
        backend="mock",
        attempt_id="audit",
        invocation_id="selection",
    )
    assert prepared.validate_workflow()["expected_batch_verified"]
    view = read_pool(pool).metric_view()
    assert verify_preparation(
        out, view, expected_batch_id=prepared.workflow.batch.batch_id
    )["expected_batch_verified"]
    with pytest.raises(c.ContractError):
        verify_preparation(out, view, expected_batch_id="0" * 64)
    assert (
        main(
            [
                "validate-workflow",
                "--workflow",
                str(out / "workflow.json"),
                "--pool",
                str(pool),
                "--expected-batch",
                str(out / "batch.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["expected_batch_verified"]
    changed = replace(
        prepared.workflow.batch,
        training_order=tuple(reversed(prepared.workflow.batch.training_order)),
    )
    write_json(tmp_path / "wrong-batch.json", changed.to_dict())
    assert (
        main(
            [
                "validate-workflow",
                "--workflow",
                str(out / "workflow.json"),
                "--pool",
                str(pool),
                "--expected-batch",
                str(tmp_path / "wrong-batch.json"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "ARTIFACT_MISMATCH"
    assert (
        main(
            [
                "inspect-preparation",
                "--prepared",
                str(out),
                "--pool",
                str(pool),
                "--expected-batch",
                str(out / "batch.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["expected_batch_verified"]
