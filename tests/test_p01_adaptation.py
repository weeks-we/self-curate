"""P0/P1 compatibility at trust, version, cost and downstream feature boundaries."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from helpers_m2 import MODEL, context, fixture_program, node, program, records
from helpers_p0 import ar, fixture_bundle, mock_gradient_spec
from self_curate import contracts as c
from self_curate.cli import main
from self_curate.data import MetricView, attach_gradients, make_gradient, read_pool
from self_curate.data.pool import build_pool, file_sha256, load_config, write_json
from self_curate.operators import (
    prepare_selection,
    prepare_to_directory,
    verify_preparation,
)

ROOT = Path(__file__).resolve().parents[1]


def objective():
    return fixture_bundle().objective


def ready(*, attempt="attempt-1", invocation="select-1", ctx=None):
    return prepare_selection(
        fixture_program(),
        ctx or context(records(), target=64, tolerance=0, batch=8, share_tolerance=0),
        objective(),
        backend="mock",
        attempt_id=attempt,
        invocation_id=invocation,
    )


def hashes(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
    }


def published(tmp_path):
    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    out = tmp_path / "prepared"
    result = prepare_to_directory(
        pool,
        ROOT / "configs/programs/m2.fixture.json",
        ROOT / "configs/m2.fixture.json",
        ROOT / "configs/p1.objective.fixture.json",
        out,
        backend="mock",
        attempt_id="attempt-1",
        invocation_id="select-1",
    )
    return pool, out, result


def test_external_anchors_reject_self_consistent_but_changed_policy():
    original = fixture_bundle()
    changed = replace(
        original.objective,
        weights=(c.SkillWeight("recovery", 0.5), c.SkillWeight("regression", 0.5)),
    )
    evaluation = replace(
        original.evaluations[0], objective_hash=changed.objective_hash, total_value=0.15
    )
    round_record = replace(original.round_record, objective_hash=changed.objective_hash)
    modified = replace(
        original,
        objective=changed,
        evaluations=(evaluation,),
        round_record=round_record,
    )
    assert c.validate_workflow(modified)["expected_objective_verified"] is False
    with pytest.raises(c.ContractError, match="fixed system objective"):
        c.validate_workflow(modified, expected_objective=original.objective)
    with pytest.raises(c.ContractError, match="system model"):
        c.validate_workflow(
            original,
            expected_model=replace(MODEL, weights_hash=c.digest("new-checkpoint")),
        )
    with pytest.raises(c.ContractError, match="trusted selection context"):
        c.validate_workflow(
            original, expected_selection_context_hash=c.digest("another-context")
        )
    verified = c.validate_workflow(
        original,
        expected_model=MODEL,
        expected_objective=original.objective,
        expected_selection_context_hash=original.selection_context_hash,
    )
    assert (
        verified["expected_model_verified"]
        and verified["expected_objective_verified"]
        and verified["selection_context_verified"]
    )


def test_cached_cost_index_preserves_serialized_identity_and_avoids_unselected_scans():
    counts = Counter()

    class CountedEvent(c.CostEvent):
        def __getattribute__(self, name):
            if name == "event_id":
                counts[object.__getattribute__(self, "subject_id")] += 1
            return super().__getattribute__(name)

    events = tuple(
        CountedEvent(
            f"e{i}",
            "a",
            "selection",
            str(i),
            None,
            "completed",
            c.ResourceUsage(c.ComputeCost(0, 0, 0)),
            ar(str(i)),
            "mock",
        )
        for i in range(500)
    )
    ledger = c.CostLedger("cached", events, "mock")
    counts.clear()
    ref = ledger.reference(("e1",))
    assert sum(counts.values()) == 1 and counts["1"] == 1
    assert "_events_by_id" not in ledger.to_dict()
    restored = c.parse_contract(
        "CostLedger", json.loads(c.canonical_json(ledger.to_dict()))
    )
    assert (
        c.canonical_json(restored.to_dict()) == c.canonical_json(ledger.to_dict())
        and restored.reference(("e1",)) == ref
    )
    assert restored.record_hash == ledger.record_hash
    with pytest.raises(c.ContractError):
        c.parse_contract("CostLedger", {**ledger.to_dict(), "_events_by_id": {}})
    with pytest.raises(TypeError):
        ledger._events_by_id["unexpected"] = events[0]
    # The old published P0 fixture retains its hash and remains readable.
    assert c.load_workflow(ROOT / "configs/p0.fixture.json") == fixture_bundle()


def test_model_operation_cost_cannot_drop_model_identity():
    event = fixture_bundle().ledger.events[0]
    with pytest.raises(c.ContractError, match="identify its model"):
        replace(event, model_hash=None)
    assert replace(event, operation="selection", model_hash=None).model_hash is None


@pytest.mark.parametrize(
    "decision,m1_label",
    [
        ("accept", "accepted"),
        ("reject", "rejected"),
        ("needs_more_evidence", "pending"),
    ],
)
def test_round_training_label_adapter_preserves_usage_identity(decision, m1_label):
    from self_curate.data import record_candidate_training, resolve_training

    b = fixture_bundle(decision)
    r = b.round_record
    assert r.training_use_decision == m1_label
    record = records()[0]
    content = record.content_hash
    receipt = r.training.receipt
    use = c.TrainingUse(r.attempt_id, r.starting_model.model_hash, receipt, "pending")
    started = record_candidate_training(record, use)
    if m1_label != "pending":
        finished = resolve_training(started, r.attempt_id, r.training_use_decision)
    else:
        finished = record_candidate_training(
            started, replace(use, decision=r.training_use_decision)
        )
    saved = finished.usage.candidate_sft[0]
    assert (
        saved.model_id == r.starting_model.model_hash
        and saved.receipt == receipt
        and saved.decision == m1_label
    )
    assert finished.content_hash == content and record.usage.candidate_sft == ()
    assert bool(finished.usage.accepted_sft) == (m1_label == "accepted")
    assert "training_use_decision" not in r.to_dict()


def test_real_workflow_rejects_case_obscured_mock_artifact():
    raw = json.loads(c.canonical_json(fixture_bundle().to_dict()))

    def rewrite(value):
        if isinstance(value, dict):
            if value.get("backend") == "mock":
                value["backend"] = "real"
            if "uri" in value:
                value["uri"] = "memory://declared-test-artifact"
            for child in value.values():
                rewrite(child)
        elif isinstance(value, list):
            for child in value:
                rewrite(child)

    rewrite(raw)
    raw["objective"]["estimator"]["uri"] = "  FiXtUrE://must-not-pass"
    with pytest.raises(c.ContractError, match="Mock artifact"):
        c.parse_workflow_bundle(raw)


def test_preparation_uses_actual_selection_and_leaves_downstream_work_unexecuted(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("P1 cannot compute gradients")

    monkeypatch.setattr("self_curate.data.features.make_gradient", forbidden)
    # Fixture construction declares its own mock features through a separate API;
    # use the already frozen objective instead of constructing the P0 fixture here.
    goal = c.parse_contract(
        "ObjectiveSpec",
        json.loads((ROOT / "configs/p1.objective.fixture.json").read_text()),
    )
    rows = records()
    before = tuple(row.record_hash for row in rows)
    ctx = context(rows, target=64, tolerance=0, batch=8, share_tolerance=0)
    result = prepare_selection(
        fixture_program(), ctx, goal, backend="mock", attempt_id="a", invocation_id="i"
    )
    assert (
        result.status == "feasible" and result.workflow.batch == result.selection.batch
    )
    assert (
        len(result.workflow.batch.members) == 8
        and result.workflow.batch.supervised_tokens == 64
    )
    assert result.workflow.batch.actual_branch_tokens == {"a": 32, "b": 32}
    assert (
        result.workflow.sample_gradients
        == result.workflow.target_gradients
        == result.workflow.evaluations
        == ()
    )
    assert (
        result.workflow.rollouts == result.workflow.capabilities == ()
        and result.workflow.round_record is None
    )
    assert result.validate_workflow()["selection_context_verified"] is True
    assert tuple(row.record_hash for row in rows) == before
    event = result.ledger.events[0]
    assert event.attempt_id == "a" and event.status == "completed"
    assert event.usage.compute.gpu_seconds == event.usage.compute.processed_tokens == 0
    assert (
        event.usage.new_gradient_tokens == event.usage.training_supervised_tokens == 0
    )
    assert event.usage.compute.wall_seconds >= 0


def test_invocations_separate_physical_calls_but_preserve_exact_batch():
    first = ready(invocation="first")
    second = ready(invocation="retry")
    assert (
        first.selection == second.selection
        and first.workflow.batch == second.workflow.batch
    )
    assert first.ledger.events[0].event_id != second.ledger.events[0].event_id
    merged = c.CostLedger("both", first.ledger.events + second.ledger.events, "mock")
    assert len(merged.events) == 2
    with pytest.raises(c.ContractError):
        c.CostLedger("duplicate", first.ledger.events + first.ledger.events, "mock")


def test_handoff_accepts_later_declared_gradient_valuation_and_training_records_without_resampling():
    prepared = ready(attempt="mock-attempt-1", invocation="new-selection")
    downstream = fixture_bundle()
    # Replace the fixture's declared selection event with the actual CPU receipt.
    events = (
        tuple(e for e in downstream.ledger.events if e.event_id != "cost-selection")
        + prepared.ledger.events
    )
    ledger = c.CostLedger("downstream-ledger", events, "mock")
    evaluation = downstream.evaluations[0]
    evaluation = replace(
        evaluation,
        cost=ledger.reference(
            tuple(i for i in evaluation.cost.event_ids if i != "cost-selection")
            + (prepared.ledger.events[0].event_id,)
        ),
    )
    round_record = replace(
        downstream.round_record,
        cost=ledger.reference(tuple(e.event_id for e in events)),
    )
    complete = replace(
        downstream,
        bundle_id=prepared.workflow.bundle_id,
        ledger=ledger,
        evaluations=(evaluation,),
        round_record=round_record,
    )
    receipt = c.validate_workflow(
        complete,
        prepared.context.view,
        expected_model=MODEL,
        expected_objective=prepared.objective,
        expected_selection_context_hash=prepared.context.context_hash,
    )
    assert receipt["status"] == "verified"
    assert complete.batch == prepared.workflow.batch
    assert (
        complete.round_record.training.training_data_hash
        == prepared.workflow.batch.training_data_hash
    )
    assert complete.batch.training_order == prepared.workflow.batch.training_order


def test_new_checkpoint_never_substitutes_previous_scores():
    from helpers_m2 import SCORE
    from self_curate.operators import SignalContext

    next_model = replace(MODEL, weights_hash=c.digest("accepted-next-model"))
    ctx = context(records(), model=next_model, signals=(SignalContext(SCORE),))
    recipe = program(
        [
            node(
                "score",
                "NLLSelect",
                {"interval": [0, 100], "missing_policy": "request"},
            )
        ]
    )
    failed = prepare_selection(
        recipe,
        ctx,
        objective(),
        backend="mock",
        attempt_id="next-round",
        invocation_id="score",
    )
    assert failed.status == "failed"
    request = failed.error["details"]["requests"][0]
    assert request["model_ref"]["weights_hash"] == next_model.weights_hash
    assert (
        len(request["data_ids"]) == 12
        and failed.ledger.events[0].model_hash == next_model.model_hash
    )


def test_infeasible_and_missing_score_outcomes_preserve_cost_without_fake_batch():
    infeasible = ready(
        ctx=context(records(), target=65, tolerance=0, batch=8, share_tolerance=0)
    )
    assert infeasible.status == "infeasible" and infeasible.workflow is None
    assert (
        infeasible.ledger.events[0].status == "completed" and infeasible.error is None
    )
    with pytest.raises(c.ContractError):
        infeasible.validate_workflow()
    rows = tuple(replace(row, signals=()) for row in records())
    recipe = program(
        [
            node(
                "score",
                "NLLSelect",
                {"interval": [0, 100], "missing_policy": "request"},
            )
        ]
    )
    failed = prepare_selection(
        recipe,
        context(rows),
        objective(),
        backend="mock",
        attempt_id="a",
        invocation_id="missing",
    )
    assert (
        failed.status == "failed"
        and failed.workflow is None
        and failed.selection is None
    )
    assert (
        failed.error["code"] == "MISSING_SIGNAL"
        and len(failed.error["details"]["requests"][0]["data_ids"]) == 12
    )
    assert failed.error["details"]["completed_traces"]
    assert (
        failed.ledger.events[0].status == "failed"
        and failed.ledger.events[0].usage.scoring_tokens == 0
    )


def test_preparation_requires_model_and_explicit_fixture_domain():
    with pytest.raises(c.ContractError, match="Fixture"):
        prepare_selection(
            fixture_program(),
            context(records()),
            objective(),
            backend="real",
            attempt_id="a",
            invocation_id="i",
        )
    with pytest.raises(c.ContractError, match="ModelRef"):
        prepare_selection(
            fixture_program(),
            context(records(), model=None, signals=()),
            objective(),
            backend="mock",
            attempt_id="a",
            invocation_id="i",
        )


def test_publish_read_replay_and_no_overwrite_leave_pool_unchanged(tmp_path):
    pool, out, result = published(tmp_path)
    before = hashes(pool)
    receipt = hashes(out)
    verified = verify_preparation(
        out,
        read_pool(pool).metric_view(),
        expected_model=MODEL,
        expected_objective=result.objective,
    )
    assert verified["replayed"] and verified["selection_status"] == "feasible"
    assert (
        verified["expected_model_verified"] and verified["expected_objective_verified"]
    )
    restored = c.load_workflow(
        out / "workflow.json",
        view=read_pool(pool).metric_view(),
        expected_model=MODEL,
        expected_objective=result.objective,
    )
    assert restored == result.workflow
    with pytest.raises(c.ContractError) as exc:
        prepare_to_directory(
            pool,
            ROOT / "configs/programs/m2.fixture.json",
            ROOT / "configs/m2.fixture.json",
            ROOT / "configs/p1.objective.fixture.json",
            out,
            backend="mock",
            attempt_id="a",
            invocation_id="i",
        )
    assert exc.value.code == c.ErrorCode.OUTPUT_EXISTS
    assert hashes(pool) == before and hashes(out) == receipt


@pytest.mark.parametrize("mode,exit_code", [("infeasible", 3), ("failed", 2)])
def test_nonfeasible_publications_keep_ledger_and_never_publish_workflow(
    mode, exit_code, tmp_path, capsys
):
    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    recipe = fixture_program()
    ctx = context(
        read_pool(pool).records,
        target=65 if mode == "infeasible" else 64,
        tolerance=0,
        batch=8,
        share_tolerance=0,
    )
    if mode == "failed":
        next_model = replace(MODEL, weights_hash=c.digest("other-active-model"))
        from helpers_m2 import SCORE
        from self_curate.operators import SignalContext

        ctx = replace(ctx, model_ref=next_model, signals=(SignalContext(SCORE),))
        recipe = program(
            [
                node(
                    "score",
                    "NLLSelect",
                    {"interval": [0, 100], "missing_policy": "request"},
                )
            ]
        )
    write_json(tmp_path / "program.json", recipe.to_dict())
    write_json(tmp_path / "context.json", ctx.to_dict())
    out = tmp_path / "nonfeasible"
    args = [
        "prepare-selection",
        "--pool",
        str(pool),
        "--program",
        str(tmp_path / "program.json"),
        "--context",
        str(tmp_path / "context.json"),
        "--objective",
        str(ROOT / "configs/p1.objective.fixture.json"),
        "--backend",
        "mock",
        "--attempt-id",
        "a",
        "--invocation-id",
        mode,
        "--output",
        str(out),
    ]
    assert main(args) == exit_code
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["status"] == mode and metadata["batch_id"] is None
    assert (out / "ledger.json").exists() and (out / "trace.json").exists()
    assert not (out / "workflow.json").exists() and not (out / "batch.json").exists()
    verified = verify_preparation(
        out,
        read_pool(pool).metric_view(),
        expected_model=ctx.model_ref,
        expected_objective=objective(),
    )
    assert verified["selection_status"] == mode


@pytest.mark.parametrize(
    "case",
    [
        "missing_file",
        "unlisted_file",
        "bad_checksum",
        "changed_outcome_resealed",
        "changed_trace_resealed",
        "changed_invocation_resealed",
    ],
)
def test_partial_or_resealed_preparation_cannot_fabricate_replay(case, tmp_path):
    pool, out, _ = published(tmp_path)
    damaged = tmp_path / "damaged"
    shutil.copytree(out, damaged)
    if case == "missing_file":
        (damaged / "workflow.json").rename(damaged / "not-workflow.json")
    elif case == "unlisted_file":
        (damaged / "extra.json").write_text("{}")
    elif case == "bad_checksum":
        (damaged / "outcome.json").write_text("{}")
    else:
        if case == "changed_outcome_resealed":
            path = damaged / "outcome.json"
            raw = json.loads(path.read_text())
            raw["selection"]["batch"]["training_seed"] += 1
        elif case == "changed_trace_resealed":
            path = damaged / "trace.json"
            raw = json.loads(path.read_text())
            raw[0]["output_ids"] = []
        else:
            path = damaged / "prepared.json"
            raw = json.loads(path.read_text())
            raw["invocation_id"] = "replaced-call"
        write_json(path, raw)
        sums = json.loads((damaged / "checksums.json").read_text())
        sums[path.name] = file_sha256(path)
        write_json(damaged / "checksums.json", sums)
    with pytest.raises(c.ContractError):
        verify_preparation(damaged, read_pool(pool).metric_view())


def test_appending_sample_features_preserves_workflow_but_requires_new_selection_snapshot(
    tmp_path,
):
    pool, out, result = published(tmp_path)
    original = read_pool(pool)
    updated = []
    for row in original.records:
        if row.data_id in result.workflow.batch.data_ids:
            gradient = make_gradient(
                row,
                MODEL,
                mock_gradient_spec(),
                vector=ar("mock-next-feature-" + row.data_id),
                dimension=8,
                dtype="float32",
                norm=1,
                provenance=(ar("mock-feature"),),
            )
            row = attach_gradients(row, (gradient,))
        updated.append(row)
    view = MetricView(updated)
    assert (
        c.load_workflow(
            out / "workflow.json",
            view=view,
            expected_model=MODEL,
            expected_objective=result.objective,
        ).batch
        == result.workflow.batch
    )
    with pytest.raises(c.ContractError, match="snapshot"):
        verify_preparation(out, view)


def test_cli_preparation_and_trusted_workflow_validation(tmp_path, capsys):
    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    out = tmp_path / "cli-prepared"
    args = [
        "prepare-selection",
        "--pool",
        str(pool),
        "--program",
        str(ROOT / "configs/programs/m2.fixture.json"),
        "--context",
        str(ROOT / "configs/m2.fixture.json"),
        "--objective",
        str(ROOT / "configs/p1.objective.fixture.json"),
        "--backend",
        "mock",
        "--attempt-id",
        "a",
        "--invocation-id",
        "i",
        "--output",
        str(out),
    ]
    assert (
        main(args) == 0 and json.loads(capsys.readouterr().out)["status"] == "feasible"
    )
    assert (
        main(
            [
                "inspect-preparation",
                "--prepared",
                str(out),
                "--pool",
                str(pool),
                "--expected-model",
                str(ROOT / "configs/p1.model.fixture.json"),
                "--expected-objective",
                str(ROOT / "configs/p1.objective.fixture.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["expected_objective_verified"]
    assert (
        main(
            [
                "validate-workflow",
                "--workflow",
                str(out / "workflow.json"),
                "--pool",
                str(pool),
                "--expected-model",
                str(ROOT / "configs/p1.model.fixture.json"),
                "--expected-objective",
                str(ROOT / "configs/p1.objective.fixture.json"),
                "--expected-context",
                str(ROOT / "configs/m2.fixture.json"),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert (
        receipt["training_inputs_verified"]
        and receipt["expected_model_verified"]
        and receipt["selection_context_verified"]
    )
    assert (
        main(
            [
                "validate-workflow",
                "--workflow",
                str(out / "workflow.json"),
                "--expected-context",
                str(ROOT / "configs/m2.fixture.json"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "INVALID_CONFIG"
