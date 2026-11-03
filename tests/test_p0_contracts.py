"""P0 interface closure using declared mock values, not an executed P2 loop."""

import copy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from helpers_p0 import ar, fixture_bundle
from helpers_m2 import MODEL, context, records
from self_curate import contracts as c
from self_curate.cli import main
from self_curate.contracts.workflow_parsing import CONTRACT_TYPES
from self_curate.contracts.workflow_validation import _walk


@pytest.fixture
def bundle():
    return fixture_bundle()


def document(bundle):
    return json.loads(c.canonical_json(bundle.to_dict()))


def put(raw, path, value):
    at = raw
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = value


def test_all_twelve_core_interfaces_and_full_chain_roundtrip(bundle):
    expected = "ModelRef TrajectoryRecord TaskRecord SignalRecord ProgramSpec BatchManifest GradientSpec GradientFeature RolloutRecord EvaluationReport CapabilityReport RoundRecord".split()
    assert all(hasattr(c, name) for name in expected)
    restored = c.parse_workflow_bundle(document(bundle))
    assert restored == bundle
    view = context(records()).view
    receipt = c.validate_workflow(restored, view)
    assert receipt["backend"] == "mock" and receipt["round_decision"] == "accept"
    assert (
        receipt["training_inputs_verified"] is True
        and receipt["artifact_contents_verified"] is False
    )
    assert receipt["batch_id"] == bundle.batch.batch_id
    assert (
        restored.round_record.training.training_data_hash
        == bundle.batch.training_data_hash
    )
    assert len(restored.tasks) == 3 and len(restored.rollouts) == 8
    with pytest.raises(FrozenInstanceError):
        restored.backend = "real"
    assert c.canonical_json(restored.to_dict()) == c.canonical_json(bundle.to_dict())


@pytest.mark.parametrize("kind", tuple(CONTRACT_TYPES))
def test_every_new_contract_has_strict_serializable_versioned_boundary(kind, bundle):
    records_by_type = {
        type(r).__name__: r for r in _walk(bundle) if type(r).__name__ in CONTRACT_TYPES
    }
    item = records_by_type[kind]
    raw = json.loads(c.canonical_json(item.to_dict()))
    assert c.parse_contract(kind, raw) == item
    assert raw["schema_version"] == "1"
    with pytest.raises(c.ContractError):
        c.parse_contract(kind, {**raw, "unrecognized_field": 1})
    no_version = copy.deepcopy(raw)
    no_version.pop("schema_version")
    with pytest.raises(c.ContractError):
        c.parse_contract(kind, no_version)
    with pytest.raises(c.ContractError):
        c.parse_contract(kind, {**raw, "schema_version": "future"})


@pytest.mark.parametrize(
    "path,value",
    [
        (("program", "seed"), 99),
        (("selection_context_hash",), c.digest("other-context")),
        (("tasks", 0, "version"), "changed"),
        (("rollouts", 0, "task_hash"), c.digest("other-task")),
        (("rollouts", 0, "scorer", "sha256"), c.digest("other-scorer")),
        (("rollouts", 0, "model_ref", "weights_hash"), c.digest("other-model")),
        (("rollouts", 0, "generated_tokens"), 33),
        (("target_gradients", 0, "rollouts_hash"), c.digest("other-rollouts")),
        (("sample_gradients", 0, "binding", "token_count"), 9),
        (
            ("sample_gradients", 0, "binding", "tokenization_hash"),
            c.digest("other-mask"),
        ),
        (("evaluations", 0, "batch_id"), c.digest("other-batch")),
        (("evaluations", 0, "objective_hash"), c.digest("other-objective")),
        (("evaluations", 0, "sample_cache_keys", 0), c.digest("missing-cache-key")),
        (("evaluations", 0, "total_value"), 99.0),
        (("evaluations", 0, "constraints", 0, "passed"), False),
        (
            ("round_record", "training", "training_data_hash"),
            c.digest("resampled-training-order"),
        ),
        (("round_record", "accepted_version_after"), 2),
        (("round_record", "decision_policy_hash"), c.digest("changed-policy")),
        (("ledger", "events", 0, "usage", "compute", "wall_seconds"), 100.0),
        (("round_record", "before_capability_id"), "unknown-report"),
        (("capabilities", 0, "metrics", 0, "sample_count"), 2),
    ],
)
def test_cross_module_mismatch_fails_instead_of_silently_rebinding(path, value, bundle):
    raw = document(bundle)
    put(raw, path, value)
    with pytest.raises(c.ContractError):
        c.parse_workflow_bundle(raw)


def test_accept_reject_and_evidence_pending_records_preserve_correct_model_identity():
    accepted = fixture_bundle("accept")
    rejected = fixture_bundle("reject")
    more = fixture_bundle("needs_more_evidence")
    assert (
        accepted.round_record.accepted_model
        == accepted.round_record.training.candidate_model
    )
    assert accepted.round_record.accepted_version_after == 1
    for b in (rejected, more):
        assert c.parse_workflow_bundle(document(b)) == b
        assert (
            b.round_record.accepted_model == MODEL
            and b.round_record.accepted_version_after == 0
        )
        assert b.round_record.training.checkpoint is not None
        assert b.round_record.cost.event_ids == tuple(
            e.event_id for e in b.ledger.events
        )
    raw = document(accepted)
    raw["round_record"]["decision"] = "reject"
    with pytest.raises(c.ContractError, match="Non-accepting"):
        c.parse_workflow_bundle(raw)


def test_pending_records_keep_unknown_outputs_missing(bundle):
    ledger = c.CostLedger("not-executed", (), "mock")
    pending_target = replace(
        bundle.target_gradients[0],
        rollout_ids=(),
        rollouts_hash=c.rollout_set_hash(()),
        vector=None,
        norm=None,
        validity="pending",
        reason="backend not executed",
        cost=ledger.reference(()),
    )
    pending_round = c.RoundRecord(
        "planned-round",
        "planned-attempt",
        MODEL,
        MODEL,
        0,
        0,
        bundle.objective.objective_hash,
        bundle.objective.acceptance_policy.sha256,
        None,
        None,
        None,
        None,
        None,
        "pending",
        None,
        None,
        ledger.reference(()),
        "mock",
    )
    pending = replace(
        bundle,
        rollouts=(),
        target_gradients=(pending_target,),
        sample_gradients=(),
        evaluations=(),
        capabilities=(),
        ledger=ledger,
        round_record=pending_round,
    )
    restored = c.parse_workflow_bundle(document(pending))
    assert (
        restored.target_gradients[0].norm is None
        and restored.round_record.training is None
    )
    pending_evaluation = replace(
        bundle.evaluations[0],
        targets=(),
        sample_cache_keys=(),
        coverage=c.GradientCoverage(bundle.batch.data_ids, (), bundle.batch.data_ids),
        skill_values=(),
        total_value=None,
        constraints=(),
        status="pending",
        reason="backends not executed",
        cost=ledger.reference(()),
    )
    assert (
        c.validate_workflow(replace(pending, evaluations=(pending_evaluation,)))[
            "evaluations"
        ]
        == 1
    )
    rejected = replace(
        pending_round,
        decision="reject",
        reason="no feasible batch",
        decision_evidence=ar("mock-no-feasible-batch"),
    )
    assert (
        c.validate_workflow(replace(pending, round_record=rejected))["round_decision"]
        == "reject"
    )
    with pytest.raises(c.ContractError):
        replace(pending_target, norm=0)


def test_infrastructure_failure_is_not_a_zero_reward(bundle):
    rollout = bundle.rollouts[0]
    with pytest.raises(c.ContractError, match="Infrastructure"):
        replace(rollout, status="environment_failed", reward=0)
    failed = replace(
        rollout,
        status="environment_failed",
        reward=None,
        stop_reason="mock tool unavailable",
    )
    assert c.parse_rollout_record(failed.to_dict()).reward is None
    scored_failure = replace(rollout, status="task_failed", reward=0)
    assert c.parse_rollout_record(scored_failure.to_dict()).reward == 0
    with pytest.raises(c.ContractError):
        replace(rollout, reward=float("nan"))


def test_zero_target_and_incomplete_gradients_cannot_be_ranked(bundle):
    target = replace(
        bundle.target_gradients[0], norm=0, validity="zero", reason="zero target norm"
    )
    references = (
        c.TargetRef(target.spec.skill, target.target_version),
        bundle.evaluations[0].targets[1],
    )
    evaluation = replace(bundle.evaluations[0], targets=references)
    with pytest.raises(c.ContractError) as exc:
        replace(
            bundle,
            target_gradients=(target, bundle.target_gradients[1]),
            evaluations=(evaluation,),
        )
    assert exc.value.code == c.ErrorCode.ZERO_TARGET_GRADIENT
    coverage = c.GradientCoverage(
        bundle.batch.data_ids, bundle.batch.data_ids[:-1], bundle.batch.data_ids[-1:]
    )
    with pytest.raises(c.ContractError):
        replace(bundle.evaluations[0], coverage=coverage)
    incomplete = replace(
        bundle.evaluations[0],
        status="invalid",
        reason="missing gradient",
        coverage=coverage,
        sample_cache_keys=bundle.evaluations[0].sample_cache_keys[:-1],
        total_value=None,
    )
    assert not c.parse_evaluation_report(incomplete.to_dict()).feasible
    with pytest.raises(c.ContractError, match="feasible"):
        replace(bundle, evaluations=(incomplete,))


def test_sample_and_target_coordinates_are_checked_without_loading_vectors(bundle):
    target = replace(
        bundle.target_gradients[0],
        spec=replace(
            bundle.target_gradients[0].spec,
            gradient=replace(
                bundle.target_gradients[0].spec.gradient,
                parameter_coordinates_hash=c.digest("other-coordinates"),
            ),
        ),
    )
    evaluation = replace(
        bundle.evaluations[0],
        targets=(
            c.TargetRef(target.spec.skill, target.target_version),
            bundle.evaluations[0].targets[1],
        ),
    )
    with pytest.raises(c.ContractError, match="coordinate"):
        replace(
            bundle,
            target_gradients=(target, bundle.target_gradients[1]),
            evaluations=(evaluation,),
        )


def test_mixed_sample_loss_and_changed_selection_mask_are_rejected(bundle):
    sample = replace(
        bundle.sample_gradients[0],
        spec=replace(bundle.sample_gradients[0].spec, reduction="token_sum"),
    )
    evaluation = replace(
        bundle.evaluations[0],
        sample_cache_keys=(
            sample.cache_key,
            *bundle.evaluations[0].sample_cache_keys[1:],
        ),
    )
    with pytest.raises(c.ContractError, match="loss/reduction"):
        replace(
            bundle,
            sample_gradients=(sample, *bundle.sample_gradients[1:]),
            evaluations=(evaluation,),
        )
    sample = replace(
        bundle.sample_gradients[0],
        binding=replace(
            bundle.sample_gradients[0].binding,
            selection_hash=c.digest("other-selection"),
        ),
    )
    evaluation = replace(
        bundle.evaluations[0],
        sample_cache_keys=(
            sample.cache_key,
            *bundle.evaluations[0].sample_cache_keys[1:],
        ),
    )
    declared = replace(
        bundle,
        sample_gradients=(sample, *bundle.sample_gradients[1:]),
        evaluations=(evaluation,),
    )
    assert c.validate_workflow(declared)["training_inputs_verified"] is False
    with pytest.raises(c.ContractError, match="mask selection"):
        c.validate_workflow(declared, context(records()).view)


def test_capability_metric_order_is_joined_by_skill_name(bundle):
    after = replace(
        bundle.capabilities[1], metrics=tuple(reversed(bundle.capabilities[1].metrics))
    )
    assert (
        c.validate_workflow(
            replace(bundle, capabilities=(bundle.capabilities[0], after))
        )["status"]
        == "verified"
    )


def test_task_mean_and_rollout_mean_cannot_be_confused(bundle):
    before, after = bundle.capabilities
    assert before.metrics[0].sample_count == 3 and len(before.metrics[0].task_ids) == 2
    assert before.metrics[0].mean_reward == 0.5
    wrong = replace(
        before,
        metrics=(
            replace(before.metrics[0], aggregation="rollout_mean"),
            before.metrics[1],
        ),
    )
    with pytest.raises(c.ContractError, match="aggregation"):
        replace(bundle, capabilities=(wrong, after))
    corrected = replace(
        before,
        metrics=(
            replace(before.metrics[0], aggregation="rollout_mean", mean_reward=1 / 3),
            before.metrics[1],
        ),
    )
    after = replace(
        after,
        metrics=(
            replace(after.metrics[0], aggregation="rollout_mean"),
            after.metrics[1],
        ),
    )
    assert (
        c.validate_workflow(replace(bundle, capabilities=(corrected, after)))["status"]
        == "verified"
    )
    with pytest.raises(c.ContractError):
        replace(before.metrics[0], uncertainty=0.1)


def test_final_test_split_and_mock_real_mixing_are_rejected(bundle):
    with pytest.raises(c.ContractError, match="Final test"):
        replace(
            bundle, tasks=(replace(bundle.tasks[0], split="test"), *bundle.tasks[1:])
        )
    raw = document(bundle)
    raw["backend"] = "real"
    with pytest.raises(c.ContractError):
        c.parse_workflow_bundle(raw)
    standalone_test = replace(bundle.tasks[0], split="test")
    assert c.parse_task_record(standalone_test.to_dict()).split == "test"


def test_costs_preserve_failures_missing_measurements_and_content_bindings(bundle):
    first = replace(
        bundle.ledger.events[0],
        event_id="one",
        usage=c.ResourceUsage(c.ComputeCost(0.2, 0, 5)),
    )
    failed = replace(
        first,
        event_id="failed",
        status="failed",
        usage=c.ResourceUsage(c.ComputeCost(0.1, None, 2)),
    )
    ledger = c.CostLedger("cost-test", (first, failed), "mock")
    summary = ledger.summary()
    assert summary["wall_seconds"]["total"] == pytest.approx(0.3)
    assert summary["processed_tokens"]["total"] == 7
    assert summary["gpu_seconds"] == {
        "known_subtotal": 0.0,
        "missing_events": 1,
        "total": None,
    }
    assert summary["agent_input_tokens"]["total"] is None
    ref = ledger.reference(("one", "failed"))
    assert len(ref.event_ids) == 2
    with pytest.raises(c.ContractError):
        ledger.reference("one")
    with pytest.raises(c.ContractError):
        ledger.reference(("unknown",))
    with pytest.raises(c.ContractError):
        replace(ledger, events=(first, first))
    changed = replace(bundle.ledger.events[0], status="failed")
    with pytest.raises(c.ContractError, match="Cost reference"):
        replace(
            bundle,
            ledger=replace(bundle.ledger, events=(changed, *bundle.ledger.events[1:])),
        )
    shortened = bundle.ledger.reference(bundle.round_record.cost.event_ids[:-1])
    with pytest.raises(c.ContractError, match="all attempt cost"):
        replace(bundle, round_record=replace(bundle.round_record, cost=shortened))


def test_invalid_capability_cannot_authorize_acceptance(bundle):
    after = bundle.capabilities[1]
    invalid = replace(
        after,
        status="invalid",
        reason="insufficient execution evidence",
        metrics=tuple(replace(m, mean_reward=None) for m in after.metrics),
    )
    with pytest.raises(c.ContractError, match="valid paired"):
        replace(bundle, capabilities=(bundle.capabilities[0], invalid))
    more = replace(
        bundle.round_record,
        decision="needs_more_evidence",
        accepted_model=MODEL,
        accepted_version_after=0,
    )
    assert (
        c.validate_workflow(
            replace(
                bundle,
                round_record=more,
                capabilities=(bundle.capabilities[0], invalid),
            )
        )["round_decision"]
        == "needs_more_evidence"
    )


def test_cli_validates_mock_contract_chain_and_optional_pool(tmp_path, capsys):
    path = tmp_path / "workflow.json"
    path.write_text(c.canonical_json(fixture_bundle().to_dict()), encoding="utf-8")
    assert main(["validate-workflow", "--workflow", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "mock"
    raw = json.loads(path.read_text())
    raw["evaluations"][0]["batch_id"] = c.digest("wrong")
    path.write_text(c.canonical_json(raw), encoding="utf-8")
    assert main(["validate-workflow", "--workflow", str(path)]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "ARTIFACT_MISMATCH"
    with pytest.raises(c.ContractError):
        c.parse_contract('__import__("os")', {})
