"""End-to-end inner search, externally supplied values, and exact-best recovery."""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from helpers_feedback import inputs
from self_curate import contracts as c
from self_curate.agents import RoleTeam
from self_curate.agents.demo import FixtureRoleClient
from self_curate.data.pool import file_sha256, write_json
from self_curate.operators import prepare_selection
from self_curate.search import InnerSearch, StopDecision, load_best


def setup(*, use_model=True, policy_step=0.125):
    request, _, ctx, policy = inputs()
    policy = replace(policy, share_step=policy_step)
    initial = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="inner-outer",
        invocation_id="seed",
    )
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="inner-session", backend="mock"
    )
    search = InnerSearch(
        team, initial, request, policy, search_id="inner-test", use_model=use_model
    )
    return search, client, request, ctx, policy


def value(request, amount, *, status="valid", result_id="fixture-value"):
    constraints = tuple(
        c.ConstraintResult(
            b.name,
            amount,
            (b.minimum is None or amount >= b.minimum)
            and (b.maximum is None or amount <= b.maximum),
        )
        for b in request.objective.constraints
    )
    return c.ValidationResult(
        result_id,
        request.request_hash,
        status,
        (
            tuple(c.SkillValue(w.skill, amount) for w in request.objective.weights)
            if status == "valid"
            else ()
        ),
        (
            sum(w.weight * amount for w in request.objective.weights)
            if status == "valid"
            else None
        ),
        constraints if status == "valid" else (),
        None if status == "valid" else "Explicit fixture pending/invalid",
        c.ArtifactRef("fixture://declared-inner-value", c.digest((amount, status))),
        c.ResourceUsage(),
        "mock",
    )


class Values:
    def __init__(self, amounts):
        self.amounts = iter(amounts)
        self.calls = []

    def evaluate(self, request):
        self.calls.append(request)
        return value(
            request, next(self.amounts), result_id=f"fixture-{len(self.calls)}"
        )


def stop_after_evaluations(count):
    # A TEST-ONLY stopping policy. Production thresholds deliberately unset.
    return lambda state: (
        StopDecision(True, "explicit test evaluation limit")
        if state.evaluated_count >= count
        else StopDecision(False)
    )


def restore(path, client, ctx, request, policy):
    return InnerSearch.restore(
        path,
        client,
        ctx.view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id="inner-test",
        expected_session_id="inner-session",
        expected_attempt_id="inner-outer",
    )


@pytest.mark.parametrize("use_model", [True, False])
def test_full_loop_returns_historical_best_actual_manifest_not_last(use_model):
    search, client, request, _, _ = setup(use_model=use_model)
    provider = Values([0.1, 0.3, 0.2])
    result = search.run(provider, stop_condition=stop_after_evaluations(3))
    assert result.status == "stopped"
    assert result.best.validation.total_value == pytest.approx(0.3)
    assert result.best.request == provider.calls[1]
    assert result.state.current.request == provider.calls[2]
    assert result.best.request.batch != result.state.current.request.batch
    assert result.best.prepared is search.history[0].candidate
    assert (
        result.best.request.batch.training_order
        == search.history[0].next_request.batch.training_order
    )
    assert len(provider.calls) == 3 and len(client.calls) == (6 if use_model else 0)
    assert result.state.stagnation_count == 1
    assert {e.attempt_id for e in result.state.ledger.events} == {"inner-outer"}


def test_no_default_stopping_rule_and_manual_steps_still_work():
    search, client, request, _, _ = setup()
    provider = Values([0.1])
    result = search.run(provider)
    assert result.status == "awaiting_stop_condition"
    assert not provider.calls and not client.calls
    assert search.state.phase == "awaiting_validation"
    search.step(provider)
    assert search.state.phase == "ready" and search.state.evaluated_count == 1
    search.step()
    assert search.state.phase == "awaiting_validation" and len(search.history) == 1
    assert len(client.calls) == 3


def test_stopping_callback_must_return_typed_decision_and_can_stop_before_work():
    search, client, _, _, _ = setup()
    with pytest.raises(c.ContractError):
        search.run(Values([0.1]), stop_condition=lambda state: True)
    result = search.run(
        Values([]), stop_condition=lambda state: StopDecision(True, "caller cancelled")
    )
    assert result.status == "no_feasible_result" and not client.calls
    with pytest.raises(c.ContractError):
        result.require_best()


def test_pending_and_invalid_results_yield_without_busy_polling_or_optimization():
    for status in ("pending", "invalid"):
        search, client, request, _, _ = setup()

        class Pending:
            calls = 0

            def evaluate(self, req):
                self.calls += 1
                return value(req, 0.0, status=status)

        provider = Pending()
        result = search.run(provider, stop_condition=stop_after_evaluations(1))
        assert result.status == "awaiting_validation" and provider.calls == 1
        assert not client.calls and result.best is None
        assert search.state.evaluated_count == 0
        final = value(request, 0.2)
        search.submit_validation(final)
        search.submit_validation(final)
        assert search.state.evaluated_count == 1 and len(search.validations) == 2


def test_imported_result_binding_and_finality():
    search, _, request, _, _ = setup()
    result = value(request, 0.1)
    with pytest.raises(c.ContractError):
        search.submit_validation(replace(result, request_hash=c.digest("wrong")))
    assert not search.validations
    search.submit_validation(result)
    with pytest.raises(c.ContractError):
        search.submit_validation(value(request, 0.2))
    assert search.state.best.validation == result


def test_valid_but_constraint_failed_is_current_not_best_and_can_be_optimized():
    search, _, request, _, _ = setup()
    search.submit_validation(value(request, -0.02))
    assert search.state.current is not None and search.state.best is None
    search.step()
    assert search.state.candidate_attempts == 1
    assert search.state.pending_request is not None


def test_equal_scores_keep_earliest_exact_batch():
    search, _, request, _, _ = setup()
    result = search.run(Values([0.1, 0.1]), stop_condition=stop_after_evaluations(2))
    assert result.best.request == request
    assert result.state.stagnation_count == 1


def test_provider_failure_preserves_best_and_unknown_costs_requires_explicit_retry():
    search, _, request, _, _ = setup()
    search.submit_validation(value(request, 0.1))
    search.step()

    class Broken:
        def evaluate(self, req):
            raise TimeoutError("explicit fixture timeout")

    failed = search.run(Broken(), stop_condition=stop_after_evaluations(2))
    assert failed.status == "blocked" and failed.best.request == request
    assert search.validations[-1].error_json is not None
    assert search.validations[-1].event.status == "failed"
    assert failed.state.ledger.summary()["gpu_seconds"]["total"] is None
    search.retry(reason="caller authorizes a new physical validation attempt")
    done = search.run(Values([0.2]), stop_condition=stop_after_evaluations(2))
    assert done.status == "stopped" and len(search.validations) == 3
    assert len({a.invocation_id for a in search.validations}) == 3


def test_wrong_provider_result_is_kept_as_failed_evidence_never_ranked():
    search, client, _, _, _ = setup()

    class Wrong:
        def evaluate(self, req):
            return replace(value(req, 100.0), request_hash=c.digest("other-batch"))

    result = search.step(Wrong())
    assert result.status == "blocked" and result.best is None and not client.calls
    assert search.validations[0].result.total_value == pytest.approx(100)


def test_infeasible_candidates_keep_feedback_without_calling_verifier():
    search, _, request, _, _ = setup(policy_step=0.01)
    provider = Values([0.1])
    result = search.run(
        provider,
        stop_condition=lambda s: (
            StopDecision(True, "test ends after failed candidate")
            if s.candidate_attempts
            else StopDecision(False)
        ),
    )
    assert result.status == "stopped" and len(provider.calls) == 1
    assert search.history[0].status == "candidate_infeasible"
    assert result.best.request == request


def test_shared_model_change_is_rejected_before_next_call():
    search, client, request, _, _ = setup()
    search.submit_validation(value(request, 0.1))
    client.model_ref = replace(request.model_ref, weights_hash=c.digest("changed"))
    with pytest.raises(c.ContractError):
        search.step()
    assert not client.calls


def test_save_restore_waiting_candidate_does_not_resample_or_recharge(tmp_path):
    search, client, request, ctx, policy = setup()
    search.submit_validation(value(request, 0.1))
    search.step()
    saved = search.state
    search.save(tmp_path / "snapshot")
    calls = len(client.calls)
    loaded = restore(tmp_path / "snapshot", client, ctx, request, policy)
    assert loaded.state == saved and len(client.calls) == calls
    final = loaded.run(Values([0.3, 0.2]), stop_condition=stop_after_evaluations(3))
    assert final.best.request == search.state.pending_request
    assert (
        final.best.request.batch.training_order
        == saved.pending_request.batch.training_order
    )
    assert len(client.calls) == calls + 3
    loaded.save(tmp_path / "finished")
    again = restore(tmp_path / "finished", client, ctx, request, policy)
    assert again.result() == final

    class Forbidden:
        def evaluate(self, req):
            raise AssertionError("Stopped search was rerun")

    assert again.run(Forbidden(), stop_condition=stop_after_evaluations(3)) == final


def test_best_publication_consumes_original_batch_and_independent_anchor(tmp_path):
    search, client, request, ctx, _ = setup()
    result = search.run(
        Values([0.1, 0.3, 0.2]), stop_condition=stop_after_evaluations(3)
    )
    search.publish_best(tmp_path / "best")
    count = len(client.calls)
    best = load_best(tmp_path / "best", ctx.view, expected_request=result.best.request)
    assert best == result.best and len(client.calls) == count
    with pytest.raises(c.ContractError):
        load_best(tmp_path / "best", ctx.view, expected_request=request)
    with pytest.raises(c.ContractError):
        search.publish_best(tmp_path / "best")


def test_unfinished_search_does_not_publish_training_handoff(tmp_path):
    search, _, request, _, _ = setup()
    search.submit_validation(value(request, 0.1))
    with pytest.raises(c.ContractError):
        search.publish_best(tmp_path / "best")


def test_journal_marks_inflight_work_and_never_automatically_reissues_it(tmp_path):
    search, client, request, ctx, policy = setup()
    search.enable_checkpoints(tmp_path / "journal")

    class SimulateCrash(BaseException):
        pass

    class Crashing:
        calls = 0

        def evaluate(self, req):
            self.calls += 1
            raise SimulateCrash()

    provider = Crashing()
    with pytest.raises(SimulateCrash):
        search.step(provider)
    loaded = InnerSearch.restore_latest(
        tmp_path / "journal",
        client,
        ctx.view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id="inner-test",
        expected_session_id="inner-session",
        expected_attempt_id="inner-outer",
    )
    assert loaded.state.phase == "interrupted"
    assert (
        loaded.run(provider, stop_condition=stop_after_evaluations(1)).status
        == "interrupted"
    )
    assert provider.calls == 1
    with pytest.raises(c.ContractError):
        loaded.stop(reason="cannot pretend unfinished call completed")


def test_checkpointed_stable_boundary_continues_without_duplicate_validation(tmp_path):
    search, client, request, ctx, policy = setup()
    search.enable_checkpoints(tmp_path / "journal")
    search.step(Values([0.1]))
    loaded = InnerSearch.restore_latest(
        tmp_path / "journal",
        client,
        ctx.view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id="inner-test",
        expected_session_id="inner-session",
        expected_attempt_id="inner-outer",
    )
    assert loaded.state.evaluated_count == 1 and len(loaded.validations) == 1
    loaded.run(Values([0.2]), stop_condition=stop_after_evaluations(2))
    assert loaded.state.evaluated_count == 2


def test_recovery_rejects_changed_best_even_if_checksums_rebuilt(tmp_path):
    search, client, request, ctx, policy = setup()
    search.run(Values([0.1, 0.3, 0.2]), stop_condition=stop_after_evaluations(3))
    root = tmp_path / "snapshot"
    search.save(root)
    meta = json.loads((root / "search.json").read_text(encoding="utf-8"))
    meta["state"]["best_request_hash"] = request.request_hash
    write_json(root / "search.json", meta)
    checks = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    checks["search.json"] = file_sha256(root / "search.json")
    write_json(root / "checksums.json", checks)
    with pytest.raises(c.ContractError):
        restore(root, client, ctx, request, policy)


def test_restore_requires_independent_search_scope(tmp_path):
    search, client, request, ctx, policy = setup()
    search.save(tmp_path / "snapshot")
    with pytest.raises(c.ContractError):
        restore(
            tmp_path / "snapshot",
            client,
            ctx,
            replace(request, request_id="another"),
            policy,
        )
    with pytest.raises(c.ContractError):
        restore(
            tmp_path / "snapshot",
            client,
            ctx,
            request,
            replace(policy, share_step=0.25),
        )


def test_blocked_validation_can_be_replaced_with_external_evidence_and_restored(
    tmp_path,
):
    search, client, request, ctx, policy = setup()

    class Broken:
        def evaluate(self, req):
            raise TimeoutError("declared failed call")

    search.step(Broken())
    assert search.state.failed_operations == 1
    search.submit_validation(value(request, 0.1))
    search.save(tmp_path / "snapshot")
    loaded = restore(tmp_path / "snapshot", client, ctx, request, policy)
    assert loaded.state == search.state


def test_failed_role_call_keeps_physical_costs_and_roundtrips(tmp_path):
    search, client, request, ctx, policy = setup()
    search.submit_validation(value(request, 0.1))
    original = client.generate_json

    def bad_execution(**kwargs):
        if kwargs["role"] == "lexec":
            return {"content": {"action": "edit-the-program"}}
        return original(**kwargs)

    client.generate_json = bad_execution
    assert search.step().status == "blocked"
    assert search.state.candidate_attempts == search.state.failed_operations == 1
    assert len(search.team.runtime.ledger().events) == 3
    search.save(tmp_path / "failed")
    loaded = restore(tmp_path / "failed", client, ctx, request, policy)
    assert loaded.state == search.state
    client.generate_json = original
    loaded.retry(reason="backend response format repaired")
    loaded.run(Values([0.2]), stop_condition=stop_after_evaluations(2))
    assert loaded.result().status == "stopped"


def test_negative_feasible_values_are_ranked_without_zero_baseline():
    search, _, _, _, _ = setup()
    result = search.run(
        Values([-0.005, -0.003, -0.004]), stop_condition=stop_after_evaluations(3)
    )
    assert result.best.validation.total_value == pytest.approx(-0.003)


def test_all_constraint_failed_observations_return_no_feasible_result():
    search, _, _, _, _ = setup()
    result = search.run(
        Values([-0.02, -0.03]), stop_condition=stop_after_evaluations(2)
    )
    assert result.status == "no_feasible_result"
    assert result.best is None and result.state.evaluated_count == 2


def test_restore_rejects_changed_outer_attempt_and_role_mode(tmp_path):
    search, client, request, ctx, policy = setup()
    search.save(tmp_path / "snapshot")
    for extra in ({"expected_attempt_id": "other"}, {"expected_use_model": False}):
        anchors = dict(
            expected_request=request,
            expected_policy=policy,
            expected_search_id="inner-test",
            expected_session_id="inner-session",
            expected_attempt_id="inner-outer",
        )
        anchors.update(extra)
        with pytest.raises(c.ContractError):
            InnerSearch.restore(tmp_path / "snapshot", client, ctx.view, **anchors)


def test_cli_mock_loop_has_explicit_demo_limit_only_and_no_overwrite(tmp_path, capsys):
    from self_curate.cli import main

    root = str(Path(__file__).resolve().parents[1])
    argv = ["inner-demo", "--project-root", root, "--output", str(tmp_path / "demo")]
    assert main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["best_value"] == pytest.approx(0.3)
    assert summary["last_value"] == pytest.approx(0.2)
    assert summary["production_stop_condition"] is None
    assert summary["history_restored"] and summary["model_calls"] == 6
    assert summary["validation_calls"] == 3 and not summary["real_model_called"]
    assert main(argv) == 2


def test_stopped_search_returns_result_without_requiring_new_stop_callback():
    search, _, _, _, _ = setup()
    final = search.run(Values([0.1]), stop_condition=stop_after_evaluations(1))
    assert search.run() == final


def test_restore_checks_cycle_interpretation_against_actual_role_history(tmp_path):
    search, client, request, ctx, policy = setup()
    search.run(Values([0.1, 0.2]), stop_condition=stop_after_evaluations(2))
    cycle = search.history[0]
    content = cycle.evaluation.interpretation.content
    content["commentary"] = "A different interpretation with self-consistent metadata"
    changed = replace(
        cycle.evaluation.interpretation, content_json=c.canonical_json(content)
    )
    search._cycles[0] = replace(
        cycle, evaluation=replace(cycle.evaluation, interpretation=changed)
    )
    search.save(tmp_path / "changed")
    with pytest.raises(c.ContractError):
        restore(tmp_path / "changed", client, ctx, request, policy)


def test_importing_existing_evidence_does_not_recharge_its_declared_usage(tmp_path):
    search, _, request, ctx, _ = setup()
    observed = replace(
        value(request, 0.1), usage=c.ResourceUsage(c.ComputeCost(5.0, 12.0, 50))
    )
    search.submit_validation(observed)
    assert search.validations[0].result.usage == observed.usage
    assert search.validations[0].event is None
    assert (
        len(search.ledger().events) == 1
    )  # Only the actually performed seed selection.
    search.stop(reason="caller finishes with existing validated seed")
    search.publish_best(tmp_path / "best")
    assert (
        load_best(tmp_path / "best", ctx.view, expected_request=request).validation
        == observed
    )


def test_external_worker_cost_keeps_original_id_and_survives_restore(tmp_path):
    search, client, request, ctx, policy = setup()
    observed = value(request, 0.1)
    event = c.CostEvent(
        "external-worker-call",
        "inner-outer",
        "evaluation",
        request.request_hash,
        request.model_ref.model_hash,
        "completed",
        observed.usage,
        c.ArtifactRef("fixture://worker-receipt", c.digest("worker-receipt")),
        "mock",
    )
    search.submit_validation(observed, cost_event=event)
    search.submit_validation(observed, cost_event=event)
    assert len(search.ledger().events) == 2 and event in search.ledger().events
    search.save(tmp_path / "snapshot")
    assert (
        restore(tmp_path / "snapshot", client, ctx, request, policy).ledger()
        == search.ledger()
    )


def test_external_cost_mismatch_is_rejected_before_state_changes():
    search, _, request, _, _ = setup()
    observed = value(request, 0.1)
    wrong = c.CostEvent(
        "wrong-worker-call",
        "other-outer",
        "evaluation",
        request.request_hash,
        request.model_ref.model_hash,
        "completed",
        observed.usage,
        c.ArtifactRef("fixture://wrong-worker", c.digest("wrong-worker")),
        "mock",
    )
    before = search.state
    with pytest.raises(c.ContractError):
        search.submit_validation(observed, cost_event=wrong)
    assert search.state == before
