"""Two-pass review: reproduction of state, protocol and provider handoff gaps."""

from dataclasses import replace
import json

import pytest

from test_outer_cycle import (
    Decisions,
    FixtureRunner,
    FixtureTrainer,
    make_cycle,
    ref,
    tiny_request,
)
from self_curate import contracts as c
from self_curate.training import TrainingOutcome, TrainingService
from self_curate.validation import TaskValidationFailure, TaskValidator


def test_insufficient_evidence_does_not_reinvoke_decider_on_repeated_run(tmp_path):
    cycle, trainer, runner, _, _ = make_cycle(tmp_path)
    decider = Decisions(["needs_more_evidence", "accept", "accept"])
    cycle.run(decider=decider)
    cycle.run(decider=decider)
    cycle.step(decider=decider)
    assert cycle.phase == "awaiting_decision" and len(decider.contexts) == 1
    assert len(trainer.requests) == 1 and len(runner.requests) == 6


def test_new_validation_needs_its_own_insufficient_evidence_decision(tmp_path):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run(decider=Decisions(["needs_more_evidence"]))
    cycle.request_revalidation(reason="explicit first repeat")
    cycle.step()
    assert cycle.phase == "awaiting_decision"
    with pytest.raises(c.ContractError):
        cycle.request_revalidation(reason="old decision cannot authorize new work")


@pytest.mark.parametrize("field", ["protocol", "checkpoint"])
def test_evaluation_metadata_cannot_be_relabelled_after_execution(tmp_path, field):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    with pytest.raises(c.ContractError):
        if field == "protocol":
            replace(
                evaluation,
                suite=replace(evaluation.suite, protocol=ref("another-protocol")),
            )
        else:
            replace(evaluation, checkpoint=ref("another-checkpoint"))


def test_second_task_setup_io_failure_preserves_completed_task_cost(
    tmp_path, monkeypatch
):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    suite = cycle._suite
    import self_curate.validation.service as module

    write = module.write_json
    request_writes = 0

    def fail_second_request(path, value):
        nonlocal request_writes
        if path.name == "request.json" and path.parent.name.startswith("run-"):
            request_writes += 1
            if request_writes == 2:
                raise OSError("intentional setup failure before dispatch")
        return write(path, value)

    monkeypatch.setattr(module, "write_json", fail_second_request)
    runner = FixtureRunner()
    with pytest.raises(TaskValidationFailure) as failure:
        TaskValidator(runner).evaluate(
            cycle.accepted_model,
            suite,
            attempt_id="review",
            evaluation_id="setup-failure",
            output=tmp_path / "setup-failure",
        )
    assert len(runner.requests) == 1 and len(failure.value.rollouts) == 1
    assert len(failure.value.ledger.events) == 1
    assert failure.value.ledger.events[0].status == "completed"


def test_outer_rejects_internally_valid_training_from_another_recipe(tmp_path):
    cycle, backend, _, _, _ = make_cycle(tmp_path)
    service = cycle._trainer
    train = service.train

    def altered(request, output):
        return train(
            replace(
                request,
                recipe=replace(
                    request.recipe, learning_rate=request.recipe.learning_rate * 2
                ),
            ),
            output,
        )

    service.train = altered
    cycle.step()
    assert cycle.phase == "blocked"
    assert len(backend.requests) == 1
    assert sum(e.operation == "training" for e in cycle.ledger.events) == 1
    assert all(not row.usage.candidate_sft for row in cycle.updated_records)


def test_outer_rejects_another_checkpoint_report_without_losing_incurred_costs(
    tmp_path,
):
    cycle, _, runner, _, _ = make_cycle(tmp_path)
    validator = cycle._validator
    evaluate = validator.evaluate

    def wrong_checkpoint(model, suite, **kwargs):
        return evaluate(
            model, suite, **{**kwargs, "checkpoint": ref("other-evaluated-weights")}
        )

    validator.evaluate = wrong_checkpoint
    cycle.run()
    assert cycle.phase == "blocked" and len(runner.requests) == 6
    assert sum(e.operation == "rollout" for e in cycle.ledger.events) == 6
    assert cycle.pending_decision is None


def test_strict_training_outcome_cannot_forge_matching_but_wrong_exposure_counts(
    tmp_path,
):
    outcome = TrainingService(FixtureTrainer()).train(tiny_request(), tmp_path / "good")
    usage = replace(outcome.result.usage, training_supervised_tokens=1)
    result = replace(outcome.result, processed_supervised_tokens=1, usage=usage)
    ledger = replace(
        outcome.ledger, events=(replace(outcome.ledger.events[0], usage=usage),)
    )
    record = replace(
        outcome.record,
        processed_supervised_tokens=1,
        cost=ledger.reference(outcome.record.cost.event_ids),
    )
    with pytest.raises(c.ContractError):
        TrainingOutcome(outcome.request, record, ledger, result)


def test_decision_attempt_cannot_differ_from_training_attempt(tmp_path):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    with pytest.raises(c.ContractError):
        replace(cycle.pending_decision, attempt_id="another-attempt")


def test_old_task_report_is_readable_but_requires_execution_evidence_before_deciding(
    tmp_path,
):
    from self_curate.orchestration import parse_outer_contract

    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    raw = json.loads(c.canonical_json(evaluation.to_dict()))
    raw.pop("execution_requests", None)
    raw.pop("execution_results", None)
    legacy = parse_outer_contract("TaskEvaluation", raw)
    assert legacy.report == evaluation.report
    assert legacy.protocol_verified is False
    with pytest.raises(c.ContractError):
        replace(cycle.pending_decision, evaluation=legacy)


def test_legacy_directory_is_hydrated_without_rewriting_any_evidence(tmp_path):
    from self_curate.orchestration import load_task_evaluation
    from self_curate.data.pool import file_sha256, write_json

    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    directory = tmp_path / "outer/validate-0"
    raw = evaluation.to_dict()
    raw.pop("execution_requests")
    raw.pop("execution_results")
    write_json(directory / "evaluation.json", raw)
    checks = {p: file_sha256(p) for p in directory.rglob("*") if p.is_file()}
    loaded = load_task_evaluation(
        directory,
        expected_model=evaluation.report.model_ref,
        expected_suite=evaluation.suite,
        expected_checkpoint=evaluation.checkpoint,
    )
    assert loaded.protocol_verified and loaded == evaluation
    assert all(file_sha256(p) == expected for p, expected in checks.items())


@pytest.mark.parametrize("tamper", ["protocol", "reward"])
def test_saved_execution_evidence_cannot_change_after_report(tmp_path, tamper):
    from self_curate.orchestration import load_task_evaluation
    from self_curate.data.pool import write_json

    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    directory = tmp_path / "outer/validate-0"
    folder = next(directory.glob("run-*"))
    if tamper == "protocol":
        path = folder / "request.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["protocol"] = ref("different-protocol").to_dict()
    else:
        path = folder / "result.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["reward"] = 1.0
    write_json(path, raw)
    with pytest.raises(c.ContractError):
        load_task_evaluation(
            directory,
            expected_model=evaluation.report.model_ref,
            expected_suite=evaluation.suite,
            expected_checkpoint=evaluation.checkpoint,
        )


def test_task_costs_cannot_be_transplanted_from_another_attempt(tmp_path):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    event = replace(evaluation.ledger.events[0], attempt_id="other-attempt")
    ledger = replace(evaluation.ledger, events=(event, *evaluation.ledger.events[1:]))
    rollout = replace(
        evaluation.rollouts[0],
        cost=ledger.reference(evaluation.rollouts[0].cost.event_ids),
    )
    with pytest.raises(c.ContractError):
        replace(evaluation, ledger=ledger, rollouts=(rollout, *evaluation.rollouts[1:]))


def test_training_outcome_must_keep_exact_optimizer_step_budget(tmp_path):
    outcome = TrainingService(FixtureTrainer()).train(tiny_request(), tmp_path / "good")
    with pytest.raises(c.ContractError):
        replace(
            outcome,
            result=replace(
                outcome.result, optimizer_steps=outcome.result.optimizer_steps + 1
            ),
        )


def test_checkpoint_continuation_cannot_change_trainable_parameter_scope(tmp_path):
    from test_outer_cycle import tiny_store
    from self_curate.training import TrainingFailure
    from self_curate.training.torch_backend import TorchSFTBackend

    _, store = tiny_store()
    service = TrainingService(TorchSFTBackend(store, backend="mock"))
    first = service.train(tiny_request(epochs=1), tmp_path / "first")
    load = store.load_checkpoint

    def changed_scope(model_ref, checkpoint):
        loaded = load(model_ref, checkpoint)
        loaded.model.embedding.weight.requires_grad_(True)
        return loaded

    store.load_checkpoint = changed_scope
    with pytest.raises(TrainingFailure, match="trainable"):
        service.train(
            tiny_request(epochs=1, starting_model=first.result.candidate_model),
            tmp_path / "changed-scope",
        )


@pytest.mark.parametrize("operation", ["training", "validation"])
def test_invalid_provider_output_still_preserves_valid_reported_usage(
    tmp_path, operation
):
    from types import SimpleNamespace
    from self_curate.training import TrainingFailure

    cycle, _, _, _, _ = make_cycle(tmp_path)
    reported = c.ResourceUsage(training_supervised_tokens=7, tool_calls=3)
    if operation == "training":
        backend = FixtureTrainer()
        backend.train = lambda request, output: SimpleNamespace(usage=reported)
        with pytest.raises(TrainingFailure) as failure:
            TrainingService(backend).train(tiny_request(), tmp_path / "bad-training")
    else:
        runner = FixtureRunner()
        runner.execute = lambda request, output: SimpleNamespace(usage=reported)
        with pytest.raises(TaskValidationFailure) as failure:
            TaskValidator(runner).evaluate(
                cycle.accepted_model,
                cycle._suite,
                attempt_id="review",
                evaluation_id="invalid-output",
                output=tmp_path / "bad-validation",
            )
    event = failure.value.ledger.events[-1]
    assert event.status == "failed"
    assert event.usage.training_supervised_tokens == 7 and event.usage.tool_calls == 3
