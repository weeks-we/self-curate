"""Synthetic outer integration and independent tiny CPU gradient references."""

from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import json
import random

import pytest

from helpers_m2 import MODEL
from helpers_p0 import fixture_bundle
from test_inner_search import Values, setup, stop_after_evaluations
from self_curate import contracts as c
from self_curate.orchestration import AcceptanceDecision, OuterCycle
from self_curate.training import (
    BackendTrainingResult,
    SFTRecipe,
    TrainingData,
    TrainingExample,
    TrainingFailure,
    TrainingRequest,
    TrainingService,
)
from self_curate.validation import TaskRunResult, TaskSuite, TaskValidator


def ref(name):
    return c.ArtifactRef("fixture://" + name, c.digest(name))


def recipe(**kwargs):
    return SFTRecipe(
        **{
            "learning_rate": 0.01,
            "epochs": 1,
            "accumulation_steps": 3,
            "max_sequence_tokens": 128,
            "seed": 7,
            **kwargs,
        }
    )


class FixtureTrainer:
    backend = "mock"

    def __init__(self):
        self.requests = []

    def train(self, request, output):
        self.requests.append(request)
        amount = request.data.supervised_tokens * request.recipe.epochs
        return BackendTrainingResult(
            replace(request.starting_model, weights_hash=c.digest(request.record_hash)),
            ref("declared-trained-" + request.record_hash),
            amount,
            request.recipe.epochs
            * (
                (len(request.data.examples) + request.recipe.accumulation_steps - 1)
                // request.recipe.accumulation_steps
            ),
            None,
            c.ResourceUsage(training_supervised_tokens=amount),
        )


class FixtureRunner:
    backend = "mock"

    def __init__(self, *, fail=False, reward=0.5):
        self.requests = []
        self.fail, self.reward = fail, reward

    def execute(self, request, output):
        self.requests.append(request)
        return TaskRunResult(
            request.record_hash,
            ref("actions"),
            ref("observations"),
            c.digest("tokens"),
            c.digest("action-mask"),
            2,
            None if self.fail else self.reward,
            "environment_failed" if self.fail else "completed",
            "explicit synthetic outcome",
            c.ResourceUsage(c.ComputeCost(0, 0, 2), tool_calls=1),
        )


class CurationProvider:
    def __init__(self):
        self.calls = []

    def prepare(self, handoff):
        self.calls.append(handoff)
        return TrainingData(
            "curation",
            handoff.record_hash,
            handoff.starting_model.tokenizer,
            (
                TrainingExample(
                    "curation-1", c.digest("curation-message"), (1, 2, 3), (-100, 2, 3)
                ),
            ),
            (ref("externally-produced-curation"),),
            "mock",
        )


class Decisions:
    def __init__(self, outcomes):
        self.outcomes, self.contexts = iter(outcomes), []

    def decide(self, context):
        self.contexts.append(context)
        return AcceptanceDecision(
            context.record_hash,
            next(self.outcomes),
            "explicit synthetic policy result",
            ref("decision"),
        )


def make_cycle(tmp_path, *, policy=True, provider=True, fail=False, historical=False):
    search, _, _, ctx, _ = setup(use_model=False)
    result = search.run(
        Values([0.1, 0.3, 0.2] if historical else [0.1]),
        stop_condition=stop_after_evaluations(3 if historical else 1),
    )
    suite = TaskSuite(
        "fixture-suite",
        fixture_bundle().tasks,
        (11, 12),
        ref("sampling"),
        ref("task-protocol"),
        "task_mean",
        "mock",
    )
    trainer, runner, producer = (
        FixtureTrainer(),
        FixtureRunner(fail=fail),
        CurationProvider(),
    )
    cycle = OuterCycle(
        result,
        ctx.view,
        expected_model=result.best.request.model_ref,
        expected_batch_id=result.best.request.batch.batch_id,
        trainer=TrainingService(trainer),
        validator=TaskValidator(runner),
        suite=suite,
        subset_recipe=recipe(),
        curation_recipe=recipe(epochs=2),
        output=tmp_path / "outer",
        curation_provider=producer if provider else None,
        acceptance_policy=ref("policy") if policy else None,
    )
    return cycle, trainer, runner, producer, result


def test_exact_best_sequential_weights_and_task_validation_are_connected(tmp_path):
    cycle, trainer, runner, producer, search = make_cycle(tmp_path, historical=True)
    decider = Decisions(["accept", "accept"])
    result = cycle.run(decider=decider)
    assert result["phase"] == "completed"
    first, second = trainer.requests
    assert (
        tuple(x.data_id for x in first.data.examples)
        == search.best.request.batch.training_order
    )
    assert (
        first.batch_id
        == search.best.request.batch.batch_id
        != search.state.current.request.batch.batch_id
    )
    assert (
        cycle.trainings[0].record.training_data_hash
        == search.best.request.batch.training_data_hash
    )
    assert second.starting_model == cycle.trainings[0].result.candidate_model
    assert cycle.accepted_model == cycle.trainings[1].result.candidate_model
    assert len(producer.calls) == 1
    assert len(runner.requests) == 12
    assert {r.model_ref for r in runner.requests[:6]} == {second.starting_model}
    assert {r.model_ref for r in runner.requests[6:]} == {cycle.accepted_model}
    assert cycle.evaluations[0].suite == cycle.evaluations[1].suite
    assert decider.contexts[0].baseline is None
    assert decider.contexts[1].baseline == cycle.evaluations[0]
    assert all(
        r.usage.candidate_sft[-1].decision == "accepted" for r in cycle.updated_records
    )
    ids = [e.event_id for e in cycle.ledger.events]
    assert len(ids) == len(set(ids))
    cycle.run(decider=decider)
    assert len(trainer.requests) == 2 and len(runner.requests) == 12


@pytest.mark.parametrize(
    "outcomes,expected_trainings", [(["reject"], 1), (["accept", "reject"], 2)]
)
def test_rejection_restores_correct_parent_and_keeps_rejected_costs(
    tmp_path, outcomes, expected_trainings
):
    cycle, trainer, runner, producer, search = make_cycle(tmp_path)
    cycle.run(decider=Decisions(outcomes))
    assert cycle.phase == "rejected" and len(trainer.requests) == expected_trainings
    expected = (
        search.best.request.model_ref
        if expected_trainings == 1
        else cycle.trainings[0].result.candidate_model
    )
    assert cycle.accepted_model == expected
    assert (
        json.loads((tmp_path / "outer/accepted.json").read_text(encoding="utf-8"))[
            "model"
        ]
        == expected.to_dict()
    )
    assert (
        sum(e.operation == "training" for e in cycle.ledger.events)
        == expected_trainings
    )
    assert all(
        r.usage.candidate_sft[-1].decision
        == ("rejected" if expected_trainings == 1 else "accepted")
        for r in cycle.updated_records
    )
    assert len(producer.calls) == expected_trainings - 1


def test_unfilled_policy_waits_without_training_second_stage_or_promoting(tmp_path):
    cycle, trainer, runner, _, search = make_cycle(tmp_path, policy=False)
    decider = Decisions(["accept", "accept"])
    cycle.run(decider=decider)
    assert cycle.phase == "awaiting_decision" and len(trainer.requests) == 1
    assert (
        not decider.contexts and cycle.accepted_model == search.best.request.model_ref
    )
    cycle.run(decider=decider)
    assert len(runner.requests) == 6 and len(trainer.requests) == 1
    cycle.configure_acceptance(ref("policy-filled-later"))
    cycle.run(decider=decider)
    assert cycle.phase == "completed"


def test_no_decider_waits_and_manual_decision_is_exactly_bound_and_idempotent(tmp_path):
    cycle, trainer, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    context = cycle.pending_decision
    decision = AcceptanceDecision(
        context.record_hash, "accept", "manual test policy", ref("manual")
    )
    with pytest.raises(c.ContractError):
        cycle.submit_decision(replace(decision, context_hash=c.digest("other-model")))
    cycle.submit_decision(decision)
    cycle.submit_decision(decision)
    assert len(cycle.decisions) == 1 and len(trainer.requests) == 1


def test_missing_curation_data_waits_at_explicit_interface(tmp_path):
    cycle, trainer, _, _, _ = make_cycle(tmp_path, provider=False)
    cycle.run(decider=Decisions(["accept"]))
    assert cycle.phase == "awaiting_curation_data" and len(trainer.requests) == 1
    data = CurationProvider().prepare(cycle.handoff)
    with pytest.raises(c.ContractError):
        cycle.supply_curation_data(
            replace(data, source_hash=c.digest("another-search"))
        )
    cycle.supply_curation_data(data)
    cycle.run(decider=Decisions(["accept"]))
    assert cycle.phase == "completed"


def test_environment_failure_is_unscored_and_cannot_be_accepted(tmp_path):
    cycle, trainer, runner, _, search = make_cycle(tmp_path, fail=True)
    cycle.run()
    assert cycle.evaluations[0].report.status == "invalid"
    assert all(r.reward is None for r in cycle.evaluations[0].rollouts)
    assert all(m.mean_reward is None for m in cycle.evaluations[0].report.metrics)
    context = cycle.pending_decision
    with pytest.raises(c.ContractError):
        cycle.submit_decision(
            AcceptanceDecision(context.record_hash, "accept", "bad", ref("bad"))
        )
    cycle.submit_decision(
        AcceptanceDecision(
            context.record_hash, "reject", "invalid evidence", ref("reject")
        )
    )
    assert cycle.accepted_model == search.best.request.model_ref
    assert len(trainer.requests) == 1


def test_insufficient_evidence_waits_and_explicit_revalidation_does_not_retrain(
    tmp_path,
):
    cycle, trainer, runner, _, _ = make_cycle(tmp_path)
    cycle.run(decider=Decisions(["needs_more_evidence"]))
    assert cycle.phase == "awaiting_decision" and len(cycle.decisions) == 1
    cycle.request_revalidation(reason="explicit extra measurement")
    cycle.step()
    assert len(cycle.evaluations) == 2 and len(trainer.requests) == 1
    assert len({r.run_id for r in runner.requests}) == 12
    cycle.run(decider=Decisions(["accept", "accept"]))
    assert cycle.phase == "completed" and len(trainer.requests) == 2


def test_failed_physical_training_is_not_reissued_and_preserves_usage(tmp_path):
    cycle, trainer, _, _, search = make_cycle(tmp_path)

    def fail(request, output):
        trainer.requests.append(request)
        raise TrainingFailure(
            "intentional partial failure",
            usage=c.ResourceUsage(training_supervised_tokens=3),
        )

    trainer.train = fail
    cycle.run()
    cycle.run()
    assert cycle.phase == "blocked" and len(trainer.requests) == 1
    assert cycle.accepted_model == search.best.request.model_ref
    event = next(e for e in cycle.ledger.events if e.operation == "training")
    assert event.status == "failed" and event.usage.training_supervised_tokens == 3


@pytest.mark.parametrize("mutation", ["model", "batch", "not_stopped"])
def test_handoff_cannot_use_alternative_model_batch_or_running_search(
    tmp_path, mutation
):
    search, _, _, ctx, _ = setup(use_model=False)
    result = search.run(Values([0.1]), stop_condition=stop_after_evaluations(1))
    kwargs = {
        "expected_model": result.best.request.model_ref,
        "expected_batch_id": result.best.request.batch.batch_id,
    }
    if mutation == "model":
        kwargs["expected_model"] = replace(
            kwargs["expected_model"], weights_hash=c.digest("other")
        )
    elif mutation == "batch":
        kwargs["expected_batch_id"] = c.digest("other")
    else:
        result = replace(result, status="awaiting_validation")
    suite = TaskSuite(
        "test", fixture_bundle().tasks, (0,), ref("s"), ref("p"), "task_mean", "mock"
    )
    with pytest.raises(c.ContractError):
        OuterCycle(
            result,
            ctx.view,
            **kwargs,
            trainer=TrainingService(FixtureTrainer()),
            validator=TaskValidator(FixtureRunner()),
            suite=suite,
            subset_recipe=recipe(),
            curation_recipe=recipe(),
            output=tmp_path / "bad",
        )


def test_final_test_split_is_rejected_by_outer_validation():
    with pytest.raises(c.ContractError):
        TaskSuite(
            "illegal-test",
            tuple(replace(t, split="test") for t in fixture_bundle().tasks),
            (0,),
            ref("s"),
            ref("p"),
            "task_mean",
            "mock",
        )


def test_mismatched_task_result_retains_costs_without_fabricating_capability(tmp_path):
    runner = FixtureRunner()
    execute = runner.execute

    def wrong(request, output):
        return replace(execute(request, output), request_hash=c.digest("wrong-model"))

    runner.execute = wrong
    suite = TaskSuite(
        "test", fixture_bundle().tasks, (0,), ref("s"), ref("p"), "task_mean", "mock"
    )
    from self_curate.validation import TaskValidationFailure

    with pytest.raises(TaskValidationFailure) as failure:
        TaskValidator(runner).evaluate(
            MODEL, suite, attempt_id="a", evaluation_id="e", output=tmp_path / "eval"
        )
    assert failure.value.ledger.events[0].usage.tool_calls == 1
    assert not (tmp_path / "eval/evaluation.json").exists()


@pytest.mark.parametrize(
    "change", ["truncate", "first_token", "no_supervision", "bad_label"]
)
def test_training_never_silently_rewrites_tokens_or_masks(change):
    ids, labels = (1, 2, 3), (-100, 2, 3)
    if change == "first_token":
        labels = (1, 2, 3)
    if change == "no_supervision":
        labels = (-100, -100, -100)
    if change == "bad_label":
        labels = (-100, 9, 3)
    with pytest.raises(c.ContractError):
        example = TrainingExample("x", c.digest("x"), ids, labels)
        data = TrainingData(
            "curation",
            c.digest("source"),
            MODEL.tokenizer,
            (example,),
            (ref("data"),),
            "mock",
        )
        TrainingRequest(
            "t",
            "a",
            MODEL,
            data,
            c.digest("batch"),
            c.digest("data"),
            recipe(max_sequence_tokens=2),
            "mock",
        )


def tiny_request(*, epochs=2, accumulation_steps=2, starting_model=MODEL):
    examples = (
        TrainingExample("a", c.digest("a"), (1, 2, 3), (-100, 2, 3)),
        TrainingExample("b", c.digest("b"), (4, 3, 2, 1), (-100, -100, -100, 1)),
        TrainingExample("c", c.digest("c"), (2, 4, 3), (-100, 4, 3)),
    )
    data = TrainingData(
        "curation",
        c.digest("source"),
        MODEL.tokenizer,
        examples,
        (ref("tiny-data"),),
        "mock",
    )
    return TrainingRequest(
        "tiny-" + starting_model.model_hash,
        "tiny-attempt",
        starting_model,
        data,
        c.digest("batch"),
        data.record_hash,
        recipe(
            epochs=epochs,
            accumulation_steps=accumulation_steps,
            optimizer="sgd",
            learning_rate=0.1,
        ),
        "mock",
    )


def tiny_store():
    torch = pytest.importorskip("torch")
    from self_curate.training.torch_backend import LoadedModel

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 3)
            self.embedding.weight.requires_grad_(False)
            self.projection = torch.nn.Linear(3, 8, bias=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))

    class Store:
        checkpoint_kind = "full"

        def __init__(self):
            with torch.random.fork_rng():
                torch.manual_seed(22)
                self.initial = Tiny()
            self.models, self.loads = [], []

        def load_branch(self, model_ref):
            self.loads.append(model_ref)
            model = deepcopy(self.initial)
            self.models.append(model)
            return LoadedModel(model, model_ref)

        def load_checkpoint(self, model_ref, checkpoint):
            self.loads.append(model_ref)
            model = deepcopy(self.initial)
            model.load_state_dict(
                torch.load(checkpoint / "weights/state.pt", weights_only=True)
            )
            self.models.append(model)
            return LoadedModel(model, model_ref)

        def save_branch(self, model, output):
            output.mkdir()
            torch.save(model.state_dict(), output / "state.pt")

    return torch, Store()


def test_actual_cpu_sft_matches_independent_token_weighted_gradient_reference(tmp_path):
    torch, store = tiny_store()
    from self_curate.training.torch_backend import TorchSFTBackend
    import torch.nn.functional as F

    request = tiny_request()
    reference = deepcopy(store.initial)
    optimizer = torch.optim.SGD(
        [p for p in reference.parameters() if p.requires_grad], lr=0.1
    )
    # Independent reference: concatenate selected causal logits, then one CE mean.
    for epoch in range(2):
        for window in (request.data.examples[:2], request.data.examples[2:]):
            optimizer.zero_grad()
            selected_logits, selected_targets = [], []
            for x in window:
                ids = torch.tensor([x.input_ids])
                logits = reference(
                    input_ids=ids, attention_mask=torch.ones_like(ids)
                ).logits[0, :-1]
                labels = torch.tensor(x.labels[1:])
                selected_logits.append(logits[labels != -100])
                selected_targets.append(labels[labels != -100])
            F.cross_entropy(
                torch.cat(selected_logits), torch.cat(selected_targets)
            ).backward()
            optimizer.step()
    parent_weights = deepcopy(store.initial.state_dict())
    torch_rng, py_rng = torch.get_rng_state().clone(), random.getstate()
    result = TrainingService(TorchSFTBackend(store, backend="mock")).train(
        request, tmp_path / "tiny"
    )
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == py_rng
    actual = store.models[0]
    assert torch.allclose(
        actual.projection.weight, reference.projection.weight, atol=1e-7, rtol=1e-6
    )
    assert not torch.equal(
        actual.projection.weight, parent_weights["projection.weight"]
    )
    assert torch.equal(actual.embedding.weight, parent_weights["embedding.weight"])
    assert all(
        torch.equal(value, parent_weights[name])
        for name, value in store.initial.state_dict().items()
    )
    assert (
        result.result.processed_supervised_tokens == 10
        and result.result.optimizer_steps == 4
    )
    assert (tmp_path / "tiny/checkpoint/training_state.pt").exists()


def test_second_actual_training_loads_first_checkpoint_and_detects_tampering(tmp_path):
    torch, store = tiny_store()
    from self_curate.training.torch_backend import TorchSFTBackend

    backend = TorchSFTBackend(store, backend="mock")
    service = TrainingService(backend)
    first = service.train(tiny_request(epochs=1), tmp_path / "one")
    second_request = tiny_request(epochs=1, starting_model=first.result.candidate_model)
    second = service.train(second_request, tmp_path / "two")
    assert store.loads == [MODEL, first.result.candidate_model]
    assert second.result.candidate_model != first.result.candidate_model
    first_weights = tmp_path / "one/checkpoint/weights/state.pt"
    first_weights.write_bytes(first_weights.read_bytes() + b"changed")
    with pytest.raises(TrainingFailure, match="changed"):
        service.train(second_request, tmp_path / "three")


def test_backend_cannot_misreport_training_exposure(tmp_path):
    backend = FixtureTrainer()
    original = backend.train

    def wrong(request, output):
        result = original(request, output)
        return replace(
            result,
            processed_supervised_tokens=1,
            usage=c.ResourceUsage(training_supervised_tokens=1),
        )

    backend.train = wrong
    with pytest.raises(TrainingFailure, match="exposure") as error:
        TrainingService(backend).train(tiny_request(), tmp_path / "bad-count")
    assert error.value.ledger.events[0].usage.training_supervised_tokens == 1


def test_second_stage_failure_keeps_the_accepted_first_stage(tmp_path):
    cycle, trainer, _, _, _ = make_cycle(tmp_path)
    original = trainer.train

    def fail_second(request, output):
        if request.data.kind == "curation":
            raise TrainingFailure(
                "second stage failed",
                usage=c.ResourceUsage(training_supervised_tokens=1),
            )
        return original(request, output)

    trainer.train = fail_second
    cycle.run(decider=Decisions(["accept"]))
    assert cycle.phase == "blocked" and cycle.stage == "curation"
    assert cycle.accepted_model == cycle.trainings[0].result.candidate_model
    assert len(cycle.evaluations) == 1
    assert any(
        e.operation == "training" and e.status == "failed" for e in cycle.ledger.events
    )


def test_capability_cannot_claim_a_different_numeric_mean(tmp_path):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    evaluation = cycle.evaluations[0]
    forged = replace(
        evaluation.report,
        metrics=tuple(replace(m, mean_reward=1.0) for m in evaluation.report.metrics),
    )
    with pytest.raises(c.ContractError, match="metrics"):
        replace(evaluation, report=forged)


def test_validation_failure_is_not_silently_retried_or_scored_as_zero(tmp_path):
    cycle, _, runner, _, search = make_cycle(tmp_path)

    def crash(request, output):
        runner.requests.append(request)
        raise RuntimeError("intentional transport failure")

    runner.execute = crash
    cycle.run()
    cycle.run()
    assert cycle.phase == "blocked" and not cycle.evaluations
    assert (
        len(runner.requests) == 1
        and cycle.accepted_model == search.best.request.model_ref
    )
    failure = next(e for e in cycle.ledger.events if e.operation == "rollout")
    assert failure.status == "failed" and failure.usage.tool_calls is None


def test_actual_state_dict_store_and_outer_cpu_pipeline(tmp_path):
    torch, example_store = tiny_store()
    from self_curate.training.torch_backend import StateDictModelStore, TorchSFTBackend
    from self_curate.data import FixtureTokenizer, normalize_trajectory
    from self_curate.data.pool import file_sha256
    from helpers_m2 import context, node, program
    from helpers_feedback import inputs
    from self_curate.operators import prepare_selection
    from self_curate.agents import RoleTeam
    from self_curate.agents.demo import FixtureRoleClient
    from self_curate.search import InnerSearch

    initial_path = tmp_path / "initial.pt"
    torch.save(example_store.initial.state_dict(), initial_path)
    tokenizer = replace(
        MODEL.tokenizer,
        tokenizer_id="fixture-compact",
        tokenizer_hash=c.digest("fixture-compact"),
    )
    model = replace(MODEL, tokenizer=tokenizer, weights_hash=file_sha256(initial_path))

    class CompactTokenizer:
        identity = tokenizer

        def encode(self, messages, provided=None):
            encoded = FixtureTokenizer().encode(messages)
            return replace(
                encoded,
                identity=tokenizer,
                input_ids=tuple(token % 8 for token in encoded.input_ids),
            )

    rows = tuple(
        normalize_trajectory(
            {
                "schema_version": "1",
                "data_id": "compact-" + str(i),
                "source": {
                    "source_id": "explicit-synthetic",
                    "revision": "1",
                    "record_id": str(i),
                    "artifact": ref("compact-" + str(i)).to_dict(),
                },
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "ab" + str(i)},
                ],
                "quality": {
                    "status": "pass",
                    "method": "manual",
                    "verifier_id": "synthetic",
                    "evidence": ref("manual-fixture").to_dict(),
                },
            },
            CompactTokenizer(),
            ("recovery", "regression"),
        )
        for i in range(3)
    )
    ctx = context(rows, target=9, tolerance=0, batch=1, model=model, signals=())
    spec = program(
        [node("random", "SampleSelect", {"method": "uniform", "size": 3, "seed": 7})]
    )
    old_request, _, _, policy = inputs()
    initial = prepare_selection(
        spec,
        ctx,
        old_request.objective,
        backend="mock",
        attempt_id="tiny-full-flow",
        invocation_id="seed",
    )
    request = replace(
        old_request,
        request_id="tiny-inner",
        model_ref=model,
        program=spec,
        batch=initial.workflow.batch,
        selection_context_hash=ctx.context_hash,
    )
    team = RoleTeam(
        FixtureRoleClient(model), model=model, session_id="tiny-inner", backend="mock"
    )
    search = InnerSearch(
        team, initial, request, policy, search_id="tiny-inner", use_model=False
    )
    result = search.run(Values([0.1]), stop_condition=stop_after_evaluations(1))
    suite = TaskSuite(
        "tiny-task-suite",
        fixture_bundle().tasks,
        (11,),
        ref("sampling"),
        ref("protocol"),
        "task_mean",
        "mock",
    )
    store = StateDictModelStore(
        type(example_store.initial), initial_model=model, initial_state=initial_path
    )
    cycle = OuterCycle(
        result,
        ctx.view,
        expected_model=model,
        expected_batch_id=result.best.request.batch.batch_id,
        trainer=TrainingService(TorchSFTBackend(store, backend="mock")),
        validator=TaskValidator(FixtureRunner()),
        suite=suite,
        subset_recipe=recipe(optimizer="sgd"),
        curation_recipe=recipe(optimizer="sgd"),
        curation_provider=CurationProvider(),
        acceptance_policy=ref("explicit-synthetic-policy"),
        output=tmp_path / "actual-outer",
    )
    cycle.run(decider=Decisions(["accept", "accept"]))
    assert cycle.phase == "completed"
    assert len(cycle.trainings) == len(cycle.evaluations) == 2
    assert (
        cycle.trainings[1].request.starting_model
        == cycle.trainings[0].result.candidate_model
    )
    assert cycle.accepted_model.weights_hash != model.weights_hash
    assert file_sha256(initial_path) == model.weights_hash
    assert (
        tmp_path / "actual-outer/train-curation/checkpoint/training_state.pt"
    ).exists()


def test_cli_outer_demo_is_explicitly_synthetic_and_complete(tmp_path, capsys):
    from self_curate.cli import main

    assert main(["outer-demo", "--output", str(tmp_path / "demo")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["phase"] == "completed"
    assert summary["same_task_suite"] and summary["sequential_weights"]
    assert summary["exact_best_training_order"]
    assert summary["training_count"] == summary["validation_count"] == 2
    assert summary["training_executed"] is summary["improvement_verified"] is False
    assert summary["production_acceptance_policy"] is None


def test_strict_json_interfaces_roundtrip_all_outer_handoffs(tmp_path):
    from self_curate.orchestration import parse_outer_contract, JsonCurationDataProvider
    from self_curate.data.pool import write_json

    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    records = [
        cycle.handoff,
        cycle.trainings[0].request.data,
        recipe(),
        cycle.trainings[0],
        cycle.evaluations[0],
        cycle.pending_decision,
    ]
    for record in records:
        raw = json.loads(c.canonical_json(record.to_dict()))
        assert parse_outer_contract(type(record).__name__, raw) == record
        with pytest.raises(c.ContractError):
            parse_outer_contract(
                type(record).__name__, {**raw, "silently_ignored": True}
            )
    data = CurationProvider().prepare(cycle.handoff)
    path = tmp_path / "external-curation.json"
    write_json(path, data.to_dict())
    assert JsonCurationDataProvider(path).prepare(cycle.handoff) == data


def test_actual_adapter_branch_keeps_base_hash_and_frozen_parameters(tmp_path):
    torch, store = tiny_store()
    from self_curate.training.torch_backend import LoadedModel, TorchSFTBackend

    store.checkpoint_kind = "adapter"

    def save_adapter(model, output):
        output.mkdir()
        torch.save(model.projection.state_dict(), output / "adapter.pt")

    def load_adapter(model_ref, checkpoint):
        model = deepcopy(store.initial)
        model.projection.load_state_dict(
            torch.load(checkpoint / "weights/adapter.pt", weights_only=True)
        )
        store.models.append(model)
        return LoadedModel(model, model_ref)

    store.save_branch, store.load_checkpoint = save_adapter, load_adapter
    model = replace(MODEL, adapter_hash=c.digest("initial-synthetic-adapter"))
    service = TrainingService(TorchSFTBackend(store, backend="mock"))
    first = service.train(
        tiny_request(epochs=1, starting_model=model), tmp_path / "adapter-one"
    )
    second = service.train(
        tiny_request(epochs=1, starting_model=first.result.candidate_model),
        tmp_path / "adapter-two",
    )
    assert (
        second.result.candidate_model.weights_hash
        == first.result.candidate_model.weights_hash
        == model.weights_hash
    )
    assert first.result.candidate_model.adapter_hash != model.adapter_hash
    assert (
        second.result.candidate_model.adapter_hash
        != first.result.candidate_model.adapter_hash
    )
    assert all(
        torch.equal(m.embedding.weight, store.initial.embedding.weight)
        for m in store.models
    )


def test_post_execution_report_io_failure_preserves_task_costs(tmp_path, monkeypatch):
    import self_curate.validation.service as module

    suite = TaskSuite(
        "test", fixture_bundle().tasks, (0,), ref("s"), ref("p"), "task_mean", "mock"
    )
    write = module.write_json

    def fail_report(path, value):
        if path.name == "evaluation.json":
            raise OSError("intentional publication failure")
        return write(path, value)

    monkeypatch.setattr(module, "write_json", fail_report)
    from self_curate.validation import TaskValidationFailure

    with pytest.raises(TaskValidationFailure) as error:
        TaskValidator(FixtureRunner()).evaluate(
            MODEL,
            suite,
            attempt_id="a",
            evaluation_id="e",
            output=tmp_path / "failed-report",
        )
    assert len(error.value.rollouts) == 3
    assert sum(e.operation == "rollout" for e in error.value.ledger.events) == 3


def test_supplied_baseline_must_use_exact_model_and_task_protocol(tmp_path):
    cycle, _, _, _, _ = make_cycle(tmp_path)
    cycle.run()
    context = cycle.pending_decision
    with pytest.raises(c.ContractError, match="Baseline"):
        replace(context, baseline=cycle.evaluations[0])


def test_reusing_output_directory_cannot_dispatch_another_physical_training(tmp_path):
    backend = FixtureTrainer()
    service = TrainingService(backend)
    request = tiny_request()
    service.train(request, tmp_path / "one")
    with pytest.raises(FileExistsError):
        service.train(request, tmp_path / "one")
    assert len(backend.requests) == 1


def test_validator_receives_exact_checkpoint_and_cannot_accept_changed_files(tmp_path):
    from self_curate.training.artifacts import checkpoint_fingerprint

    cycle, trainer, runner, _, search = make_cycle(tmp_path)
    original = trainer.train

    def local_checkpoint(request, output):
        result = original(request, output)
        output.mkdir()
        (output / "synthetic-weights.txt").write_text(
            "declared synthetic weights", encoding="utf-8"
        )
        return replace(
            result,
            checkpoint=c.ArtifactRef(
                output.as_uri(), checkpoint_fingerprint(output)[0]
            ),
        )

    trainer.train = local_checkpoint
    cycle.run()
    checkpoint = cycle.trainings[0].result.checkpoint
    assert all(request.checkpoint == checkpoint for request in runner.requests)
    assert cycle.evaluations[0].checkpoint == checkpoint
    context = cycle.pending_decision
    for record in cycle.updated_records:
        assert (
            record.usage.candidate_sft[-1].model_id
            == search.best.request.model_ref.model_hash
        )
    (tmp_path / "outer/train-subset/checkpoint/synthetic-weights.txt").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(c.ContractError, match="changed"):
        cycle.submit_decision(
            AcceptanceDecision(context.record_hash, "accept", "test", ref("decision"))
        )
    assert cycle.accepted_model == search.best.request.model_ref
