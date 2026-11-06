"""NICE numeric references, candidate SFT isolation, and history/prompt handoff."""

from dataclasses import replace
from copy import deepcopy
import json
import math
import random

import pytest

from helpers_feedback import inputs
from helpers_m2 import context, node, program, MODEL
from helpers_p0 import fixture_bundle
from test_outer_cycle import FixtureTrainer, FixtureRunner, recipe, ref, tiny_store
from test_inner_search import value, stop_after_evaluations
from self_curate import contracts as c
from self_curate.agents import RoleTeam
from self_curate.agents.demo import FixtureRoleClient
from self_curate.agents.feedback_context import (
    LevalContext,
    ValidationObservation,
    parse_leval_context,
)
from self_curate.data import FixtureTokenizer, normalize_trajectory
from self_curate.operators import prepare_selection
from self_curate.search import InnerSearch
from self_curate.training import TrainingExample, TrainingService
from self_curate.validation import TaskSuite, TaskValidator
from self_curate.valuation import (
    NICERecipe,
    NICEReference,
    NICEScores,
    ReplayTrajectory,
    NICEValidator,
    SFTValidator,
    ValuationFailure,
    create_validator,
)
from self_curate.valuation.common import candidate_data
from self_curate.valuation.io import parse_valuation_contract, load_method
from self_curate.valuation.nice import fresh_adam, count_sketch, score_matrix
from self_curate.valuation.torch_nice import (
    TorchNICEBackend,
    parameter_coordinates,
    state_hash,
)


def suite(seeds=(1, 2)):
    return TaskSuite(
        "inner-development",
        fixture_bundle().tasks,
        seeds,
        ref("sampling"),
        ref("task-protocol"),
        "task_mean",
        "mock",
    )


def reference(tmp_path, model=MODEL, runner=None):
    evaluation = TaskValidator(runner or FixtureRunner(reward=1.0)).evaluate(
        model,
        suite(),
        attempt_id="reference",
        evaluation_id="reference",
        output=tmp_path / "reference",
    )
    replay = TrainingExample("action", c.digest("action"), (1, 2, 3), (-100, 2, 3))
    return NICEReference(
        evaluation,
        tuple(
            ReplayTrajectory(
                r.record_hash, (replay,), ref(r.rollout_id), model.tokenizer
            )
            for r in evaluation.rollouts
        ),
    )


class MatrixBackend:
    backend = "mock"
    implementation_hash = c.digest("explicit-fixture-vector-engine")

    def __init__(self, *, zero=False):
        self.zero, self.calls = zero, []

    def compute(self, request, data, reference, recipe, output):
        self.calls.append(request)
        row = (0.0, 0.0, 0.0) if self.zero else (0.9, 0.0, -0.4)
        return NICEScores(
            request.request_hash,
            data.record_hash,
            reference.record_hash,
            recipe.record_hash,
            tuple(x.data_id for x in data.examples),
            tuple(t.task_id for t in reference.evaluation.suite.tasks),
            tuple(row for _ in data.examples),
            tuple(row for _ in data.examples),
            (0.0, 0.0, 0.0) if self.zero else (1.0, 0.0, 1.0),
            ref("declared-vectors"),
            c.ResourceUsage(new_gradient_samples=len(data.examples)),
            "mock",
        )


def nice_provider(tmp_path, *, zero=False):
    request, _, ctx, policy = inputs()
    engine = MatrixBackend(zero=zero)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=reference(tmp_path),
        recipe=NICERecipe(c.digest("coordinates"), 2, 1e-8, 1.0),
        engine=engine,
        output=tmp_path / "nice",
    )
    request = replace(request, protocol=provider.protocol)
    return provider, request, ctx, policy


def sft_provider(tmp_path, *, runner=None, trainer=None):
    request, _, ctx, policy = inputs()
    runner, trainer = runner or FixtureRunner(), trainer or FixtureTrainer()
    provider = SFTValidator(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(trainer),
        validator=TaskValidator(runner),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "sft",
    )
    return (
        provider,
        replace(request, measure="reward_delta", protocol=provider.protocol),
        ctx,
        policy,
        trainer,
        runner,
    )


def test_nice_fixed_task_denominator_not_mean_direction_renormalization():
    scores = score_matrix(
        ([1.0, 0.0],), ([1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 2.0])
    )
    assert scores == ((1.0, 0.0, -1.0, 0.0),)
    assert score_matrix(([1.0, 0.0],), ([1.0, 0.0], [0.0, 0.0])) == ((1.0, 0.0),)
    assert math.fsum(scores[0]) / 4 == 0


def test_fresh_adam_matches_independent_first_update_and_precedes_projection():
    np = pytest.importorskip("numpy")
    g = np.array([0.01, -2.0, 0.0, 0.4])
    norm = math.sqrt(sum(float(x * x) for x in g))
    clipped = g * min(1.0, 0.2 / (norm + 1e-6))
    expected = [float(x) / (abs(float(x)) + 1e-4) for x in clipped]
    assert fresh_adam(g, 1e-4, 0.2) == pytest.approx(expected)
    # Coordinate hashing is checked independently with integer, wrapped arithmetic.
    mask, projected = 2**64 - 1, [0.0] * 3
    for i, x in enumerate(expected):
        bits = (i + 7 + 0x9E3779B97F4A7C15) & mask
        bits = ((bits ^ (bits >> 30)) * 0xBF58476D1CE4E5B9) & mask
        bits = ((bits ^ (bits >> 27)) * 0x94D049BB133111EB) & mask
        bits ^= bits >> 31
        projected[bits % 3] += x if bits >> 63 == 0 else -x
    assert count_sketch(np.asarray(expected), 3, 7) == pytest.approx(projected)
    assert not np.allclose(
        count_sketch(fresh_adam(g, 1e-4, 0.2), 3, 7),
        fresh_adam(count_sketch(g, 3, 7), 1e-4, 0.2),
    )


def test_nice_sign_matches_analytic_reward_finite_difference():
    # Bernoulli policy, reward=1 for action=1, success NLL=-log(sigmoid(theta)).
    probability = lambda x: 1 / (1 + math.exp(-x))
    theta = 0.3
    p = probability(theta)
    target_nll = (p - 1) / 2  # one success + one reward-zero failure; fixed M=2
    for gradient, direction in ((p - 1, 1), (p, -1)):
        score = score_matrix(([gradient],), ([target_nll],))[0][0]
        change = probability(theta - 1e-4 * gradient) - p
        assert score * direction > 0 and change * direction > 0


def test_nice_provider_member_mean_skill_task_mean_and_constraints(tmp_path):
    provider, request, ctx, _ = nice_provider(tmp_path)
    result = provider.evaluate(request)
    assert result.status == "valid"
    assert {v.skill: v.value for v in result.skill_values} == pytest.approx(
        {"recovery": 0.45, "regression": -0.4}
    )
    assert result.total_value == pytest.approx(0.7 * 0.45 + 0.3 * (-0.4))
    assert result.constraints[0].passed is False
    assert provider.engine.calls[0].batch.training_order == request.batch.training_order
    assert ctx.view.record(request.batch.training_order[0]).usage.candidate_sft == ()


def test_all_zero_nice_target_blocks_ranking_not_zero_score(tmp_path):
    provider, request, _, _ = nice_provider(tmp_path, zero=True)
    result = provider.evaluate(request)
    assert result.status == "invalid" and result.total_value is None
    assert result.reason == "ALL_VALIDATION_DIRECTIONS_ZERO"


@pytest.mark.parametrize(
    "kind", ["NICERecipe", "NICEReference", "NICEScores", "ReplayTrajectory"]
)
def test_valuation_json_roundtrip_and_unknown_fields(tmp_path, kind):
    provider, request, _, _ = nice_provider(tmp_path)
    scores = provider.engine.compute(
        request,
        candidate_data(request, provider.view),
        provider.reference,
        provider.recipe,
        tmp_path,
    )
    obj = {
        "NICERecipe": provider.recipe,
        "NICEReference": provider.reference,
        "NICEScores": scores,
        "ReplayTrajectory": provider.reference.replays[0],
    }[kind]
    assert parse_valuation_contract(kind, obj.to_dict()) == obj
    with pytest.raises(c.ContractError):
        parse_valuation_contract(kind, {**obj.to_dict(), "injected": 1})


@pytest.mark.parametrize("field", ["measure", "protocol", "model_ref"])
def test_methods_reject_wrong_measure_protocol_or_model_before_calls(tmp_path, field):
    provider, request, _, _ = nice_provider(tmp_path)
    changes = {
        "measure": "reward_delta",
        "protocol": ref("other"),
        "model_ref": replace(request.model_ref, revision="other"),
    }
    # ModelRef change is already rejected by the request's batch model binding.
    with pytest.raises(c.ContractError):
        provider.evaluate(replace(request, **{field: changes[field]}))
    assert not provider.engine.calls


def test_nice_reference_requires_every_seed_and_exact_action_count(tmp_path):
    refdata = reference(tmp_path)
    with pytest.raises(c.ContractError):
        replace(refdata, replays=refdata.replays[:-1])
    with pytest.raises(c.ContractError):
        replace(
            refdata,
            replays=(replace(refdata.replays[0], steps=()),) + refdata.replays[1:],
        )
    legacy = replace(refdata.evaluation, execution_requests=(), execution_results=())
    with pytest.raises(c.ContractError):
        replace(refdata, evaluation=legacy)


class DeltaRunner(FixtureRunner):
    def execute(self, request, output):
        self.reward = (
            0.2
            if request.checkpoint is None
            else (0.6 if request.task.task_id != "dev-c" else 0.1)
        )
        return super().execute(request, output)


def test_candidate_sft_delta_exact_data_baseline_once_and_no_promotion(tmp_path):
    provider, request, ctx, _, trainer, runner = sft_provider(
        tmp_path, runner=DeltaRunner()
    )
    before = {i: ctx.view.record(i).record_hash for i in ctx.view.data_ids}
    first, second = provider.evaluate(request), provider.evaluate(request)
    assert first.status == second.status == "valid"
    assert {v.skill: v.value for v in first.skill_values} == pytest.approx(
        {"recovery": 0.4, "regression": -0.1}
    )
    assert len(trainer.requests) == 2 and all(
        r.starting_model == request.model_ref for r in trainer.requests
    )
    assert len(runner.requests) == len(suite().tasks) * len(suite().seeds) * 3
    assert (
        tuple(x.data_id for x in trainer.requests[0].data.examples)
        == request.batch.training_order
    )
    assert trainer.requests[0].training_data_hash == request.batch.training_data_hash
    evidence = json.loads(
        next((tmp_path / "sft").glob("*/evidence.json")).read_text(encoding="utf-8")
    )
    assert (
        sum(
            e["usage"]["training_supervised_tokens"] or 0
            for e in evidence["ledger"]["events"]
        )
        == request.batch.supervised_tokens
    )
    assert provider.model == request.model_ref
    assert all(
        len(r.usage.candidate_sft) == 2 and not r.usage.accepted_sft
        for r in provider.updated_records
    )
    assert {i: ctx.view.record(i).record_hash for i in ctx.view.data_ids} == before
    assert provider.baseline.report.model_ref == request.model_ref


def test_failed_baseline_never_trains_or_manufactures_reward_zero(tmp_path):
    provider, request, _, _, trainer, _ = sft_provider(
        tmp_path, runner=FixtureRunner(fail=True)
    )
    result = provider.evaluate(request)
    assert result.status == "invalid" and result.total_value is None
    assert not trainer.requests and provider.baseline is None
    assert provider.results == [result]


def test_sft_baseline_restore_is_exact_read_only_and_not_recharged(tmp_path):
    provider, request, ctx, _, _, _ = sft_provider(tmp_path, runner=DeltaRunner())
    provider.evaluate(request)
    saved = provider.save_reference(tmp_path / "saved-baseline")
    runner = DeltaRunner()
    restored = SFTValidator.restore_reference(
        tmp_path / "saved-baseline/reference.json",
        expected_hash=saved.sha256,
        expected_protocol=provider.protocol,
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(FixtureTrainer()),
        validator=TaskValidator(runner),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "restored-sft",
    )
    assert restored.baseline == provider.baseline
    assert restored.evaluate(request).status == "valid"
    assert len(runner.requests) == len(suite().tasks) * len(suite().seeds)
    with pytest.raises(c.ContractError):
        SFTValidator.restore_reference(
            tmp_path / "saved-baseline/reference.json",
            expected_hash=c.digest("wrong"),
            expected_protocol=provider.protocol,
        )


def test_failed_candidate_reward_preserves_training_evidence(tmp_path):
    class Runner(DeltaRunner):
        def execute(self, request, output):
            self.fail = request.checkpoint is not None
            return super().execute(request, output)

    provider, request, _, _, trainer, _ = sft_provider(tmp_path, runner=Runner())
    result = provider.evaluate(request)
    assert result.status == "invalid" and result.total_value is None
    assert (
        len(trainer.requests) == 1
        and provider.trainings[0].result.processed_supervised_tokens > 0
    )
    assert provider.updated_records


def test_failed_sft_keeps_training_costs_and_inner_search_failure_usage(tmp_path):
    class FailingTrainer:
        backend = "mock"

        def train(self, request, output):
            from self_curate.training import TrainingFailure

            raise TrainingFailure(
                "known partial training",
                usage=c.ResourceUsage(training_supervised_tokens=7),
            )

    provider, request, ctx, policy, _, _ = sft_provider(
        tmp_path, trainer=FailingTrainer()
    )
    prepared = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="attempt",
        invocation_id="seed",
    )
    team = RoleTeam(
        FixtureRoleClient(request.model_ref),
        model=request.model_ref,
        session_id="failure",
        backend="mock",
    )
    search = InnerSearch(
        team, prepared, request, policy, search_id="failure", use_model=False
    )
    result = search.step(provider)
    assert result.status == "blocked"
    event = next(e for e in search.ledger().events if e.operation == "evaluation")
    # The aggregate remains unknown if a rollout didn't declare this counter;
    # partial known costs must still be preserved in the nested physical ledger.
    failure = next((tmp_path / "sft").glob("*/failure.json"))
    ledger = json.loads(failure.read_text(encoding="utf-8"))["ledger"]
    assert any(e["usage"]["training_supervised_tokens"] == 7 for e in ledger["events"])
    assert event.status == "failed"


def test_inner_search_captures_explicit_backend_failure_usage(tmp_path):
    from test_inner_search import setup

    search, _, _, _, _ = setup(use_model=False)

    class Provider:
        def evaluate(self, request):
            raise ValuationFailure(
                "reported failure", usage=c.ResourceUsage(new_gradient_samples=3)
            )

    search.step(Provider())
    assert (
        next(
            e for e in search.ledger().events if e.operation == "evaluation"
        ).usage.new_gradient_samples
        == 3
    )


def test_history_slots_compare_valid_prior_results_and_arrive_at_lopt(tmp_path):
    request, _, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="history", backend="mock"
    )
    team.leval.evaluate(
        request, value(request, 0.1, result_id="first"), use_model=False
    )
    cycle = team.feedback_step(
        request,
        value(request, 0.3, result_id="second"),
        ctx,
        policy,
        attempt_id="history",
        invocation_id="cycle",
        use_model=True,
    )
    slots = cycle.evaluation.to_dict()["prompt_slots"]
    assert slots["comparison"]["delta_previous_total"] == pytest.approx(0.2)
    assert len(slots["related_history"]) == 1
    leval_call = next(v for v in client.calls if v["role"] == "leval")
    lopt_call = next(v for v in client.calls if v["role"] == "lopt")
    assert leval_call["payload"]["work"]["prompt_slots"] == slots
    assert lopt_call["payload"]["work"]["handoff"]["prompt_slots"] == slots
    assert all(e.role == "leval" for e in team.leval.history)
    assert (
        parse_leval_context(
            json.loads(c.canonical_json(cycle.evaluation.context.to_dict()))
        )
        == cycle.evaluation.context
    )
    from self_curate.agents import load_role_cycle

    team.publish_cycle(cycle, tmp_path / "cycle")
    loaded = load_role_cycle(
        tmp_path / "cycle", ctx.view, expected_request=request, expected_policy=policy
    )
    assert loaded.evaluation.context == cycle.evaluation.context


def test_history_rejects_cross_method_and_never_uses_model_prose_as_scores():
    request, _, _, _ = inputs()
    current = ValidationObservation(request, value(request, 0.1))
    other = replace(request, measure="reward_delta")
    with pytest.raises(c.ContractError):
        LevalContext(current, (ValidationObservation(other, value(other, 0.2)),))
    pending = ValidationObservation(request, value(request, 0.9, status="pending"))
    ctx = LevalContext(current, (pending,))
    from self_curate.agents.leval import build_feedback

    slots = ctx.prompt_slots(build_feedback(request, current.result))
    assert slots["comparison"]["delta_previous_total"] is None


def compact_setup(tmp_path):
    torch, store = tiny_store()
    tokenizer = replace(
        MODEL.tokenizer, tokenizer_id="compact", tokenizer_hash=c.digest("compact")
    )
    model = replace(MODEL, tokenizer=tokenizer)

    class CompactTokenizer:
        identity = tokenizer

        def encode(self, messages, provided=None):
            encoded = FixtureTokenizer().encode(messages)
            return replace(
                encoded,
                identity=tokenizer,
                input_ids=tuple(i % 8 for i in encoded.input_ids),
            )

    rows = tuple(
        normalize_trajectory(
            {
                "schema_version": "1",
                "data_id": "compact-" + str(i),
                "source": {
                    "source_id": "synthetic",
                    "revision": "1",
                    "record_id": str(i),
                    "artifact": ref(str(i)).to_dict(),
                },
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "ab" + str(i)},
                ],
                "quality": {
                    "status": "pass",
                    "method": "manual",
                    "verifier_id": "fixture",
                    "evidence": ref("manual").to_dict(),
                },
            },
            CompactTokenizer(),
            ("recovery", "regression"),
        )
        for i in range(3)
    )
    ctx = context(rows, target=9, tolerance=0, batch=1, model=model, signals=())
    spec = program(
        [node("sample", "SampleSelect", {"method": "uniform", "size": 3, "seed": 7})]
    )
    request, _, _, policy = inputs()
    prepared = prepare_selection(
        spec,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="tiny",
        invocation_id="seed",
    )
    request = replace(
        request,
        model_ref=model,
        program=spec,
        batch=prepared.workflow.batch,
        selection_context_hash=ctx.context_hash,
    )
    return torch, store, request, ctx, policy, prepared


def test_actual_torch_nice_reference_sum_mask_sign_rng_frozen_and_cache(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    refdata = reference(tmp_path, request.model_ref)
    coords, count = parameter_coordinates(store.initial)
    engine = TorchNICEBackend(store, backend="mock")
    rec = NICERecipe(coords, count, 1e-4, 0.2)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=refdata,
        recipe=rec,
        engine=engine,
        output=tmp_path / "nice",
    )
    request = replace(request, protocol=provider.protocol)
    original_state, rng, py_rng = (
        state_hash(store.initial),
        torch.get_rng_state().clone(),
        random.getstate(),
    )
    result = provider.evaluate(request)
    assert result.status == "valid" and result.usage.training_supervised_tokens == 0
    assert state_hash(store.initial) == original_state
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py_rng
    import numpy as np

    vectors = np.load(next((tmp_path / "nice").glob("*/vectors.npz")))
    # Independent CE on only the supervised generated actions, token sum for h.
    independent = deepcopy(store.initial)
    example = refdata.replays[0].steps[0]
    ids = torch.tensor([example.input_ids])
    logits = independent(ids, torch.ones_like(ids)).logits[:, :-1]
    nll = -torch.log_softmax(logits, -1)[0, torch.arange(2), torch.tensor([2, 3])].sum()
    params = tuple(p for p in independent.parameters() if p.requires_grad)
    expected = torch.autograd.grad(nll, params)[0].reshape(-1).numpy()
    assert vectors["targets"][0] == pytest.approx(expected, abs=2e-7)
    sample = candidate_data(request, ctx.view).examples[0]
    logits = independent(
        torch.tensor([sample.input_ids]),
        torch.ones(1, len(sample.input_ids), dtype=torch.long),
    ).logits[0, :-1]
    labels = torch.tensor(sample.labels[1:])
    selected = labels != -100
    independent_loss = -torch.log_softmax(logits[selected], -1)[
        torch.arange(selected.sum()), labels[selected]
    ].mean()
    expected_sample = (
        torch.autograd.grad(independent_loss, params)[0].reshape(-1).numpy()
    )
    assert vectors["raw"][0] == pytest.approx(expected_sample, abs=2e-7)
    first_skill = next(v.value for v in result.skill_values if v.skill == "recovery")
    ref_cosine = math.fsum(
        float(
            np.dot(fresh_adam(s, rec.adam_epsilon, rec.max_grad_norm), expected)
            / np.linalg.norm(fresh_adam(s, rec.adam_epsilon, rec.max_grad_norm))
            / np.linalg.norm(expected)
        )
        for s in vectors["raw"]
    ) / len(vectors["raw"])
    assert first_skill == pytest.approx(ref_cosine, abs=2e-6)
    second = provider.evaluate(request)
    assert (
        second.usage.new_gradient_samples == 0 and second.usage.new_gradient_tokens == 0
    )
    assert second.usage.cache_hits == len(request.batch.data_ids) + len(suite().tasks)
    assert second.total_value == pytest.approx(result.total_value)


def test_actual_candidate_sft_starts_each_time_from_frozen_model(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    from self_curate.training.torch_backend import TorchSFTBackend

    provider = SFTValidator(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(TorchSFTBackend(store, backend="mock")),
        validator=TaskValidator(DeltaRunner()),
        suite=suite(),
        recipe=recipe(optimizer="sgd", accumulation_steps=2),
        output=tmp_path / "sft",
    )
    request = replace(request, measure="reward_delta", protocol=provider.protocol)
    original = state_hash(store.initial)
    provider.evaluate(request)
    provider.evaluate(request)
    assert all(
        r.request.starting_model == request.model_ref for r in provider.trainings
    )
    assert (
        provider.trainings[0].result.candidate_model
        == provider.trainings[1].result.candidate_model
    )
    assert state_hash(store.initial) == original
    assert len(store.loads) == 2


@pytest.mark.parametrize("method", ["nice", "sft"])
def test_full_inner_search_switches_kernels_and_keeps_role_model(tmp_path, method):
    if method == "nice":
        provider, request, ctx, policy = nice_provider(tmp_path)
    else:
        provider, request, ctx, policy, _, _ = sft_provider(
            tmp_path, runner=DeltaRunner()
        )
    initial = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="switch",
        invocation_id="seed",
    )
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(client, model=request.model_ref, session_id=method, backend="mock")
    search = InnerSearch(
        team, initial, request, policy, search_id=method, use_model=False
    )
    # Explicit test-only stop condition; no production stopping threshold supplied.
    search.run(provider, stop_condition=stop_after_evaluations(2))
    assert len(provider.results) == 2
    assert team.runtime.model_ref == request.model_ref
    assert len(search.validations) == 2
    assert search.validations[1].request.protocol == provider.protocol
    search.save(tmp_path / "snapshot")
    restored = InnerSearch.restore(
        tmp_path / "snapshot",
        FixtureRoleClient(request.model_ref),
        ctx.view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id=method,
        expected_session_id=method,
        expected_attempt_id="switch",
        expected_use_model=False,
    )
    assert restored.state == search.state


def test_method_switch_is_strict_and_missing_compute_adapter_never_mock_falls_back(
    tmp_path,
):
    path = tmp_path / "method.json"
    for method in ("nice", "sft"):
        path.write_text(
            json.dumps({"schema_version": "1", "method": method}), encoding="utf-8"
        )
        assert load_method(path) == method
    with pytest.raises(c.ContractError):
        create_validator("typo")
    with pytest.raises(TypeError):
        create_validator("nice")


@pytest.mark.parametrize("link", ["task-id", "task-family", "sample-family"])
def test_candidate_task_isolation_checks_explicit_cross_family_links(tmp_path, link):
    from self_curate.valuation.common import require_isolation
    from self_curate.data import MetricView

    request, _, ctx, _ = inputs()
    data = candidate_data(request, ctx.view)
    selected = data.examples[0].data_id
    row = ctx.view.record(selected)
    tasks = suite().tasks
    if link == "task-id":
        tasks = (replace(tasks[0], task_id=row.source.record_id),) + tasks[1:]
    elif link == "task-family":
        tasks = (replace(tasks[0], family_ids=(row.source.record_id,)),) + tasks[1:]
    else:
        row = replace(row, source=replace(row.source, family_ids=(tasks[0].task_id,)))
    view = MetricView(
        tuple(row if i == selected else ctx.view.record(i) for i in ctx.view.data_ids)
    )
    with pytest.raises(c.ContractError):
        require_isolation(view, data, replace(suite(), tasks=tasks))


def test_nice_bad_return_preserves_usage_and_not_ranked(tmp_path):
    from types import SimpleNamespace

    provider, request, _, _ = nice_provider(tmp_path)
    provider.engine.compute = lambda *a: SimpleNamespace(
        usage=c.ResourceUsage(new_gradient_samples=7)
    )
    with pytest.raises(ValuationFailure) as error:
        provider.evaluate(request)
    assert error.value.usage.new_gradient_samples == 7
    assert not provider.results


def test_nice_rejects_reordered_score_rows_after_cost_preservation(tmp_path):
    provider, request, _, _ = nice_provider(tmp_path)
    original = provider.engine.compute

    def wrong(*args):
        result = original(*args)
        return replace(result, sample_ids=tuple(reversed(result.sample_ids)))

    provider.engine.compute = wrong
    with pytest.raises(ValuationFailure) as error:
        provider.evaluate(request)
    assert error.value.usage.new_gradient_samples == len(request.batch.data_ids)


def test_sft_refuses_self_consistent_training_for_another_physical_request(tmp_path):
    provider, request, _, _, _, _ = sft_provider(tmp_path)
    original = provider.trainer.train

    def wrong(req, output):
        return original(replace(req, training_id="different-job"), output)

    provider.trainer.train = wrong
    with pytest.raises(ValuationFailure):
        provider.evaluate(request)
    failure = json.loads(
        next((tmp_path / "sft").glob("*/failure.json")).read_text(encoding="utf-8")
    )
    assert any(e["operation"] == "training" for e in failure["ledger"]["events"])
    assert not provider.results and provider.model == request.model_ref


def test_nice_wrong_parameter_scope_stops_before_backward(tmp_path):
    _, store, request, ctx, _, _ = compact_setup(tmp_path)
    reference_data = reference(tmp_path, request.model_ref)
    engine = TorchNICEBackend(store, backend="mock")
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=reference_data,
        recipe=NICERecipe(c.digest("wrong-scope"), 1, 1e-8, 1.0),
        engine=engine,
        output=tmp_path / "bad-scope",
    )
    with pytest.raises(ValuationFailure) as error:
        provider.evaluate(replace(request, protocol=provider.protocol))
    assert error.value.usage.new_gradient_samples == 0


def test_nice_target_failed_reward_keeps_rollout_denominator(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)

    class SeedRunner(FixtureRunner):
        def execute(self, req, output):
            self.reward = 1.0 if req.seed == 1 else 0.0
            return super().execute(req, output)

    refdata = reference(tmp_path, request.model_ref, SeedRunner())
    coords, count = parameter_coordinates(store.initial)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=refdata,
        recipe=NICERecipe(coords, count, 1e-8, 1.0),
        engine=TorchNICEBackend(store, backend="mock"),
        output=tmp_path / "half",
    )
    provider.evaluate(replace(request, protocol=provider.protocol))
    import numpy as np

    targets = np.load(next((tmp_path / "half").glob("*/vectors.npz")))["targets"]
    model = deepcopy(store.initial)
    ids = torch.tensor([[1, 2, 3]])
    logits = model(ids, torch.ones_like(ids)).logits[0, :-1]
    loss = -torch.log_softmax(logits, -1)[torch.arange(2), torch.tensor([2, 3])].sum()
    expected = torch.autograd.grad(
        loss, tuple(p for p in model.parameters() if p.requires_grad)
    )[0]
    assert targets[0] == pytest.approx(expected.reshape(-1).numpy() / 2, abs=2e-7)


def test_evaluator_setting_mutation_is_not_silent_protocol_drift(tmp_path):
    provider, request, _, _, trainer, runner = sft_provider(tmp_path)
    provider.recipe = replace(provider.recipe, learning_rate=0.5)
    with pytest.raises(c.ContractError):
        provider.evaluate(request)
    assert not trainer.requests and not runner.requests


def test_history_tool_after_must_match_its_recorded_before_observation():
    from self_curate.agents.feedback_context import context_from_history

    request, _, _, _ = inputs()
    team = RoleTeam(
        FixtureRoleClient(request.model_ref),
        model=request.model_ref,
        session_id="history-binding",
        backend="mock",
    )
    team.leval.evaluate(request, value(request, 0.1), use_model=False)
    before, after = team.leval.history
    data = after.payload
    data["result"]["summary"] = "changed numeric interpretation"
    forged = replace(after, payload_json=c.canonical_json(data))
    with pytest.raises(c.ContractError):
        context_from_history(request, value(request, 0.2), (before, forged))


def test_nice_detects_trainable_scope_drift_even_if_weight_values_unchanged(tmp_path):
    _, store, request, ctx, _, _ = compact_setup(tmp_path)
    load = store.load_branch

    def changing_scope(model_ref):
        loaded = load(model_ref)
        forward = loaded.model.forward

        def changed(*args, **kwargs):
            result = forward(*args, **kwargs)
            loaded.model.embedding.weight.requires_grad_(True)
            return result

        loaded.model.forward = changed
        return loaded

    store.load_branch = changing_scope
    coords, count = parameter_coordinates(store.initial)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=reference(tmp_path, request.model_ref),
        recipe=NICERecipe(coords, count, 1e-8, 1.0),
        engine=TorchNICEBackend(store, backend="mock"),
        output=tmp_path / "drift",
    )
    with pytest.raises(ValuationFailure, match="trainable scope"):
        provider.evaluate(replace(request, protocol=provider.protocol))


@pytest.mark.parametrize("method", ["nice", "sft"])
def test_leval_demo_cli_has_history_and_honest_execution_flags(
    tmp_path, method, capsys
):
    from pathlib import Path
    from self_curate.cli import main

    assert (
        main(
            [
                "leval-demo",
                "--method",
                method,
                "--output",
                str(tmp_path / method),
                "--project-root",
                str(Path(__file__).parents[1]),
            ]
        )
        == 0
    )
    raw = json.loads(capsys.readouterr().out)
    assert raw["evaluated_count"] == 3 and raw["history_comparisons"] >= 1
    assert raw["shared_model_unchanged"]
    assert not raw["real_task_scoring"] and not raw["actual_sft_executed"]
