"""Second review: document promises versus versioned kernels and handoffs."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from helpers_feedback import inputs
from test_outer_cycle import (
    FixtureTrainer,
    FixtureRunner,
    recipe,
    tiny_request,
    tiny_store,
)
from test_valuation import compact_setup, reference, sft_provider, suite
from self_curate import contracts as c
from self_curate.training import TrainingService
from self_curate.training.torch_backend import TorchSFTBackend
from self_curate.validation import TaskValidator
from self_curate.valuation import (
    NICERecipe,
    NICEValidator,
    SFTValidator,
    ValuationFailure,
)
from self_curate.valuation.torch_nice import TorchNICEBackend, parameter_coordinates


def test_sft_protocol_distinguishes_backend_implementation_versions(tmp_path):
    original, request, ctx, _, _, _ = sft_provider(tmp_path)

    class OtherTrainer(FixtureTrainer):
        implementation_hash = c.digest("different-training-implementation")

    class OtherRunner(FixtureRunner):
        implementation_hash = c.digest("different-task-runner-implementation")

    new = SFTValidator(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(OtherTrainer()),
        validator=TaskValidator(OtherRunner()),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "new",
    )
    # Isolate implementation versioning from the distinct automatic-baseline namespace.
    new._reference_namespace = getattr(original, "_reference_namespace", None)
    assert original.protocol != new.protocol


def test_sft_protocol_rejects_runtime_runner_replacement_before_execution(tmp_path):
    provider, request, _, _, trainer, runner = sft_provider(tmp_path)

    class DifferentRunner(FixtureRunner):
        implementation_hash = c.digest("changed-runner")

    provider.validator.runner = DifferentRunner()
    with pytest.raises(c.ContractError):
        provider.evaluate(request)
    assert not trainer.requests and not runner.requests


def test_nice_owns_autocast_policy_and_matches_independent_default_context(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    ref = reference(tmp_path, request.model_ref)
    coordinates, count = parameter_coordinates(store.initial)
    rec = NICERecipe(coordinates, count, 1e-4, 0.2)
    results = []
    for label, enabled in (("plain", False), ("ambient-amp", True)):
        provider = NICEValidator(
            model=request.model_ref,
            view=ctx.view,
            reference=ref,
            recipe=rec,
            engine=TorchNICEBackend(store, backend="mock"),
            output=tmp_path / label,
        )
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            results.append(
                provider.evaluate(replace(request, protocol=provider.protocol))
            )
    import numpy as np

    plain = np.load(next((tmp_path / "plain").glob("*/vectors.npz")))
    amp = np.load(next((tmp_path / "ambient-amp").glob("*/vectors.npz")))
    np.testing.assert_allclose(plain["raw"], amp["raw"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(plain["targets"], amp["targets"], atol=1e-8, rtol=1e-8)
    assert results[0].total_value == pytest.approx(results[1].total_value, abs=1e-10)


def test_sft_owns_autocast_policy_and_keeps_equal_parameter_updates(tmp_path):
    torch, store = tiny_store()
    request = tiny_request(epochs=1)
    states = []
    for label, enabled in (("plain", False), ("ambient-amp", True)):
        backend = TorchSFTBackend(store, backend="mock")
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            TrainingService(backend).train(request, tmp_path / label)
        states.append(deepcopy(store.models[-1].state_dict()))
    for name in states[0]:
        torch.testing.assert_close(
            states[0][name], states[1][name], atol=1e-8, rtol=1e-8
        )


def test_restoring_sft_baseline_also_preserves_candidate_exposure_history(tmp_path):
    provider, request, ctx, _, _, _ = sft_provider(tmp_path)
    provider.evaluate(request)
    anchor = provider.save_reference(tmp_path / "saved")
    restored = SFTValidator.restore_reference(
        tmp_path / "saved/reference.json",
        expected_hash=anchor.sha256,
        expected_protocol=provider.protocol,
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(FixtureTrainer()),
        validator=TaskValidator(FixtureRunner()),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "restored",
    )
    restored.evaluate(request)
    assert all(len(r.usage.candidate_sft) == 2 for r in restored.updated_records)
    old = {
        u.attempt_id for r in provider.updated_records for u in r.usage.candidate_sft
    }
    assert old.issubset(
        {u.attempt_id for r in restored.updated_records for u in r.usage.candidate_sft}
    )


def test_oversize_sft_recipe_is_rejected_before_baseline_or_training_cost(tmp_path):
    request, _, ctx, _ = inputs()
    trainer, runner = FixtureTrainer(), FixtureRunner()
    provider = SFTValidator(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(trainer),
        validator=TaskValidator(runner),
        suite=suite(),
        recipe=recipe(max_sequence_tokens=2),
        output=tmp_path / "sft",
    )
    with pytest.raises(c.ContractError):
        provider.evaluate(
            replace(request, measure="reward_delta", protocol=provider.protocol)
        )
    assert not trainer.requests and not runner.requests


def test_bad_baseline_return_keeps_reported_usage_and_unknown_fields(tmp_path):
    provider, request, _, _, _, _ = sft_provider(tmp_path)
    provider.validator.evaluate = lambda *a, **k: SimpleNamespace(
        usage=c.ResourceUsage(tool_calls=19)
    )
    with pytest.raises(ValuationFailure) as error:
        provider.evaluate(request)
    assert error.value.usage.tool_calls == 19
    assert error.value.usage.training_supervised_tokens is None


def test_unknown_baseline_failure_is_not_reported_as_zero_cost(tmp_path):
    provider, request, _, _, _, _ = sft_provider(tmp_path)

    def fail(*a, **k):
        raise RuntimeError("remote status unavailable")

    provider.validator.evaluate = fail
    with pytest.raises(ValuationFailure) as error:
        provider.evaluate(request)
    assert error.value.usage.compute.processed_tokens is None
    assert error.value.usage.tool_calls is None


def test_training_exception_without_ledger_keeps_reported_partial_usage(tmp_path):
    provider, request, _, _, _, _ = sft_provider(tmp_path)

    def fail(*a, **k):
        raise ValuationFailure(
            "partial external training",
            usage=c.ResourceUsage(training_supervised_tokens=7),
        )

    provider.trainer.train = fail
    with pytest.raises(ValuationFailure):
        provider.evaluate(request)
    evidence = json.loads(
        next((tmp_path / "sft").glob("*/failure.json")).read_text(encoding="utf-8")
    )
    assert any(
        e["usage"]["training_supervised_tokens"] == 7
        for e in evidence["ledger"]["events"]
    )


def test_invalid_baseline_evidence_contains_the_actual_attempted_report(tmp_path):
    provider, request, _, _, _, _ = sft_provider(
        tmp_path, runner=FixtureRunner(fail=True)
    )
    result = provider.evaluate(request)
    assert result.status == "invalid" and provider.baseline is None
    evidence = json.loads(
        next((tmp_path / "sft").glob("*/evidence.json")).read_text(encoding="utf-8")
    )
    assert evidence["attempted_baseline"]["report"]["status"] == "invalid"


def test_model_factory_is_part_of_real_torch_implementation_identity(tmp_path):
    torch, source = tiny_store()
    from self_curate.training.torch_backend import StateDictModelStore
    from self_curate.data.pool import file_sha256

    path = tmp_path / "initial.pt"
    torch.save(source.initial.state_dict(), path)
    request, _, _, _ = inputs()
    model = replace(request.model_ref, weights_hash=file_sha256(path))

    class OtherFactory(type(source.initial)):
        pass

    original = StateDictModelStore(
        type(source.initial), initial_model=model, initial_state=path
    )
    different = StateDictModelStore(
        OtherFactory, initial_model=model, initial_state=path
    )
    assert (
        TorchNICEBackend(original, backend="mock").implementation_hash
        != TorchNICEBackend(different, backend="mock").implementation_hash
    )
    assert (
        TorchSFTBackend(original, backend="mock").implementation_hash
        != TorchSFTBackend(different, backend="mock").implementation_hash
    )


def test_numeric_policy_change_invalidates_nice_protocol_before_model_load(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    coords, count = parameter_coordinates(store.initial)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=reference(tmp_path, request.model_ref),
        recipe=NICERecipe(coords, count, 1e-8, 1.0),
        engine=TorchNICEBackend(store, backend="mock"),
        output=tmp_path / "policy",
    )
    request = replace(request, protocol=provider.protocol)
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high" if old != "high" else "highest")
        with pytest.raises(c.ContractError):
            provider.evaluate(request)
        assert not store.loads
    finally:
        torch.set_float32_matmul_precision(old)


def test_overlapping_batches_reuse_only_compatible_sample_gradients(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    from helpers_m2 import context, node, program
    from self_curate.operators import prepare_selection

    coords, count = parameter_coordinates(store.initial)
    provider = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=reference(tmp_path, request.model_ref),
        recipe=NICERecipe(coords, count, 1e-8, 1.0),
        engine=TorchNICEBackend(store, backend="mock"),
        output=tmp_path / "overlap",
    )
    provider.evaluate(replace(request, protocol=provider.protocol))
    smaller = context(
        tuple(ctx.view.record(i) for i in ctx.view.data_ids),
        target=6,
        tolerance=0,
        batch=1,
        model=request.model_ref,
        signals=(),
    )
    spec = program(
        [node("sample", "SampleSelect", {"method": "uniform", "size": 2, "seed": 7})]
    )
    prepared = prepare_selection(
        spec,
        smaller,
        request.objective,
        backend="mock",
        attempt_id="overlap",
        invocation_id="smaller",
    )
    next_request = replace(
        request,
        request_id="overlap-2",
        program=spec,
        batch=prepared.workflow.batch,
        selection_context_hash=smaller.context_hash,
        protocol=provider.protocol,
    )
    result = provider.evaluate(next_request)
    assert result.usage.new_gradient_samples == 0
    assert result.usage.cache_hits == 2 + len(provider.reference.evaluation.suite.tasks)
    # Projection changes must not reuse the prior feature coordinates.
    changed = NICEValidator(
        model=request.model_ref,
        view=ctx.view,
        reference=provider.reference,
        recipe=replace(provider.recipe, projection_dimension=7, projection_seed=11),
        engine=provider.engine,
        output=tmp_path / "changed",
    )
    assert (
        changed.evaluate(
            replace(next_request, protocol=changed.protocol)
        ).usage.new_gradient_samples
        == 2
    )


def test_reference_restore_does_not_reverse_existing_terminal_training_use(tmp_path):
    from self_curate.data import MetricView, resolve_training

    provider, request, ctx, _, _, _ = sft_provider(tmp_path)
    provider.evaluate(request)
    anchor = provider.save_reference(tmp_path / "terminal-save")
    changed = {
        r.data_id: resolve_training(r, r.usage.candidate_sft[-1].attempt_id, "rejected")
        for r in provider.updated_records
    }
    view = MetricView(
        tuple(changed.get(i, ctx.view.record(i)) for i in ctx.view.data_ids)
    )
    restored = SFTValidator.restore_reference(
        tmp_path / "terminal-save/reference.json",
        expected_hash=anchor.sha256,
        expected_protocol=provider.protocol,
        model=request.model_ref,
        view=view,
        trainer=TrainingService(FixtureTrainer()),
        validator=TaskValidator(FixtureRunner()),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "terminal-restore",
    )
    assert {r.data_id for r in restored.updated_records} == set(changed)
    assert all(
        r.usage.candidate_sft[-1].decision == "rejected"
        for r in restored.updated_records
    )


def test_reference_restore_refuses_exposures_for_changed_data_content(tmp_path):
    from self_curate.data import MetricView, normalize_trajectory, FixtureTokenizer
    from helpers_m2 import raw_rows

    provider, request, ctx, _, _, _ = sft_provider(tmp_path)
    provider.evaluate(request)
    anchor = provider.save_reference(tmp_path / "changed-save")
    changed_id = provider.updated_records[0].data_id
    raw = next(r for r in raw_rows() if r["data_id"] == changed_id)
    raw["messages"][1]["content"] = "xyz-0000"
    raw.pop("signals", None)  # Old-content scores cannot be attached to new content.
    new_record = normalize_trajectory(
        raw, FixtureTokenizer(), ("recovery", "regression")
    )
    rows = tuple(
        new_record if i == changed_id else ctx.view.record(i) for i in ctx.view.data_ids
    )
    with pytest.raises(c.ContractError):
        SFTValidator.restore_reference(
            tmp_path / "changed-save/reference.json",
            expected_hash=anchor.sha256,
            expected_protocol=provider.protocol,
            model=request.model_ref,
            view=MetricView(rows),
            trainer=TrainingService(FixtureTrainer()),
            validator=TaskValidator(FixtureRunner()),
            suite=suite(),
            recipe=recipe(),
            output=tmp_path / "changed-restore",
        )


def declared_real_reference(tmp_path):
    """Negative-domain contract fixtures only: no real model/task execution occurs."""
    from test_valuation import ref
    from self_curate.validation import TaskRunResult

    def clean(a):
        return c.ArtifactRef("record://explicit-negative-fixture/" + a.sha256, a.sha256)

    request, _, _, _ = inputs()
    model = replace(
        request.model_ref,
        tokenizer=replace(request.model_ref.tokenizer, backend="precomputed"),
    )
    source = suite()
    tasks = tuple(
        replace(
            t,
            backend="real",
            payload=clean(t.payload),
            environment=clean(t.environment),
            scorer=clean(t.scorer),
        )
        for t in source.tasks
    )
    declared = replace(
        source,
        tasks=tasks,
        backend="real",
        sampling=clean(source.sampling),
        protocol=clean(source.protocol),
    )

    class MetadataRunner:
        backend = "real"

        def execute(self, req, output):
            return TaskRunResult(
                req.record_hash,
                clean(ref("synthetic-actions")),
                clean(ref("synthetic-context")),
                c.digest("tokens"),
                c.digest("mask"),
                2,
                1.0,
                "completed",
                "negative-fixture-only",
                c.ResourceUsage(),
            )

    return TaskValidator(MetadataRunner()).evaluate(
        model,
        declared,
        attempt_id="negative-fixture",
        evaluation_id="negative-fixture",
        output=tmp_path / "declared",
    )


@pytest.mark.parametrize("uri", [" fixture://replay", "\tMOCK://replay"])
def test_real_reference_rejects_whitespace_disguised_mock_replay(tmp_path, uri):
    from self_curate.valuation import NICEReference, ReplayTrajectory
    from self_curate.training import TrainingExample

    evaluated = declared_real_reference(tmp_path)
    step = TrainingExample("action", c.digest("action"), (1, 2, 3), (-100, 2, 3))
    replays = tuple(
        ReplayTrajectory(
            r.record_hash,
            (step,),
            c.ArtifactRef(uri, c.digest(uri)),
            evaluated.report.model_ref.tokenizer,
        )
        for r in evaluated.rollouts
    )
    with pytest.raises(c.ContractError, match="Mock replay"):
        NICEReference(evaluated, replays)


def test_new_automatic_baseline_cannot_impersonate_an_existing_search_reference(
    tmp_path,
):
    original, request, ctx, _, _, _ = sft_provider(tmp_path)
    runner = FixtureRunner(reward=0.9)
    other = SFTValidator(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(FixtureTrainer()),
        validator=TaskValidator(runner),
        suite=suite(),
        recipe=recipe(),
        output=tmp_path / "new-reference",
    )
    assert original.protocol != other.protocol
    with pytest.raises(c.ContractError):
        other.evaluate(request)
    assert not runner.requests


def test_importing_the_same_explicit_baseline_has_stable_reference_identity(tmp_path):
    original, request, ctx, _, _, _ = sft_provider(tmp_path)
    original.evaluate(request)
    settings = dict(
        model=request.model_ref,
        view=ctx.view,
        trainer=TrainingService(FixtureTrainer()),
        validator=TaskValidator(FixtureRunner()),
        suite=suite(),
        recipe=recipe(),
        baseline=original.baseline,
    )
    first = SFTValidator(**settings, output=tmp_path / "first-import")
    second = SFTValidator(**settings, output=tmp_path / "second-import")
    assert first.protocol == second.protocol


def test_real_sft_adapter_must_supply_an_explicit_execution_identity(tmp_path):
    evaluated = declared_real_reference(tmp_path)

    class NoIdentityTrainer:
        backend = "real"

    class NoIdentityRunner:
        backend = "real"

    with pytest.raises(c.ContractError, match="implementation_hash"):
        SFTValidator(
            model=evaluated.report.model_ref,
            view=None,
            trainer=TrainingService(NoIdentityTrainer()),
            validator=TaskValidator(NoIdentityRunner()),
            suite=evaluated.suite,
            recipe=recipe(),
            output=tmp_path / "missing-identity",
        )
