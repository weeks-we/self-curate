"""Cross-module regression tests discovered by the full interface review."""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from helpers_feedback import inputs
from helpers_p0 import fixture_bundle, ar
from test_feedback_lopt import Client, envelope
from self_curate import contracts as c
from self_curate.agents import (
    build_feedback,
    propose_program,
    result_from_evaluation,
    feedback_step_to_directory,
)
from self_curate.data.pool import build_pool, load_config, write_json
from self_curate.operators import prepare_to_directory, verify_preparation

ROOT = Path(__file__).parents[1]


def published_step(tmp_path, *, policy_override=None):
    request, result, _, policy = inputs()
    policy = policy_override or policy
    pool = tmp_path / "pool"
    parent = tmp_path / "parent"
    output = tmp_path / "step"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    paths = []
    for name, value in [
        ("request", request),
        ("result", result),
        ("policy", policy),
        ("program", request.program),
    ]:
        path = tmp_path / (name + ".json")
        write_json(path, value.to_dict())
        paths.append(path)
    prepare_to_directory(
        pool,
        paths[-1],
        ROOT / "configs/m2.fixture.json",
        ROOT / "configs/p1.objective.fixture.json",
        parent,
        backend="mock",
        attempt_id="a",
        invocation_id="parent",
    )
    step = feedback_step_to_directory(
        parent, pool, *paths[:3], output, attempt_id="a", invocation_id="first"
    )
    return request, result, pool, parent, output, step


def test_candidate_handoff_is_directly_consumable_and_two_steps_chain(tmp_path):
    request, result, pool, _, output, step = published_step(tmp_path)
    from self_curate.data import read_pool

    candidate_prepared = output / "candidate_prepared"
    assert candidate_prepared.is_dir()
    receipt = verify_preparation(
        candidate_prepared,
        read_pool(pool).metric_view(),
        expected_batch_id=step.next_request.batch.batch_id,
    )
    assert receipt["selection_status"] == "feasible"
    next_result = replace(
        result,
        result_id="second-external-result",
        request_hash=step.next_request.request_hash,
    )
    result_path = tmp_path / "next-result.json"
    write_json(result_path, next_result.to_dict())
    second = feedback_step_to_directory(
        candidate_prepared,
        pool,
        output / "next_validation_request.json",
        result_path,
        output / "edit_policy.json",
        tmp_path / "step2",
        attempt_id="a",
        invocation_id="second",
    )
    assert second.request.batch == step.next_request.batch
    assert second.proposal.program.parent_id == step.proposal.program.program_id
    assert second.status == "awaiting_validation"
    assert second.ledger.events[-1].event_id != step.ledger.events[-1].event_id
    assert second.next_request.model_ref == request.model_ref
    for name, expected in json.loads((output / "checksums.json").read_text()).items():
        from self_curate.data.pool import file_sha256

        assert file_sha256(output / name) == expected


def test_evaluation_adapter_rejects_coverage_from_another_batch():
    bundle = fixture_bundle()
    request, _, _, _ = inputs()
    request = replace(
        request,
        program=bundle.program,
        batch=bundle.batch,
        selection_context_hash=bundle.selection_context_hash,
    )
    report = bundle.evaluations[0]
    wrong = tuple("other-" + i for i in report.coverage.required_ids)
    report = replace(report, coverage=c.GradientCoverage(wrong, wrong, ()))
    with pytest.raises(c.ContractError):
        result_from_evaluation(request, report, ar("external-report"))


def test_rule_policy_rejection_preserves_failure_instead_of_raising(tmp_path):
    *_, policy = inputs()
    _, _, _, _, output, step = published_step(
        tmp_path, policy_override=replace(policy, max_nodes=1)
    )
    assert step.status == "proposal_blocked"
    assert len(step.attempts) == 1 and step.attempts[0].error_json
    assert step.ledger.events[-1].status == "failed"
    assert (output / "lopt_attempts.json").is_file()


def test_rule_call_request_hash_binds_saved_request():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    _, attempts = propose_program(request, result, feedback, policy)
    attempt = attempts[0].to_dict()
    assert attempt["request_hash"] == c.digest(attempt["request"])


def test_renamed_seed_node_cannot_bypass_lock():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    response = envelope(request, feedback)
    node = next(n for n in response["program"]["nodes"] if n["id"] == "b")
    node["id"] = "renamed-b"
    node["params"]["seed"] += 1
    mix = next(n for n in response["program"]["nodes"] if n["id"] == "mix")
    mix["inputs"] = ["a", "renamed-b"]
    mix["params"]["branch_token_shares"] = {"a": 0.375, "renamed-b": 0.625}
    client = Client(request, [{"content": response}] * policy.max_model_attempts)
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "blocked" and all(a.error_json for a in attempts)


def test_client_model_switch_during_call_blocks_without_retrying_other_weights():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)

    class SwitchingClient(Client):
        def generate_json(self, **kwargs):
            response = super().generate_json(**kwargs)
            self.model_ref = replace(
                self.model_ref, weights_hash=c.digest("another-checkpoint")
            )
            return response

    client = SwitchingClient(
        request, [{"content": envelope(request, feedback)}] * policy.max_model_attempts
    )
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "blocked" and len(client.calls) == len(attempts) == 1
    assert json.loads(attempts[0].error_json)["code"] == "STALE_CACHE"
    assert attempts[0].response_json is not None


def test_service_supplies_full_frozen_context_to_model():
    from self_curate.agents import run_feedback_step

    request, result, ctx, policy = inputs()
    feedback = build_feedback(request, result)
    client = Client(request, [{"content": envelope(request, feedback)}])
    step = run_feedback_step(
        request,
        result,
        ctx,
        policy,
        method="model",
        client=client,
        attempt_id="a",
        invocation_id="i",
    )
    assert step.status == "awaiting_validation"
    supplied = client.calls[0]["payload"]["selection_context"]
    assert c.canonical_json(supplied) == c.canonical_json(ctx.role_summary())
    assert supplied["budget_pack"] == ctx.budget.to_dict()


def test_model_context_uses_semantic_artifact_summary_without_materializing_pairs(
    monkeypatch,
):
    from self_curate.operators import SimilarityPair, SimilarityTable, execute_program

    request, result, ctx, policy = inputs()
    left, right = (ctx.view.record(i) for i in ctx.view.data_ids[:2])
    table = SimilarityTable(
        ar("semantic-table"),
        (
            SimilarityPair(
                left.data_id, right.data_id, left.content_hash, right.content_hash, 0.5
            ),
        ),
    )
    ctx = replace(ctx, similarities=table)
    request = replace(
        request,
        selection_context_hash=ctx.context_hash,
        batch=execute_program(request.program, ctx).batch,
    )
    result = replace(result, request_hash=request.request_hash)
    feedback = build_feedback(request, result)
    client = Client(request, [{"content": envelope(request, feedback)}])

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not materialize pairwise features for a role prompt")

    monkeypatch.setattr(SimilarityTable, "to_dict", forbidden)
    proposal, _ = propose_program(
        request, result, feedback, policy, method="model", client=client, context=ctx
    )
    assert proposal.status == "proposed"
    prompt = client.calls[0]["payload"]["selection_context"]
    assert prompt["context_hash"] == ctx.context_hash
    assert prompt["similarities"] == {
        "identity": table.identity.to_dict(),
        "pair_count": 1,
    }
    assert "pairs" not in prompt["similarities"]


def test_proposal_ids_distinguish_different_model_outputs():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    response = envelope(request, feedback)
    another = json.loads(json.dumps(response))
    another["rationale"] = "Another independently generated explanation"
    first, _ = propose_program(
        request,
        result,
        feedback,
        policy,
        method="model",
        client=Client(request, [{"content": response}]),
    )
    second, _ = propose_program(
        request,
        result,
        feedback,
        policy,
        method="model",
        client=Client(request, [{"content": another}]),
    )
    assert first.proposal_id != second.proposal_id


def test_publish_already_executed_selection_does_not_rerun(tmp_path, monkeypatch):
    from self_curate.operators import (
        prepare_selection,
        publish_preparation,
        load_preparation,
    )
    import self_curate.operators.preparation as module

    request, _, ctx, _ = inputs()
    prepared = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="a",
        invocation_id="i",
    )
    saved = module.execute_program

    def forbidden(*args, **kwargs):
        raise AssertionError("Publication must not execute selectors again")

    monkeypatch.setattr(module, "execute_program", forbidden)
    assert publish_preparation(prepared, tmp_path / "prepared") is prepared
    monkeypatch.setattr(module, "execute_program", saved)
    restored = load_preparation(
        tmp_path / "prepared",
        ctx.view,
        expected_model=request.model_ref,
        expected_objective=request.objective,
        expected_batch_id=request.batch.batch_id,
    )
    assert restored == prepared and restored.ledger == prepared.ledger


@pytest.mark.parametrize(
    "change",
    [
        "next_protocol",
        "missing_cost",
        "false_diff",
        "shared_memory",
        "missing_candidate",
    ],
)
def test_feedback_step_rejects_self_inconsistent_aggregate(change):
    from self_curate.agents import run_feedback_step

    request, result, ctx, policy = inputs()
    step = run_feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    with pytest.raises(c.ContractError):
        if change == "next_protocol":
            replace(
                step,
                next_request=replace(
                    step.next_request, protocol=ar("unrelated-protocol")
                ),
            )
        elif change == "missing_cost":
            replace(step, ledger=replace(step.ledger, events=step.ledger.events[:-1]))
        elif change == "false_diff":
            replace(
                step,
                proposal=replace(
                    step.proposal, changes=c.ProgramChange((), (), (), False, False)
                ),
            )
        elif change == "shared_memory":
            replace(
                step,
                proposal=replace(
                    step.proposal, memory_namespace=step.feedback.memory_namespace
                ),
            )
        else:
            replace(step, candidate=None, next_request=None)


@pytest.mark.parametrize("outcome", ["infeasible", "failed"])
def test_p1_publication_roundtrips_nonfeasible_states(tmp_path, outcome):
    from self_curate.operators import (
        prepare_selection,
        publish_preparation,
        load_preparation,
    )
    from helpers_m2 import context, records

    request, _, ctx, _ = inputs()
    if outcome == "infeasible":
        ctx = context(records(), target=65, tolerance=0, batch=8, share_tolerance=0)
    else:
        ctx = context(
            tuple(replace(row, signals=()) for row in records()),
            target=64,
            tolerance=0,
            batch=8,
            share_tolerance=0,
        )
    result = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="a",
        invocation_id="i",
    )
    assert result.status == outcome
    publish_preparation(result, tmp_path / "prepared")
    assert load_preparation(tmp_path / "prepared", ctx.view) == result
    assert not (tmp_path / "prepared/workflow.json").exists()


def test_real_request_rejects_mock_objective_artifacts():
    request, _, _, _ = inputs()
    tokenizer = replace(request.model_ref.tokenizer, backend="precomputed")
    model = replace(request.model_ref, tokenizer=tokenizer)
    batch = replace(request.batch, model_hash=model.model_hash, synthetic=None)
    with pytest.raises(c.ContractError):
        replace(
            request,
            model_ref=model,
            batch=batch,
            protocol=c.ArtifactRef("artifact://real-protocol", c.digest("protocol")),
            backend="real",
        )


def test_real_p1_rejects_registered_fixture_cdf_context():
    # Deliberately relabel synthetic metadata to exercise a negative boundary;
    # these token IDs remain fixture-only and never become research evidence.
    from self_curate.data import MetricView
    from self_curate.operators import ExecutionContext, SignalContext, prepare_selection

    request, _, ctx, _ = inputs()
    tokenizer = replace(request.model_ref.tokenizer, backend="precomputed")
    model = replace(request.model_ref, tokenizer=tokenizer)
    rows = tuple(
        replace(
            row, tokenization=replace(row.tokenization, identity=tokenizer), signals=()
        )
        for row in (ctx.view.record(i) for i in ctx.view.data_ids)
    )
    signals = tuple(
        SignalContext(
            s.spec, tuple(replace(r, model_hash=model.model_hash) for r in s.references)
        )
        for s in ctx.signals
    )
    forged = ExecutionContext(MetricView(rows), ctx.quality, ctx.budget, model, signals)
    objective = replace(
        request.objective, estimator=ar("estimator"), acceptance_policy=ar("policy")
    )
    objective = replace(
        objective,
        estimator=replace(objective.estimator, uri="artifact://estimator"),
        acceptance_policy=replace(objective.acceptance_policy, uri="artifact://policy"),
    )
    with pytest.raises(c.ContractError):
        prepare_selection(
            request.program,
            forged,
            objective,
            backend="real",
            attempt_id="negative-only",
            invocation_id="not-a-model-run",
        )


@pytest.mark.parametrize(
    "field", ["quality_counts", "missing_skill_annotations", "backend"]
)
def test_pool_read_checks_all_audit_labels(tmp_path, field):
    from self_curate.data import read_pool
    from self_curate.data.pool import file_sha256

    pool = tmp_path / "pool"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    audit = json.loads((pool / "audit.json").read_text())
    if field == "quality_counts":
        audit[field]["pass"] += 1
    elif field == "missing_skill_annotations":
        audit[field]["recovery"] += 1
    else:
        audit[field] = "not-the-actual-backend"
    write_json(pool / "audit.json", audit)
    manifest = json.loads((pool / "manifest.json").read_text())
    manifest["artifacts"]["audit.json"]["sha256"] = file_sha256(pool / "audit.json")
    manifest["pool_id"] = c.digest(
        {k: v for k, v in manifest.items() if k != "pool_id"}
    )
    write_json(pool / "manifest.json", manifest)
    with pytest.raises(c.ContractError):
        read_pool(pool)
