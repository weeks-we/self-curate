"""Independent boundaries beyond the ordinary fixture happy paths."""

from dataclasses import replace

import pytest

from helpers_feedback import inputs
from helpers_m2 import MODEL, records
from self_curate import contracts as c
from self_curate.agents import RoleRuntime, RoleTeam, build_feedback, propose_program
from self_curate.agents.demo import FixtureRoleClient
from self_curate.data import MetricView
from self_curate.operators import prepare_selection
from self_curate.operators import BudgetPolicy, ExecutionContext, QualityPolicy
from helpers_m2 import node, program
from self_curate.search import InnerSearch
from self_curate.search.demo import FixtureValidationProvider


def search_fixture():
    request, _, ctx, policy = inputs()
    seed = prepare_selection(
        request.program,
        ctx,
        request.objective,
        backend="mock",
        attempt_id="full-audit",
        invocation_id="seed",
    )
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="full-audit", backend="mock"
    )
    search = InnerSearch(team, seed, request, policy, search_id="full-audit")
    return search, client


@pytest.mark.parametrize(
    "field,value", [("density", 0.99), ("share", 0.01), ("complete_events", 99)]
)
def test_metric_view_recomputes_in_memory_evidence_before_operators_can_use_it(
    field, value
):
    row = replace(records(1)[0], signals=())
    changed = replace(row.evidence_metrics[0], **{field: value})
    row = replace(row, evidence_metrics=(changed, *row.evidence_metrics[1:]))
    with pytest.raises(c.ContractError):
        MetricView((row,))


def test_metric_view_rejects_duplicate_evidence_rows_instead_of_silent_overwrite():
    row = replace(records(1)[0], signals=())
    changed = replace(row.evidence_metrics[0], density=0.99)
    row = replace(row, evidence_metrics=(*row.evidence_metrics, changed))
    with pytest.raises(c.ContractError):
        MetricView((row,))


def test_metric_view_rechecks_supervision_alignment_for_direct_record_api():
    row = replace(records(1)[0], signals=())
    mask = list(row.tokenization.assistant_mask)
    mask[row.tokenization.message_spans[0].start] = True
    mask[row.tokenization.message_spans[1].start] = False
    row = replace(
        row, tokenization=replace(row.tokenization, assistant_mask=tuple(mask))
    )
    with pytest.raises(c.ContractError):
        MetricView((row,))


def test_metric_view_rechecks_evidence_union_count():
    row = replace(records(1)[0], signals=())
    row = replace(
        row,
        stats=replace(
            row.stats, evidence_union_tokens=row.stats.evidence_union_tokens + 1
        ),
    )
    with pytest.raises(c.ContractError):
        MetricView((row,))


def test_failed_write_ahead_checkpoint_does_not_mark_unstarted_job_inflight(
    tmp_path, monkeypatch
):
    search, client = search_fixture()
    search.enable_checkpoints(tmp_path / "journal")
    saved = search.save
    before = search.state
    provider = FixtureValidationProvider((0.1,))

    def denied(*args, **kwargs):
        raise OSError("explicit simulated checkpoint write failure")

    monkeypatch.setattr(search, "save", denied)
    with pytest.raises(OSError):
        search.step(provider)
    assert not provider.calls and not client.calls
    assert search.state == before
    monkeypatch.setattr(search, "save", saved)
    search.step(provider)
    assert len(provider.calls) == 1 and search.state.evaluated_count == 1


def test_failed_post_operation_save_preserves_completed_validation_without_reissuing(
    tmp_path, monkeypatch
):
    search, client = search_fixture()
    search.enable_checkpoints(tmp_path / "journal")
    original = search.save
    saves = 0

    def deny_completion(path):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise OSError("explicit post-operation save failure")
        return original(path)

    provider = FixtureValidationProvider((0.1,))
    monkeypatch.setattr(search, "save", deny_completion)
    with pytest.raises(OSError):
        search.step(provider)
    assert len(provider.calls) == 1 and search.state.phase == "ready"
    assert (
        search.state.evaluated_count == 1
        and search.state.unresolved_operation_json is None
    )
    monkeypatch.setattr(search, "save", original)
    search.save(tmp_path / "recovered")
    loaded = InnerSearch.restore(
        tmp_path / "recovered",
        client,
        search._context.view,
        expected_request=search._initial_request,
        expected_policy=search._policy,
        expected_search_id="full-audit",
        expected_session_id="full-audit",
        expected_attempt_id="full-audit",
    )
    assert loaded.state == search.state and len(provider.calls) == 1


class NonJsonReply:
    model_ref = MODEL

    def generate_json(self, **kwargs):
        return {
            "content": {"invalid_number": float("nan")},
            "usage": c.ResourceUsage(
                c.ComputeCost(2.0, 3.0, 5),
                agent_input_tokens=10,
                agent_output_tokens=11,
            ).to_dict(),
        }


def test_invalid_non_json_role_output_still_preserves_reported_usage():
    runtime = RoleRuntime(
        NonJsonReply(), model=MODEL, session_id="malformed-output", backend="mock"
    )
    with pytest.raises(c.ContractError):
        runtime.generate("leval", {"fixture": "no numeric evaluation"})
    event = runtime.ledger().events[0]
    assert event.status == "failed"
    assert event.usage.compute.gpu_seconds == 3.0
    assert event.usage.agent_input_tokens == 10
    assert event.usage.agent_output_tokens == 11


def test_standalone_lopt_invalid_non_json_output_preserves_each_physical_usage():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=NonJsonReply()
    )
    assert proposal.status == "blocked" and len(attempts) == policy.max_model_attempts
    assert all(
        a.usage.compute.gpu_seconds == 3.0 and a.usage.agent_input_tokens == 10
        for a in attempts
    )


def relabelled_fixture_context():
    # Explicitly relabel a synthetic tokenizer only to test a negative boundary.
    # These token IDs and provenance never become actual research evidence.
    request, _, _, _ = inputs()
    row = replace(records(1)[0], signals=())
    tokenizer = replace(row.tokenization.identity, backend="precomputed")
    row = replace(row, tokenization=replace(row.tokenization, identity=tokenizer))
    model = replace(MODEL, tokenizer=tokenizer)
    ctx = ExecutionContext(
        MetricView((row,)), QualityPolicy(), BudgetPolicy(8, 0, 1), model
    )
    objective = replace(
        request.objective,
        estimator=c.ArtifactRef("artifact://estimator", c.digest("estimator")),
        acceptance_policy=c.ArtifactRef("artifact://policy", c.digest("policy")),
    )
    recipe = program(
        [node("length", "LengthSelect", {"supervised_token_range": [8, 8]})]
    )
    return ctx, recipe, objective


def test_precomputed_declaration_cannot_promote_known_mock_source_to_real_p1():
    ctx, recipe, objective = relabelled_fixture_context()
    with pytest.raises(c.ContractError):
        prepare_selection(
            recipe,
            ctx,
            objective,
            backend="real",
            attempt_id="negative-test",
            invocation_id="never-real",
        )


def test_real_workflow_checks_selected_record_provenance_when_view_is_supplied():
    ctx, recipe, objective = relabelled_fixture_context()
    seed = prepare_selection(
        recipe,
        ctx,
        objective,
        backend="mock",
        attempt_id="negative-test",
        invocation_id="never-real",
    )
    ledger = replace(
        seed.ledger,
        backend="real",
        events=tuple(replace(e, backend="real") for e in seed.ledger.events),
    )
    bundle = replace(seed.workflow, backend="real", ledger=ledger)
    with pytest.raises(c.ContractError):
        c.validate_workflow(bundle, ctx.view)


def test_standalone_optimizer_rejects_previous_cycle_from_different_edit_policy():
    request, result, ctx, policy = inputs()
    first = RoleTeam(
        FixtureRoleClient(request.model_ref),
        model=request.model_ref,
        session_id="original-policy",
        backend="mock",
    )
    previous = first.feedback_step(
        request,
        result,
        ctx,
        policy,
        attempt_id="first",
        invocation_id="first",
        use_model=False,
    )
    second = RoleTeam(
        FixtureRoleClient(request.model_ref),
        model=request.model_ref,
        session_id="changed-policy",
        backend="mock",
    )
    changed = replace(policy, share_step=0.25)
    with pytest.raises(c.ContractError):
        second.lopt.optimize(
            request,
            result,
            changed,
            context=ctx,
            method="rule",
            previous_cycle=previous,
        )
