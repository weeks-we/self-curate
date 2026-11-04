"""Pass 1: original framework invariants and independent consumer anchors."""

from dataclasses import replace

import pytest

from helpers_feedback import inputs
from self_curate import contracts as c
from self_curate.agents import RoleTeam
from self_curate.agents.demo import FixtureRoleClient
from self_curate.operators import execute_program, prepare_selection


def ready():
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client,
        model=request.model_ref,
        session_id="review",
        attempt_id="a",
        backend="mock",
    )
    cycle = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="first"
    )
    return team, client, request, result, ctx, policy, cycle


@pytest.mark.parametrize("change", ["objective", "context", "measure", "protocol"])
def test_session_rejects_changes_to_fixed_experiment_before_another_model_call(change):
    team, client, request, result, ctx, policy, _ = ready()
    if change == "objective":
        request = replace(
            request,
            objective=replace(request.objective, objective_id="different-objective"),
        )
    elif change == "context":
        ctx = replace(ctx, training_seed=99)
        request = replace(
            request,
            batch=execute_program(request.program, ctx).batch,
            selection_context_hash=ctx.context_hash,
        )
    elif change == "measure":
        request = replace(request, measure="reward_delta")
    else:
        request = replace(
            request,
            protocol=c.ArtifactRef(
                "fixture://another-protocol", c.digest("another-protocol")
            ),
        )
    result = replace(result, request_hash=request.request_hash)
    calls = len(client.calls)
    with pytest.raises(c.ContractError):
        team.feedback_step(
            request, result, ctx, policy, attempt_id="a", invocation_id="second"
        )
    assert len(client.calls) == calls


def test_cycle_rejects_backend_change_in_proposal():
    *_, cycle = ready()
    with pytest.raises(c.ContractError):
        replace(cycle, proposal=replace(cycle.proposal, backend="real"))


def test_cycle_rejects_self_consistent_candidate_from_another_context():
    _, _, request, _, ctx, _, cycle = ready()
    changed = replace(ctx, training_seed=99)
    candidate = prepare_selection(
        cycle.proposal.program,
        changed,
        request.objective,
        backend="mock",
        attempt_id="a",
        invocation_id="another-context",
    )
    next_request = replace(
        cycle.next_request,
        batch=candidate.workflow.batch,
        selection_context_hash=changed.context_hash,
    )
    with pytest.raises(c.ContractError):
        replace(cycle, candidate=candidate, next_request=next_request)


def test_team_runtime_binding_cannot_be_replaced_halfway():
    team, *_ = ready()
    with pytest.raises(AttributeError):
        team.runtime = object()


def test_restore_accepts_independent_session_attempt_and_backend_anchors(tmp_path):
    team, client, request, *_ = ready()
    team.save(tmp_path / "history")
    with pytest.raises(c.ContractError):
        RoleTeam.restore(
            tmp_path / "history",
            client,
            expected_model=request.model_ref,
            expected_session_id="not-review",
        )
    restored = RoleTeam.restore(
        tmp_path / "history",
        client,
        expected_model=request.model_ref,
        expected_session_id="review",
        expected_attempt_id="a",
        expected_backend="mock",
    )
    assert restored.runtime.attempt_id == "a"
