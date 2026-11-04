"""Integration boundaries checked along the code-reading tutorial's data flow."""

import pytest

from helpers_feedback import inputs
from helpers_m2 import context, node, program, records
from self_curate.agents import RoleTeam, load_role_cycle
from self_curate.agents.demo import FixtureRoleClient
from self_curate.operators import execute_program


def test_metric_view_membership_cannot_drift_under_frozen_execution_identity():
    ctx = context(records(2), target=8, tolerance=0, batch=1)
    recipe = program([node("sample", "SampleSelect", {"method": "ranked", "size": 1})])
    selected = execute_program(recipe, ctx)
    # A normal public assignment must not change the candidate universe while
    # the context continues to advertise the same cached snapshot/hash.
    with pytest.raises(AttributeError):
        ctx.view.data_ids = ("data-01",)
    assert ctx.view.data_ids == ("data-00", "data-01")
    assert execute_program(recipe, ctx) == selected


def test_complete_cycle_publishes_below_windows_path_limit(tmp_path):
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="path-review", backend="mock"
    )
    cycle = team.feedback_step(
        request, result, ctx, policy, attempt_id="path-review", invocation_id="first"
    )
    # Every final artifact is comfortably below 260 characters. The former
    # pair of output-name + 32-character UUID staging names exceeded that limit.
    parent = tmp_path / ("p" * max(1, 185 - len(str(tmp_path)) - 1))
    output = parent / "cycle"
    assert len(str(output / "candidate_prepared" / "checksums.json")) < 250
    team.publish_cycle(cycle, output)
    loaded = load_role_cycle(
        output, ctx.view, expected_request=request, expected_policy=policy
    )
    assert loaded.next_request == cycle.next_request
    assert len(client.calls) == 3
    assert not list(parent.glob("*.staging"))
