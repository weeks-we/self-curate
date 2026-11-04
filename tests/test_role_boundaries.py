"""Role tools preserve numerical authority, checkpoint pins and failure evidence."""

from dataclasses import replace

import pytest

from helpers_feedback import inputs
from self_curate import contracts as c
from self_curate.agents import RoleTeam, RoleRuntime, default_prompts, ROLES
from self_curate.agents.demo import FixtureRoleClient, run_role_demo
from test_role_runtime import EchoClient, runtime


def team():
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    value = RoleTeam(
        client, model=request.model_ref, session_id="boundaries", backend="mock"
    )
    return value, client, request, result, ctx, policy


def test_lexec_cannot_change_program_or_invoke_p1_after_bad_control_reply(monkeypatch):
    value, client, request, _, ctx, _ = team()
    client.generate_json = lambda **kwargs: {
        "content": {
            "action": "execute",
            "program_hash": request.program.recipe_hash,
            "context_hash": ctx.context_hash,
            "program": {"changed": True},
        }
    }
    import self_curate.agents.roles as module

    invoked = []
    monkeypatch.setattr(
        module, "prepare_selection", lambda *a, **kw: invoked.append(True)
    )
    with pytest.raises(c.ContractError):
        value.lexec.execute(
            request.program, ctx, request.objective, attempt_id="a", invocation_id="i"
        )
    assert not invoked and len(value.lexec.history) == 2
    assert value.lexec.history[-1].payload["status"] == "failed"
    assert not value.leval.history and not value.lopt.history


def test_leval_model_cannot_override_numeric_feedback():
    value, client, request, result, _, _ = team()
    original = result.to_dict()
    client.generate_json = lambda **kwargs: {
        "content": {
            "source_feedback_hash": kwargs["payload"]["work"]["feedback_hash"],
            "commentary": "Untrusted numeric override",
            "hypotheses": [],
            "total_value": 999,
        }
    }
    with pytest.raises(c.ContractError):
        value.leval.evaluate(request, result)
    assert result.to_dict() == original and len(value.leval.history) == 4
    facts = value.leval.history[1].payload["result"]
    assert facts["total_value"] == 0.134
    assert value.leval.history[-1].payload["status"] == "failed"


def test_lopt_invalid_json_retries_preserve_every_model_call_and_cost():
    request, result, ctx, policy = inputs()
    client = EchoClient(request.model_ref)
    client.fail = {"content": "{"}
    value = RoleTeam(
        client, model=request.model_ref, session_id="invalid-json", backend="mock"
    )
    proposal, attempts = value.lopt.optimize(request, result, policy, context=ctx)
    assert (
        proposal.status == "blocked"
        and len(attempts) == len(client.calls) == policy.max_model_attempts
    )
    ledger = value.runtime.ledger()
    assert len(ledger.events) == policy.max_model_attempts and all(
        e.status == "failed" for e in ledger.events
    )
    summary = value.lopt.history[-1].payload["result"]
    assert set(summary["role_call_ids"]) == {e.event_id for e in ledger.events}
    assert all(a.error_json for a in attempts)
    assert not value.lexec.history and not value.leval.history


def test_valid_json_with_invalid_program_is_not_accepted():
    request, result, ctx, policy = inputs()
    client = EchoClient(request.model_ref)
    client.fail = {
        "content": {
            "program": {},
            "rationale": "bad",
            "hypothesis": "bad",
            "cited_feedback_id": "wrong",
        }
    }
    value = RoleTeam(
        client, model=request.model_ref, session_id="invalid-program", backend="mock"
    )
    proposal, attempts = value.lopt.optimize(request, result, policy, context=ctx)
    assert proposal.status == "blocked" and all(a.error_json for a in attempts)
    # A completed generation is not an accepted selection program.
    assert all(e.status == "completed" for e in value.runtime.ledger().events)
    assert value.lopt.history[-1].payload["result"]["proposal"]["status"] == "blocked"


def test_role_payload_cannot_select_another_memory_namespace():
    value, client = runtime()
    value.generate(
        "leval", {"memory_namespace": value.namespace("lexec"), "role": "lexec"}
    )
    assert client.calls[0]["payload"]["memory_namespace"] == value.namespace("leval")
    assert not value.history("lexec") and len(value.history("leval")) == 2


def test_wrong_model_team_stops_before_any_role_call():
    request, result, ctx, policy = inputs()
    model = replace(request.model_ref, weights_hash=c.digest("new-model"))
    client = FixtureRoleClient(model)
    value = RoleTeam(client, model=model, session_id="next-checkpoint", backend="mock")
    with pytest.raises(c.ContractError):
        value.feedback_step(
            request, result, ctx, policy, attempt_id="a", invocation_id="i"
        )
    assert not client.calls and all(not value.runtime.history(role) for role in ROLES)


@pytest.mark.parametrize("change", ["policy", "attempt", "invocation"])
def test_invalid_team_configuration_stops_before_model_work(change):
    value, client, request, result, ctx, policy = team()
    attempt, invocation = "a", "i"
    if change == "policy":
        policy = replace(policy, max_nodes=129)
    elif change == "attempt":
        attempt = ""
    else:
        invocation = ""
    with pytest.raises(c.ContractError):
        value.feedback_step(
            request, result, ctx, policy, attempt_id=attempt, invocation_id=invocation
        )
    assert not client.calls


def test_keyboard_cancellation_preserves_failed_history_and_propagates():
    value, client = runtime()

    def cancelled(**kwargs):
        raise KeyboardInterrupt("cancelled")

    client.generate_json = cancelled
    with pytest.raises(KeyboardInterrupt):
        value.generate("lexec", {"job": "cancel"})
    assert value.history("lexec")[-1].payload["status"] == "failed"
    assert (
        value.history("lexec")[-1].payload["error"]["details"]["exception"]
        == "KeyboardInterrupt"
    )
    assert not value._active


def test_missing_snapshot_is_a_structured_artifact_error(tmp_path):
    request, *_ = inputs()
    with pytest.raises(c.ContractError) as error:
        RoleRuntime.restore(
            tmp_path / "missing",
            EchoClient(request.model_ref),
            expected_model=request.model_ref,
        )
    assert error.value.code == c.ErrorCode.ARTIFACT_MISMATCH


def test_default_prompts_are_distinct_even_with_empty_bodies():
    prompts = default_prompts()
    assert len({p.prompt_hash for p in prompts.values()}) == 3
    assert all(p.instructions == "" for p in prompts.values())


def test_real_runtime_refuses_fixture_model():
    request, *_ = inputs()
    with pytest.raises(c.ContractError):
        RoleRuntime(
            EchoClient(request.model_ref),
            model=request.model_ref,
            session_id="no",
            backend="real",
        )


def test_new_physical_calls_do_not_reuse_event_ids_in_fresh_runtime():
    first, _ = runtime(session="same-metadata")
    second, _ = runtime(session="same-metadata")
    a = first.generate("lexec", {"same": True})
    b = second.generate("lexec", {"same": True})
    assert a.call_id != b.call_id
    merged = c.CostLedger(
        "physical-calls", first.ledger().events + second.ledger().events, "mock"
    )
    assert len(merged.events) == 2


def test_shared_client_protocol_keeps_legacy_import_path():
    from self_curate.agents import SharedModelClient
    from self_curate.agents.client import SharedModelClient as current
    from self_curate.agents.lopt import SharedModelClient as legacy

    assert SharedModelClient is current is legacy


def test_model_costs_bind_outer_attempt_separately_from_session_and_survive_restore(
    tmp_path,
):
    value, client, request, result, ctx, policy = team()
    cycle = value.feedback_step(
        request, result, ctx, policy, attempt_id="outer-17", invocation_id="child"
    )
    assert (
        value.runtime.session_id == "boundaries"
        and value.runtime.attempt_id == "outer-17"
    )
    assert {
        e.attempt_id
        for e in value.runtime.ledger().events + cycle.candidate.ledger.events
    } == {"outer-17"}
    value.save(tmp_path / "snapshot")
    restored = RoleTeam.restore(
        tmp_path / "snapshot", client, expected_model=request.model_ref
    )
    assert restored.runtime.attempt_id == "outer-17"
    calls = len(client.calls)
    with pytest.raises(c.ContractError):
        restored.feedback_step(
            request,
            result,
            ctx,
            policy,
            attempt_id="other-outer",
            invocation_id="wrong",
        )
    assert len(client.calls) == calls


def test_standalone_model_history_cannot_be_relabelled_to_another_attempt():
    value, _ = runtime(session="standalone-attempt")
    value.generate("lexec", {})
    with pytest.raises(c.ContractError):
        value.bind_attempt("relabelled-attempt")


@pytest.mark.parametrize("role", ROLES)
def test_invalid_role_method_inputs_fail_before_model(role):
    value, client, request, result, ctx, policy = team()
    with pytest.raises(c.ContractError):
        if role == "lexec":
            value.lexec.execute(
                {}, ctx, request.objective, attempt_id="a", invocation_id="i"
            )
        elif role == "leval":
            value.leval.evaluate({}, result)
        else:
            value.lopt.optimize({}, result, policy, context=ctx)
    assert not client.calls


def test_role_demo_runs_restores_and_is_explicitly_mock(tmp_path):
    from pathlib import Path

    root = Path(__file__).parents[1]
    summary = run_role_demo(tmp_path / "demo", project_root=root)
    assert (
        summary["shared_client"]
        and summary["model_calls"] == 6
        and summary["history_restored"]
    )
    assert (
        len(set(summary["memory_namespaces"].values())) == 3
        and len(set(summary["role_models"].values())) == 1
    )
    assert all(n == 8 for n in summary["history_events"].values())
    assert (
        summary["real_model_called"] is False
        and summary["validation_implemented"] is False
    )
    with pytest.raises(c.ContractError):
        run_role_demo(tmp_path / "demo", project_root=root)
