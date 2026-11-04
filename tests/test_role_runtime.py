"""Shared client identity, role-private prompt/context, and durable snapshots."""

from dataclasses import replace, FrozenInstanceError
import json

import pytest

from helpers_feedback import inputs
from self_curate import contracts as c
from self_curate.agents import (
    RoleRuntime,
    RoleTeam,
    default_prompts,
    load_prompts,
    ROLES,
)
from self_curate.agents.demo import FixtureRoleClient
from self_curate.data.pool import write_json, file_sha256


class EchoClient:
    def __init__(self, model):
        self.model_ref = model
        self.calls = []
        self.fail = None
        self.hook = None

    def generate_json(self, **packet):
        self.calls.append(json.loads(c.canonical_json(packet)))
        if self.hook:
            self.hook()
        if self.fail is not None:
            if isinstance(self.fail, Exception):
                raise self.fail
            return self.fail
        return {"content": {"message": "response-for-" + packet["role"]}}


def runtime(*, session="session", window=8):
    request, *_ = inputs()
    client = EchoClient(request.model_ref)
    prompts = {
        r: replace(p, instructions=r.upper() + "_PRIVATE_PROMPT")
        for r, p in default_prompts().items()
    }
    return (
        RoleRuntime(
            client,
            model=request.model_ref,
            session_id=session,
            prompts=prompts,
            backend="mock",
            history_window=window,
        ),
        client,
    )


def rehash(root, name):
    path = root / "checksums.json"
    checks = json.loads(path.read_text())
    checks[name] = file_sha256(root / name)
    write_json(path, checks)


def test_three_agents_share_exact_runtime_client_and_frozen_model():
    request, *_ = inputs()
    client = EchoClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="shared", backend="mock"
    )
    assert team.lexec.runtime is team.leval.runtime is team.lopt.runtime is team.runtime
    assert (
        team.lexec.model_ref
        is team.leval.model_ref
        is team.lopt.model_ref
        is request.model_ref
    )
    assert all(
        agent.runtime.shared_client is client
        for agent in (team.lexec, team.leval, team.lopt)
    )
    assert (
        len({agent.memory_namespace for agent in (team.lexec, team.leval, team.lopt)})
        == 3
    )


def test_prompts_and_history_are_separate_on_every_call():
    value, client = runtime()
    for role in ROLES:
        value.generate(role, {"private_marker": role.upper() + "_PRIVATE_WORK"})
    for role in ROLES:
        value.generate(role, {"second": role})
    for call in client.calls:
        role = call["role"]
        packet = c.canonical_json(call)
        assert role.upper() + "_PRIVATE_PROMPT" in call["system_prompt"]
        for other in set(ROLES) - {role}:
            assert other.upper() + "_PRIVATE_PROMPT" not in packet
            assert other.upper() + "_PRIVATE_WORK" not in packet
        assert call["payload"]["memory_namespace"] == value.namespace(role)
        assert call["payload"]["role"] == role
        assert all(
            item["call_id"] in {e.call_id for e in value.history(role)}
            for item in call["payload"]["history"]
        )
    assert all(len(value.history(role)) == 4 for role in ROLES)


def test_prompt_body_may_be_empty_and_files_fill_without_code_changes(tmp_path):
    for role, prompt in default_prompts().items():
        assert prompt.instructions == ""
        write_json(tmp_path / (role + ".json"), prompt.to_dict())
    first = load_prompts(tmp_path)
    updated = replace(
        first["leval"], instructions="Diagnose code-owned observations.", revision="2"
    )
    write_json(tmp_path / "leval.json", updated.to_dict())
    second = load_prompts(tmp_path)
    assert (
        second["leval"] == updated
        and second["lopt"] == first["lopt"]
        and second["lexec"] == first["lexec"]
    )
    assert second["leval"].prompt_hash != first["leval"].prompt_hash


def test_prompts_are_fixed_within_a_runtime():
    value, _ = runtime()
    with pytest.raises(TypeError):
        value.prompts["lexec"] = value.prompts["leval"]
    with pytest.raises(FrozenInstanceError):
        value.prompts["lexec"].instructions = "changed"
    updated = dict(value.prompts)
    updated["lexec"] = replace(updated["lexec"], instructions="New version")
    next_runtime = RoleRuntime(
        value.shared_client,
        model=value.model_ref,
        session_id=value.session_id,
        prompts=updated,
        backend="mock",
    )
    assert next_runtime.namespace("lexec") != value.namespace("lexec")
    assert next_runtime.namespace("leval") == value.namespace("leval")


def test_own_history_before_after_and_window_preserves_archived_events():
    value, client = runtime(window=1)
    for i in range(4):
        value.generate("lexec", {"turn": i})
    history = client.calls[-1]["payload"]["history"]
    assert len(history) == 1 and history[0]["before"] == {"turn": 2}
    assert history[0]["after"] == {"message": "response-for-lexec"}
    assert len(value.history("lexec")) == 8 and not value.history("leval")
    assert (
        value.history("lexec")[0].phase == "before"
        and value.history("lexec")[1].phase == "after"
    )
    before = value.history("lexec")[0].record_hash
    exposed = value.history("lexec")[0].payload
    exposed["work"]["turn"] = 100
    assert value.history("lexec")[0].record_hash == before


def test_no_history_window_still_records_full_history():
    value, client = runtime(window=0)
    value.generate("leval", {"turn": 1})
    value.generate("leval", {"turn": 2})
    assert (
        client.calls[-1]["payload"]["history"] == []
        and len(value.history("leval")) == 4
    )


def test_session_and_checkpoint_namespaces_do_not_mix():
    old, client = runtime(session="first")
    new, _ = runtime(session="second")
    assert old.namespace("lexec") != new.namespace("lexec")
    newer_model = replace(old.model_ref, weights_hash=c.digest("new-checkpoint"))
    newer = RoleRuntime(
        EchoClient(newer_model),
        model=newer_model,
        session_id=old.session_id,
        backend="mock",
    )
    assert old.namespace("lexec") != newer.namespace("lexec") and not newer.history(
        "lexec"
    )
    with pytest.raises(c.ContractError):
        RoleRuntime(client, model=newer_model, session_id="wrong", backend="mock")


@pytest.mark.parametrize(
    "bad",
    [
        {"content": "{"},
        {"content": []},
        {"content": {}, "unexpected": 1},
        TimeoutError("transport failed"),
    ],
)
def test_bad_responses_preserve_failed_before_after_and_unknown_cost(bad):
    value, client = runtime()
    client.fail = bad
    with pytest.raises(c.ContractError):
        value.generate("lexec", {"job": "failure"})
    events = value.history("lexec")
    assert len(events) == 2 and events[1].payload["status"] == "failed"
    assert events[1].payload["error"]
    assert events[1].payload["usage"]["agent_input_tokens"] is None
    assert value.ledger().events[0].status == "failed"
    assert value.ledger().summary()["agent_input_tokens"]["total"] is None
    assert not value.history("leval")


def test_known_backend_usage_is_preserved_once_in_model_ledger():
    value, client = runtime()
    usage = c.ResourceUsage(c.ComputeCost(0.1, 0.2, 7), 3, 4, 0, 0, 0, 0, 0, 0)
    client.fail = {"content": {"okay": True}, "usage": usage.to_dict()}
    reply = value.generate("lexec", {})
    assert reply.usage == usage and len(value.ledger().events) == 1
    assert value.ledger().summary()["agent_input_tokens"]["total"] == 3


def test_model_switch_is_rejected_and_original_history_is_kept():
    value, client = runtime()
    client.hook = lambda: setattr(
        client, "model_ref", replace(client.model_ref, weights_hash=c.digest("changed"))
    )
    with pytest.raises(c.ContractError):
        value.generate("leval", {})
    assert value.history("leval")[-1].payload["error"]["code"] == "STALE_CACHE"
    with pytest.raises(c.ContractError):
        value.generate("lopt", {})
    assert not value.history("lopt") and len(client.calls) == 1


def test_snapshot_roundtrip_then_continue_only_own_history(tmp_path):
    value, client = runtime()
    for role in ROLES:
        value.generate(role, {"private_marker": role})
    value.save(tmp_path / "snapshot")
    loaded = RoleRuntime.restore(
        tmp_path / "snapshot",
        client,
        expected_model=value.model_ref,
        expected_prompts=value.prompts,
    )
    assert loaded.ledger() == value.ledger()
    for role in ROLES:
        assert loaded.history(role) == value.history(role)
    loaded.generate("lopt", {"next": 1})
    assert len(loaded.history("lopt")) == 4 and len(loaded.history("lexec")) == 2
    assert len(value.history("lopt")) == 2
    with pytest.raises(c.ContractError):
        value.save(tmp_path / "snapshot")


@pytest.mark.parametrize(
    "change",
    [
        "role_swap",
        "event_payload",
        "prompt",
        "session",
        "model",
        "ledger",
        "unknown_file",
    ],
)
def test_snapshot_tampering_is_rejected_even_if_file_checksum_is_refreshed(
    tmp_path, change
):
    value, client = runtime()
    value.generate("lexec", {"job": 1})
    value.save(tmp_path / "snapshot")
    root = tmp_path / "snapshot"
    if change == "role_swap":
        a = (root / "lexec.json").read_bytes()
        b = (root / "leval.json").read_bytes()
        (root / "lexec.json").write_bytes(b)
        (root / "leval.json").write_bytes(a)
        rehash(root, "lexec.json")
        rehash(root, "leval.json")
    elif change == "unknown_file":
        (root / "extra.json").write_text("{}")
    else:
        name = (
            "lexec.json"
            if change == "event_payload"
            else ("ledger.json" if change == "ledger" else "session.json")
        )
        raw = json.loads((root / name).read_text())
        if change == "event_payload":
            raw["events"][0]["payload_json"] = c.canonical_json(
                {"work": {"forged": True}}
            )
        elif change == "prompt":
            raw["prompts"]["lexec"]["instructions"] = "forged"
        elif change == "session":
            raw["session_id"] = "other-session"
        elif change == "model":
            raw["model_ref"]["weights_hash"] = c.digest("forged")
        else:
            raw["events"] = []
        write_json(root / name, raw)
        rehash(root, name)
    with pytest.raises(c.ContractError):
        RoleRuntime.restore(
            root, client, expected_model=value.model_ref, expected_prompts=value.prompts
        )


def test_prompt_change_or_model_change_requires_new_session_instead_of_history_rebinding(
    tmp_path,
):
    value, client = runtime()
    value.save(tmp_path / "snapshot")
    updated = dict(value.prompts)
    updated["lexec"] = replace(updated["lexec"], instructions="new instructions")
    with pytest.raises(c.ContractError):
        RoleRuntime.restore(
            tmp_path / "snapshot",
            client,
            expected_model=value.model_ref,
            expected_prompts=updated,
        )
    with pytest.raises(c.ContractError):
        RoleRuntime.restore(
            tmp_path / "snapshot",
            client,
            expected_model=replace(value.model_ref, weights_hash=c.digest("new")),
        )


def test_interrupted_snapshot_requires_explicit_reconciliation_without_reexecuting(
    tmp_path,
):
    value, client = runtime()
    client.hook = lambda: value.save(tmp_path / "interrupted")
    value.generate("lexec", {"job": "may-have-run"})
    loaded = RoleRuntime.restore(
        tmp_path / "interrupted", client, expected_model=value.model_ref
    )
    assert len(loaded.history("lexec")) == 1
    calls = len(client.calls)
    with pytest.raises(c.ContractError):
        loaded.generate("lexec", {})
    assert len(client.calls) == calls
    pending = loaded.history("lexec")[0]
    loaded.resolve_interrupted(
        "lexec",
        pending.call_id,
        reason="Reviewed upstream call; result unavailable, preserve failure",
    )
    client.hook = None
    loaded.generate("lexec", {"job": "new-explicit-job"})
    assert loaded.history("lexec")[1].payload["usage"]["agent_input_tokens"] is None
    assert len(client.calls) == calls + 1


def test_tool_whitelist_prevents_other_role_execution():
    value, _ = runtime()
    called = []
    with pytest.raises(c.ContractError):
        value.run_tool(
            "leval", "execute_selection", {}, lambda: called.append(1), lambda x: {}
        )
    assert not called and not value.history("leval")


def test_three_roles_reuse_existing_pipeline_and_restore(tmp_path):
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(client, model=request.model_ref, session_id="cycle", backend="mock")
    first = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="first"
    )
    assert first.status == "awaiting_validation" and {
        call["role"] for call in client.calls
    } == set(ROLES)
    assert (
        first.evaluation.feedback.total_value == result.total_value
        and first.candidate.workflow.batch.supervised_tokens == 64
    )
    assert len(team.runtime.ledger().events) == 3
    team.save(tmp_path / "history")
    restored = RoleTeam.restore(
        tmp_path / "history", client, expected_model=request.model_ref
    )
    second_result = replace(
        result,
        result_id="next-declared-result",
        request_hash=first.next_request.request_hash,
    )
    second = restored.feedback_step(
        first.next_request,
        second_result,
        ctx,
        policy,
        attempt_id="a",
        invocation_id="second",
    )
    assert (
        second.status == "awaiting_validation"
        and len(restored.runtime.ledger().events) == 6
    )
    assert all(len(restored.runtime.history(role)) == 8 for role in ROLES)
    for call in client.calls:
        if call["role"] == "lopt":
            handoff = call["payload"]["work"]["handoff"]
            assert handoff["feedback"]["model_ref"] == request.model_ref.to_dict()
            assert "history" not in handoff and "system_prompt" not in handoff


def test_explicit_rule_mode_uses_no_model_calls_but_keeps_role_work_history():
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="no-model", backend="mock"
    )
    cycle = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i", use_model=False
    )
    assert (
        cycle.status == "awaiting_validation"
        and not client.calls
        and not team.runtime.ledger().events
    )
    assert all(len(team.runtime.history(role)) == 2 for role in ROLES)


def test_pending_evidence_does_not_trigger_role_models_or_candidate():
    request, result, ctx, policy = inputs()
    result = replace(
        result,
        status="pending",
        skill_values=(),
        constraints=(),
        total_value=None,
        reason="Await verifier",
    )
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="pending", backend="mock"
    )
    cycle = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    assert (
        cycle.status == "needs_evidence"
        and cycle.candidate is None
        and not client.calls
        and not team.lexec.history
    )
