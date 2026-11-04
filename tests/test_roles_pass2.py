"""Pass 2: restore, P0/P1/feature consumers, costs and complete publications."""

from dataclasses import replace
import json

import pytest

from helpers_feedback import inputs
from helpers_p0 import ar, mock_gradient_spec
from self_curate import contracts as c
from self_curate.agents import RoleTeam, RoleRuntime, load_role_cycle
from self_curate.agents.demo import FixtureRoleClient
from self_curate.data import MetricView, attach_gradients, make_gradient
from self_curate.data.pool import write_json, file_sha256
from self_curate.operators import load_preparation


def setup_cycle(*, mode="model", status="valid", policy_step=0.125):
    request, result, ctx, policy = inputs()
    policy = replace(policy, share_step=policy_step)
    if status != "valid":
        result = replace(
            result,
            status=status,
            skill_values=(),
            constraints=(),
            total_value=None,
            reason="External evidence not ready",
        )
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client,
        model=request.model_ref,
        session_id="pass2",
        attempt_id="a",
        backend="mock",
    )
    cycle = team.feedback_step(
        request,
        result,
        ctx,
        policy,
        attempt_id="a",
        invocation_id="first",
        use_model=mode == "model",
    )
    return team, client, request, result, ctx, policy, cycle


@pytest.mark.parametrize(
    "mode,status,step,expected",
    [
        ("model", "valid", 0.125, "awaiting_validation"),
        ("rule", "valid", 0.125, "awaiting_validation"),
        ("model", "pending", 0.125, "needs_evidence"),
        ("model", "invalid", 0.125, "needs_evidence"),
        ("model", "valid", 0.01, "candidate_infeasible"),
    ],
)
def test_cycle_publishes_and_loads_without_model_calls_or_resampling(
    tmp_path, monkeypatch, mode, status, step, expected
):
    team, client, request, _, ctx, _, cycle = setup_cycle(
        mode=mode, status=status, policy_step=step
    )
    assert cycle.status == expected
    import self_curate.operators.preparation as prep

    saved = prep.execute_program

    def forbidden(*args, **kwargs):
        raise AssertionError("Publishing must not rerun selectors")

    monkeypatch.setattr(prep, "execute_program", forbidden)
    team.publish_cycle(cycle, tmp_path / "cycle")
    monkeypatch.setattr(prep, "execute_program", saved)
    calls = len(client.calls)
    loaded = load_role_cycle(
        tmp_path / "cycle",
        ctx.view,
        expected_request=request,
        expected_candidate_batch_id=(
            None if cycle.next_request is None else cycle.next_request.batch.batch_id
        ),
    )
    assert loaded.status == cycle.status and loaded.next_request == cycle.next_request
    assert (
        loaded.proposal == cycle.proposal
        and loaded.model_call_ids == cycle.model_call_ids
    )
    assert len(client.calls) == calls
    for name, sha in json.loads(
        (tmp_path / "cycle/checksums.json").read_text(encoding="utf-8")
    ).items():
        assert file_sha256(tmp_path / "cycle" / name) == sha
    with pytest.raises(c.ContractError):
        team.publish_cycle(cycle, tmp_path / "cycle")


def test_current_cycle_costs_exclude_previous_cycle_prefix_and_fit_p0(tmp_path):
    team, _, request, result, ctx, policy, first = setup_cycle()
    team.publish_cycle(first, tmp_path / "one")
    next_result = replace(
        result, request_hash=first.next_request.request_hash, result_id="declared-next"
    )
    second = team.feedback_step(
        first.next_request,
        next_result,
        ctx,
        policy,
        attempt_id="a",
        invocation_id="second",
    )
    team.publish_cycle(second, tmp_path / "two")

    def ledger(path):
        return c.parse_contract(
            "CostLedger", json.loads(path.read_text(encoding="utf-8"))
        )

    a, b = ledger(tmp_path / "one/ledger.json"), ledger(tmp_path / "two/ledger.json")
    assert (
        len(a.events) == len(b.events) == 4 and len(team.runtime.ledger().events) == 6
    )
    merged = c.CostLedger("attempt", a.events + b.events, "mock")
    assert len(merged.events) == 8 and {e.attempt_id for e in merged.events} == {"a"}
    prepared = load_preparation(tmp_path / "two/candidate_prepared", ctx.view)
    workflow = replace(prepared.workflow, ledger=merged)
    receipt = c.validate_workflow(
        workflow,
        ctx.view,
        expected_model=request.model_ref,
        expected_objective=request.objective,
        expected_selection_context_hash=ctx.context_hash,
        expected_batch_id=second.next_request.batch.batch_id,
    )
    assert receipt["status"] == "verified"


def test_next_verification_and_gradient_consumers_keep_exact_batch(tmp_path):
    team, _, request, result, ctx, policy, first = setup_cycle()
    team.publish_cycle(first, tmp_path / "cycle")
    loaded = load_role_cycle(tmp_path / "cycle", ctx.view, expected_request=request)
    enriched = []
    for data_id in ctx.view.data_ids:
        row = ctx.view.record(data_id)
        feature = make_gradient(
            row,
            request.model_ref,
            mock_gradient_spec(),
            vector=ar("mock-vector-" + data_id),
            dimension=8,
            dtype="float32",
            norm=1.0,
            provenance=(ar("declared-mock-gradient"),),
        )
        enriched.append(attach_gradients(row, (feature,)))
    view = MetricView(enriched)
    wf = c.load_workflow(
        tmp_path / "cycle/candidate_prepared/workflow.json",
        view=view,
        expected_model=request.model_ref,
        expected_objective=request.objective,
        expected_selection_context_hash=ctx.context_hash,
        expected_batch_id=loaded.next_request.batch.batch_id,
    )
    assert wf.batch == loaded.candidate.workflow.batch
    with pytest.raises(c.ContractError):
        load_role_cycle(tmp_path / "cycle", view, expected_request=request)
    external = replace(
        result, result_id="next-external", request_hash=loaded.next_request.request_hash
    )
    next_cycle = team.feedback_step(
        loaded.next_request, external, ctx, policy, attempt_id="a", invocation_id="next"
    )
    assert next_cycle.proposal.program.parent_id == loaded.proposal.program.program_id


@pytest.mark.parametrize(
    "artifact", ["evaluation.json", "proposal.json", "attempts.json", "ledger.json"]
)
def test_consumer_rejects_semantic_tampering_even_when_checksum_is_updated(
    tmp_path, artifact
):
    team, _, request, _, ctx, _, cycle = setup_cycle()
    root = tmp_path / "cycle"
    team.publish_cycle(cycle, root)
    value = json.loads((root / artifact).read_text(encoding="utf-8"))
    if artifact == "evaluation.json":
        value["feedback"]["total_value"] = 999
    elif artifact == "proposal.json":
        value["changes"]["modified"] = []
    elif artifact == "attempts.json":
        value[0]["request_hash"] = c.digest("wrong-input")
    else:
        value["events"] = []
    write_json(root / artifact, value)
    checks = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    checks[artifact] = file_sha256(root / artifact)
    write_json(root / "checksums.json", checks)
    with pytest.raises(c.ContractError):
        load_role_cycle(root, ctx.view, expected_request=request)


def test_dropped_logical_attempts_are_not_accepted_as_complete_evidence(tmp_path):
    team, _, request, _, ctx, _, cycle = setup_cycle()
    root = tmp_path / "cycle"
    team.publish_cycle(cycle, root)
    write_json(root / "attempts.json", [])
    checks = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    checks["attempts.json"] = file_sha256(root / "attempts.json")
    write_json(root / "checksums.json", checks)
    with pytest.raises(c.ContractError):
        load_role_cycle(root, ctx.view, expected_request=request)


def test_consumer_can_pin_the_system_edit_policy(tmp_path):
    team, _, request, _, ctx, policy, cycle = setup_cycle()
    root = tmp_path / "cycle"
    team.publish_cycle(cycle, root)
    with pytest.raises(c.ContractError):
        load_role_cycle(
            root,
            ctx.view,
            expected_request=request,
            expected_policy=replace(policy, share_step=0.2),
        )
    assert (
        load_role_cycle(
            root, ctx.view, expected_request=request, expected_policy=policy
        ).source_policy
        == policy
    )


def test_wrong_independent_source_and_best_batch_are_rejected(tmp_path):
    team, _, request, _, ctx, _, cycle = setup_cycle()
    team.publish_cycle(cycle, tmp_path / "cycle")
    with pytest.raises(c.ContractError):
        load_role_cycle(
            tmp_path / "cycle",
            ctx.view,
            expected_request=replace(request, request_id="another"),
        )
    with pytest.raises(c.ContractError):
        load_role_cycle(
            tmp_path / "cycle",
            ctx.view,
            expected_request=request,
            expected_candidate_batch_id=c.digest("another-best"),
        )


def test_missing_model_cost_cannot_be_published_as_free_work(tmp_path):
    from self_curate.agents import publish_role_cycle

    *_, cycle = setup_cycle()
    with pytest.raises(c.ContractError):
        publish_role_cycle(
            cycle, tmp_path / "lost", model_ledger=c.CostLedger("missing", (), "mock")
        )
    assert not (tmp_path / "lost").exists()


def test_failed_execution_can_be_explicitly_fed_back_without_inventing_new_validation(
    tmp_path,
):
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client, model=request.model_ref, session_id="repair", backend="mock"
    )
    original = client.generate_json
    first_lopt = True

    def response(**kwargs):
        nonlocal first_lopt
        reply = original(**kwargs)
        if kwargs["role"] == "lopt" and first_lopt:
            first_lopt = False
            program = reply["content"]["program"]
            next(n for n in program["nodes"] if n["id"] == "entropy")["params"][
                "reference_group"
            ] = "unregistered"
        return reply

    client.generate_json = response
    failed = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="bad"
    )
    assert failed.status == "candidate_failed"
    team.publish_cycle(failed, tmp_path / "failed")
    loaded = load_role_cycle(tmp_path / "failed", ctx.view, expected_request=request)
    assert loaded.status == "candidate_failed" and loaded.candidate.error
    repaired = team.feedback_step(
        request,
        result,
        ctx,
        policy,
        attempt_id="a",
        invocation_id="repair",
        previous_cycle=loaded,
    )
    assert repaired.status == "awaiting_validation" and repaired.source_result == result
    lopt = next(call for call in reversed(client.calls) if call["role"] == "lopt")
    handoff = lopt["payload"]["work"]["handoff"]["execution_feedback"]
    assert (
        handoff["status"] == "candidate_failed"
        and handoff["error"]["code"] == failed.candidate.error["code"]
    )
    assert (
        "completed_traces" not in handoff["error"]["details"]
        and "history" not in handoff
    )
    assert handoff["improvement_verified"] is False


def test_blocked_proposal_can_be_loaded_without_fake_candidate(tmp_path):
    request, result, ctx, policy = inputs()
    client = FixtureRoleClient(request.model_ref)
    original = client.generate_json

    def bad(**kwargs):
        reply = original(**kwargs)
        if kwargs["role"] == "lopt":
            reply["content"]["cited_feedback_id"] = "unrelated"
        return reply

    client.generate_json = bad
    team = RoleTeam(
        client, model=request.model_ref, session_id="blocked", backend="mock"
    )
    cycle = team.feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="blocked"
    )
    assert cycle.status == "proposal_blocked"
    team.publish_cycle(cycle, tmp_path / "blocked")
    loaded = load_role_cycle(tmp_path / "blocked", ctx.view, expected_request=request)
    assert (
        loaded.candidate is None
        and len(loaded.model_call_ids) == 1 + policy.max_model_attempts
    )


def test_legacy_060_snapshot_can_infer_fixed_scope_without_rewriting_old_records(
    tmp_path,
):
    from self_curate.agents.history import RoleHistory, HistoryEvent

    team, client, request, result, ctx, policy, _ = setup_cycle()
    root = tmp_path / "legacy"
    team.save(root)
    meta = json.loads((root / "session.json").read_text(encoding="utf-8"))
    meta.pop("scope")
    reconstructed = RoleRuntime(
        client,
        model=request.model_ref,
        session_id=meta["session_id"],
        prompts=team.runtime.prompts,
        backend="mock",
        attempt_id="a",
    )
    for role in ("lexec", "leval", "lopt"):
        raw = json.loads((root / (role + ".json")).read_text(encoding="utf-8"))
        history = RoleHistory(
            role, raw["namespace"], raw["model_hash"], raw["prompt_hash"], "mock"
        )
        for row in raw["events"]:
            payload = json.loads(row["payload_json"])
            if row["phase"] == "before" and row["operation"].startswith("tool:"):
                payload["work"].pop("scope", None)
            if row["phase"] == "before" and row["operation"] == "model":
                payload["history_hash"] = c.digest(
                    history.context(len(payload["history_call_ids"]))
                )
            event = HistoryEvent(
                **{
                    **row,
                    "payload_json": c.canonical_json(payload),
                    "previous_hash": history.root_hash,
                }
            )
            history._append(event)
        write_json(root / (role + ".json"), history.to_dict())
        reconstructed._histories[role] = history
    write_json(root / "session.json", meta)
    write_json(root / "ledger.json", reconstructed.ledger().to_dict())
    write_json(
        root / "checksums.json",
        {p.name: file_sha256(p) for p in root.iterdir() if p.name != "checksums.json"},
    )
    before = {p.name: file_sha256(p) for p in root.iterdir()}
    restored = RoleTeam.restore(
        root,
        client,
        expected_model=request.model_ref,
        expected_session_id="pass2",
        expected_attempt_id="a",
    )
    assert restored.runtime.scope.objective_hash == request.objective.objective_hash
    assert restored.runtime.scope.selection_context_hash == ctx.context_hash
    changed = replace(request, measure="reward_delta")
    changed_result = replace(result, request_hash=changed.request_hash)
    calls = len(client.calls)
    with pytest.raises(c.ContractError):
        restored.feedback_step(
            changed, changed_result, ctx, policy, attempt_id="a", invocation_id="bad"
        )
    assert len(client.calls) == calls and before == {
        p.name: file_sha256(p) for p in root.iterdir()
    }
