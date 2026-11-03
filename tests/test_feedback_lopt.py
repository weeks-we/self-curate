"""Grounded feedback, bounded edits, exact handoffs and unverified candidates."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from helpers_feedback import inputs
from self_curate import contracts as c
from self_curate.agents import (
    build_feedback,
    check_external_result,
    program_diff,
    propose_program,
    run_feedback_step,
    feedback_step_to_directory,
)
from self_curate.cli import main
from self_curate.data.pool import build_pool, load_config, write_json, file_sha256
from self_curate.operators import prepare_to_directory

ROOT = Path(__file__).parents[1]


def envelope(request, feedback):
    program = request.program.to_dict()
    program["program_id"] = "model-child"
    program["parent_id"] = request.program.program_id
    next(n for n in program["nodes"] if n["id"] == "mix")["params"][
        "branch_token_shares"
    ] = {"a": 0.375, "b": 0.625}
    return {
        "program": program,
        "rationale": "Fixed constraint failed; explore regression exposure.",
        "hypothesis": "Must revalidate; effect is not known.",
        "cited_feedback_id": feedback.feedback_id,
    }


class Client:
    def __init__(self, request, responses):
        self.model_ref = request.model_ref
        self.responses = iter(responses)
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def test_feedback_preserves_observations_and_separates_hypothesis():
    request, result, _, _ = inputs()
    original = result.to_dict()
    feedback = build_feedback(request, result)
    assert feedback.total_value == 0.134
    assert {f.skill: f.value for f in feedback.facts if f.code == "OBSERVED_VALUE"} == {
        "recovery": 0.2,
        "regression": -0.02,
    }
    assert [(h.skill, h.direction) for h in feedback.hypotheses] == [
        ("regression", "increase")
    ]
    assert result.to_dict() == original
    assert feedback == build_feedback(request, result)


@pytest.mark.parametrize("status", ["pending", "invalid"])
def test_no_evidence_blocks_proposal_without_zero_or_model_calls(status):
    request, result, ctx, policy = inputs()
    result = replace(
        result,
        status=status,
        skill_values=(),
        constraints=(),
        total_value=None,
        reason="External verifier unavailable",
    )
    client = Client(request, [])
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
    assert step.status == "needs_evidence" and step.proposal.program is None
    assert step.feedback.total_value is None and not step.feedback.hypotheses
    assert step.next_request is None and not client.calls


@pytest.mark.parametrize(
    "change",
    [
        "request",
        "backend",
        "total",
        "skills",
        "constraint_flag",
        "constraint_value",
        "coverage",
    ],
)
def test_import_rejects_wrong_identity_or_inconsistent_values(change):
    request, result, _, _ = inputs()
    if change == "request":
        result = replace(result, request_hash=c.digest("other"))
    elif change == "backend":
        result = replace(result, backend="real")
    elif change == "total":
        result = replace(result, total_value=0.5)
    elif change == "skills":
        result = replace(result, skill_values=(c.SkillValue("invented", 0.2),))
    elif change == "constraint_flag":
        result = replace(
            result, constraints=(c.ConstraintResult("retain-regression", -0.02, True),)
        )
    elif change == "constraint_value":
        result = replace(
            result, constraints=(c.ConstraintResult("retain-regression", -0.5, False),)
        )
    else:
        result = replace(result, constraints=())
    with pytest.raises(c.ContractError):
        check_external_result(request, result)


@pytest.mark.parametrize(
    "change", ["protocol", "objective", "model", "context", "program", "batch"]
)
def test_request_anchors_all_versions(change):
    request, result, _, _ = inputs()
    with pytest.raises(c.ContractError):
        if change == "protocol":
            request = replace(
                request, protocol=c.ArtifactRef("fixture://new", c.digest("new"))
            )
        elif change == "objective":
            request = replace(
                request,
                objective=replace(request.objective, objective_id="another-objective"),
            )
        elif change == "model":
            request = replace(
                request,
                model_ref=replace(
                    request.model_ref, weights_hash=c.digest("new-model")
                ),
            )
        elif change == "context":
            request = replace(request, selection_context_hash=c.digest("new-context"))
        elif change == "program":
            request = replace(
                request,
                program=c.parse_program({**request.program.to_dict(), "seed": 55}),
            )
        else:
            request = replace(
                request,
                batch=replace(
                    request.batch, execution_hash=c.digest("another-execution")
                ),
            )
        check_external_result(request, result)


def test_rule_child_diff_replay_and_new_request():
    request, result, ctx, policy = inputs()
    step = run_feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    assert step.status == "awaiting_validation"
    assert step.proposal.program.parent_id == request.program.program_id
    assert step.proposal.changes == program_diff(request.program, step.proposal.program)
    assert step.proposal.changes.modified == ("mix",)
    mix = next(n for n in step.proposal.program.nodes if n.op == "MixAllocate")
    assert mix.params["branch_token_shares"] == {"a": 0.375, "b": 0.625}
    assert step.candidate.workflow.batch.supervised_tokens == 64
    assert step.next_request.batch == step.candidate.workflow.batch
    assert (
        step.next_request.protocol == request.protocol
        and step.next_request.objective == request.objective
    )
    assert step.next_request.model_ref == request.model_ref
    assert step.next_request.request_hash != request.request_hash
    assert step.to_dict()["improvement_verified"] is False
    with pytest.raises(c.ContractError):
        check_external_result(step.next_request, result)


def test_model_uses_same_model_separate_memory_and_unknown_usage():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    client = Client(request, [{"content": envelope(request, feedback)}])
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "proposed" and proposal.method == "model"
    assert (
        client.calls[0]["role"] == "lopt" and client.calls[0]["timeout_seconds"] == 30
    )
    assert client.calls[0]["payload"]["model_ref"] == request.model_ref.to_dict()
    assert client.calls[0]["payload"]["memory_namespace"] != feedback.memory_namespace
    assert attempts[0].usage.agent_input_tokens is None
    assert attempts[0].usage.compute.gpu_seconds is None
    assert attempts[0].usage.compute.wall_seconds >= 0


@pytest.mark.parametrize(
    "bad",
    [
        "extra",
        "wrong_parent",
        "fake_id_only",
        "seed",
        "node_seed",
        "shares",
        "node_budget",
        "citation",
        "evaluator",
    ],
)
def test_model_rejections_are_bounded_and_preserved_without_rule_fallback(bad):
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    raw = envelope(request, feedback)
    if bad == "extra":
        raw["score"] = 100
    elif bad == "wrong_parent":
        raw["program"]["parent_id"] = "wrong"
    elif bad == "fake_id_only":
        raw["program"] = request.program.to_dict()
        raw["program"]["program_id"] = "fake"
        raw["program"]["parent_id"] = request.program.program_id
    elif bad == "seed":
        raw["program"]["seed"] += 1
    elif bad == "node_seed":
        next(n for n in raw["program"]["nodes"] if n["id"] == "b")["params"][
            "seed"
        ] += 1
    elif bad == "shares":
        next(n for n in raw["program"]["nodes"] if n["id"] == "mix")["params"][
            "branch_token_shares"
        ] = {"a": 0.001, "b": 0.999}
    elif bad == "node_budget":
        policy = replace(policy, max_nodes=1)
    elif bad == "citation":
        raw["cited_feedback_id"] = "invented"
    else:
        raw["program"]["quality_gate"] = {"allow_unknown": True}
    client = Client(request, [{"content": raw}] * policy.max_model_attempts)
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "blocked" and proposal.program is None
    assert len(attempts) == len(client.calls) == policy.max_model_attempts
    assert all(a.error_json and a.status == "invalid_proposal" for a in attempts)
    assert client.calls[1]["payload"]["previous_errors"]


def test_retry_can_repair_json_and_keeps_failed_call_cost():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    client = Client(
        request,
        [{"content": "{"}, {"content": json.dumps(envelope(request, feedback))}],
    )
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "proposed" and len(attempts) == 2
    assert attempts[0].error_json and attempts[1].error_json is None
    assert attempts[0].to_dict()["response"]["content"] == "{"
    assert attempts[0].to_dict()["request"]["system_prompt"]


def test_transport_error_is_bounded_unknown_cost():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    client = Client(request, [TimeoutError("remote timeout")] * 2)
    proposal, attempts = propose_program(
        request, result, feedback, policy, method="model", client=client
    )
    assert proposal.status == "blocked" and len(attempts) == 2
    assert all(a.usage.agent_input_tokens is None for a in attempts)


def test_forged_feedback_and_mismatched_role_model_rejected():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    with pytest.raises(c.ContractError):
        propose_program(request, result, replace(feedback, total_value=100), policy)
    with pytest.raises(c.ContractError):
        propose_program(request, result, feedback, policy, memory_namespace="leval")
    client = Client(request, [])
    client.model_ref = replace(request.model_ref, weights_hash=c.digest("other-model"))
    with pytest.raises(c.ContractError):
        propose_program(
            request, result, feedback, policy, method="model", client=client
        )
    assert not client.calls


def test_contracts_strict_roundtrip():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result)
    proposal, _ = propose_program(request, result, feedback, policy)
    for value in (
        request,
        result,
        policy,
        feedback,
        proposal,
        proposal.changes,
        *feedback.facts,
        *feedback.hypotheses,
    ):
        raw = json.loads(c.canonical_json(value.to_dict()))
        assert c.parse_feedback_contract(type(value).__name__, raw) == value
        with pytest.raises(c.ContractError):
            c.parse_feedback_contract(type(value).__name__, {**raw, "unknown": 1})
        with pytest.raises(c.ContractError):
            c.parse_feedback_contract(
                type(value).__name__, {**raw, "schema_version": "2"}
            )


def test_candidate_infeasible_is_retained_without_next_validation():
    request, result, ctx, policy = inputs()
    # Smaller share change cannot match 8-token indivisible records with exact quotas.
    step = run_feedback_step(
        request,
        result,
        ctx,
        replace(policy, share_step=0.01),
        attempt_id="a",
        invocation_id="i",
    )
    assert step.status == "candidate_infeasible"
    assert (
        step.proposal.program and step.candidate.outcome and step.next_request is None
    )


def test_directory_cli_chain_checksums_and_readonly_sources(tmp_path, capsys):
    request, result, _, policy = inputs()
    pool, prepared, output = (tmp_path / n for n in ("pool", "prepared", "feedback"))
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    paths = {}
    for name, value in [
        ("request", request),
        ("result", result),
        ("policy", policy),
        ("program", request.program),
    ]:
        paths[name] = tmp_path / (name + ".json")
        write_json(paths[name], value.to_dict())
    prepare_to_directory(
        pool,
        paths["program"],
        ROOT / "configs/m2.fixture.json",
        ROOT / "configs/p1.objective.fixture.json",
        prepared,
        backend="mock",
        attempt_id="a",
        invocation_id="parent",
    )
    before = {
        str(p): file_sha256(p)
        for root in (pool, prepared)
        for p in root.rglob("*")
        if p.is_file()
    }
    assert (
        main(
            [
                "validate-feedback",
                "--request",
                str(paths["request"]),
                "--result",
                str(paths["result"]),
            ]
        )
        == 0
    )
    argv = [
        "feedback-step",
        "--prepared",
        str(prepared),
        "--pool",
        str(pool),
        "--request",
        str(paths["request"]),
        "--result",
        str(paths["result"]),
        "--policy",
        str(paths["policy"]),
        "--output",
        str(output),
        "--attempt-id",
        "a",
        "--invocation-id",
        "child",
    ]
    assert main(argv) == 0
    assert (
        json.loads((output / "step.json").read_text())["status"]
        == "awaiting_validation"
    )
    for name, sha in json.loads((output / "checksums.json").read_text()).items():
        assert file_sha256(output / name) == sha
    assert main(argv) == 2
    assert before == {name: file_sha256(Path(name)) for name in before}
    assert not (output / "candidate_validation.json").exists()
    assert "OUTPUT_EXISTS" in capsys.readouterr().err
    from self_curate.agents import request_from_preparation

    assert (
        request_from_preparation(
            prepared,
            pool,
            request_id=request.request_id,
            measure=request.measure,
            protocol=request.protocol,
        )
        == request
    )
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, request.protocol.to_dict())
    requested = tmp_path / "requested.json"
    request_argv = [
        "request-validation",
        "--prepared",
        str(prepared),
        "--pool",
        str(pool),
        "--protocol",
        str(protocol_path),
        "--request-id",
        request.request_id,
        "--measure",
        request.measure,
        "--output",
        str(requested),
    ]
    assert main(request_argv) == 0
    assert (
        c.parse_feedback_contract(
            "ValidationRequest", json.loads(requested.read_text())
        )
        == request
    )
    assert main(request_argv) == 2


def test_saved_preparation_rejects_unrelated_request(tmp_path):
    request, result, _, policy = inputs()
    pool, prepared = tmp_path / "pool", tmp_path / "prepared"
    build_pool(load_config(ROOT / "configs/m2.data.fixture.json"), pool)
    prepare_to_directory(
        pool,
        ROOT / "configs/programs/m2.fixture.json",
        ROOT / "configs/m2.fixture.json",
        ROOT / "configs/p1.objective.fixture.json",
        prepared,
        backend="mock",
        attempt_id="a",
        invocation_id="parent",
    )
    paths = []
    for name, value in [("request", request), ("result", result), ("policy", policy)]:
        path = tmp_path / (name + ".json")
        write_json(path, value.to_dict())
        paths.append(path)
    with pytest.raises(c.ContractError):
        feedback_step_to_directory(
            prepared,
            pool,
            *paths,
            tmp_path / "out",
            attempt_id="a",
            invocation_id="child",
        )
    assert not (tmp_path / "out").exists()
