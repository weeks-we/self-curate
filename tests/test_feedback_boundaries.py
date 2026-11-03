"""Independent edge checks for external validation and proposal boundaries."""

from dataclasses import replace

import pytest

from helpers_feedback import inputs
from helpers_p0 import fixture_bundle, ar
from helpers_m2 import context, records, node, program
from self_curate import contracts as c
from self_curate.agents import (
    build_feedback,
    propose_program,
    run_feedback_step,
    result_from_evaluation,
)
from self_curate.operators import execute_program


def test_all_constraints_pass_explores_lower_metric_without_causal_claim():
    request, result, _, _ = inputs()
    result = replace(
        result,
        skill_values=(c.SkillValue("recovery", 0.2), c.SkillValue("regression", 0.1)),
        total_value=0.17,
        constraints=(c.ConstraintResult("retain-regression", 0.1, True),),
    )
    feedback = build_feedback(request, result)
    assert not any(f.code == "CONSTRAINT_FAILED" for f in feedback.facts)
    assert feedback.hypotheses[0].skill == "regression"
    assert feedback.hypotheses[0].direction == "increase"


def test_upper_bound_failure_reduces_related_share():
    request, result, ctx, policy = inputs()
    objective = replace(
        request.objective,
        constraints=(
            c.SkillConstraint("retain-regression", "regression", None, -0.03),
        ),
    )
    request = replace(request, objective=objective)
    result = replace(result, request_hash=request.request_hash)
    step = run_feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    assert step.feedback.hypotheses[0].direction == "decrease"
    assert (
        next(n for n in step.proposal.program.nodes if n.id == "mix").params[
            "branch_token_shares"
        ]["b"]
        == 0.375
    )
    assert step.status == "awaiting_validation"


def test_no_skill_branch_blocks_instead_of_seed_search():
    request, result, ctx, policy = inputs()
    parent = program(
        [node("sample", "SampleSelect", {"method": "uniform", "size": 8, "seed": 11})]
    )
    request = replace(request, program=parent, batch=execute_program(parent, ctx).batch)
    result = replace(result, request_hash=request.request_hash)
    step = run_feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    assert step.status == "proposal_blocked" and step.next_request is None
    assert step.proposal.changes.seed_changed is False


def test_evidence_fallback_retains_infeasible_outcome():
    request, result, _, policy = inputs()
    ctx = context(records(), target=32, tolerance=0, batch=4, share_tolerance=0)
    parent = program(
        [
            node("r", "SkillSelect", {"skills": ["regression"]}),
            node(
                "e",
                "EvidenceSelect",
                {"target_skill": "regression", "min_tokens": 4},
                ("r",),
            ),
        ]
    )
    request = replace(
        request,
        program=parent,
        batch=execute_program(parent, ctx).batch,
        selection_context_hash=ctx.context_hash,
    )
    result = replace(result, request_hash=request.request_hash)
    step = run_feedback_step(
        request, result, ctx, policy, attempt_id="a", invocation_id="i"
    )
    assert step.proposal.changes.modified == ("e",)
    assert step.status == "candidate_infeasible" and step.next_request is None


def test_existing_p0_evaluation_adapts_with_exact_anchors():
    bundle = fixture_bundle()
    request, _, _, _ = inputs()
    request = replace(
        request,
        program=bundle.program,
        batch=bundle.batch,
        selection_context_hash=bundle.selection_context_hash,
    )
    report = bundle.evaluations[0]
    result = result_from_evaluation(request, report, ar("external-report"))
    assert (
        result.skill_values == report.skill_values
        and result.total_value == report.total_value
    )
    assert result.usage.compute.gpu_seconds is None
    with pytest.raises(c.ContractError):
        result_from_evaluation(
            replace(request, measure="reward_delta"), report, ar("report")
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_model_attempts", 4),
        ("max_nodes", 257),
        ("share_step", 0),
        ("minimum_share", 0.99),
        ("model_timeout_seconds", 0),
        ("allow_seed_change", 1),
    ],
)
def test_invalid_edit_policy_rejected(field, value):
    *_, policy = inputs()
    with pytest.raises(c.ContractError):
        replace(policy, **{field: value})


def test_role_memories_independent_with_same_model():
    request, result, _, policy = inputs()
    feedback = build_feedback(request, result, memory_namespace="attempt/Leval")
    proposal, _ = propose_program(
        request, result, feedback, policy, memory_namespace="attempt/Lopt"
    )
    assert proposal.memory_namespace != feedback.memory_namespace
    assert proposal.model_ref == feedback.model_ref == request.model_ref


def test_context_node_limit_cannot_be_raised():
    request, result, ctx, policy = inputs()
    with pytest.raises(c.ContractError):
        run_feedback_step(
            request,
            result,
            ctx,
            replace(policy, max_nodes=129),
            attempt_id="a",
            invocation_id="i",
        )
