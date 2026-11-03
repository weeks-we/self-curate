"""Explicit external mock evidence. No verification algorithm is simulated."""

from helpers_m2 import MODEL, records, context, fixture_program
from self_curate.contracts import (
    ArtifactRef,
    ConstraintResult,
    EditPolicy,
    ResourceUsage,
    SkillValue,
    ValidationRequest,
    ValidationResult,
    digest,
    parse_contract,
    parse_program,
)
from self_curate.contracts.parsing import load_json
from self_curate.operators import execute_program
from pathlib import Path


def inputs():
    ctx = context(records(), target=64, tolerance=0, batch=8, share_tolerance=0)
    raw = fixture_program().to_dict()
    raw["program_id"] = "feedback-parent"
    for n in raw["nodes"]:
        if n["id"] in ("a", "b"):
            n["params"]["size"] = 6
    parent = parse_program(raw)
    objective = parse_contract(
        "ObjectiveSpec",
        load_json(
            (Path(__file__).parents[1] / "configs/p1.objective.fixture.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    request = ValidationRequest(
        "external-fixture-validation",
        MODEL,
        parent,
        execute_program(parent, ctx).batch,
        objective,
        ctx.context_hash,
        "predicted_gain",
        ArtifactRef("fixture://external-protocol", digest("external-protocol-v1")),
        "mock",
    )
    result = ValidationResult(
        "declared-external-result",
        request.request_hash,
        "valid",
        (SkillValue("recovery", 0.2), SkillValue("regression", -0.02)),
        0.134,
        (ConstraintResult("retain-regression", -0.02, False),),
        None,
        ArtifactRef("fixture://external-result", digest("declared-external-values")),
        ResourceUsage(),
        "mock",
    )
    policy = EditPolicy(0.125, 0.01, 0.99, 4, 4, 128, 2, 30)
    return request, result, ctx, policy
