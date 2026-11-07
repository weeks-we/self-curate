"""Lopt direction is input data. No skill guessing, ranking, or model updates here."""

from dataclasses import dataclass

from self_curate.contracts import ArtifactRef, ModelRef, digest
from self_curate.contracts.parsing import artifact, fields, model_ref
from self_curate.contracts.types import integer, require, text


@dataclass(frozen=True)
class SkillFocus:
    outer_round: int
    model_ref: ModelRef
    skills: tuple[str, ...]
    feedback: ArtifactRef
    rationale: str
    schema_version: str = "1"

    def __post_init__(self):
        integer(self.outer_round, "outer refresh round", 2)
        require(
            self.schema_version == "1" and isinstance(self.model_ref, ModelRef),
            "Invalid skill focus",
        )
        require(
            type(self.skills) is tuple and len(set(self.skills)) == len(self.skills),
            "Unique immutable focus skills required",
        )
        for skill in self.skills:
            text(skill, "focus skill")
        require(
            isinstance(self.feedback, ArtifactRef), "Outer feedback reference required"
        )
        text(self.rationale, "focus rationale")

    @property
    def focus_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "outer_round": self.outer_round,
            "model_ref": self.model_ref.to_dict(),
            "skills": list(self.skills),
            "feedback": self.feedback.to_dict(),
            "rationale": self.rationale,
        }


def parse_skill_focus(value):
    row = fields(
        value,
        (
            "schema_version",
            "outer_round",
            "model_ref",
            "skills",
            "feedback",
            "rationale",
        ),
    )
    require(type(row["skills"]) is list, "Focus skills must be an array")
    return SkillFocus(
        row["outer_round"],
        model_ref(row["model_ref"]),
        tuple(row["skills"]),
        artifact(row["feedback"]),
        row["rationale"],
        row["schema_version"],
    )


def focus_from_outer_decision(context, decision, *, outer_round, skills, rationale):
    """Optional checked bridge; actual Lopt direction selection is supplied later."""
    from self_curate.orchestration.contracts import DecisionContext, AcceptanceDecision

    require(
        isinstance(context, DecisionContext)
        and isinstance(decision, AcceptanceDecision),
        "Typed outer decision evidence required",
    )
    require(
        context.policy is not None
        and decision.context_hash == context.record_hash
        and decision.outcome in ("accept", "reject"),
        "Outer decision is pending or belongs to another candidate",
    )
    require(
        context.evaluation.protocol_verified
        and context.evaluation.report.status == "valid",
        "Valid outer task measurement required",
    )
    model = (
        context.training.result.candidate_model
        if decision.outcome == "accept"
        else context.accepted_model
    )
    evidence_hash = digest(
        {"context": context.record_hash, "decision": decision.record_hash}
    )
    return SkillFocus(
        outer_round,
        model,
        tuple(skills),
        ArtifactRef("record://outer-refresh-feedback/" + evidence_hash, evidence_hash),
        rationale,
    )
