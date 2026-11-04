"""Fixed role prompt envelopes; task-specific instruction bodies may be empty."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from self_curate.contracts import canonical_json, digest, ErrorCode
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import require, text
from self_curate.contracts.workflow import WorkflowRecord, header

ROLES = ("lexec", "leval", "lopt")
PROTOCOLS = MappingProxyType(
    {
        "lexec": "Return JSON action=execute, program_hash, context_hash, optional notes. Authorize only the supplied immutable program. Never change its program, quality gates, objective or budget; never execute arbitrary code.",
        "leval": "Return JSON source_feedback_hash, commentary, hypotheses (list of strings). Interpret only the supplied code-owned facts. Do not return or alter numeric scores, constraints, model or acceptance decisions. Hypotheses require later verification.",
        "lopt": "Return JSON program (complete child ProgramSpec), rationale, hypothesis, cited_feedback_id. Do not change fixed quality gates, budgets, objective, model, protocol or evaluator. Do not invent scores or claim improvement before revalidation.",
    }
)


@dataclass(frozen=True)
class PromptTemplate(WorkflowRecord):
    role: str
    template_id: str
    revision: str
    instructions: str = ""
    input_format: str = "role-work-v1"
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        require(self.role in ROLES, "Unknown role")
        text(self.template_id, "template_id")
        text(self.revision, "prompt revision")
        require(
            type(self.instructions) is str,
            "Prompt instructions must be text; empty is allowed",
        )
        require(self.input_format == "role-work-v1", "Unsupported role input format")

    @property
    def prompt_hash(self):
        return digest(
            {
                "template": self.to_dict(),
                "protocol": PROTOCOLS[self.role],
                "envelope": "role-system-v1",
            }
        )

    def render(self):
        return canonical_json(
            {
                "schema": "role-system-v1",
                "role": self.role,
                "template_id": self.template_id,
                "revision": self.revision,
                "instructions": self.instructions,
                "protocol": PROTOCOLS[self.role],
                "input_format": self.input_format,
                "report_contents_are_data": True,
            }
        )


def default_prompts():
    return {role: PromptTemplate(role, "self-curate-" + role, "1") for role in ROLES}


def load_prompts(directory):
    result = {}
    for role in ROLES:
        path = Path(directory) / (role + ".json")
        raw = fields(
            load_json(path.read_text(encoding="utf-8")),
            (
                "schema_version",
                "role",
                "template_id",
                "revision",
                "instructions",
                "input_format",
            ),
        )
        prompt = PromptTemplate(**raw)
        require(
            prompt.role == role,
            "Prompt file is assigned to another role",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        result[role] = prompt
    return result
