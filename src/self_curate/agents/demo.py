"""Explicit CPU fixture client and a three-role example; no model weights."""

from pathlib import Path
import uuid

from self_curate.contracts import (
    canonical_json,
    digest,
    parse_feedback_contract,
    parse_program,
    ErrorCode,
)
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require
from self_curate.data.pool import (
    build_pool,
    load_config,
    write_json,
    _publish,
    file_sha256,
)
from self_curate.operators import load_preparation, publish_preparation
from self_curate.data import read_pool
from .lopt import _rule_candidate
from .prompts import ROLES, load_prompts
from .roles import RoleTeam


class FixtureRoleClient:
    """A declared mock that follows the role output schemas, never a fallback."""

    def __init__(self, model_ref):
        self.model_ref = model_ref
        self.calls = []

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        require(role in ROLES and payload["role"] == role, "Fixture role mismatch")
        self.calls.append(
            load_json(
                canonical_json(
                    {
                        "role": role,
                        "system_prompt": system_prompt,
                        "payload": payload,
                        "timeout_seconds": timeout_seconds,
                    }
                )
            )
        )
        work = payload["work"]
        if role == "lexec":
            content = {
                "action": "execute",
                "program_hash": work["program_hash"],
                "context_hash": work["context_hash"],
                "notes": "Explicit mock control decision",
            }
        elif role == "leval":
            content = {
                "source_feedback_hash": work["feedback_hash"],
                "commentary": "Explicit fixture interpretation; external numbers are unchanged.",
                "hypotheses": [
                    "Explore the supplied constraint hypothesis; improvement is unverified."
                ],
            }
        else:
            source = work["input"]
            parent = parse_program(source["parent_program"])
            feedback = parse_feedback_contract("LevalFeedback", source["feedback"])
            policy = parse_feedback_contract("EditPolicy", source["fixed_policy"])
            raw, rationale, hypothesis = _rule_candidate(parent, feedback, policy)
            require(raw is not None, "Fixture cannot produce this proposal")
            raw["program_id"] = "fixture-role-" + digest(source)[:24]
            raw["parent_id"] = parent.program_id
            content = {
                "program": raw,
                "rationale": "Explicit fixture: " + rationale,
                "hypothesis": hypothesis,
                "cited_feedback_id": feedback.feedback_id,
            }
        return {"content": content}


def run_role_demo(output, *, project_root=".", prompt_directory=None):
    """Publish a local fixture cycle, restored private histories and exact outputs."""
    output = Path(output).resolve()
    root = Path(project_root).resolve()
    require(
        not output.exists(), "Role demo output already exists", ErrorCode.OUTPUT_EXISTS
    )
    request = parse_feedback_contract(
        "ValidationRequest",
        load_json(
            (root / "configs/feedback.request.fixture.json").read_text(encoding="utf-8")
        ),
    )
    result = parse_feedback_contract(
        "ValidationResult",
        load_json(
            (root / "configs/feedback.result.fixture.json").read_text(encoding="utf-8")
        ),
    )
    policy = parse_feedback_contract(
        "EditPolicy",
        load_json(
            (root / "configs/feedback.policy.fixture.json").read_text(encoding="utf-8")
        ),
    )
    prompts = load_prompts(prompt_directory or root / "configs/roles")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    build_pool(load_config(root / "configs/m2.data.fixture.json"), staging / "pool")
    from self_curate.operators import prepare_to_directory

    prepare_to_directory(
        staging / "pool",
        root / "configs/programs/feedback.parent.fixture.json",
        root / "configs/m2.fixture.json",
        root / "configs/p1.objective.fixture.json",
        staging / "parent",
        backend="mock",
        attempt_id="role-demo",
        invocation_id="parent",
    )
    context = load_preparation(
        staging / "parent", read_pool(staging / "pool").metric_view()
    ).context
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client,
        model=request.model_ref,
        session_id="role-demo",
        prompts=prompts,
        backend="mock",
    )
    first = team.feedback_step(
        request,
        result,
        context,
        policy,
        attempt_id="role-demo",
        invocation_id="child-1",
    )
    require(
        first.status == "awaiting_validation", "Fixture first cycle must be feasible"
    )
    publish_preparation(first.candidate, staging / "candidate-1")
    write_json(staging / "cycle-1.json", first.to_dict())
    write_json(staging / "evaluation-1.json", first.evaluation.to_dict())
    write_json(staging / "proposal-1.json", first.proposal.to_dict())
    write_json(staging / "next-validation-1.json", first.next_request.to_dict())
    team.save(staging / "history-1")
    restored = RoleTeam.restore(
        staging / "history-1",
        client,
        expected_model=request.model_ref,
        expected_prompts=prompts,
    )
    from dataclasses import replace

    external = replace(
        result,
        result_id="declared-fixture-second-result",
        request_hash=first.next_request.request_hash,
    )
    second = restored.feedback_step(
        first.next_request,
        external,
        context,
        policy,
        attempt_id="role-demo",
        invocation_id="child-2",
    )
    publish_preparation(second.candidate, staging / "candidate-2")
    write_json(staging / "cycle-2.json", second.to_dict())
    write_json(staging / "external-fixture-result-2.json", external.to_dict())
    restored.save(staging / "history-2")
    write_json(staging / "client_calls.json", client.calls)
    require(
        {call["role"] for call in client.calls} == set(ROLES),
        "All three roles must call the same client",
    )
    summary = {
        "schema_version": "1",
        "backend": "mock",
        "model_hash": request.model_ref.model_hash,
        "shared_client": True,
        "role_models": {r: getattr(restored, r).model_ref.model_hash for r in ROLES},
        "memory_namespaces": {r: restored.runtime.namespace(r) for r in ROLES},
        "history_events": {r: len(restored.runtime.history(r)) for r in ROLES},
        "prompt_hashes": {r: prompts[r].prompt_hash for r in ROLES},
        "model_calls": len(client.calls),
        "cycle_statuses": [first.status, second.status],
        "history_restored": True,
        "validation_implemented": False,
        "real_model_called": False,
        "improvement_verified": False,
    }
    write_json(staging / "summary.json", summary)
    write_json(
        staging / "checksums.json",
        {
            p.relative_to(staging).as_posix(): file_sha256(p)
            for p in staging.rglob("*")
            if p.is_file()
        },
    )
    _publish(staging, output)
    return summary
