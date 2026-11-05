"""Portable, checked three-role cycle handoff. No model calls or resampling."""

from pathlib import Path

from self_curate.contracts import (
    ContractError,
    CostLedger,
    ErrorCode,
    ValidationRequest,
    canonical_json,
    digest,
    parse_contract,
    parse_feedback_contract,
    verify_batch,
)
from self_curate.contracts.parsing import fields, load_json, sequence
from self_curate.contracts.types import require, integer, sha256
from self_curate.data.pool import _publish, _staging_directory, write_json, file_sha256
from self_curate.operators import load_context, load_preparation, publish_preparation
from .lopt import ProposalAttempt
from .roles import RoleCycle, RoleEvaluation
from .runtime import RoleReply
from .prompts import ROLES

ROOT_FILES = {
    "cycle.json",
    "source_request.json",
    "source_result.json",
    "context.json",
    "evaluation.json",
    "proposal.json",
    "edit_policy.json",
    "attempts.json",
    "model_ledger.json",
    "ledger.json",
}
P1_FILES = {
    "prepared.json",
    "program.json",
    "context.json",
    "objective.json",
    "outcome.json",
    "ledger.json",
    "trace.json",
    "workflow.json",
    "batch.json",
    "checksums.json",
}


def _costs(cycle, model_ledger):
    require(
        isinstance(model_ledger, CostLedger)
        and model_ledger.backend == cycle.proposal.backend,
        "Cycle model ledger backend mismatch",
    )
    require(
        cycle.model_call_ids is not None and cycle.attempt_id is not None,
        "Cycle lacks physical call/attempt anchors; use RoleTeam.feedback_step",
    )
    indexed = {e.event_id: e for e in model_ledger.events}
    require(
        set(cycle.model_call_ids).issubset(indexed),
        "Cycle physical model costs are missing",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    selected = tuple(indexed[key] for key in cycle.model_call_ids)
    if cycle.proposal.method == "model":
        require(
            sum(e.subject_id == "lopt" for e in selected) == len(cycle.attempts),
            "Logical proposal attempts differ from physical calls",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    require(
        all(
            e.operation == "role_call"
            and e.subject_id in ROLES
            and e.model_hash == cycle.proposal.model_ref.model_hash
            and e.attempt_id == cycle.attempt_id
            for e in selected
        ),
        "Cycle physical model cost identity mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    reply = cycle.evaluation.interpretation
    if reply is not None:
        require(
            reply.call_id in cycle.model_call_ids
            and indexed[reply.call_id].subject_id == "leval"
            and indexed[reply.call_id].status == "completed"
            and indexed[reply.call_id].usage == reply.usage,
            "Leval interpretation cost reference mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    model = CostLedger(
        "cycle-model-" + digest(cycle.model_call_ids), selected, cycle.proposal.backend
    )
    events = selected + (
        () if cycle.candidate is None else cycle.candidate.ledger.events
    )
    total = CostLedger(
        "cycle-"
        + digest(
            (
                cycle.source_request.request_hash,
                cycle.proposal.record_hash,
                tuple(e.event_id for e in events),
            )
        ),
        events,
        cycle.proposal.backend,
    )
    return model, total


def _write_role_cycle(cycle, staging, *, model_ledger):
    """Fill an empty caller-owned unpublished directory; outer owner publishes it."""
    require(
        isinstance(cycle, RoleCycle)
        and cycle.source_request is not None
        and cycle.source_policy is not None,
        "Expected complete RoleTeam cycle",
    )
    cycle.validate_source(
        cycle.source_request, cycle.source_result, cycle.source_context
    )
    model, total = _costs(cycle, model_ledger)
    require(
        staging.is_dir() and not any(staging.iterdir()), "Cycle staging must be empty"
    )
    payloads = {
        "cycle.json": cycle.to_dict(),
        "source_request.json": cycle.source_request.to_dict(),
        "source_result.json": cycle.source_result.to_dict(),
        "context.json": cycle.source_context.to_dict(),
        "evaluation.json": cycle.evaluation.to_dict(),
        "proposal.json": cycle.proposal.to_dict(),
        "edit_policy.json": cycle.source_policy.to_dict(),
        "attempts.json": [a.to_dict() for a in cycle.attempts],
        "model_ledger.json": model.to_dict(),
        "ledger.json": total.to_dict(),
    }
    if cycle.next_request is not None:
        payloads["next_validation_request.json"] = cycle.next_request.to_dict()
    for name, value in payloads.items():
        write_json(staging / name, value)
    if cycle.candidate is not None:
        publish_preparation(cycle.candidate, staging / "candidate_prepared")
    write_json(
        staging / "checksums.json",
        {
            p.relative_to(staging).as_posix(): file_sha256(p)
            for p in staging.rglob("*")
            if p.is_file()
        },
    )
    return {
        "status": cycle.status,
        "ledger_hash": total.record_hash,
        "next_request_hash": (
            None if cycle.next_request is None else cycle.next_request.request_hash
        ),
    }


def publish_role_cycle(cycle, output, *, model_ledger):
    require(
        isinstance(cycle, RoleCycle)
        and cycle.source_request is not None
        and cycle.source_policy is not None,
        "Expected complete RoleTeam cycle",
    )
    output = Path(output).resolve()
    require(not output.exists(), "Cycle output already exists", ErrorCode.OUTPUT_EXISTS)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_directory(output)
    summary = _write_role_cycle(cycle, staging, model_ledger=model_ledger)
    _publish(staging, output)
    return {**summary, "output": str(output)}


def _attempt(raw):
    fields(
        raw,
        (
            "index",
            "status",
            "request_hash",
            "response_hash",
            "error",
            "usage",
            "request",
            "response",
        ),
    )
    integer(raw["index"], "attempt index", 1)
    require(
        raw["status"] in ("accepted_proposal", "invalid_proposal"),
        "Invalid proposal attempt status",
    )
    sha256(raw["request_hash"], "attempt request hash")
    if raw["response_hash"] is not None:
        sha256(raw["response_hash"], "attempt response hash")
    for name in ("request", "response"):
        if raw[name] is not None:
            require(
                digest(raw[name]) == raw[name + "_hash"],
                "Proposal attempt content/hash mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
    return ProposalAttempt(
        raw["index"],
        raw["status"],
        raw["request_hash"],
        raw["response_hash"],
        None if raw["error"] is None else canonical_json(raw["error"]),
        parse_contract("ResourceUsage", raw["usage"]),
        None if raw["request"] is None else canonical_json(raw["request"]),
        None if raw["response"] is None else canonical_json(raw["response"]),
    )


def load_role_cycle(
    path,
    view,
    *,
    expected_request,
    expected_candidate_batch_id=None,
    expected_policy=None,
):
    try:
        return _load_role_cycle(
            path,
            view,
            expected_request=expected_request,
            expected_candidate_batch_id=expected_candidate_batch_id,
            expected_policy=expected_policy,
        )
    except ContractError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        UnicodeError,
    ) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Missing or malformed role cycle",
            reason=type(exc).__name__,
        ) from exc


def _load_role_cycle(
    path, view, *, expected_request, expected_candidate_batch_id, expected_policy
):
    require(
        isinstance(expected_request, ValidationRequest),
        "Supply independent source ValidationRequest",
    )
    root = Path(path)
    checks = load_json((root / "checksums.json").read_text(encoding="utf-8"))
    require(
        type(checks) is dict and ROOT_FILES.issubset(checks),
        "Incomplete cycle publication",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    for name, expected in checks.items():
        require(
            name in ROOT_FILES
            or name == "next_validation_request.json"
            or name in {"candidate_prepared/" + n for n in P1_FILES},
            "Unknown cycle artifact",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        sha256(expected, "file checksum")
        require(
            file_sha256(root / name) == expected,
            "Cycle file checksum mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
            file=name,
        )
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"
    }
    require(
        actual == set(checks),
        "Unknown/missing cycle files",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        {p.name for p in root.iterdir()}
        == {name.split("/")[0] for name in checks} | {"checksums.json"},
        "Unknown cycle directories",
        ErrorCode.ARTIFACT_MISMATCH,
    )

    def read(name):
        return load_json((root / name).read_text(encoding="utf-8"))

    request = parse_feedback_contract("ValidationRequest", read("source_request.json"))
    require(
        request == expected_request,
        "Cycle source differs from independent request",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    result = parse_feedback_contract("ValidationResult", read("source_result.json"))
    context = load_context(root / "context.json", view)
    verify_batch(request.batch, view, model_ref=request.model_ref)
    model_ledger = parse_contract("CostLedger", read("model_ledger.json"))
    evaluation = fields(
        read("evaluation.json"),
        (
            "feedback",
            "feedback_hash",
            "interpretation",
            "interpretation_call_id",
            "numeric_source",
            "validation_implemented",
        ),
    )
    feedback = parse_feedback_contract("LevalFeedback", evaluation["feedback"])
    reply = None
    if evaluation["interpretation"] is not None:
        events = {e.event_id: e for e in model_ledger.events}
        key = evaluation["interpretation_call_id"]
        require(
            key in events,
            "Missing Leval interpretation call",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        reply = RoleReply(
            "leval",
            key,
            canonical_json(evaluation["interpretation"]),
            events[key].usage,
        )
    interpreted = RoleEvaluation(feedback, reply)
    require(
        canonical_json(interpreted.to_dict()) == canonical_json(evaluation),
        "Leval artifact metadata mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    proposal = parse_feedback_contract("LoptProposal", read("proposal.json"))
    policy = parse_feedback_contract("EditPolicy", read("edit_policy.json"))
    if expected_policy is not None:
        require(
            policy == expected_policy,
            "Cycle differs from independent edit policy",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    attempts = tuple(
        _attempt(v) for v in sequence(read("attempts.json"), "proposal attempts")
    )
    candidate = (
        load_preparation(
            root / "candidate_prepared",
            view,
            expected_model=request.model_ref,
            expected_objective=request.objective,
            expected_batch_id=expected_candidate_batch_id,
        )
        if (root / "candidate_prepared").exists()
        else None
    )
    if expected_candidate_batch_id is not None:
        require(
            candidate is not None,
            "Expected candidate is absent",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    next_request = (
        parse_feedback_contract(
            "ValidationRequest", read("next_validation_request.json")
        )
        if (root / "next_validation_request.json").exists()
        else None
    )
    meta = read("cycle.json")
    calls = tuple(sequence(meta["model_call_ids"], "cycle model call IDs"))
    cycle = RoleCycle(
        interpreted,
        proposal,
        attempts,
        candidate,
        next_request,
        request,
        result,
        context,
        calls,
        meta["attempt_id"],
        policy,
    )
    require(
        canonical_json(cycle.to_dict()) == canonical_json(meta),
        "Cycle summary differs from exact artifacts",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    subset, total = _costs(cycle, model_ledger)
    require(
        subset == model_ledger
        and total == parse_contract("CostLedger", read("ledger.json")),
        "Cycle cost ledger differs from physical calls/selection",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    return cycle
