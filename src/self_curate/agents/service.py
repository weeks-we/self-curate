"""One imported-result feedback/proposal step, with P1 execution and no evaluation."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ComputeCost,
    ResourceUsage,
    ErrorCode,
    digest,
    parse_feedback_contract,
)
from self_curate.contracts.feedback import (
    ValidationRequest,
    ValidationResult,
    LevalFeedback,
    LoptProposal,
)
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require, text
from self_curate.data import read_pool
from self_curate.data.pool import _publish, _staging_directory, write_json, file_sha256
from self_curate.operators import (
    load_preparation,
    publish_preparation,
    PreparedSelection,
    prepare_selection,
    execute_program,
)
from .leval import build_feedback
from .lopt import propose_program, ProposalAttempt, program_diff


def request_from_preparation(
    prepared_path, pool_path, *, request_id, measure, protocol
):
    """Produce a pinned external-verification request from an existing P1 handoff."""
    root = Path(prepared_path)
    view = read_pool(pool_path).metric_view()
    prepared = load_preparation(root, view)
    require(
        prepared.status == "feasible",
        "Only feasible P1 handoffs can request validation",
    )
    context = prepared.context
    workflow = prepared.workflow
    return ValidationRequest(
        request_id,
        workflow.model_ref,
        workflow.program,
        workflow.batch,
        workflow.objective,
        context.context_hash,
        measure,
        protocol,
        workflow.backend,
    )


@dataclass(frozen=True)
class FeedbackStep:
    request: ValidationRequest
    result: ValidationResult
    feedback: LevalFeedback
    proposal: LoptProposal
    attempts: tuple[ProposalAttempt, ...]
    ledger: CostLedger
    candidate: PreparedSelection | None
    next_request: ValidationRequest | None

    def __post_init__(self):
        require(
            isinstance(self.request, ValidationRequest)
            and isinstance(self.result, ValidationResult)
            and isinstance(self.feedback, LevalFeedback)
            and isinstance(self.proposal, LoptProposal)
            and isinstance(self.ledger, CostLedger),
            "Invalid feedback step contracts",
        )
        require(
            type(self.attempts) is tuple
            and all(isinstance(a, ProposalAttempt) for a in self.attempts),
            "Invalid proposal attempts",
        )
        require(
            self.feedback
            == build_feedback(
                self.request,
                self.result,
                memory_namespace=self.feedback.memory_namespace,
            )
            and self.proposal.feedback_hash == self.feedback.record_hash
            and self.proposal.parent_recipe_hash == self.request.program.recipe_hash
            and self.proposal.model_ref == self.request.model_ref
            and self.proposal.backend == self.request.backend
            and self.ledger.backend == self.request.backend,
            "Feedback step identity mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            self.proposal.memory_namespace != self.feedback.memory_namespace,
            "Role memories must remain separate",
        )
        require(
            (self.candidate is not None) == (self.proposal.status == "proposed"),
            "Proposed program needs a P1 execution result",
        )
        if self.proposal.program is not None:
            require(
                self.proposal.changes
                == program_diff(self.request.program, self.proposal.program)
                and self.proposal.program.parent_id == self.request.program.program_id,
                "Proposal diff/parent lineage mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        if self.candidate is not None:
            require(
                isinstance(self.candidate, PreparedSelection)
                and self.candidate.program == self.proposal.program
                and self.candidate.objective == self.request.objective
                and self.candidate.context.context_hash
                == self.request.selection_context_hash
                and self.candidate.backend == self.request.backend,
                "Candidate P1 handoff mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            indexed = {e.event_id: e for e in self.ledger.events}
            require(
                all(indexed.get(e.event_id) == e for e in self.candidate.ledger.events),
                "Candidate selection cost missing from feedback ledger",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        feasible = self.candidate is not None and self.candidate.status == "feasible"
        require(
            (self.next_request is not None) == feasible,
            "Only a feasible candidate gets the next validation request",
        )
        if self.next_request is not None:
            expected = ValidationRequest(
                "validate-" + self.proposal.record_hash,
                self.request.model_ref,
                self.proposal.program,
                self.candidate.workflow.batch,
                self.request.objective,
                self.request.selection_context_hash,
                self.request.measure,
                self.request.protocol,
                self.request.backend,
            )
            require(
                self.next_request == expected,
                "Next verification request changed fixed anchors",
                ErrorCode.ARTIFACT_MISMATCH,
            )

    @property
    def status(self):
        if self.proposal.status == "blocked":
            return (
                "needs_evidence"
                if self.feedback.status != "valid"
                else "proposal_blocked"
            )
        if self.candidate.status != "feasible":
            return "candidate_" + self.candidate.status
        return "awaiting_validation"

    def to_dict(self):
        return {
            "schema_version": "1",
            "status": self.status,
            "backend": self.request.backend,
            "source_request_hash": self.request.request_hash,
            "source_result_hash": self.result.record_hash,
            "feedback_hash": self.feedback.record_hash,
            "proposal_hash": self.proposal.record_hash,
            "parent_batch_id": self.request.batch.batch_id,
            "candidate_batch_id": (
                None
                if self.candidate is None or self.candidate.workflow is None
                else self.candidate.workflow.batch.batch_id
            ),
            "next_request_hash": (
                None if self.next_request is None else self.next_request.request_hash
            ),
            "improvement_verified": False,
            "validation_implemented": False,
            "ledger_hash": self.ledger.record_hash,
        }


def run_feedback_step(
    request,
    result,
    context,
    policy,
    *,
    attempt_id,
    invocation_id,
    method="rule",
    client=None,
    leval_memory="leval",
    lopt_memory="lopt",
):
    text(attempt_id, "attempt_id")
    text(invocation_id, "invocation_id")
    require(
        context.model_ref == request.model_ref
        and context.context_hash == request.selection_context_hash,
        "Feedback step uses another model/context",
        ErrorCode.STALE_CACHE,
    )
    require(
        policy.max_nodes <= context.max_nodes,
        "Edit policy cannot increase fixed system node limit",
    )
    parent = execute_program(request.program, context)
    require(
        parent.batch == request.batch,
        "Source validation is not bound to the exact executed parent batch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    started = perf_counter()
    feedback = build_feedback(request, result, memory_namespace=leval_memory)
    elapsed = max(0.0, perf_counter() - started)
    proposal_started = perf_counter()
    proposal, attempts = propose_program(
        request,
        result,
        feedback,
        policy,
        method=method,
        client=client,
        memory_namespace=lopt_memory,
        context=context,
    )
    proposal_elapsed = max(0.0, perf_counter() - proposal_started)
    events = []

    def event(suffix, subject, status, usage, payload_hash):
        event_id = digest(
            {"attempt": attempt_id, "invocation": invocation_id, "role_call": suffix}
        )
        events.append(
            CostEvent(
                event_id,
                attempt_id,
                "role_call",
                subject,
                request.model_ref.model_hash,
                status,
                usage,
                ArtifactRef("record://role-output/" + payload_hash, payload_hash),
                request.backend,
            )
        )

    event(
        "leval",
        feedback.feedback_id,
        "completed",
        ResourceUsage(ComputeCost(elapsed, 0, 0), 0, 0, 0, 0, 0, 0, 0, 0),
        feedback.record_hash,
    )
    for call in attempts:
        event(
            "lopt-" + str(call.index),
            proposal.proposal_id,
            "completed" if call.status == "accepted_proposal" else "failed",
            call.usage,
            digest(call.to_dict()),
        )
    if not attempts:
        event(
            "lopt-blocked",
            proposal.proposal_id,
            "completed",
            ResourceUsage(ComputeCost(proposal_elapsed, 0, 0), 0, 0, 0, 0, 0, 0, 0, 0),
            proposal.record_hash,
        )
    candidate = None
    next_request = None
    if proposal.program is not None:
        candidate = prepare_selection(
            proposal.program,
            context,
            request.objective,
            backend=request.backend,
            attempt_id=attempt_id,
            invocation_id=invocation_id + "-candidate",
        )
        events.extend(candidate.ledger.events)
        if candidate.workflow is not None:
            next_request = ValidationRequest(
                "validate-" + proposal.record_hash,
                request.model_ref,
                proposal.program,
                candidate.workflow.batch,
                request.objective,
                context.context_hash,
                request.measure,
                request.protocol,
                request.backend,
            )
    return FeedbackStep(
        request,
        result,
        feedback,
        proposal,
        attempts,
        CostLedger(
            "feedback-step-" + digest((attempt_id, invocation_id)),
            tuple(events),
            request.backend,
        ),
        candidate,
        next_request,
    )


def feedback_step_to_directory(
    prepared_path,
    pool_path,
    request_path,
    result_path,
    policy_path,
    output,
    *,
    attempt_id,
    invocation_id,
    method="rule",
    client=None,
):
    output = Path(output).resolve()
    require(
        not output.exists(), "Feedback output already exists", ErrorCode.OUTPUT_EXISTS
    )
    root = Path(prepared_path)
    view = read_pool(pool_path).metric_view()
    request = parse_feedback_contract(
        "ValidationRequest", load_json(Path(request_path).read_text(encoding="utf-8"))
    )
    result = parse_feedback_contract(
        "ValidationResult", load_json(Path(result_path).read_text(encoding="utf-8"))
    )
    policy = parse_feedback_contract(
        "EditPolicy", load_json(Path(policy_path).read_text(encoding="utf-8"))
    )
    prepared = load_preparation(
        root,
        view,
        expected_model=request.model_ref,
        expected_objective=request.objective,
        expected_batch_id=request.batch.batch_id,
    )
    require(
        prepared.status == "feasible"
        and request.program == prepared.program
        and request.batch == prepared.workflow.batch,
        "Request differs from saved selection",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    step = run_feedback_step(
        request,
        result,
        prepared.context,
        policy,
        attempt_id=attempt_id,
        invocation_id=invocation_id,
        method=method,
        client=client,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_directory(output)
    payloads = {
        "step.json": step.to_dict(),
        "validation_request.json": request.to_dict(),
        "external_validation.json": result.to_dict(),
        "edit_policy.json": policy.to_dict(),
        "leval_feedback.json": step.feedback.to_dict(),
        "lopt_proposal.json": step.proposal.to_dict(),
        "lopt_attempts.json": [a.to_dict() for a in step.attempts],
        "ledger.json": step.ledger.to_dict(),
        "role_memories.json": {
            "leval": {
                "namespace": step.feedback.memory_namespace,
                "model_hash": request.model_ref.model_hash,
                "feedback_hash": step.feedback.record_hash,
            },
            "lopt": {
                "namespace": step.proposal.memory_namespace,
                "model_hash": request.model_ref.model_hash,
                "proposal_hash": step.proposal.record_hash,
            },
        },
    }
    if step.proposal.program is not None:
        payloads["candidate_program.json"] = step.proposal.program.to_dict()
    if step.candidate is not None:
        payloads["candidate_selection.json"] = step.candidate.outcome
        if step.candidate.workflow is not None:
            payloads["candidate_workflow.json"] = step.candidate.workflow.to_dict()
    if step.next_request is not None:
        payloads["next_validation_request.json"] = step.next_request.to_dict()
    for name, value in payloads.items():
        write_json(staging / name, value)
    if step.candidate is not None:
        publish_preparation(step.candidate, staging / "candidate_prepared")
    write_json(
        staging / "checksums.json",
        {
            path.relative_to(staging).as_posix(): file_sha256(path)
            for path in staging.rglob("*")
            if path.is_file()
        },
    )
    _publish(staging, output)
    return step
