"""Three thin role agents around one parameter identity and existing tools."""

from dataclasses import dataclass

from self_curate.contracts import (
    ErrorCode,
    ValidationRequest,
    ValidationResult,
    digest,
    LevalFeedback,
    LoptProposal,
    EditPolicy,
    ProgramSpec,
    ObjectiveSpec,
)
from self_curate.contracts.parsing import fields
from self_curate.contracts.types import require, text
from self_curate.operators import (
    execute_program,
    prepare_selection,
    PreparedSelection,
    ExecutionContext,
)
from self_curate.operators.preparation import _domain
from .leval import build_feedback
from .lopt import propose_program, SYSTEM_PROMPT, ProposalAttempt, program_diff
from .runtime import RoleRuntime, RoleReply
from .scope import RoleScope, request_scope


class RoleAgent:
    role = None

    def __init__(self, runtime):
        require(isinstance(runtime, RoleRuntime), "Expected shared RoleRuntime")
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime

    @property
    def model_ref(self):
        return self._runtime.model_ref

    @property
    def prompt(self):
        return self._runtime.prompts[self.role]

    @property
    def memory_namespace(self):
        return self._runtime.namespace(self.role)

    @property
    def history(self):
        return self._runtime.history(self.role)

    def generate(self, work, *, timeout_seconds=30, validator=None):
        return self._runtime.generate(
            self.role, work, timeout_seconds=timeout_seconds, validator=validator
        )


class ExecutionAgent(RoleAgent):
    role = "lexec"

    def execute(
        self,
        program: ProgramSpec,
        context: ExecutionContext,
        objective: ObjectiveSpec,
        *,
        attempt_id,
        invocation_id,
        use_model=True,
        timeout_seconds=30,
    ) -> PreparedSelection:
        require(
            isinstance(program, ProgramSpec)
            and isinstance(context, ExecutionContext)
            and isinstance(objective, ObjectiveSpec),
            "Executor needs typed program/context/objective",
        )
        text(attempt_id, "attempt_id")
        text(invocation_id, "invocation_id")
        _domain(context, objective, self.runtime.backend)
        require(
            context.model_ref == self.model_ref,
            "Executor uses another checkpoint",
            ErrorCode.STALE_CACHE,
        )
        require(type(use_model) is bool, "use_model must be explicit boolean")
        self.runtime.bind_attempt(attempt_id)
        scope = RoleScope(
            selection_context_hash=context.context_hash,
            objective_hash=objective.objective_hash,
        )
        self.runtime.bind_scope(scope)
        work = {
            "program": program.to_dict(),
            "program_hash": program.recipe_hash,
            "context_hash": context.context_hash,
            "selection_context": context.role_summary(),
            "objective": objective.to_dict(),
            "scope": scope.to_dict(),
        }
        if use_model:

            def validate(content):
                fields(content, ("action", "program_hash", "context_hash"), ("notes",))
                require(
                    content["action"] == "execute"
                    and content["program_hash"] == program.recipe_hash
                    and content["context_hash"] == context.context_hash,
                    "Executor may only authorize the exact supplied program/context",
                    ErrorCode.ARTIFACT_MISMATCH,
                )
                require(
                    "notes" not in content or type(content["notes"]) is str,
                    "Executor notes must be text",
                )

            self.generate(work, timeout_seconds=timeout_seconds, validator=validate)
        return self.runtime.run_tool(
            self.role,
            "execute_selection",
            work,
            lambda: prepare_selection(
                program,
                context,
                objective,
                backend=self.runtime.backend,
                attempt_id=attempt_id,
                invocation_id=invocation_id,
            ),
            lambda result: {**result.to_dict(), "error": result.error},
        )


@dataclass(frozen=True)
class RoleEvaluation:
    feedback: LevalFeedback
    interpretation: RoleReply | None

    def __post_init__(self):
        require(
            isinstance(self.feedback, LevalFeedback),
            "Expected code-owned Leval feedback",
        )
        if self.interpretation is not None:
            require(
                self.feedback.status == "valid"
                and isinstance(self.interpretation, RoleReply)
                and self.interpretation.role == "leval"
                and self.interpretation.content.get("source_feedback_hash")
                == self.feedback.record_hash,
                "Interpretation/feedback mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            content = fields(
                self.interpretation.content,
                ("source_feedback_hash", "commentary", "hypotheses"),
            )
            text(content["commentary"], "Leval commentary")
            require(
                type(content["hypotheses"]) is list
                and all(
                    type(v) is str and bool(v.strip()) for v in content["hypotheses"]
                ),
                "Interpretation hypotheses must be text",
            )

    def to_dict(self):
        return {
            "feedback": self.feedback.to_dict(),
            "feedback_hash": self.feedback.record_hash,
            "interpretation": (
                None if self.interpretation is None else self.interpretation.content
            ),
            "interpretation_call_id": (
                None if self.interpretation is None else self.interpretation.call_id
            ),
            "numeric_source": "external_validation",
            "validation_implemented": False,
        }


class EvaluationAgent(RoleAgent):
    role = "leval"

    def evaluate(
        self,
        request: ValidationRequest,
        result: ValidationResult,
        *,
        use_model=True,
        timeout_seconds=30,
    ) -> RoleEvaluation:
        require(
            isinstance(request, ValidationRequest)
            and isinstance(result, ValidationResult),
            "Evaluator needs typed request/result",
        )
        require(
            request.model_ref == self.model_ref
            and request.backend == self.runtime.backend,
            "Evaluator model/backend mismatch",
            ErrorCode.STALE_CACHE,
        )
        require(type(use_model) is bool, "use_model must be explicit boolean")
        scope = request_scope(request)
        self.runtime.bind_scope(scope)
        feedback = self.runtime.run_tool(
            self.role,
            "interpret_validation",
            {
                "request_hash": request.request_hash,
                "external_result": result.to_dict(),
                "scope": scope.to_dict(),
            },
            lambda: build_feedback(
                request, result, memory_namespace=self.memory_namespace
            ),
            lambda value: value.to_dict(),
        )
        interpretation = None
        if use_model and feedback.status == "valid":

            def validate(content):
                fields(content, ("source_feedback_hash", "commentary", "hypotheses"))
                require(
                    content["source_feedback_hash"] == feedback.record_hash,
                    "Interpretation cites another feedback",
                    ErrorCode.ARTIFACT_MISMATCH,
                )
                text(content["commentary"], "Leval commentary")
                require(
                    type(content["hypotheses"]) is list
                    and all(
                        type(v) is str and bool(v.strip())
                        for v in content["hypotheses"]
                    ),
                    "Interpretation hypotheses must be text",
                )

            interpretation = self.generate(
                {"feedback_hash": feedback.record_hash, "feedback": feedback.to_dict()},
                timeout_seconds=timeout_seconds,
                validator=validate,
            )
        return RoleEvaluation(feedback, interpretation)


class _OptimizerClient:
    def __init__(self, agent, handoff):
        self.agent = agent
        self.handoff = handoff
        self.call_ids = []

    @property
    def model_ref(self):
        # Expose the live backend identity to the old bounded proposal adapter.
        return self.agent.runtime.shared_client.model_ref

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        require(
            role == "lopt" and system_prompt == SYSTEM_PROMPT,
            "Optimizer adapter protocol mismatch",
        )
        require(
            payload.get("memory_namespace") == self.agent.memory_namespace,
            "Optimizer requested another role memory",
        )
        before = {
            e.call_id
            for e in self.agent.history
            if e.operation == "model" and e.phase == "before"
        }
        try:
            reply = self.agent.generate(
                {
                    "input": payload,
                    "handoff": self.handoff,
                    "adapter_protocol_hash": digest(system_prompt),
                },
                timeout_seconds=timeout_seconds,
            )
        finally:
            self.call_ids.extend(
                e.call_id
                for e in self.agent.history
                if e.operation == "model"
                and e.phase == "before"
                and e.call_id not in before
            )
        return {"content": reply.content, "usage": reply.usage.to_dict()}


class OptimizationAgent(RoleAgent):
    role = "lopt"

    def optimize(
        self,
        request,
        result,
        policy,
        *,
        context,
        evaluation=None,
        method="model",
        previous_cycle=None,
    ):
        require(
            isinstance(request, ValidationRequest)
            and isinstance(result, ValidationResult)
            and isinstance(policy, EditPolicy)
            and isinstance(context, ExecutionContext),
            "Optimizer needs typed request/result/policy/context",
        )
        require(
            request.model_ref == self.model_ref
            and request.backend == self.runtime.backend,
            "Optimizer model/backend mismatch",
            ErrorCode.STALE_CACHE,
        )
        require(
            context.model_ref == request.model_ref
            and context.context_hash == request.selection_context_hash
            and policy.max_nodes <= context.max_nodes,
            "Optimizer context/edit policy differs from fixed request",
            ErrorCode.STALE_CACHE,
        )
        _domain(context, request.objective, self.runtime.backend)
        if evaluation is None:
            feedback = build_feedback(
                request, result, memory_namespace=self.runtime.namespace("leval")
            )
            handoff = None
        else:
            require(
                isinstance(evaluation, RoleEvaluation),
                "Expected explicit Leval artifact",
            )
            feedback = evaluation.feedback
            require(
                feedback
                == build_feedback(
                    request, result, memory_namespace=self.runtime.namespace("leval")
                ),
                "Leval handoff belongs to another request/role",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            handoff = evaluation.to_dict()
        if previous_cycle is not None:
            require(
                isinstance(previous_cycle, RoleCycle)
                and previous_cycle.source_request is not None,
                "Expected complete previous RoleCycle",
            )
            previous_cycle.validate_source(
                previous_cycle.source_request,
                previous_cycle.source_result,
                previous_cycle.source_context,
            )
            require(
                previous_cycle.proposal.model_ref == request.model_ref
                and previous_cycle.source_context.context_hash == context.context_hash
                and previous_cycle.source_request.objective == request.objective
                and previous_cycle.source_request.measure == request.measure
                and previous_cycle.source_request.protocol == request.protocol
                and previous_cycle.source_policy == policy
                and request
                in (previous_cycle.source_request, previous_cycle.next_request),
                "Previous execution is unrelated to current source/conditions",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            error = (
                None
                if previous_cycle.candidate is None
                else previous_cycle.candidate.error
            )
            if error is not None:
                error = {
                    **error,
                    "details": {
                        k: v
                        for k, v in error["details"].items()
                        if k != "completed_traces"
                    },
                }
            execution = {
                "status": previous_cycle.status,
                "source_request_hash": previous_cycle.source_request.request_hash,
                "proposal_hash": previous_cycle.proposal.record_hash,
                "program": (
                    None
                    if previous_cycle.proposal.program is None
                    else previous_cycle.proposal.program.to_dict()
                ),
                "candidate": (
                    None
                    if previous_cycle.candidate is None
                    else previous_cycle.candidate.to_dict()
                ),
                "error": error,
                "improvement_verified": False,
            }
            handoff = {**(handoff or {}), "execution_feedback": execution}
        client = _OptimizerClient(self, handoff)
        scope = request_scope(request, policy)
        self.runtime.bind_scope(scope)
        work = {
            "request_hash": request.request_hash,
            "feedback_hash": feedback.record_hash,
            "policy": policy.to_dict(),
            "method": method,
            "context_hash": context.context_hash,
            "leval_handoff": handoff,
            "scope": scope.to_dict(),
        }
        return self.runtime.run_tool(
            self.role,
            "optimize_program",
            work,
            lambda: propose_program(
                request,
                result,
                feedback,
                policy,
                method=method,
                client=client,
                context=context,
                memory_namespace=self.memory_namespace,
            ),
            lambda value: {
                "proposal": value[0].to_dict(),
                "attempts": [a.to_dict() for a in value[1]],
                "role_call_ids": list(client.call_ids),
            },
        )


@dataclass(frozen=True)
class RoleCycle:
    evaluation: RoleEvaluation
    proposal: LoptProposal
    attempts: tuple[ProposalAttempt, ...]
    candidate: PreparedSelection | None
    next_request: ValidationRequest | None
    source_request: ValidationRequest | None = None
    source_result: ValidationResult | None = None
    source_context: ExecutionContext | None = None
    model_call_ids: tuple[str, ...] | None = None
    attempt_id: str | None = None
    source_policy: EditPolicy | None = None

    def __post_init__(self):
        require(
            isinstance(self.evaluation, RoleEvaluation)
            and isinstance(self.proposal, LoptProposal)
            and type(self.attempts) is tuple
            and all(isinstance(a, ProposalAttempt) for a in self.attempts),
            "Invalid role cycle contracts",
        )
        feedback = self.evaluation.feedback
        require(
            tuple(a.index for a in self.attempts)
            == tuple(range(1, len(self.attempts) + 1))
            and all(
                (a.error_json is None) == (a.status == "accepted_proposal")
                and a.status in ("accepted_proposal", "invalid_proposal")
                for a in self.attempts
            ),
            "Cycle proposal attempt sequence/status mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        if self.proposal.status == "proposed":
            require(
                bool(self.attempts)
                and self.attempts[-1].status == "accepted_proposal"
                and all(a.status == "invalid_proposal" for a in self.attempts[:-1]),
                "Proposed cycle lacks accepted final attempt",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        else:
            require(
                all(a.status == "invalid_proposal" for a in self.attempts),
                "Blocked cycle contains accepted proposal",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        if self.model_call_ids is not None:
            require(
                type(self.model_call_ids) is tuple
                and all(type(v) is str and v.strip() for v in self.model_call_ids)
                and len(set(self.model_call_ids)) == len(self.model_call_ids),
                "Invalid cycle model call IDs",
            )
        if self.attempt_id is not None:
            text(self.attempt_id, "cycle attempt_id")
        require(
            self.proposal.model_ref == feedback.model_ref
            and self.proposal.feedback_hash == feedback.record_hash,
            "Cycle proposal/feedback mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            self.proposal.backend == feedback.backend
            and self.proposal.memory_namespace != feedback.memory_namespace,
            "Cycle proposal backend/memory mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            (self.candidate is not None) == (self.proposal.status == "proposed"),
            "Proposed program needs executed P1 result",
        )
        if self.candidate is not None:
            require(
                isinstance(self.candidate, PreparedSelection)
                and self.candidate.program == self.proposal.program
                and self.candidate.context.model_ref == feedback.model_ref
                and self.candidate.objective.objective_hash == feedback.objective_hash,
                "Cycle candidate identity mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            require(
                self.candidate.backend == feedback.backend,
                "Cycle candidate backend mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            require(
                self.attempt_id is None or self.candidate.attempt_id == self.attempt_id,
                "Cycle candidate outer attempt mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        feasible = self.candidate is not None and self.candidate.status == "feasible"
        require(
            (self.next_request is not None) == feasible,
            "Only feasible role candidate gets next validation request",
        )
        if self.next_request is not None:
            require(
                self.next_request.program == self.proposal.program
                and self.next_request.batch == self.candidate.workflow.batch
                and self.next_request.model_ref == feedback.model_ref
                and self.next_request.objective == self.candidate.objective
                and self.next_request.protocol.sha256 == feedback.protocol_hash
                and self.next_request.measure == feedback.measure
                and self.next_request.selection_context_hash
                == self.candidate.context.context_hash
                and self.next_request.backend == feedback.backend,
                "Cycle next-request anchors changed",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        sources = (self.source_request, self.source_result, self.source_context)
        require(
            all(v is None for v in sources) or all(v is not None for v in sources),
            "Cycle source records must be supplied together",
        )
        if self.source_request is not None:
            self.validate_source(
                self.source_request, self.source_result, self.source_context
            )
        if self.source_policy is not None:
            require(
                isinstance(self.source_policy, EditPolicy)
                and self.source_context is not None
                and self.source_policy.max_nodes <= self.source_context.max_nodes,
                "Cycle source edit policy mismatch",
            )
            if self.proposal.program is not None:
                from .lopt import _validate_candidate

                _validate_candidate(
                    self.source_request.program,
                    feedback,
                    self.source_policy,
                    {
                        "program": self.proposal.program.to_dict(),
                        "rationale": self.proposal.rationale,
                        "hypothesis": self.proposal.hypothesis,
                        "cited_feedback_id": feedback.feedback_id,
                    },
                )

    def validate_source(self, request, result, context):
        require(
            isinstance(request, ValidationRequest)
            and isinstance(result, ValidationResult)
            and isinstance(context, ExecutionContext),
            "Expected exact cycle source records",
        )
        feedback = self.evaluation.feedback
        require(
            feedback
            == build_feedback(
                request, result, memory_namespace=feedback.memory_namespace
            )
            and request.model_ref == context.model_ref
            and request.selection_context_hash == context.context_hash
            and self.proposal.parent_recipe_hash == request.program.recipe_hash,
            "Cycle differs from original source/model/context",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        if self.proposal.program is not None:
            require(
                self.proposal.program.parent_id == request.program.program_id
                and self.proposal.changes
                == program_diff(request.program, self.proposal.program),
                "Cycle parent/diff mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        if self.candidate is not None:
            require(
                self.candidate.context.context_hash == context.context_hash
                and self.candidate.objective == request.objective,
                "Cycle candidate changed fixed source conditions",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        if self.next_request is not None:
            expected = ValidationRequest(
                "validate-" + self.proposal.record_hash,
                request.model_ref,
                self.proposal.program,
                self.candidate.workflow.batch,
                request.objective,
                context.context_hash,
                request.measure,
                request.protocol,
                request.backend,
            )
            require(
                self.next_request == expected,
                "Cycle next request differs from pinned source",
                ErrorCode.ARTIFACT_MISMATCH,
            )

    @property
    def status(self):
        if self.proposal.status == "blocked":
            return (
                "needs_evidence"
                if self.evaluation.feedback.status != "valid"
                else "proposal_blocked"
            )
        if self.candidate.status != "feasible":
            return "candidate_" + self.candidate.status
        return "awaiting_validation"

    def to_dict(self):
        return {
            "schema_version": "1",
            "status": self.status,
            "model_hash": self.proposal.model_ref.model_hash,
            "feedback_hash": self.evaluation.feedback.record_hash,
            "proposal_hash": self.proposal.record_hash,
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "next_request_hash": (
                None if self.next_request is None else self.next_request.request_hash
            ),
            "improvement_verified": False,
            "validation_implemented": False,
            "source_request_hash": (
                None
                if self.source_request is None
                else self.source_request.request_hash
            ),
            "source_result_hash": (
                None if self.source_result is None else self.source_result.record_hash
            ),
            "model_call_ids": self.model_call_ids,
            "attempt_id": self.attempt_id,
            "edit_policy_hash": (
                None if self.source_policy is None else self.source_policy.record_hash
            ),
        }


class RoleTeam:
    def __init__(
        self,
        client,
        *,
        model,
        session_id,
        prompts=None,
        backend,
        history_window=8,
        attempt_id=None,
    ):
        self._attach(
            RoleRuntime(
                client,
                model=model,
                session_id=session_id,
                prompts=prompts,
                backend=backend,
                history_window=history_window,
                attempt_id=attempt_id,
            )
        )

    def _attach(self, runtime):
        self._runtime = runtime
        self._lexec = ExecutionAgent(runtime)
        self._leval = EvaluationAgent(runtime)
        self._lopt = OptimizationAgent(runtime)

    @property
    def runtime(self):
        return self._runtime

    @property
    def lexec(self):
        return self._lexec

    @property
    def leval(self):
        return self._leval

    @property
    def lopt(self):
        return self._lopt

    @classmethod
    def restore(
        cls,
        path,
        client,
        *,
        expected_model,
        expected_prompts=None,
        expected_session_id=None,
        expected_attempt_id=None,
        expected_backend=None,
        expected_scope=None,
    ):
        result = cls.__new__(cls)
        result._attach(
            RoleRuntime.restore(
                path,
                client,
                expected_model=expected_model,
                expected_prompts=expected_prompts,
                expected_session_id=expected_session_id,
                expected_attempt_id=expected_attempt_id,
                expected_backend=expected_backend,
                expected_scope=expected_scope,
            )
        )
        return result

    def save(self, output):
        return self.runtime.save(output)

    def feedback_step(
        self,
        request,
        result,
        context,
        policy,
        *,
        attempt_id,
        invocation_id,
        use_model=True,
        previous_cycle=None,
    ):
        with self.runtime._lock:
            return self._feedback_step(
                request,
                result,
                context,
                policy,
                attempt_id=attempt_id,
                invocation_id=invocation_id,
                use_model=use_model,
                previous_cycle=previous_cycle,
            )

    def _feedback_step(
        self,
        request,
        result,
        context,
        policy,
        *,
        attempt_id,
        invocation_id,
        use_model,
        previous_cycle,
    ):
        before = {
            e.call_id
            for role in ("lexec", "leval", "lopt")
            for e in self.runtime.history(role)
            if e.operation == "model"
        }
        require(
            isinstance(request, ValidationRequest)
            and isinstance(result, ValidationResult)
            and isinstance(context, ExecutionContext),
            "Team needs typed request/result/context",
        )
        require(type(use_model) is bool, "use_model must be explicit boolean")
        require(
            isinstance(policy, EditPolicy) and policy.max_nodes <= context.max_nodes,
            "Invalid/excessive role edit policy",
        )
        text(attempt_id, "attempt_id")
        text(invocation_id, "invocation_id")
        require(
            request.model_ref == self.runtime.model_ref
            and request.backend == self.runtime.backend
            and context.model_ref == self.runtime.model_ref
            and context.context_hash == request.selection_context_hash,
            "Team request uses another model/context/backend",
            ErrorCode.STALE_CACHE,
        )
        _domain(context, request.objective, self.runtime.backend)
        require(
            execute_program(request.program, context).batch == request.batch,
            "Team source differs from exact parent batch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        self.runtime.bind_attempt(attempt_id)
        self.runtime.bind_scope(request_scope(request, policy))
        evaluation = self.leval.evaluate(
            request,
            result,
            use_model=use_model,
            timeout_seconds=policy.model_timeout_seconds,
        )
        proposal, attempts = self.lopt.optimize(
            request,
            result,
            policy,
            context=context,
            evaluation=evaluation,
            method="model" if use_model else "rule",
            previous_cycle=previous_cycle,
        )
        candidate = None
        next_request = None
        if proposal.program is not None:
            candidate = self.lexec.execute(
                proposal.program,
                context,
                request.objective,
                attempt_id=attempt_id,
                invocation_id=invocation_id,
                use_model=use_model,
                timeout_seconds=policy.model_timeout_seconds,
            )
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
        calls = tuple(
            e.call_id
            for role in ("lexec", "leval", "lopt")
            for e in self.runtime.history(role)
            if e.operation == "model" and e.phase == "after" and e.call_id not in before
        )
        return RoleCycle(
            evaluation,
            proposal,
            attempts,
            candidate,
            next_request,
            request,
            result,
            context,
            calls,
            attempt_id,
            policy,
        )

    def publish_cycle(self, cycle, output):
        from .handoff import publish_role_cycle

        with self.runtime._lock:
            require(
                cycle.attempt_id == self.runtime.attempt_id
                and cycle.proposal.model_ref == self.runtime.model_ref,
                "Cycle belongs to another team attempt",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            return publish_role_cycle(cycle, output, model_ledger=self.runtime.ledger())
