"""Frozen-model inner feedback loop. Numeric validation remains an injected tool."""

from dataclasses import replace
from pathlib import Path
from threading import RLock
from time import perf_counter

from self_curate.agents import (
    RoleTeam,
    RoleCycle,
    ValidationProvider,
    check_external_result,
    request_scope,
)
from self_curate.contracts import (
    ContractError,
    CostLedger,
    EditPolicy,
    ErrorCode,
    ResourceUsage,
    ValidationRequest,
    ValidationResult,
    canonical_json,
    digest,
)
from self_curate.contracts.types import require, text
from self_curate.contracts.parsing import load_json
from self_curate.operators import PreparedSelection, execute_program
from .contracts import (
    EvaluatedCandidate,
    SearchState,
    SearchResult,
    StopCondition,
    StopDecision,
    ValidationAttempt,
)


class InnerSearch:
    def __init__(
        self,
        team,
        initial,
        request,
        policy,
        *,
        search_id,
        use_model=True,
        _restoring=False,
    ):
        require(
            isinstance(team, RoleTeam) and isinstance(initial, PreparedSelection),
            "Search needs RoleTeam and actual P1 preparation",
        )
        require(
            isinstance(request, ValidationRequest) and isinstance(policy, EditPolicy),
            "Search needs fixed request/edit policy",
        )
        require(type(use_model) is bool, "use_model must be explicit boolean")
        text(search_id, "search_id")
        require(
            initial.workflow is not None
            and initial.program == request.program
            and initial.workflow.batch == request.batch
            and initial.objective == request.objective
            and initial.context.context_hash == request.selection_context_hash
            and initial.context.model_ref == request.model_ref
            and initial.backend == request.backend,
            "Initial request differs from exact P1 handoff",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            team.runtime.model_ref == request.model_ref
            and team.runtime.backend == request.backend,
            "Search model/backend mismatch",
            ErrorCode.STALE_CACHE,
        )
        require(
            policy.max_nodes <= initial.context.max_nodes,
            "Search edit policy exceeds fixed controls",
        )
        require(
            execute_program(request.program, initial.context).batch == request.batch,
            "Initial batch differs from selection replay",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        if not _restoring:
            require(
                not any(team.runtime.history(r) for r in ("lexec", "leval", "lopt")),
                "Use a new role session for a new search; restore existing searches explicitly",
            )
        team.runtime.bind_attempt(initial.attempt_id)
        team.runtime.bind_scope(request_scope(request, policy))
        self._team, self._initial, self._initial_request, self._policy = (
            team,
            initial,
            request,
            policy,
        )
        self._search_id, self._use_model = search_id, use_model
        self._context = initial.context
        self._requests = {request.request_hash: request}
        self._prepared = {request.request_hash: initial}
        self._validations, self._cycles, self._failures, self._timeline = [], [], [], []
        self._evaluated = {}
        self._current = self._best = None
        self._pending_request = request
        self._phase, self._reason = "awaiting_validation", None
        self._stagnation = self._operation_counter = 0
        self._pending_operation = None
        self._checkpoint_root = None
        self._checkpoint_sequence = 0
        self._lock = RLock()

    @property
    def team(self):
        return self._team

    @property
    def history(self):
        return tuple(self._cycles)

    @property
    def validations(self):
        return tuple(self._validations)

    def _assert_fixed(self):
        require(
            self._team.runtime.model_ref == self._initial_request.model_ref
            and self._team.runtime.shared_client.model_ref
            == self._initial_request.model_ref
            and self._team.runtime.backend == self._initial_request.backend
            and self._team.runtime.attempt_id == self._initial.attempt_id
            and self._team.runtime.scope
            == request_scope(self._initial_request, self._policy),
            "Frozen search identity/conditions changed",
            ErrorCode.STALE_CACHE,
        )

    def ledger(self):
        with self._lock:
            events = list(self._initial.ledger.events) + list(
                self._team.runtime.ledger().events
            )
            for cycle in self._cycles:
                if cycle.candidate is not None:
                    events.extend(cycle.candidate.ledger.events)
            events.extend(v.event for v in self._validations if v.event is not None)
            indexed = {}
            for event in events:
                require(
                    event.event_id not in indexed or indexed[event.event_id] == event,
                    "Search cost event conflict",
                    ErrorCode.ARTIFACT_MISMATCH,
                )
                indexed[event.event_id] = event
            return CostLedger(
                "inner-" + digest(self._namespace()),
                tuple(indexed.values()),
                self._initial_request.backend,
            )

    @property
    def state(self):
        with self._lock:
            return SearchState(
                self._search_id,
                "interrupted" if self._pending_operation is not None else self._phase,
                len(self._cycles) + len(self._failures),
                len(self._evaluated),
                len(self._validations),
                self._stagnation,
                self._current,
                self._best,
                self._pending_request,
                self.ledger(),
                self._reason,
                (
                    None
                    if self._pending_operation is None
                    else canonical_json(self._pending_operation)
                ),
                self._operation_counter,
                len(self._failures)
                + sum(v.error_json is not None for v in self._validations),
            )

    def result(self):
        state = self.state
        status = (
            "no_feasible_result"
            if state.phase == "stopped" and state.best is None
            else state.phase
        )
        return SearchResult(status, state, state.reason)

    def _namespace(self):
        return {
            "search_id": self._search_id,
            "attempt_id": self._initial.attempt_id,
            "session_id": self._team.runtime.session_id,
            "model_hash": self._initial_request.model_ref.model_hash,
        }

    def _operation_id(self, counter=None):
        return digest(
            {
                **self._namespace(),
                "operation": self._operation_counter if counter is None else counter,
            }
        )

    def _begin(self, kind):
        previous_counter = self._operation_counter
        self._operation_counter += 1
        self._pending_operation = {"kind": kind, "invocation_id": self._operation_id()}
        try:
            self._checkpoint()
        except BaseException:
            # The external job has not been dispatched. A failed write-ahead
            # save must not turn unstarted work into an unresolved operation.
            self._operation_counter = previous_counter
            self._pending_operation = None
            raise
        return self._operation_id()

    def _complete(self):
        self._pending_operation = None
        self._checkpoint()

    def _accept_validation(self, attempt):
        require(
            self._pending_request == attempt.request
            and attempt.outer_attempt_id == self._initial.attempt_id,
            "Validation is unrelated to pending candidate",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        self._validations.append(attempt)
        self._timeline.append(
            {"kind": "validation", "index": len(self._validations) - 1}
        )
        if attempt.error_json is not None:
            self._phase, self._reason = (
                "blocked",
                "Validation call or result failed; explicit retry/replacement required",
            )
        elif attempt.result.status != "valid":
            self._phase, self._reason = "awaiting_validation", attempt.result.reason
        else:
            item = EvaluatedCandidate(
                self._prepared[attempt.request.request_hash],
                attempt.request,
                attempt.result,
            )
            require(
                item.request.request_hash not in self._evaluated,
                "Final validation cannot be replaced",
            )
            self._evaluated[item.request.request_hash] = item
            self._current = item
            self._pending_request = None
            self._phase, self._reason = "ready", None
            if item.feasible and (
                self._best is None
                or item.validation.total_value > self._best.validation.total_value
            ):
                self._best, self._stagnation = item, 0
            else:
                self._stagnation += 1

    def _accept_cycle(self, cycle):
        require(
            self._current is not None and self._pending_request is None,
            "No valid parent for proposal",
        )
        require(
            isinstance(cycle, RoleCycle)
            and cycle.source_request == self._current.request
            and cycle.source_result == self._current.validation
            and cycle.source_context.context_hash == self._context.context_hash
            and cycle.source_policy == self._policy
            and cycle.attempt_id == self._initial.attempt_id,
            "Cycle changed frozen search source",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        if cycle.candidate is not None:
            require(
                cycle.candidate.invocation_id == self._operation_id(),
                "Cycle invocation differs from search operation",
            )
        self._check_cycle_evidence(cycle)
        if cycle.next_request is not None:
            key = cycle.next_request.request_hash
            require(
                key not in self._requests,
                "Search produced a duplicate validation request",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        self._cycles.append(cycle)
        self._timeline.append({"kind": "cycle", "index": len(self._cycles) - 1})
        self._reason = None
        if cycle.next_request is not None:
            self._requests[key], self._prepared[key] = (
                cycle.next_request,
                cycle.candidate,
            )
            self._pending_request = cycle.next_request
            self._phase = "awaiting_validation"
        else:
            self._phase = "ready"
            self._stagnation += 1

    def _check_cycle_evidence(self, cycle):
        require(
            cycle.evaluation.feedback.memory_namespace
            == self._team.leval.memory_namespace
            and cycle.proposal.memory_namespace == self._team.lopt.memory_namespace
            and cycle.proposal.method == ("model" if self._use_model else "rule"),
            "Cycle role namespace/execution mode changed",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        previous = {key for old in self._cycles for key in old.model_call_ids}
        require(
            not previous.intersection(cycle.model_call_ids),
            "Cycle reused earlier physical calls",
        )
        before, after = {}, {}
        for role in ("lexec", "leval", "lopt"):
            for event in self._team.runtime.history(role):
                (before if event.phase == "before" else after)[event.call_id] = event
        require(
            all(
                key in before and key in after and before[key].operation == "model"
                for key in cycle.model_call_ids
            ),
            "Cycle model calls lack role history",
        )
        reply = cycle.evaluation.interpretation
        if reply is not None:
            require(
                after[reply.call_id].payload["result"] == reply.content
                and before[reply.call_id].payload["work"].get("feedback_hash")
                == cycle.evaluation.feedback.record_hash,
                "Leval output differs from physical role history",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        optimizer_calls = [
            key for key in cycle.model_call_ids if after[key].role == "lopt"
        ]
        if self._use_model:
            require(
                len(optimizer_calls) == len(cycle.attempts),
                "Proposal call count mismatch",
            )
            for key, attempt in zip(optimizer_calls, cycle.attempts):
                payload = before[key].payload["work"]["input"]
                require(
                    payload["parent_program"] == cycle.source_request.program.to_dict()
                    and payload["feedback"] == cycle.evaluation.feedback.to_dict()
                    and payload["fixed_policy"] == self._policy.to_dict(),
                    "Lopt input differs from search source",
                    ErrorCode.ARTIFACT_MISMATCH,
                )
                if attempt.response_json is not None:
                    require(
                        load_json(attempt.response_json)["content"]
                        == after[key].payload["result"],
                        "Lopt response differs from physical history",
                        ErrorCode.ARTIFACT_MISMATCH,
                    )
        else:
            require(
                not cycle.model_call_ids and reply is None,
                "Rule search contains model calls",
            )
        for key in cycle.model_call_ids:
            if after[key].role == "lexec":
                work = before[key].payload["work"]
                require(
                    work["program_hash"] == cycle.proposal.program.recipe_hash
                    and work["context_hash"] == self._context.context_hash,
                    "Lexec authorized another candidate",
                    ErrorCode.ARTIFACT_MISMATCH,
                )

    @staticmethod
    def _error(exc):
        return (
            exc.to_dict()
            if isinstance(exc, ContractError)
            else {
                "code": ErrorCode.ENVIRONMENT_FAILURE.value,
                "message": str(exc),
                "details": {"exception": type(exc).__name__},
            }
        )

    def _provider_validation(self, provider, invocation):
        request = self._pending_request
        started, result, error, usage = perf_counter(), None, None, ResourceUsage()
        try:
            result = provider.evaluate(request)
            require(
                isinstance(result, ValidationResult),
                "ValidationProvider must return ValidationResult",
            )
            usage = result.usage
            self._assert_fixed()
            check_external_result(request, result)
        except Exception as exc:
            error = canonical_json(self._error(exc))
            if not isinstance(result, ValidationResult):
                result = None
        if usage.compute.wall_seconds is None:
            usage = replace(
                usage,
                compute=replace(
                    usage.compute, wall_seconds=max(0.0, perf_counter() - started)
                ),
            )
        self._accept_validation(
            ValidationAttempt(
                invocation,
                self._initial.attempt_id,
                request,
                result,
                error,
                usage,
                "provider",
            )
        )

    def step(self, validator: ValidationProvider | None = None) -> SearchResult:
        """Perform one proposal/execution OR one external validation operation."""
        with self._lock:
            if (
                self._phase in ("stopped", "blocked")
                or self._pending_operation is not None
            ):
                return self.result()
            self._assert_fixed()
            if self._pending_request is not None:
                if validator is None:
                    return self.result()
                require(
                    callable(getattr(validator, "evaluate", None)),
                    "Expected ValidationProvider",
                )
                invocation = self._begin("validation")
                self._provider_validation(validator, invocation)
                self._complete()
                return self.result()
            invocation = self._begin("cycle")
            try:
                cycle = self._team.feedback_step(
                    self._current.request,
                    self._current.validation,
                    self._context,
                    self._policy,
                    attempt_id=self._initial.attempt_id,
                    invocation_id=invocation,
                    use_model=self._use_model,
                    previous_cycle=self._cycles[-1] if self._cycles else None,
                )
                self._accept_cycle(cycle)
            except Exception as exc:
                self._failures.append(
                    {"invocation_id": invocation, "error": self._error(exc)}
                )
                self._timeline.append(
                    {"kind": "failure", "index": len(self._failures) - 1}
                )
                self._phase, self._reason = (
                    "blocked",
                    "Role operation failed; inspect evidence before retry",
                )
            self._complete()
            return self.result()

    def submit_validation(
        self, result: ValidationResult, *, cost_event=None
    ) -> SearchResult:
        """Accept a bound external result once; identical re-delivery is a no-op."""
        with self._lock:
            require(
                isinstance(result, ValidationResult),
                "Expected external ValidationResult",
            )
            delivered = next(
                (
                    v
                    for v in self._validations
                    if v.result == result and v.error_json is None
                ),
                None,
            )
            if delivered is not None:
                require(
                    cost_event is None or cost_event == delivered.event,
                    "Cannot replace the cost identity of delivered evidence",
                )
                return self.result()
            require(
                self._pending_operation is None
                and self._phase != "stopped"
                and self._pending_request is not None,
                "No available pending validation request",
            )
            self._assert_fixed()
            check_external_result(self._pending_request, result)
            attempt = ValidationAttempt(
                self._operation_id(self._operation_counter + 1),
                self._initial.attempt_id,
                self._pending_request,
                result,
                None,
                result.usage,
                "imported",
                cost_event,
            )
            if attempt.event is not None:
                for existing in self.ledger().events:
                    require(
                        existing.event_id != attempt.event.event_id
                        or existing == attempt.event,
                        "External validation cost conflicts with existing event",
                    )
            self._begin("validation")
            self._accept_validation(attempt)
            self._complete()
            return self.result()

    def retry(self, *, reason):
        with self._lock:
            text(reason, "retry reason")
            require(
                self._phase == "blocked" and self._pending_operation is None,
                "Only a completed failed operation can be retried",
            )
            self._assert_fixed()
            self._phase = (
                "awaiting_validation" if self._pending_request is not None else "ready"
            )
            self._reason = None
            self._timeline.append({"kind": "retry", "reason": reason})
            self._checkpoint()
            return self.result()

    def stop(self, *, reason):
        with self._lock:
            text(reason, "stop reason")
            require(
                self._pending_operation is None,
                "Unresolved operation cannot be marked complete",
            )
            if self._phase != "stopped":
                self._phase, self._reason = "stopped", reason
                self._timeline.append({"kind": "stop", "reason": reason})
                self._checkpoint()
            return self.result()

    def run(
        self,
        validator: ValidationProvider | None = None,
        *,
        stop_condition: StopCondition | None = None,
    ) -> SearchResult:
        """Drive the loop using an explicit caller-owned stopping callback."""
        with self._lock:
            if (
                self._phase in ("stopped", "blocked")
                or self._pending_operation is not None
            ):
                return self.result()
            if stop_condition is None:
                return SearchResult(
                    "awaiting_stop_condition",
                    self.state,
                    "Configure stop_condition, or use step() manually",
                )
            require(callable(stop_condition), "Expected StopCondition callback")
            while True:
                if (
                    self._phase in ("stopped", "blocked")
                    or self._pending_operation is not None
                ):
                    return self.result()
                decision = stop_condition(self.state)
                require(
                    isinstance(decision, StopDecision),
                    "Stopping callback must return StopDecision",
                )
                if decision.stop:
                    return self.stop(reason=decision.reason)
                observations = len(self._validations)
                result = self.step(validator)
                if result.status in (
                    "blocked",
                    "interrupted",
                    "stopped",
                    "no_feasible_result",
                ):
                    return result
                if self._pending_request is not None and (
                    validator is None or len(self._validations) != observations
                ):
                    return result  # A pending/invalid provider result yields; never busy-poll.

    def enable_checkpoints(self, directory):
        with self._lock:
            require(self._checkpoint_root is None, "Checkpoint root already configured")
            root = Path(directory).resolve()
            require(
                not root.exists(), "Checkpoint root exists", ErrorCode.OUTPUT_EXISTS
            )
            root.mkdir(parents=True)
            self._checkpoint_root = root
            self._checkpoint()

    def _checkpoint(self):
        if self._checkpoint_root is not None:
            self._checkpoint_sequence += 1
            self.save(self._checkpoint_root / f"{self._checkpoint_sequence:06d}")

    def save(self, output):
        from .storage import save_search

        with self._lock:
            return save_search(self, output)

    def publish_best(self, output):
        from .storage import publish_best

        with self._lock:
            return publish_best(self, output)

    @classmethod
    def restore(cls, path, client, view, **anchors):
        from .storage import restore_search

        return restore_search(path, client, view, **anchors)

    @classmethod
    def restore_latest(cls, directory, client, view, **anchors):
        directories = sorted(
            (
                p
                for p in Path(directory).iterdir()
                if p.is_dir() and p.name.isdigit() and p.name == f"{int(p.name):06d}"
            ),
            key=lambda p: int(p.name),
        )
        require(bool(directories), "No complete search checkpoint")
        return cls.restore(directories[-1], client, view, **anchors)
