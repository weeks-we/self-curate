"""Inner-search results and an injectable stopping interface; no default limits."""

from dataclasses import dataclass
from typing import Protocol

from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ResourceUsage,
    ValidationRequest,
    ValidationResult,
    digest,
)
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require, text
from self_curate.agents.leval import check_external_result
from self_curate.operators import PreparedSelection


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str | None = None

    def __post_init__(self):
        require(type(self.stop) is bool, "Stop decision must be boolean")
        if self.stop:
            text(self.reason, "stop reason")
        else:
            require(self.reason is None, "Continue decision cannot carry a stop reason")


class StopCondition(Protocol):
    def __call__(self, state: "SearchState") -> StopDecision: ...


@dataclass(frozen=True)
class EvaluatedCandidate:
    prepared: PreparedSelection
    request: ValidationRequest
    validation: ValidationResult

    def __post_init__(self):
        require(
            isinstance(self.prepared, PreparedSelection), "Expected actual P1 result"
        )
        check_external_result(self.request, self.validation)
        require(self.validation.status == "valid", "Candidate has no valid observation")
        require(
            self.prepared.workflow is not None
            and self.prepared.program == self.request.program
            and self.prepared.workflow.batch == self.request.batch
            and self.prepared.context.context_hash
            == self.request.selection_context_hash
            and self.prepared.objective == self.request.objective
            and self.prepared.backend == self.request.backend,
            "Evaluated candidate changed its exact selection",
        )

    @property
    def feasible(self):
        return all(c.passed for c in self.validation.constraints)


@dataclass(frozen=True)
class ValidationAttempt:
    invocation_id: str
    outer_attempt_id: str
    request: ValidationRequest
    result: ValidationResult | None
    error_json: str | None
    usage: ResourceUsage
    source: str
    external_cost: CostEvent | None = None

    def __post_init__(self):
        text(self.invocation_id, "validation invocation")
        text(self.outer_attempt_id, "outer attempt")
        require(
            isinstance(self.request, ValidationRequest), "Expected validation request"
        )
        require(
            self.result is None or isinstance(self.result, ValidationResult),
            "Invalid validation output",
        )
        require(
            isinstance(self.usage, ResourceUsage),
            "Expected actual/unknown validation usage",
        )
        require(self.source in ("provider", "imported"), "Unknown validation source")
        if self.external_cost is not None:
            event = self.external_cost
            require(
                self.source == "imported"
                and isinstance(event, CostEvent)
                and event.operation == "evaluation"
                and event.status == "completed"
                and event.attempt_id == self.outer_attempt_id
                and event.subject_id == self.request.request_hash
                and event.model_hash == self.request.model_ref.model_hash
                and event.backend == self.request.backend
                and event.usage == self.usage,
                "External validation cost belongs to another physical operation",
            )
        if self.error_json is None:
            require(self.result is not None, "Completed validation requires a result")
            check_external_result(self.request, self.result)
        else:
            require(
                type(load_json(self.error_json)) is dict, "Invalid validation error"
            )

    def to_dict(self):
        return {
            "invocation_id": self.invocation_id,
            "outer_attempt_id": self.outer_attempt_id,
            "request": self.request.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
            "error": None if self.error_json is None else load_json(self.error_json),
            "usage": self.usage.to_dict(),
            "source": self.source,
            "external_cost": (
                None if self.external_cost is None else self.external_cost.to_dict()
            ),
        }

    @property
    def event(self):
        if self.source == "imported":
            return (
                self.external_cost
            )  # Reading old evidence is not a new model evaluation.
        identity = digest(self.to_dict())
        return CostEvent(
            "validation-" + self.invocation_id,
            self.outer_attempt_id,
            "evaluation",
            self.request.request_hash,
            self.request.model_ref.model_hash,
            "completed" if self.error_json is None else "failed",
            self.usage,
            ArtifactRef("record://search-validation/" + identity, identity),
            self.request.backend,
        )


@dataclass(frozen=True)
class SearchState:
    search_id: str
    phase: str
    candidate_attempts: int
    evaluated_count: int
    validation_attempts: int
    stagnation_count: int
    current: EvaluatedCandidate | None
    best: EvaluatedCandidate | None
    pending_request: ValidationRequest | None
    ledger: CostLedger
    reason: str | None
    unresolved_operation_json: str | None = None
    operation_count: int = 0
    failed_operations: int = 0

    def to_dict(self):
        return {
            "search_id": self.search_id,
            "phase": self.phase,
            "candidate_attempts": self.candidate_attempts,
            "evaluated_count": self.evaluated_count,
            "validation_attempts": self.validation_attempts,
            "stagnation_count": self.stagnation_count,
            "operation_count": self.operation_count,
            "failed_operations": self.failed_operations,
            "current_request_hash": (
                None if self.current is None else self.current.request.request_hash
            ),
            "best_request_hash": (
                None if self.best is None else self.best.request.request_hash
            ),
            "best_batch_id": (
                None if self.best is None else self.best.request.batch.batch_id
            ),
            "best_value": (
                None if self.best is None else self.best.validation.total_value
            ),
            "pending_request_hash": (
                None
                if self.pending_request is None
                else self.pending_request.request_hash
            ),
            "ledger_hash": self.ledger.record_hash,
            "reason": self.reason,
            "unresolved_operation": (
                None
                if self.unresolved_operation_json is None
                else load_json(self.unresolved_operation_json)
            ),
        }


@dataclass(frozen=True)
class SearchResult:
    status: str
    state: SearchState
    reason: str | None = None

    @property
    def best(self):
        return self.state.best

    def require_best(self):
        from self_curate.contracts import ContractError, ErrorCode

        if self.best is None:
            raise ContractError(
                ErrorCode.INSUFFICIENT_POOL, "Search has no validated feasible batch"
            )
        return self.best

    def to_dict(self):
        return {
            "schema_version": "1",
            "status": self.status,
            "reason": self.reason,
            "state": self.state.to_dict(),
            "best_program": (
                None if self.best is None else self.best.request.program.to_dict()
            ),
            "best_batch": (
                None if self.best is None else self.best.request.batch.to_dict()
            ),
            "best_validation": (
                None if self.best is None else self.best.validation.to_dict()
            ),
            "model_updated": False,
        }
