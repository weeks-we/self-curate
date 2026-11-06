"""Code-owned current/history prompt slots; no cross-condition score comparison."""

from dataclasses import dataclass
import math

from self_curate.contracts import (
    ValidationRequest,
    ValidationResult,
    parse_feedback_contract,
)
from self_curate.contracts.parsing import fields
from self_curate.contracts.types import require, integer
from self_curate.contracts.workflow import WorkflowRecord
from .leval import check_external_result, build_feedback
from .scope import request_scope


@dataclass(frozen=True)
class ValidationObservation(WorkflowRecord):
    request: ValidationRequest
    result: ValidationResult
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported validation observation")
        check_external_result(self.request, self.result)


@dataclass(frozen=True)
class LevalContext(WorkflowRecord):
    current: ValidationObservation
    history: tuple[ValidationObservation, ...]
    limit: int = 8
    schema_version: str = "1"

    def __post_init__(self):
        require(
            self.schema_version == "1"
            and isinstance(self.current, ValidationObservation),
            "Invalid Leval context",
        )
        integer(self.limit, "history limit")
        require(
            self.limit <= 256
            and type(self.history) is tuple
            and len(self.history) <= self.limit,
            "History limit exceeded",
        )
        request = self.current.request
        seen = {self.current.result.record_hash}
        for prior in self.history:
            require(isinstance(prior, ValidationObservation), "Typed history required")
            require(
                prior.request.model_ref == request.model_ref
                and prior.request.backend == request.backend
                and request_scope(prior.request) == request_scope(request),
                "Historical evaluation uses another model/objective/context/method/protocol",
            )
            require(
                prior.result.record_hash not in seen, "Duplicate historical evidence"
            )
            seen.add(prior.result.record_hash)

    def validate_source(self, request, result):
        require(
            self.current == ValidationObservation(request, result),
            "History context cites another current result",
        )

    @staticmethod
    def _summary(item):
        request, result = item.request, item.result
        return {
            "request_hash": request.request_hash,
            "result_hash": result.record_hash,
            "program_id": request.program.program_id,
            "program_hash": request.program.recipe_hash,
            "batch_id": request.batch.batch_id,
            "sample_count": len(request.batch.data_ids),
            "supervised_tokens": request.batch.supervised_tokens,
            "status": result.status,
            "total_value": result.total_value,
            "skill_values": {v.skill: v.value for v in result.skill_values},
            "constraints": [c.to_dict() for c in result.constraints],
            "reason": result.reason,
            "evidence": result.evidence.to_dict(),
        }

    def prompt_slots(self, feedback):
        require(
            feedback.request_hash == self.current.request.request_hash
            and feedback.result_hash == self.current.result.record_hash,
            "Prompt feedback differs from current evidence",
        )
        valid = [o for o in self.history if o.result.status == "valid"]
        feasible = [o for o in valid if all(c.passed for c in o.result.constraints)]
        previous = valid[-1] if valid else None
        best = max(feasible, key=lambda o: o.result.total_value) if feasible else None
        comparison = {
            "previous_result_hash": (
                None if previous is None else previous.result.record_hash
            ),
            "best_feasible_result_hash": (
                None if best is None else best.result.record_hash
            ),
            "delta_previous_total": None,
            "delta_best_feasible_total": None,
            "skill_delta_previous": {},
            "measure": self.current.request.measure,
            "scope_hash": request_scope(self.current.request).record_hash,
            "causal_explanation_verified": False,
        }
        current = self.current.result
        if current.status == "valid":
            for name, prior in (
                ("delta_previous_total", previous),
                ("delta_best_feasible_total", best),
            ):
                if prior is not None:
                    delta = current.total_value - prior.result.total_value
                    require(
                        math.isfinite(delta), "Historical score difference overflows"
                    )
                    comparison[name] = delta
            if previous is not None:
                old = {v.skill: v.value for v in previous.result.skill_values}
                comparison["skill_delta_previous"] = {
                    v.skill: v.value - old[v.skill] for v in current.skill_values
                }
        return {
            "schema": "leval-feedback-slots-v1",
            "context_hash": self.record_hash,
            "current_result": self._summary(self.current),
            "related_history": [self._summary(o) for o in self.history],
            "comparison": comparison,
            "feedback": feedback.to_dict(),
        }


def context_from_history(request, result, events, *, limit=8):
    """Only completed Leval tools with exact typed observations are eligible.

    Legacy tools without saved requests remain readable but are not guessed into
    comparable observations. Memory prose and model replies never become scores.
    """
    starts, observed, seen = {}, [], {result.record_hash}
    for event in events:
        if event.role != "leval" or event.operation != "tool:interpret_validation":
            continue
        payload = event.payload
        if event.phase == "before":
            starts[event.call_id] = payload["work"]
        elif payload["status"] == "completed" and event.call_id in starts:
            work = starts[event.call_id]
            if "current_request" not in work:
                continue
            old_request = parse_feedback_contract(
                "ValidationRequest", work["current_request"]
            )
            old_result = parse_feedback_contract(
                "ValidationResult", work["external_result"]
            )
            observation = ValidationObservation(old_request, old_result)
            saved_feedback = parse_feedback_contract("LevalFeedback", payload["result"])
            require(
                saved_feedback
                == build_feedback(
                    old_request, old_result, memory_namespace=event.namespace
                ),
                "Historical tool result differs from its saved numeric input",
            )
            if (
                old_request.model_ref == request.model_ref
                and old_request.backend == request.backend
                and request_scope(old_request) == request_scope(request)
                and old_result.record_hash not in seen
            ):
                observed.append(observation)
                seen.add(old_result.record_hash)
    return LevalContext(
        ValidationObservation(request, result),
        tuple(observed[-limit:] if limit else ()),
        limit,
    )


def parse_leval_context(value):
    row = fields(value, ("schema_version", "current", "history", "limit"))

    def observation(raw):
        r = fields(raw, ("schema_version", "request", "result"))
        return ValidationObservation(
            parse_feedback_contract("ValidationRequest", r["request"]),
            parse_feedback_contract("ValidationResult", r["result"]),
            r["schema_version"],
        )

    require(type(row["history"]) is list, "History must be an array")
    return LevalContext(
        observation(row["current"]),
        tuple(observation(o) for o in row["history"]),
        row["limit"],
        row["schema_version"],
    )
