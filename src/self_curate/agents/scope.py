"""Fixed scientific conditions for one role session; no scoring algorithms."""

from dataclasses import dataclass

from self_curate.contracts import ErrorCode, parse_contract, parse_feedback_contract
from self_curate.contracts.parsing import fields
from self_curate.contracts.types import require, sha256
from self_curate.contracts.workflow import WorkflowRecord, header


@dataclass(frozen=True)
class RoleScope(WorkflowRecord):
    selection_context_hash: str | None = None
    objective_hash: str | None = None
    measure: str | None = None
    protocol_hash: str | None = None
    edit_policy_hash: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        for name in (
            "selection_context_hash",
            "objective_hash",
            "protocol_hash",
            "edit_policy_hash",
        ):
            if getattr(self, name) is not None:
                sha256(getattr(self, name), name)
        require(
            self.measure is None or self.measure in ("predicted_gain", "reward_delta"),
            "Invalid session measure",
        )

    def merge(self, other):
        require(isinstance(other, RoleScope), "Expected RoleScope")
        values = {}
        for name in (
            "selection_context_hash",
            "objective_hash",
            "measure",
            "protocol_hash",
            "edit_policy_hash",
        ):
            a, b = getattr(self, name), getattr(other, name)
            require(
                a is None or b is None or a == b,
                "Role session changed fixed experiment conditions",
                ErrorCode.STALE_CACHE,
                field=name,
            )
            values[name] = a if a is not None else b
        return RoleScope(**values)


def parse_role_scope(value):
    return RoleScope(
        **fields(
            value,
            (
                "schema_version",
                "selection_context_hash",
                "objective_hash",
                "measure",
                "protocol_hash",
                "edit_policy_hash",
            ),
        )
    )


def request_scope(request, policy=None):
    return RoleScope(
        request.selection_context_hash,
        request.objective.objective_hash,
        request.measure,
        request.protocol.sha256,
        None if policy is None else policy.record_hash,
    )


def infer_scope(histories):
    """Recover available anchors from 0.6.0 business histories, never model prose."""
    scope = RoleScope()
    for history in histories:
        for event in history.events:
            payload = event.payload
            if event.phase == "before" and event.operation.startswith("tool:"):
                work = payload["work"]
                if "scope" in work:
                    scope = scope.merge(parse_role_scope(work["scope"]))
                elif event.operation == "tool:execute_selection" and {
                    "objective",
                    "context_hash",
                }.issubset(work):
                    objective = parse_contract("ObjectiveSpec", work["objective"])
                    scope = scope.merge(
                        RoleScope(
                            selection_context_hash=work["context_hash"],
                            objective_hash=objective.objective_hash,
                        )
                    )
                elif event.operation == "tool:optimize_program" and {
                    "policy",
                    "context_hash",
                }.issubset(work):
                    policy = parse_feedback_contract("EditPolicy", work["policy"])
                    scope = scope.merge(
                        RoleScope(
                            selection_context_hash=work["context_hash"],
                            edit_policy_hash=policy.record_hash,
                        )
                    )
            elif (
                event.operation == "tool:interpret_validation"
                and event.phase == "after"
                and payload["status"] == "completed"
                and type(payload["result"]) is dict
                and "feedback_id" in payload["result"]
            ):
                feedback = parse_feedback_contract("LevalFeedback", payload["result"])
                scope = scope.merge(
                    RoleScope(
                        objective_hash=feedback.objective_hash,
                        measure=feedback.measure,
                        protocol_hash=feedback.protocol_hash,
                    )
                )
    return scope
