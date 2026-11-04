"""Append-only per-role before/after events; histories never contain other roles."""

from dataclasses import dataclass

from self_curate.contracts import canonical_json, digest, ErrorCode
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import require, text, sha256, integer
from self_curate.contracts.workflow import WorkflowRecord, header
from .prompts import ROLES


@dataclass(frozen=True)
class HistoryEvent(WorkflowRecord):
    role: str
    namespace: str
    model_hash: str
    prompt_hash: str
    sequence: int
    call_id: str
    operation: str
    phase: str
    previous_hash: str
    payload_json: str
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        header(self)
        require(self.role in ROLES, "Unknown history role")
        text(self.namespace, "history namespace")
        text(self.call_id, "call_id")
        text(self.operation, "operation")
        integer(self.sequence, "event sequence")
        for name in ("model_hash", "prompt_hash", "previous_hash"):
            sha256(getattr(self, name), name)
        require(self.phase in ("before", "after"), "History phase must be before/after")
        require(
            type(self.payload_json) is str, "History payload must be immutable JSON"
        )
        require(
            type(load_json(self.payload_json)) is dict,
            "History payload must be a JSON object",
        )
        require(
            canonical_json(load_json(self.payload_json)) == self.payload_json,
            "History JSON must be canonical",
        )

    @property
    def payload(self):
        return load_json(self.payload_json)


class RoleHistory:
    def __init__(self, role, namespace, model_hash, prompt_hash, backend, events=()):
        require(role in ROLES, "Unknown history role")
        text(namespace, "history namespace")
        sha256(model_hash, "model hash")
        sha256(prompt_hash, "prompt hash")
        require(backend in ("mock", "real"), "Invalid history backend")
        self.role = role
        self.namespace = namespace
        self.model_hash = model_hash
        self.prompt_hash = prompt_hash
        self.backend = backend
        self._events = []
        self._starts = {}
        self._ends = {}
        self._order = []
        self._pending = {}
        self._root = digest(
            {
                "role": self.role,
                "namespace": self.namespace,
                "model": self.model_hash,
                "prompt": self.prompt_hash,
                "backend": self.backend,
            }
        )
        for event in events:
            self._append(event)

    @property
    def events(self):
        return tuple(self._events)

    @property
    def root_hash(self):
        return self._root

    @property
    def event_count(self):
        return len(self._events)

    def _append(self, event):
        require(isinstance(event, HistoryEvent), "Expected HistoryEvent")
        require(
            (
                event.role,
                event.namespace,
                event.model_hash,
                event.prompt_hash,
                event.backend,
            )
            == (
                self.role,
                self.namespace,
                self.model_hash,
                self.prompt_hash,
                self.backend,
            ),
            "History role/model/prompt namespace mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            event.sequence == len(self._events)
            and event.previous_hash == self.root_hash,
            "Broken history hash chain",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        before = self._starts
        after = self._ends
        if event.phase == "before":
            require(event.call_id not in before, "Duplicate history call")
            fields(
                event.payload,
                ("work",),
                (
                    "system_prompt",
                    "history_call_ids",
                    "history_hash",
                    "timeout_seconds",
                ),
            )
            require(
                type(event.payload["work"]) is dict, "History work must be an object"
            )
            if event.operation == "model":
                payload = fields(
                    event.payload,
                    (
                        "work",
                        "system_prompt",
                        "history_call_ids",
                        "history_hash",
                        "timeout_seconds",
                    ),
                )
                require(
                    type(payload["history_call_ids"]) is list,
                    "History call IDs must be a list",
                )
                context = self.context(len(payload["history_call_ids"]))
                require(
                    payload["history_call_ids"] == [v["call_id"] for v in context]
                    and payload["history_hash"] == digest(context),
                    "Model call history differs from its own prior work",
                    ErrorCode.ARTIFACT_MISMATCH,
                )
        else:
            require(
                event.call_id in before
                and event.call_id not in after
                and before[event.call_id].operation == event.operation,
                "History completion lacks matching start",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            fields(event.payload, ("status", "result", "error", "usage", "raw_reply"))
            require(
                event.payload["status"] in ("completed", "failed"),
                "Invalid completion status",
            )
            require(
                (event.payload["error"] is None)
                == (event.payload["status"] == "completed"),
                "History error/status mismatch",
            )
            from self_curate.contracts import parse_contract

            parse_contract("ResourceUsage", event.payload["usage"])
        self._events.append(event)
        self._root = event.record_hash
        if event.phase == "before":
            self._starts[event.call_id] = event
            self._order.append(event.call_id)
            self._pending[event.call_id] = event
        else:
            self._ends[event.call_id] = event
            self._pending.pop(event.call_id)
        return event

    def append(self, call_id, operation, phase, payload):
        return self._append(
            HistoryEvent(
                self.role,
                self.namespace,
                self.model_hash,
                self.prompt_hash,
                len(self._events),
                call_id,
                operation,
                phase,
                self.root_hash,
                canonical_json(payload),
                self.backend,
            )
        )

    def pending(self):
        return tuple(self._pending.values())

    def context(self, window):
        integer(window, "history window")
        result = []
        for call_id in self._order[-window:] if window else []:
            event = self._starts[call_id]
            end = self._ends.get(call_id)
            result.append(
                {
                    "call_id": event.call_id,
                    "operation": event.operation,
                    "before": event.payload["work"],
                    "after": None if end is None else end.payload["result"],
                    "status": "pending" if end is None else end.payload["status"],
                    "error": None if end is None else end.payload["error"],
                }
            )
        return result

    def to_dict(self):
        return {
            "schema_version": "1",
            "role": self.role,
            "namespace": self.namespace,
            "model_hash": self.model_hash,
            "prompt_hash": self.prompt_hash,
            "backend": self.backend,
            "events": [e.to_dict() for e in self._events],
            "root_hash": self.root_hash,
        }

    @classmethod
    def from_dict(cls, value):
        row = fields(
            value,
            (
                "schema_version",
                "role",
                "namespace",
                "model_hash",
                "prompt_hash",
                "backend",
                "events",
                "root_hash",
            ),
        )
        require(
            row["schema_version"] == "1" and type(row["events"]) is list,
            "Invalid role history schema",
        )
        keys = (
            "schema_version",
            "role",
            "namespace",
            "model_hash",
            "prompt_hash",
            "sequence",
            "call_id",
            "operation",
            "phase",
            "previous_hash",
            "payload_json",
            "backend",
        )
        events = tuple(HistoryEvent(**fields(e, keys)) for e in row["events"])
        result = cls(
            row["role"],
            row["namespace"],
            row["model_hash"],
            row["prompt_hash"],
            row["backend"],
            events,
        )
        require(
            result.root_hash == row["root_hash"],
            "History root checksum mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        return result
