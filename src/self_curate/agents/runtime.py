"""One frozen model client, fixed role prompts, and independent role histories."""

from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from time import perf_counter
from types import MappingProxyType
import uuid

from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ContractError,
    CostEvent,
    CostLedger,
    ErrorCode,
    ModelRef,
    ResourceUsage,
    canonical_json,
    digest,
    parse_contract,
)
from self_curate.contracts.parsing import fields, load_json, model_ref
from self_curate.contracts.types import integer, require, text
from self_curate.contracts.workflow import finite
from self_curate.data.pool import _publish, _staging_directory, write_json, file_sha256
from .prompts import ROLES, PromptTemplate, default_prompts
from .history import RoleHistory
from .client import SharedModelClient
from .scope import RoleScope, parse_role_scope, infer_scope

TOOL_NAMES = {
    "lexec": ("execute_selection",),
    "leval": ("interpret_validation",),
    "lopt": ("optimize_program",),
}


@dataclass(frozen=True)
class RoleReply:
    role: str
    call_id: str
    content_json: str
    usage: ResourceUsage

    def __post_init__(self):
        require(
            self.role in ROLES and isinstance(self.usage, ResourceUsage),
            "Invalid role reply identity/usage",
        )
        text(self.call_id, "reply call_id")
        require(
            type(self.content_json) is str
            and type(load_json(self.content_json)) is dict,
            "Role reply must contain immutable JSON object",
        )

    @property
    def content(self):
        return load_json(self.content_json)


class RoleRuntime:
    """The client must be stateless: each call receives its complete role context.

    Parameter identity is pinned here; an actual remote client must additionally
    attest the revision loaded by its service. This module never loads weights.
    """

    def __init__(
        self,
        client: SharedModelClient,
        *,
        model: ModelRef,
        session_id: str,
        prompts=None,
        backend: str,
        history_window: int = 8,
        attempt_id=None,
    ):
        require(isinstance(model, ModelRef), "Expected pinned ModelRef")
        text(session_id, "session_id")
        require(backend in ("mock", "real"), "Declare execution backend")
        if attempt_id is not None:
            text(attempt_id, "attempt_id")
        require(
            backend != "real" or model.tokenizer.backend != "fixture",
            "Fixture model cannot be real",
        )
        integer(history_window, "history_window")
        require(history_window <= 256, "History context window is too large")
        require(
            client is not None and client.model_ref == model,
            "Shared client must match pinned ModelRef",
            ErrorCode.STALE_CACHE,
        )
        templates = default_prompts() if prompts is None else dict(prompts)
        require(
            set(templates) == set(ROLES)
            and all(
                isinstance(templates[r], PromptTemplate) and templates[r].role == r
                for r in ROLES
            ),
            "Provide one prompt for each role",
        )
        self._model = model
        self._client = client
        self._session_id = session_id
        self._backend = backend
        self._attempt_id = attempt_id
        self._scope = RoleScope()
        self._prompts = MappingProxyType(templates)
        self._window = history_window
        self._histories = {
            r: RoleHistory(
                r,
                self.namespace(r),
                model.model_hash,
                templates[r].prompt_hash,
                backend,
            )
            for r in ROLES
        }
        self._active = set()
        self._lock = RLock()

    @property
    def model_ref(self):
        return self._model

    @property
    def session_id(self):
        return self._session_id

    @property
    def attempt_id(self):
        return self._attempt_id or self._session_id

    def bind_attempt(self, attempt_id):
        text(attempt_id, "attempt_id")
        with self._lock:
            require(
                self._attempt_id is None or self._attempt_id == attempt_id,
                "Role session cannot mix outer attempts",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            require(
                self._attempt_id is not None
                or not any(h.event_count for h in self._histories.values())
                or attempt_id == self._session_id,
                "Existing role work cannot be relabelled to another attempt",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            self._attempt_id = attempt_id

    @property
    def backend(self):
        return self._backend

    @property
    def scope(self):
        return self._scope

    def bind_scope(self, scope):
        with self._lock:
            self._scope = self._scope.merge(scope)

    @property
    def prompts(self):
        return self._prompts

    @property
    def shared_client(self):
        return self._client

    def namespace(self, role):
        require(role in ROLES, "Unknown role")
        return (
            "roles/"
            + digest(self._session_id)
            + "/"
            + self._model.model_hash
            + "/"
            + role
            + "/"
            + self._prompts[role].prompt_hash
        )

    def history(self, role):
        require(role in ROLES, "Unknown role")
        with self._lock:
            return self._histories[role].events

    def history_context(self, role):
        require(role in ROLES, "Unknown role")
        with self._lock:
            return self._histories[role].context(self._window)

    def _check_ready(self):
        require(
            self._client.model_ref == self._model,
            "Shared client changed model parameters",
            ErrorCode.STALE_CACHE,
        )
        interrupted = [
            e.call_id
            for history in self._histories.values()
            for e in history.pending()
            if e.call_id not in self._active
        ]
        require(
            not interrupted,
            "Unresolved interrupted calls; reconcile explicitly before continuing",
            ErrorCode.ENVIRONMENT_FAILURE,
            call_ids=interrupted,
        )

    def _begin(self, role, operation, payload):
        self._check_ready()
        if self._attempt_id is None:
            self._attempt_id = self._session_id
        history = self._histories[role]
        call_id = "call-" + digest(
            {
                "namespace": history.namespace,
                "sequence": history.event_count,
                "parent": history.root_hash,
                "invocation": uuid.uuid4().hex,
            }
        )
        history.append(call_id, operation, "before", payload)
        self._active.add(call_id)
        return call_id

    def _finish(
        self,
        role,
        call_id,
        operation,
        *,
        result=None,
        error=None,
        usage=None,
        raw_reply=None,
    ):
        try:
            return self._histories[role].append(
                call_id,
                operation,
                "after",
                {
                    "status": "completed" if error is None else "failed",
                    "result": result,
                    "error": error,
                    "usage": (usage or ResourceUsage()).to_dict(),
                    "raw_reply": raw_reply,
                },
            )
        finally:
            self._active.discard(call_id)

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

    def generate(self, role, work, *, timeout_seconds=30, validator=None):
        require(role in ROLES, "Unknown role")
        finite(timeout_seconds, "timeout_seconds", 0)
        require(timeout_seconds > 0, "Timeout must be positive")
        require(type(work) is dict, "Role work must be JSON object")
        with self._lock:
            self._check_ready()
            work = load_json(canonical_json(work))
            history = self._histories[role].context(self._window)
            system = self._prompts[role].render()
            payload = {
                "schema": "role-work-v1",
                "role": role,
                "session_id": self._session_id,
                "model_ref": self._model.to_dict(),
                "memory_namespace": self.namespace(role),
                "prompt_hash": self._prompts[role].prompt_hash,
                "work": work,
                "history": history,
            }
            call_id = self._begin(
                role,
                "model",
                {
                    "work": work,
                    "system_prompt": system,
                    "history_call_ids": [v["call_id"] for v in history],
                    "history_hash": digest(history),
                    "timeout_seconds": timeout_seconds,
                },
            )
            usage = ResourceUsage()
            raw_reply = None
            started = perf_counter()
            try:
                reply = self._client.generate_json(
                    role=role,
                    system_prompt=system,
                    payload=load_json(canonical_json(payload)),
                    timeout_seconds=timeout_seconds,
                )
                raw_reply = load_json(canonical_json(reply))
                if type(reply) is dict and "usage" in reply:
                    usage = parse_contract("ResourceUsage", reply["usage"])
                require(
                    self._client.model_ref == self._model,
                    "Shared client changed model during generation",
                    ErrorCode.STALE_CACHE,
                )
                fields(reply, ("content",), ("usage",))
                content = (
                    load_json(reply["content"])
                    if type(reply["content"]) is str
                    else load_json(canonical_json(reply["content"]))
                )
                require(type(content) is dict, "Role response must be a JSON object")
                if validator is not None:
                    validator(content)
                if usage.compute.wall_seconds is None:
                    usage = replace(
                        usage,
                        compute=replace(
                            usage.compute,
                            wall_seconds=max(0.0, perf_counter() - started),
                        ),
                    )
                self._finish(
                    role,
                    call_id,
                    "model",
                    result=content,
                    usage=usage,
                    raw_reply=raw_reply,
                )
                return RoleReply(role, call_id, canonical_json(content), usage)
            except BaseException as exc:
                if usage.compute.wall_seconds is None:
                    usage = replace(
                        usage,
                        compute=replace(
                            usage.compute,
                            wall_seconds=max(0.0, perf_counter() - started),
                        ),
                    )
                error = self._error(exc)
                self._finish(
                    role,
                    call_id,
                    "model",
                    error=error,
                    usage=usage,
                    raw_reply=raw_reply,
                )
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, ContractError):
                    raise
                raise ContractError(
                    ErrorCode.ENVIRONMENT_FAILURE,
                    "Role backend call failed",
                    role=role,
                    error=error,
                ) from exc

    def run_tool(self, role, tool, work, function, summarize):
        require(
            role in ROLES and tool in TOOL_NAMES[role],
            "Tool is not allowed for this role",
        )
        require(type(work) is dict, "Tool work must be JSON object")
        with self._lock:
            call_id = self._begin(
                role, "tool:" + tool, {"work": load_json(canonical_json(work))}
            )
            started = perf_counter()
            try:
                result = function()
                require(
                    self._client.model_ref == self._model,
                    "Shared client changed model during tool work",
                    ErrorCode.STALE_CACHE,
                )
                summary = load_json(canonical_json(summarize(result)))
                usage = ResourceUsage(
                    ComputeCost(max(0.0, perf_counter() - started), 0, 0),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                self._finish(role, call_id, "tool:" + tool, result=summary, usage=usage)
                return result
            except BaseException as exc:
                usage = ResourceUsage(
                    ComputeCost(max(0.0, perf_counter() - started), 0, 0),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                self._finish(
                    role, call_id, "tool:" + tool, error=self._error(exc), usage=usage
                )
                raise

    def resolve_interrupted(self, role, call_id, *, reason):
        require(role in ROLES, "Unknown role")
        text(reason, "interruption reason")
        with self._lock:
            event = next(
                (e for e in self._histories[role].pending() if e.call_id == call_id),
                None,
            )
            require(
                event is not None and call_id not in self._active,
                "Call is not an unresolved interruption",
            )
            self._finish(
                role,
                call_id,
                event.operation,
                error={
                    "code": ErrorCode.ENVIRONMENT_FAILURE.value,
                    "message": reason,
                    "details": {"interrupted": True},
                },
                usage=ResourceUsage(),
            )

    def ledger(self):
        """Model-call costs only. Tool work keeps its existing P1/business ledger."""
        with self._lock:
            events = []
            for role in ROLES:
                for event in self._histories[role].events:
                    if event.operation == "model" and event.phase == "after":
                        payload = event.payload
                        events.append(
                            CostEvent(
                                event.call_id,
                                self.attempt_id,
                                "role_call",
                                role,
                                self._model.model_hash,
                                payload["status"],
                                parse_contract("ResourceUsage", payload["usage"]),
                                ArtifactRef(
                                    "record://role-history/" + event.record_hash,
                                    event.record_hash,
                                ),
                                self._backend,
                            )
                        )
            return CostLedger(
                "roles-"
                + digest((self._session_id, self._model.model_hash, self.attempt_id)),
                tuple(events),
                self._backend,
            )

    def save(self, output):
        with self._lock:
            output = Path(output).resolve()
            require(
                not output.exists(), "History snapshot exists", ErrorCode.OUTPUT_EXISTS
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            staging = _staging_directory(output)
            meta = {
                "schema_version": "1",
                "session_id": self._session_id,
                "attempt_id": self._attempt_id,
                "scope": self._scope.to_dict(),
                "model_ref": self._model.to_dict(),
                "backend": self._backend,
                "history_window": self._window,
                "prompts": {r: self._prompts[r].to_dict() for r in ROLES},
                "prompt_hashes": {r: self._prompts[r].prompt_hash for r in ROLES},
            }
            payloads = {
                "session.json": meta,
                "ledger.json": self.ledger().to_dict(),
                **{r + ".json": self._histories[r].to_dict() for r in ROLES},
            }
            for name, value in payloads.items():
                write_json(staging / name, value)
            write_json(
                staging / "checksums.json",
                {name: file_sha256(staging / name) for name in payloads},
            )
            _publish(staging, output)
            return {
                "session_id": self._session_id,
                "model_hash": self._model.model_hash,
                "roles": list(ROLES),
                "output": str(output),
            }

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
        try:
            return cls._restore(
                path,
                client,
                expected_model=expected_model,
                expected_prompts=expected_prompts,
                expected_session_id=expected_session_id,
                expected_attempt_id=expected_attempt_id,
                expected_backend=expected_backend,
                expected_scope=expected_scope,
            )
        except ContractError:
            raise
        except (
            FileNotFoundError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            UnicodeError,
        ) as exc:
            raise ContractError(
                ErrorCode.ARTIFACT_MISMATCH,
                "Missing or malformed role history snapshot",
                reason=type(exc).__name__,
            ) from exc

    @classmethod
    def _restore(
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
        require(isinstance(expected_model, ModelRef), "Expected pinned ModelRef")
        root = Path(path)
        checks = load_json((root / "checksums.json").read_text(encoding="utf-8"))
        names = {"session.json", "ledger.json", *(r + ".json" for r in ROLES)}
        require(
            type(checks) is dict
            and set(checks) == names
            and {p.name for p in root.iterdir()} == names | {"checksums.json"},
            "Incomplete/unknown history snapshot files",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        for name, sha in checks.items():
            require(
                file_sha256(root / name) == sha,
                "History file checksum mismatch",
                ErrorCode.ARTIFACT_MISMATCH,
                file=name,
            )
        meta = fields(
            load_json((root / "session.json").read_text(encoding="utf-8")),
            (
                "schema_version",
                "session_id",
                "attempt_id",
                "model_ref",
                "backend",
                "history_window",
                "prompts",
                "prompt_hashes",
            ),
            ("scope",),
        )
        for name, expected in (
            ("session_id", expected_session_id),
            ("attempt_id", expected_attempt_id),
            ("backend", expected_backend),
        ):
            if expected is not None:
                require(
                    meta[name] == expected,
                    "History differs from independent session anchor",
                    ErrorCode.ARTIFACT_MISMATCH,
                    field=name,
                )
        require(
            meta["schema_version"] == "1"
            and model_ref(meta["model_ref"]) == expected_model,
            "History belongs to another model",
            ErrorCode.STALE_CACHE,
        )
        require(
            type(meta["prompts"]) is dict
            and set(meta["prompts"]) == set(ROLES)
            and type(meta["prompt_hashes"]) is dict
            and set(meta["prompt_hashes"]) == set(ROLES),
            "Missing role prompt identities",
        )
        prompt_keys = (
            "schema_version",
            "role",
            "template_id",
            "revision",
            "instructions",
            "input_format",
        )
        prompts = {
            r: PromptTemplate(**fields(meta["prompts"][r], prompt_keys)) for r in ROLES
        }
        require(
            {r: prompts[r].prompt_hash for r in ROLES} == meta["prompt_hashes"],
            "Prompt protocol identity changed",
            ErrorCode.STALE_CACHE,
        )
        if expected_prompts is not None:
            require(
                dict(expected_prompts) == prompts,
                "History prompt configuration changed",
                ErrorCode.STALE_CACHE,
            )
        runtime = cls(
            client,
            model=expected_model,
            session_id=meta["session_id"],
            prompts=prompts,
            backend=meta["backend"],
            history_window=meta["history_window"],
            attempt_id=meta["attempt_id"],
        )
        for role in ROLES:
            history = RoleHistory.from_dict(
                load_json((root / (role + ".json")).read_text(encoding="utf-8"))
            )
            require(
                (
                    history.role,
                    history.namespace,
                    history.model_hash,
                    history.prompt_hash,
                    history.backend,
                )
                == (
                    role,
                    runtime.namespace(role),
                    expected_model.model_hash,
                    prompts[role].prompt_hash,
                    runtime.backend,
                ),
                "History file belongs to another role/session/model",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            runtime._histories[role] = history
            require(
                meta["attempt_id"] is not None or history.event_count == 0,
                "Nonempty role history needs a bound outer attempt",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            require(
                all(
                    e.payload["system_prompt"] == prompts[role].render()
                    for e in history.events
                    if e.operation == "model" and e.phase == "before"
                ),
                "History call used another role prompt",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        ledger = parse_contract(
            "CostLedger", load_json((root / "ledger.json").read_text(encoding="utf-8"))
        )
        require(
            ledger == runtime.ledger(),
            "History model costs differ from events",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        observed = infer_scope(runtime._histories.values())
        if "scope" in meta:
            declared = parse_role_scope(meta["scope"])
            require(
                declared.merge(observed) == declared,
                "Saved session scope omits observed conditions",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            runtime._scope = declared
        else:
            runtime._scope = observed
        if expected_scope is not None:
            require(
                runtime.scope.merge(expected_scope) == runtime.scope,
                "History lacks required session scope anchors",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        return runtime
