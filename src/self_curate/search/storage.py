"""Immutable search checkpoints and exact-best publication; no backend calls."""

from pathlib import Path
import re

from self_curate.agents import RoleTeam, default_prompts, load_role_cycle, request_scope
from self_curate.contracts import (
    ContractError,
    ErrorCode,
    ValidationRequest,
    canonical_json,
    parse_contract,
    parse_feedback_contract,
)
from self_curate.contracts.parsing import fields, load_json, sequence
from self_curate.contracts.types import integer, require
from self_curate.data.pool import _publish, _staging_directory, file_sha256, write_json
from self_curate.operators import load_preparation, publish_preparation
from .contracts import EvaluatedCandidate, ValidationAttempt
from .inner_loop import InnerSearch

ROOT_FILES = {
    "search.json",
    "initial_request.json",
    "policy.json",
    "validations.json",
    "failures.json",
    "timeline.json",
    "ledger.json",
}
BEST_FILES = {
    "request.json",
    "validation.json",
    "best_attempt.json",
    "result.json",
    "ledger.json",
}


def _finish(staging, output):
    write_json(
        staging / "checksums.json",
        {
            p.relative_to(staging).as_posix(): file_sha256(p)
            for p in staging.rglob("*")
            if p.is_file()
        },
    )
    _publish(staging, output)


def _directory(output):
    output = Path(output).resolve()
    require(not output.exists(), "Search output exists", ErrorCode.OUTPUT_EXISTS)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output, _staging_directory(output)


def _read(root, name):
    return load_json((root / name).read_text(encoding="utf-8"))


def _verify_tree(root, required, allowed):
    checks = _read(root, "checksums.json")
    require(
        type(checks) is dict and required.issubset(checks),
        "Incomplete search artifacts",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    for name, expected in checks.items():
        require(
            type(name) is str
            and allowed(name)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and Path(name).as_posix() == name,
            "Unknown search artifact",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            file_sha256(root / name) == expected,
            "Search checksum mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
            file=name,
        )
    actual = set()
    for path in root.rglob("*"):
        require(
            not path.is_symlink(),
            "Search artifacts cannot be symlinks",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        if path.is_file() and path != root / "checksums.json":
            actual.add(path.relative_to(root).as_posix())
    require(
        actual == set(checks),
        "Unknown/missing search files",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        {p.name for p in root.iterdir()}
        == {n.split("/")[0] for n in checks} | {"checksums.json"},
        "Unknown search directories",
        ErrorCode.ARTIFACT_MISMATCH,
    )


def save_search(search, output):
    output, staging = _directory(output)
    meta = {
        "schema_version": "1",
        "search_id": search._search_id,
        "use_model": search._use_model,
        "session_id": search.team.runtime.session_id,
        "operation_counter": search._operation_counter,
        "pending_operation": search._pending_operation,
        "phase": search._phase,
        "reason": search._reason,
        "state": search.state.to_dict(),
    }
    for name, value in {
        "search.json": meta,
        "initial_request.json": search._initial_request.to_dict(),
        "policy.json": search._policy.to_dict(),
        "validations.json": [v.to_dict() for v in search._validations],
        "failures.json": search._failures,
        "timeline.json": search._timeline,
        "ledger.json": search.ledger().to_dict(),
    }.items():
        write_json(staging / name, value)
    publish_preparation(search._initial, staging / "initial")
    search.team.save(staging / "roles")
    from self_curate.agents.handoff import _write_role_cycle

    for i, cycle in enumerate(search._cycles):
        destination = staging / f"cycle-{i:06d}"
        destination.mkdir()
        _write_role_cycle(cycle, destination, model_ledger=search.team.runtime.ledger())
    _finish(staging, output)
    return {"output": str(output), "state": search.state.to_dict()}


def _validation(row):
    fields(
        row,
        (
            "invocation_id",
            "outer_attempt_id",
            "request",
            "result",
            "error",
            "usage",
            "source",
            "external_cost",
        ),
    )
    return ValidationAttempt(
        row["invocation_id"],
        row["outer_attempt_id"],
        parse_feedback_contract("ValidationRequest", row["request"]),
        (
            None
            if row["result"] is None
            else parse_feedback_contract("ValidationResult", row["result"])
        ),
        None if row["error"] is None else canonical_json(row["error"]),
        parse_contract("ResourceUsage", row["usage"]),
        row["source"],
        (
            None
            if row["external_cost"] is None
            else parse_contract("CostEvent", row["external_cost"])
        ),
    )


def restore_search(
    path,
    client,
    view,
    *,
    expected_request,
    expected_policy,
    expected_search_id,
    expected_session_id,
    expected_attempt_id,
    expected_prompts=None,
    expected_use_model=True,
):
    """Restore a stable boundary; an in-flight checkpoint stays interrupted."""
    try:
        return _restore(
            path,
            client,
            view,
            expected_request=expected_request,
            expected_policy=expected_policy,
            expected_search_id=expected_search_id,
            expected_session_id=expected_session_id,
            expected_attempt_id=expected_attempt_id,
            expected_prompts=expected_prompts,
            expected_use_model=expected_use_model,
        )
    except ContractError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Missing/malformed search checkpoint",
            reason=type(exc).__name__,
        ) from exc


def _restore(
    path,
    client,
    view,
    *,
    expected_request,
    expected_policy,
    expected_search_id,
    expected_session_id,
    expected_attempt_id,
    expected_prompts,
    expected_use_model,
):
    root = Path(path)
    _verify_tree(
        root,
        ROOT_FILES,
        lambda n: n in ROOT_FILES
        or n.startswith(("initial/", "roles/"))
        or re.fullmatch(r"cycle-[0-9]{6,}/.+", n) is not None,
    )
    require(
        isinstance(expected_request, ValidationRequest),
        "Supply independent initial request",
    )
    request = parse_feedback_contract(
        "ValidationRequest", _read(root, "initial_request.json")
    )
    policy = parse_feedback_contract("EditPolicy", _read(root, "policy.json"))
    require(
        request == expected_request and policy == expected_policy,
        "Search request/edit policy changed",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    meta = fields(
        _read(root, "search.json"),
        (
            "schema_version",
            "search_id",
            "use_model",
            "session_id",
            "operation_counter",
            "pending_operation",
            "phase",
            "reason",
            "state",
        ),
    )
    require(
        meta["schema_version"] == "1"
        and meta["search_id"] == expected_search_id
        and meta["session_id"] == expected_session_id,
        "Search/session anchor mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    integer(meta["operation_counter"], "operation_counter")
    require(
        type(expected_use_model) is bool and meta["use_model"] == expected_use_model,
        "Search role execution mode changed",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    initial = load_preparation(
        root / "initial",
        view,
        expected_model=request.model_ref,
        expected_objective=request.objective,
        expected_batch_id=request.batch.batch_id,
    )
    require(
        initial.attempt_id == expected_attempt_id,
        "Search outer attempt changed",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    team = RoleTeam.restore(
        root / "roles",
        client,
        expected_model=request.model_ref,
        expected_prompts=(
            default_prompts() if expected_prompts is None else expected_prompts
        ),
        expected_session_id=expected_session_id,
        expected_attempt_id=initial.attempt_id,
        expected_backend=request.backend,
        expected_scope=request_scope(request, policy),
    )
    search = InnerSearch(
        team,
        initial,
        request,
        policy,
        search_id=expected_search_id,
        use_model=meta["use_model"],
        _restoring=True,
    )
    attempts = sequence(_read(root, "validations.json"), "validation attempts")
    failures = sequence(_read(root, "failures.json"), "search failures")
    timeline = sequence(_read(root, "timeline.json"), "search timeline")
    for entry in timeline:
        kind = entry.get("kind")
        if kind in ("validation", "cycle", "failure"):
            fields(entry, ("kind", "index"))
            integer(entry["index"], "search history index")
            require(
                search._phase != "stopped"
                and (
                    search._phase != "blocked"
                    or kind == "validation"
                    and search._pending_request is not None
                ),
                "Operation bypassed stopped/blocked state",
            )
            search._operation_counter += 1
            if kind == "validation":
                require(
                    entry["index"] == len(search._validations),
                    "Validation order changed",
                )
                attempt = _validation(attempts[entry["index"]])
                require(
                    attempt.invocation_id == search._operation_id(),
                    "Validation invocation changed",
                )
                search._accept_validation(attempt)
            elif kind == "cycle":
                require(
                    entry["index"] == len(search._cycles)
                    and search._current is not None,
                    "Cycle order/source changed",
                )
                cycle = load_role_cycle(
                    root / f"cycle-{entry['index']:06d}",
                    view,
                    expected_request=search._current.request,
                    expected_policy=policy,
                )
                search._accept_cycle(cycle)
            else:
                require(
                    entry["index"] == len(search._failures)
                    and search._pending_request is None,
                    "Failure order/phase changed",
                )
                failure = fields(failures[entry["index"]], ("invocation_id", "error"))
                require(
                    failure["invocation_id"] == search._operation_id()
                    and type(failure["error"]) is dict,
                    "Failure invocation changed",
                )
                search._failures.append(failure)
                search._timeline.append(entry)
                search._phase, search._reason = (
                    "blocked",
                    "Role operation failed; inspect evidence before retry",
                )
        elif kind == "retry":
            fields(entry, ("kind", "reason"))
            search.retry(reason=entry["reason"])
        elif kind == "stop":
            fields(entry, ("kind", "reason"))
            require(search._phase != "stopped", "Duplicate stop event")
            search.stop(reason=entry["reason"])
        else:
            require(False, "Unknown search event", ErrorCode.ARTIFACT_MISMATCH)
    require(
        len(attempts) == len(search._validations)
        and len(failures) == len(search._failures),
        "Unreferenced search evidence",
    )
    cycle_dirs = {p.name for p in root.iterdir() if p.name.startswith("cycle-")}
    require(
        cycle_dirs == {f"cycle-{i:06d}" for i in range(len(search._cycles))},
        "Unreferenced cycles",
    )
    pending = meta["pending_operation"]
    if pending is not None:
        fields(pending, ("kind", "invocation_id"))
        require(
            search._phase not in ("stopped", "blocked")
            and pending["kind"]
            == ("validation" if search._pending_request is not None else "cycle"),
            "Interrupted operation phase mismatch",
        )
        search._operation_counter += 1
        require(
            pending["invocation_id"] == search._operation_id(),
            "Interrupted invocation changed",
        )
        search._pending_operation = pending
    require(
        meta["phase"] == search._phase
        and meta["reason"] == search._reason
        and meta["operation_counter"] == search._operation_counter,
        "Search phase/counter replay mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        canonical_json(meta["state"]) == canonical_json(search.state.to_dict()),
        "Search best/current/cost replay mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        parse_contract("CostLedger", _read(root, "ledger.json")) == search.ledger(),
        "Search ledger changed",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    # Verify every cycle's physical model costs against the restored role log.
    from self_curate.agents.handoff import _costs

    for cycle in search._cycles:
        _costs(cycle, team.runtime.ledger())
    return search


def publish_best(search, output):
    require(
        search.state.phase == "stopped",
        "Finalize search with caller's stop decision before publishing best",
    )
    best = search.result().require_best()
    output, staging = _directory(output)
    publish_preparation(best.prepared, staging / "prepared")
    write_json(staging / "request.json", best.request.to_dict())
    write_json(staging / "validation.json", best.validation.to_dict())
    attempt = next(
        v
        for v in search.validations
        if v.result == best.validation and v.error_json is None
    )
    write_json(staging / "best_attempt.json", attempt.to_dict())
    write_json(staging / "result.json", search.result().to_dict())
    write_json(staging / "ledger.json", search.ledger().to_dict())
    _finish(staging, output)
    return {
        "output": str(output),
        "batch_id": best.request.batch.batch_id,
        "request_hash": best.request.request_hash,
    }


def load_best(path, view, *, expected_request):
    try:
        root = Path(path)
        _verify_tree(
            root, BEST_FILES, lambda n: n in BEST_FILES or n.startswith("prepared/")
        )
        request = parse_feedback_contract(
            "ValidationRequest", _read(root, "request.json")
        )
        require(
            request == expected_request,
            "Best differs from independently saved request",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        prepared = load_preparation(
            root / "prepared",
            view,
            expected_model=request.model_ref,
            expected_objective=request.objective,
            expected_batch_id=request.batch.batch_id,
        )
        candidate = EvaluatedCandidate(
            prepared,
            request,
            parse_feedback_contract("ValidationResult", _read(root, "validation.json")),
        )
        result = _read(root, "result.json")
        ledger = parse_contract("CostLedger", _read(root, "ledger.json"))
        attempt = _validation(_read(root, "best_attempt.json"))
        require(
            candidate.feasible
            and result["status"] == "stopped"
            and result["state"]["phase"] == "stopped"
            and result["state"]["unresolved_operation"] is None
            and result["state"]["best_request_hash"] == request.request_hash
            and result["state"]["best_batch_id"] == request.batch.batch_id
            and result["state"]["best_value"] == candidate.validation.total_value
            and result["state"]["ledger_hash"] == ledger.record_hash
            and ledger.backend == request.backend
            and attempt.request == request
            and attempt.result == candidate.validation
            and attempt.error_json is None
            and (attempt.event is None or attempt.event in ledger.events)
            and all(
                e.attempt_id == prepared.attempt_id
                and e.model_hash == request.model_ref.model_hash
                for e in ledger.events
            )
            and canonical_json(result["best_program"])
            == canonical_json(request.program.to_dict())
            and canonical_json(result["best_batch"])
            == canonical_json(request.batch.to_dict())
            and canonical_json(result["best_validation"])
            == canonical_json(candidate.validation.to_dict()),
            "Best result metadata changed",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        for event in prepared.ledger.events:
            require(event in ledger.events, "Best selection costs missing")
        return candidate
    except ContractError:
        raise
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH, "Missing/malformed best selection"
        ) from exc
