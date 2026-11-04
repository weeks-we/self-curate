"""P1 selection -> P0 handoff. No score/gradient/model/training backend calls."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ContractError,
    CostEvent,
    CostLedger,
    ErrorCode,
    ObjectiveSpec,
    ResourceUsage,
    SelectionResult,
    WorkflowBundle,
    canonical_json,
    digest,
    load_workflow,
    parse_contract,
    validate_workflow,
)
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.program import ProgramSpec, selection_identity
from self_curate.contracts.types import require, sha256, text
from self_curate.data import MetricView, read_pool
from self_curate.data.pool import _publish, _staging_directory, file_sha256, write_json
from .context import ExecutionContext, load_context, parse_context
from .io import load_program
from .runtime import execute_program


def _ledger(program, context, backend, attempt_id, invocation_id, outcome, wall):
    identity = digest(
        {
            "operation": "selection",
            "attempt_id": attempt_id,
            "invocation_id": invocation_id,
        }
    )
    receipt_hash = digest(outcome)
    # This is metadata-only CPU work: no forward/gradient/GPU tokens were spent.
    usage = ResourceUsage(ComputeCost(wall, 0, 0), 0, 0, 0, 0, 0, 0, 0, 0)
    event = CostEvent(
        identity,
        attempt_id,
        "selection",
        program.program_id,
        context.model_ref.model_hash,
        "failed" if outcome["error"] is not None else "completed",
        usage,
        ArtifactRef("record://selection-outcome/" + receipt_hash, receipt_hash),
        backend,
    )
    return CostLedger("selection-" + identity, (event,), backend)


def _outcome(selection, error):
    return {
        "selection": None if selection is None else selection.to_dict(),
        "error": error,
    }


def _domain(context, objective, backend):
    require(
        backend in ("mock", "real") and context.model_ref is not None,
        "Preparation requires explicit backend and ModelRef",
        ErrorCode.INVALID_CONFIG,
    )
    if backend == "real":
        require(
            context.model_ref.tokenizer.backend != "fixture",
            "Fixture cannot be prepared as real",
            ErrorCode.INVALID_CONFIG,
        )
        references = [objective.estimator, objective.acceptance_policy]
        references.extend(
            r.artifact for signal in context.signals for r in signal.references
        )
        if context.similarities is not None:
            references.append(context.similarities.identity)
        require(
            all(
                not ref.uri.strip().lower().startswith(("fixture://", "mock://"))
                for ref in references
            ),
            "Mock objective/reference context cannot be prepared as real",
            ErrorCode.INVALID_CONFIG,
        )


@dataclass(frozen=True)
class PreparedSelection:
    program: ProgramSpec
    context: ExecutionContext
    objective: ObjectiveSpec
    backend: str
    attempt_id: str
    invocation_id: str
    selection: SelectionResult | None
    error_json: str | None
    ledger: CostLedger
    workflow: WorkflowBundle | None

    def __post_init__(self):
        require(
            isinstance(self.program, ProgramSpec)
            and isinstance(self.context, ExecutionContext)
            and isinstance(self.objective, ObjectiveSpec),
            "Invalid preparation inputs",
        )
        _domain(self.context, self.objective, self.backend)
        text(self.attempt_id, "attempt_id")
        text(self.invocation_id, "invocation_id")
        require(
            (self.selection is None) != (self.error_json is None),
            "Preparation must have either selection or structured error",
        )
        require(
            self.selection is None or isinstance(self.selection, SelectionResult),
            "Invalid selection result",
        )
        if self.selection is not None:
            require(
                (self.selection.program_hash, self.selection.execution_hash)
                == selection_identity(
                    self.program,
                    self.context.policy_hash,
                    self.context.context_hash,
                    self.context.snapshot_hash,
                ),
                "Selection belongs to another program/context",
                ErrorCode.ARTIFACT_MISMATCH,
            )
        if self.error_json is not None:
            fields(self.error, ("code", "message", "details"))
        require(
            isinstance(self.ledger, CostLedger) and len(self.ledger.events) == 1,
            "Selection invocation needs one cost event",
        )
        wall = self.ledger.events[0].usage.compute.wall_seconds
        require(wall is not None, "Selection wall time must be recorded")
        expected = _ledger(
            self.program,
            self.context,
            self.backend,
            self.attempt_id,
            self.invocation_id,
            self.outcome,
            wall,
        )
        require(
            expected == self.ledger,
            "Selection cost identity/outcome mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        require(
            (self.workflow is not None) == (self.status == "feasible"),
            "Only feasible selection can produce a workflow",
        )
        if self.workflow is not None:
            require(
                self.workflow.program == self.program
                and self.workflow.batch == self.selection.batch
                and self.workflow.ledger == self.ledger,
                "Handoff changed program/batch/cost",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            require(
                not any(
                    (
                        self.workflow.tasks,
                        self.workflow.rollouts,
                        self.workflow.target_gradients,
                        self.workflow.sample_gradients,
                        self.workflow.evaluations,
                        self.workflow.capabilities,
                    )
                )
                and self.workflow.round_record is None,
                "P1 handoff cannot fabricate downstream execution",
            )
            self.validate_workflow()

    @property
    def error(self):
        return None if self.error_json is None else load_json(self.error_json)

    @property
    def status(self):
        return "failed" if self.selection is None else self.selection.status

    @property
    def outcome(self):
        return _outcome(self.selection, self.error)

    def validate_workflow(self):
        require(
            self.workflow is not None,
            "No feasible workflow to hand off",
            ErrorCode.INSUFFICIENT_POOL,
        )
        return validate_workflow(
            self.workflow,
            self.context.view,
            expected_model=self.context.model_ref,
            expected_objective=self.objective,
            expected_selection_context_hash=self.context.context_hash,
            expected_batch_id=self.selection.batch.batch_id,
        )

    def to_dict(self):
        return {
            "schema_version": "1",
            "status": self.status,
            "backend": self.backend,
            "attempt_id": self.attempt_id,
            "invocation_id": self.invocation_id,
            "request_hash": digest(
                {
                    "recipe_hash": self.program.recipe_hash,
                    "context_hash": self.context.context_hash,
                    "objective_hash": self.objective.objective_hash,
                    "backend": self.backend,
                }
            ),
            "context_hash": self.context.context_hash,
            "objective_hash": self.objective.objective_hash,
            "program_hash": (
                None if self.selection is None else self.selection.program_hash
            ),
            "execution_hash": (
                None if self.selection is None else self.selection.execution_hash
            ),
            "batch_id": None if self.workflow is None else self.workflow.batch.batch_id,
            "ledger_hash": self.ledger.record_hash,
            "workflow_hash": (
                None if self.workflow is None else self.workflow.record_hash
            ),
        }


def prepare_selection(
    program, context, objective, *, backend, attempt_id, invocation_id
):
    """Execute a single P1 candidate and emit an unvalued, untrained P0 bundle.

    `invocation_id` identifies this physical call; retries use new invocation IDs.
    P2 must restore already completed calls instead of invoking them twice.
    """
    require(
        isinstance(program, ProgramSpec)
        and isinstance(context, ExecutionContext)
        and isinstance(objective, ObjectiveSpec),
        "Expected typed program/context/objective",
    )
    _domain(context, objective, backend)
    text(attempt_id, "attempt_id")
    text(invocation_id, "invocation_id")
    started = perf_counter()
    selection, error = None, None
    try:
        selection = execute_program(program, context)
    except ContractError as exc:
        error = exc.to_dict()
    elapsed = max(0.0, perf_counter() - started)
    ledger = _ledger(
        program,
        context,
        backend,
        attempt_id,
        invocation_id,
        _outcome(selection, error),
        elapsed,
    )
    workflow = None
    if selection is not None and selection.batch is not None:
        workflow = WorkflowBundle(
            "selection-" + ledger.events[0].event_id,
            backend,
            context.model_ref,
            program,
            context.context_hash,
            selection.batch,
            objective,
            (),
            (),
            (),
            (),
            (),
            (),
            ledger,
            None,
        )
    return PreparedSelection(
        program,
        context,
        objective,
        backend,
        attempt_id,
        invocation_id,
        selection,
        None if error is None else canonical_json(error),
        ledger,
        workflow,
    )


def prepare_to_directory(
    pool_path,
    program_path,
    context_path,
    objective_path,
    output,
    *,
    backend,
    attempt_id,
    invocation_id,
):
    output = Path(output).resolve()
    require(
        not output.exists(),
        "Preparation output already exists",
        ErrorCode.OUTPUT_EXISTS,
        path=str(output),
    )
    pool = read_pool(pool_path)
    program = load_program(program_path)
    context = load_context(context_path, pool.metric_view())
    objective = parse_contract(
        "ObjectiveSpec", load_json(Path(objective_path).read_text(encoding="utf-8"))
    )
    result = prepare_selection(
        program,
        context,
        objective,
        backend=backend,
        attempt_id=attempt_id,
        invocation_id=invocation_id,
    )
    return publish_preparation(result, output)


def publish_preparation(result, output):
    """Publish an already executed P1 result without rerunning or recharging it."""
    require(isinstance(result, PreparedSelection), "Expected PreparedSelection")
    output = Path(output).resolve()
    require(
        not output.exists(),
        "Preparation output already exists",
        ErrorCode.OUTPUT_EXISTS,
        path=str(output),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_directory(output)
    payloads = {
        "prepared.json": result.to_dict(),
        "program.json": result.program.to_dict(),
        "context.json": result.context.to_dict(),
        "objective.json": result.objective.to_dict(),
        "outcome.json": result.outcome,
        "ledger.json": result.ledger.to_dict(),
        "trace.json": (
            [t.to_dict() for t in result.selection.traces]
            if result.selection is not None
            else result.error["details"].get("completed_traces", [])
        ),
    }
    if result.workflow is not None:
        payloads.update(
            {
                "workflow.json": result.workflow.to_dict(),
                "batch.json": result.workflow.batch.to_dict(),
            }
        )
    for name, payload in payloads.items():
        write_json(staging / name, payload)
    write_json(
        staging / "checksums.json",
        {name: file_sha256(staging / name) for name in payloads},
    )
    _publish(staging, output)
    return result


def _verify_preparation(
    path, view, *, expected_model=None, expected_objective=None, expected_batch_id=None
):
    """Check publication hashes and replay only CPU operators on the frozen pool.

    Uses the original selection snapshot. After enriching features, consume
    workflow.json via load_workflow(view=...) instead of replaying this snapshot.
    """
    require(isinstance(view, MetricView), "Expected original selection MetricView")
    root = Path(path)
    meta = load_json((root / "prepared.json").read_text(encoding="utf-8"))
    fields(
        meta,
        (
            "schema_version",
            "status",
            "backend",
            "attempt_id",
            "invocation_id",
            "request_hash",
            "context_hash",
            "objective_hash",
            "program_hash",
            "execution_hash",
            "batch_id",
            "ledger_hash",
            "workflow_hash",
        ),
    )
    require(
        meta["schema_version"] == "1"
        and meta["status"] in ("feasible", "infeasible", "failed"),
        "Invalid preparation schema/status",
    )
    names = {
        "prepared.json",
        "program.json",
        "context.json",
        "objective.json",
        "outcome.json",
        "ledger.json",
        "trace.json",
    }
    if meta["status"] == "feasible":
        names.update(("workflow.json", "batch.json"))
    checks = load_json((root / "checksums.json").read_text(encoding="utf-8"))
    require(
        type(checks) is dict
        and set(checks) == names
        and {p.name for p in root.iterdir()} == names | {"checksums.json"},
        "Incomplete/unknown preparation files",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    for name, expected in checks.items():
        sha256(expected, "file checksum")
        require(
            file_sha256(root / name) == expected,
            "Preparation file checksum mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
            file=name,
        )
    program = load_program(root / "program.json")
    context = parse_context(
        load_json((root / "context.json").read_text(encoding="utf-8")), view
    )
    objective = parse_contract(
        "ObjectiveSpec",
        load_json((root / "objective.json").read_text(encoding="utf-8")),
    )
    ledger = parse_contract(
        "CostLedger", load_json((root / "ledger.json").read_text(encoding="utf-8"))
    )
    outcome = load_json((root / "outcome.json").read_text(encoding="utf-8"))
    fields(outcome, ("selection", "error"))
    selection, error = None, None
    try:
        selection = execute_program(program, context)
    except ContractError as exc:
        error = exc.to_dict()
    require(
        canonical_json(_outcome(selection, error)) == canonical_json(outcome),
        "Stored outcome differs from deterministic replay",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    workflow = (
        load_workflow(
            root / "workflow.json",
            view=view,
            expected_model=expected_model or context.model_ref,
            expected_objective=expected_objective or objective,
            expected_selection_context_hash=context.context_hash,
            expected_batch_id=expected_batch_id
            or (
                None
                if selection is None or selection.batch is None
                else selection.batch.batch_id
            ),
        )
        if meta["status"] == "feasible"
        else None
    )
    result = PreparedSelection(
        program,
        context,
        objective,
        meta["backend"],
        meta["attempt_id"],
        meta["invocation_id"],
        selection,
        None if error is None else canonical_json(error),
        ledger,
        workflow,
    )
    require(
        canonical_json(result.to_dict()) == canonical_json(meta),
        "Preparation metadata mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    traces = (
        [t.to_dict() for t in selection.traces]
        if selection is not None
        else error["details"].get("completed_traces", [])
    )
    require(
        canonical_json(traces)
        == canonical_json(load_json((root / "trace.json").read_text(encoding="utf-8"))),
        "Trace differs from replay",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    if workflow is not None:
        from .io import load_batch

        require(
            load_batch(root / "batch.json") == workflow.batch,
            "Handoff batch mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    if expected_model is not None:
        require(
            context.model_ref == expected_model,
            "Preparation model mismatch",
            ErrorCode.STALE_CACHE,
        )
    if expected_objective is not None:
        require(
            objective == expected_objective,
            "Preparation objective mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    if expected_batch_id is not None:
        sha256(expected_batch_id, "expected batch ID")
        require(
            meta["batch_id"] == expected_batch_id,
            "Preparation differs from the exact selected batch",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    return result


def load_preparation(
    path, view, *, expected_model=None, expected_objective=None, expected_batch_id=None
):
    """Load a typed P1 handoff after checksums, exact anchors and CPU replay."""
    try:
        return _verify_preparation(
            path,
            view,
            expected_model=expected_model,
            expected_objective=expected_objective,
            expected_batch_id=expected_batch_id,
        )
    except ContractError:
        raise
    except (FileNotFoundError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Missing or malformed preparation artifact",
            reason=type(exc).__name__,
        ) from exc


def verify_preparation(
    path, view, *, expected_model=None, expected_objective=None, expected_batch_id=None
):
    result = load_preparation(
        path,
        view,
        expected_model=expected_model,
        expected_objective=expected_objective,
        expected_batch_id=expected_batch_id,
    )
    return {
        "status": "verified",
        "selection_status": result.status,
        "backend": result.backend,
        "replayed": True,
        "batch_id": None if result.workflow is None else result.workflow.batch.batch_id,
        "expected_model_verified": expected_model is not None,
        "expected_objective_verified": expected_objective is not None,
        "expected_batch_verified": expected_batch_id is not None,
    }
