"""Exact candidate inputs, conservative costs, immutable result evidence."""

from dataclasses import fields, replace
from pathlib import Path
import uuid
import base64
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ConstraintResult,
    CostLedger,
    CostEvent,
    ResourceUsage,
    SkillValue,
    TrainingUse,
    ValidationRequest,
    ValidationResult,
    digest,
    verify_batch,
)
from self_curate.contracts.types import require
from self_curate.agents.leval import check_external_result
from self_curate.data import record_candidate_training, attach_usage
from self_curate.training.execution import code_identity
from self_curate.data.pool import file_sha256, write_json
from self_curate.training import TrainingData, TrainingExample


def candidate_data(request, view):
    require(isinstance(request, ValidationRequest), "Expected ValidationRequest")
    order = verify_batch(request.batch, view, model_ref=request.model_ref)
    for i in order:
        record = view.record(i)
        require(
            record.tokenization.identity == request.model_ref.tokenizer,
            "Candidate tokenizer/template mismatch",
        )
        require(
            request.backend != "real" or not record.known_mock_artifacts,
            "Mock candidate cannot be real",
        )
    return TrainingData(
        "subset",
        request.batch.batch_id,
        request.model_ref.tokenizer,
        tuple(
            TrainingExample(
                i,
                view.record(i).content_hash,
                view.record(i).tokenization.input_ids,
                view.record(i).labels,
            )
            for i in order
        ),
        (
            ArtifactRef(
                "record://selected-batch/" + request.batch.batch_id,
                request.batch.batch_id,
            ),
        ),
        request.backend,
    )


def require_isolation(view, data, suite):
    """Explicit ID/family guards; source-level near-duplicate audits remain upstream."""
    task_ids = {t.task_id for t in suite.tasks} | {
        f for t in suite.tasks for f in t.family_ids
    }
    for item in data.examples:
        row = view.record(item.data_id)
        require(
            not {
                row.data_id,
                row.source.record_id,
                *row.source.family_ids,
            }.intersection(task_ids),
            "Candidate overlaps development tasks by ID/source family",
        )


def usage_from_ledger(ledger):
    summary = ledger.summary()
    compute = ComputeCost(
        **{
            n: summary[n]["total"]
            for n in ("wall_seconds", "gpu_seconds", "processed_tokens")
        }
    )
    return ResourceUsage(
        compute,
        **{
            f.name: summary[f.name]["total"]
            for f in fields(ResourceUsage)
            if f.name not in ("compute", "schema_version")
        },
    )


class ValuationFailure(RuntimeError):
    def __init__(self, message, *, usage, evidence=None, ledger=None):
        super().__init__(message)
        self.usage, self.evidence = usage, evidence
        self.ledger = ledger


def implementation_identity(*adapters):
    """Pin the valuation kernels and the implementations they actually consume."""
    root = Path(__file__).resolve().parents[1]
    names = ("training/contracts.py", "validation/contracts.py")
    paths = [root / n for n in names] + list(Path(__file__).parent.glob("*.py"))
    return digest(
        {
            "sources": {p.relative_to(root).as_posix(): file_sha256(p) for p in paths},
            "adapters": [code_identity(a) for a in adapters],
        }
    )


def invoke_service(
    function, *, run, stage, operation, subject_id, model, attempt_id, backend
):
    """Keep a reported ledger; otherwise retain a boundary call with unknown costs.

    A typed empty ledger means no physical work dispatched. A missing ledger
    cannot prove that. Never invent rollout counts when a service reply is lost.
    """
    start, value, error = perf_counter(), None, None
    try:
        value = function()
    except Exception as exc:
        error = exc
    observed = error if error is not None else value
    ledger = getattr(observed, "ledger", None)
    if not isinstance(ledger, CostLedger):
        usage = getattr(observed, "usage", None)
        if not isinstance(usage, ResourceUsage):
            usage = ResourceUsage()
        if usage.compute.wall_seconds is None:
            usage = replace(
                usage,
                compute=replace(usage.compute, wall_seconds=perf_counter() - start),
            )
        receipt = {
            "run": run.name,
            "stage": stage,
            "operation": operation,
            "subject_id": subject_id,
            "model_hash": model.model_hash,
            "return_type": type(value).__name__,
            "error": None if error is None else str(error),
            "usage": usage.to_dict(),
            "missing_service_ledger": True,
        }
        event = CostEvent(
            "service-" + digest((run.name, stage)),
            attempt_id,
            operation,
            subject_id,
            model.model_hash,
            "failed",
            usage,
            ArtifactRef(
                "record://incomplete-service-receipt/" + digest(receipt),
                digest(receipt),
            ),
            backend,
        )
        ledger = cost_ledger(run, (event,), backend)
    if error is not None:
        raise ValuationFailure(
            str(error), usage=usage_from_ledger(ledger), ledger=ledger
        ) from error
    return value, ledger.events


def restore_usage(view, entries):
    from self_curate.contracts.parsing import fields as json_fields, usage_history

    require(type(entries) is list, "Saved exposure history must be an array")
    records = {}
    for entry in entries:
        row = json_fields(
            entry, ("data_id", "content_hash", "tokenization_hash", "usage")
        )
        require(row["data_id"] not in records, "Duplicate restored exposure ID")
        record = view.record(row["data_id"])
        require(
            record.content_hash == row["content_hash"]
            and digest(record.tokenization.to_dict()) == row["tokenization_hash"],
            "Exposure history belongs to changed training inputs",
        )
        incoming = usage_history(row["usage"])
        for use in incoming.candidate_sft:
            existing = next(
                (
                    u
                    for u in record.usage.candidate_sft
                    if u.attempt_id == use.attempt_id
                ),
                None,
            )
            if (
                existing is not None
                and existing.decision != "pending"
                and use.decision == "pending"
            ):
                require(
                    (existing.model_id, existing.receipt)
                    == (use.model_id, use.receipt),
                    "Restored training identity changed",
                )
                continue
            record = record_candidate_training(record, use)
        merged = replace(
            record.usage,
            forward_scoring=tuple(
                sorted(
                    set(record.usage.forward_scoring) | set(incoming.forward_scoring),
                    key=lambda a: (a.uri, a.sha256),
                )
            ),
            gradient_extraction=tuple(
                sorted(
                    set(record.usage.gradient_extraction)
                    | set(incoming.gradient_extraction),
                    key=lambda a: (a.uri, a.sha256),
                )
            ),
        )
        records[record.data_id] = attach_usage(record, merged)
    return records


def new_run(output, request):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Preserve all 128 UUID bits while leaving room for nested task receipts on Windows.
    label = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")
    run = root / ("eval-" + label)
    run.mkdir(exist_ok=False)
    write_json(run / "request.json", request.to_dict())
    return run


def artifact(path):
    return ArtifactRef(path.resolve().as_uri(), file_sha256(path))


def make_result(request, run, *, values, usage, details, reason=None):
    write_json(run / "evidence.json", details)
    if reason is not None:
        result = ValidationResult(
            "leval-" + digest((request.request_hash, run.name)),
            request.request_hash,
            "invalid",
            (),
            None,
            (),
            reason,
            artifact(run / "evidence.json"),
            usage,
            request.backend,
        )
    else:
        import math

        require(
            set(values) == {w.skill for w in request.objective.weights},
            "Incomplete objective skill coverage",
        )
        outcomes = tuple(
            ConstraintResult(
                b.name,
                values[b.skill],
                (b.minimum is None or values[b.skill] >= b.minimum)
                and (b.maximum is None or values[b.skill] <= b.maximum),
            )
            for b in request.objective.constraints
        )
        total = math.fsum(w.weight * values[w.skill] for w in request.objective.weights)
        result = ValidationResult(
            "leval-" + digest((request.request_hash, run.name)),
            request.request_hash,
            "valid",
            tuple(
                SkillValue(w.skill, values[w.skill]) for w in request.objective.weights
            ),
            total,
            outcomes,
            None,
            artifact(run / "evidence.json"),
            usage,
            request.backend,
        )
    check_external_result(request, result)
    write_json(run / "result.json", result.to_dict())
    write_json(
        run / "checksums.json",
        {
            p.relative_to(run).as_posix(): file_sha256(p)
            for p in run.rglob("*")
            if p.is_file()
        },
    )
    return result


def training_usage_records(request, outcome, view, previous):
    use = TrainingUse(
        outcome.request.training_id,
        request.model_ref.model_hash,
        outcome.record.receipt,
        "pending",
    )
    for i in request.batch.training_order:
        previous[i] = record_candidate_training(previous.get(i, view.record(i)), use)


def combine_events(events, extra=()):
    by_id = {e.event_id: e for e in events}
    for e in extra:
        require(
            e.event_id not in by_id or by_id[e.event_id] == e, "Conflicting cost event"
        )
        by_id[e.event_id] = e
    return tuple(by_id.values())


def cost_ledger(run, events, backend):
    return CostLedger("leval-" + run.name, tuple(events), backend)
