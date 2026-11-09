"""Final-test fanout. No optimization input, retry, checkpoint choice or split cast."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math

from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ResourceUsage,
    digest,
)
from self_curate.contracts.types import integer, require, sha256
from self_curate.data.pool import write_json
from self_curate.training.artifacts import verify_local_checkpoint
from self_curate.validation import TaskRunRequest, TaskRunResult
from .inputs import BenchmarkSuite


class FinalEvaluationFailure(RuntimeError):
    def __init__(self, message, *, ledger, report):
        super().__init__(message)
        self.ledger, self.report = ledger, report


def evaluate_benchmark(
    runner,
    model,
    suite,
    *,
    checkpoint,
    output,
    evaluation_id,
    sealed_training_hash,
    workers=1,
    cache=None,
    resume=False,
    progress=None,
):
    require(
        isinstance(suite, BenchmarkSuite) and suite.backend == runner.backend,
        "Final benchmark domain mismatch",
    )
    sha256(sealed_training_hash, "frozen training/checkpoint seal")
    integer(workers, "benchmark worker slots", 1)
    require(
        workers <= 16, "EXP001 supports at most eight GPUs times two final task slots"
    )
    if checkpoint is not None:
        verified = verify_local_checkpoint(checkpoint)
        require(
            suite.backend != "real" or verified,
            "Real benchmark needs byte-verified checkpoint",
        )
    if suite.backend == "real":
        require(
            model.tokenizer.backend != "fixture",
            "Fixture model cannot enter real benchmark",
        )
    from self_curate.training.recovery import (
        read_json,
        atomic_json,
        commit_directory,
        verify_directory,
    )

    output = Path(output)
    cache = {} if cache is None else cache
    envelope = {
        "model_ref": model.to_dict(),
        "suite": suite.to_dict(),
        "checkpoint": None if checkpoint is None else checkpoint.to_dict(),
        "sealed_training_hash": sealed_training_hash,
        "evaluation_id": evaluation_id,
    }
    if output.exists():
        require(
            resume and read_json(output / "request.json") == envelope,
            "Benchmark recovery request/seal changed",
        )
        if (output / "COMMITTED.json").exists():
            verify_directory(output)
            for folder in output.glob("run-*"):
                row = read_json(folder / "outcome.json")
                if row["result"] is not None:
                    from self_curate.orchestration.io import parse_outer_contract

                    origin = parse_outer_contract("TaskRunRequest", row["origin"])
                    result = parse_outer_contract("TaskRunResult", row["result"])
                    require(
                        result.request_hash == origin.record_hash,
                        "Recovered result identity changed",
                    )
                    cache[row["cache_key"]] = (origin, result)
            return read_json(output / "report.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "request.json", envelope)
    cache = {} if cache is None else cache
    jobs = []
    for task in suite.tasks:
        for seed in suite.seeds:
            request = TaskRunRequest(
                evaluation_id + "-" + digest((task.task_hash, seed))[:16],
                evaluation_id,
                model,
                task,
                seed,
                suite.sampling,
                suite.protocol,
                suite.record_hash,
                suite.backend,
                checkpoint,
            )
            folder = output / ("run-" + digest((task.task_hash, seed))[:16])
            if folder.exists():
                require(
                    resume and read_json(folder / "request.json") == request.to_dict(),
                    "Benchmark task recovery identity changed",
                )
                if (folder / "COMMITTED.json").exists():
                    verify_directory(folder)
                else:
                    require(
                        not (folder / "dispatched.json").exists(),
                        "In-flight benchmark call is unresolved; reconcile, do not retry model failures",
                    )
            else:
                folder.mkdir()
                write_json(folder / "request.json", request.to_dict())
            key = digest(
                {
                    "model": model.to_dict(),
                    "checkpoint": None if checkpoint is None else checkpoint.to_dict(),
                    "task": task.to_dict(),
                    "seed": seed,
                    "sampling": suite.sampling.to_dict(),
                    "protocol": suite.protocol.to_dict(),
                    "backend": suite.backend,
                }
            )
            if (folder / "COMMITTED.json").exists():
                row = read_json(folder / "outcome.json")
                require(
                    row["cache_key"] == key, "Recovered final task identity changed"
                )
                if row["reused"] and key not in cache:
                    from self_curate.orchestration.io import parse_outer_contract

                    cache[key] = (
                        parse_outer_contract("TaskRunRequest", row["origin"]),
                        parse_outer_contract("TaskRunResult", row["result"]),
                    )
            jobs.append((request, folder, key, cache.get(key)))

    def execute(job):
        request, folder, key, cached = job
        usage = ResourceUsage()
        try:
            if (folder / "COMMITTED.json").exists():
                from self_curate.orchestration.io import parse_outer_contract
                from self_curate.contracts import parse_contract

                row = read_json(folder / "outcome.json")
                origin = (
                    None
                    if row["origin"] is None
                    else parse_outer_contract("TaskRunRequest", row["origin"])
                )
                result = (
                    None
                    if row["result"] is None
                    else parse_outer_contract("TaskRunResult", row["result"])
                )
                require(
                    row["request_hash"] == request.record_hash
                    and row["reused"] == (cached is not None),
                    "Recovered benchmark outcome identity changed",
                )
                return (
                    job,
                    origin,
                    result,
                    parse_contract("ResourceUsage", row["usage"]),
                    row["error"],
                )
            if cached is None:
                atomic_json(
                    folder / "dispatched.json", {"request_hash": request.record_hash}
                )
                result = runner.execute(request, folder / "execution")
                if isinstance(getattr(result, "usage", None), ResourceUsage):
                    usage = result.usage
                require(
                    isinstance(result, TaskRunResult)
                    and result.request_hash == request.record_hash,
                    "Benchmark runner substituted task/model/protocol",
                )
                origin = request
            else:
                origin, result = cached
                require(
                    origin.model_ref == request.model_ref
                    and origin.task == request.task
                    and origin.seed == request.seed
                    and origin.sampling == request.sampling
                    and origin.protocol == request.protocol
                    and origin.backend == request.backend
                    and origin.checkpoint == request.checkpoint
                    and result.request_hash == origin.record_hash,
                    "Wrong final-test reuse identity",
                )
                write_json(
                    folder / "reuse.json",
                    {
                        "origin_request": origin.to_dict(),
                        "origin_result_hash": result.record_hash,
                        "new_model_calls": 0,
                        "origin_cost_recharged": False,
                    },
                )
            if suite.backend == "real":
                require(
                    not any(
                        r.uri.strip().lower().startswith(("mock://", "fixture://"))
                        for r in (result.actions, result.observations)
                    ),
                    "Synthetic final benchmark trace",
                )
            write_json(folder / "result.json", result.to_dict())
            atomic_json(
                folder / "outcome.json",
                {
                    "request_hash": request.record_hash,
                    "origin": origin.to_dict(),
                    "result": result.to_dict(),
                    "usage": usage.to_dict(),
                    "error": None,
                    "reused": cached is not None,
                    "cache_key": key,
                },
            )
            commit_directory(folder)
            return job, origin, result, usage, None
        except Exception as exc:
            if isinstance(getattr(exc, "usage", None), ResourceUsage):
                usage = exc.usage
            try:
                write_json(
                    folder / "failure.json",
                    {"error": str(exc), "usage": usage.to_dict()},
                )
            except OSError:
                pass
            atomic_json(
                folder / "outcome.json",
                {
                    "request_hash": request.record_hash,
                    "origin": None,
                    "result": None,
                    "usage": usage.to_dict(),
                    "error": str(exc),
                    "reused": cached is not None,
                    "cache_key": key,
                },
            )
            commit_directory(folder)
            return job, None, None, usage, str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        indexed = {}
        futures = {pool.submit(execute, job): index for index, job in enumerate(jobs)}
        if progress:
            progress.update("benchmark", 0, len(jobs), evaluation_id=evaluation_id)
        for future in as_completed(futures):
            indexed[futures[future]] = future.result()
            if progress:
                progress.update(
                    "benchmark", len(indexed), len(jobs), evaluation_id=evaluation_id
                )
        outcomes = [indexed[i] for i in range(len(jobs))]
    events, runs = [], []
    for job, origin, result, usage, error in outcomes:
        request, _, key, cached = job
        if cached is None:
            receipt_hash = digest(error) if error is not None else result.record_hash
            events.append(
                CostEvent(
                    "final-" + request.record_hash,
                    evaluation_id,
                    "rollout",
                    request.run_id,
                    model.model_hash,
                    (
                        "completed"
                        if result is not None and result.reward is not None
                        else "failed"
                    ),
                    usage,
                    ArtifactRef(
                        "record://benchmark-outcome/" + receipt_hash, receipt_hash
                    ),
                    suite.backend,
                )
            )
        if result is not None:
            cache[key] = (origin, result)
        runs.append(
            {
                "task_id": request.task.task_id,
                "seed": request.seed,
                "request_hash": request.record_hash,
                "result": None if result is None else result.to_dict(),
                "error": error,
                "reused": cached is not None,
            }
        )
    valid = all(
        v["result"] is not None and v["result"]["reward"] is not None for v in runs
    )
    report = {
        "suite_id": suite.suite_id,
        "model_ref": model.to_dict(),
        "suite_hash": suite.record_hash,
        "sealed_training_hash": sealed_training_hash,
        "status": "valid" if valid else "invalid",
        "runs": runs,
        "mean_reward": (
            math.fsum(v["result"]["reward"] / len(runs) for v in runs)
            if valid
            else None
        ),
        "ledger": CostLedger(evaluation_id, tuple(events), suite.backend).to_dict(),
        "worker_slots": workers,
    }
    try:
        write_json(output / "report.json", report)
        commit_directory(output)
    except OSError as exc:
        raise FinalEvaluationFailure(
            str(exc),
            ledger=CostLedger(evaluation_id, tuple(events), suite.backend),
            report=report,
        ) from exc
    return report
