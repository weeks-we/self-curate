"""Parallel development calls with ordered aggregation and per-call recovery evidence."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ResourceUsage,
    RolloutRecord,
    digest,
)
from self_curate.contracts.types import require
from self_curate.training.recovery import (
    atomic_json,
    read_json,
    commit_directory,
    verify_directory,
)
from self_curate.training.artifacts import verify_local_checkpoint
from self_curate.orchestration.io import parse_outer_contract, load_task_evaluation
from .contracts import TaskRunRequest, TaskRunResult
from .service import TaskValidator, TaskValidationFailure


class ParallelTaskValidator(TaskValidator):
    def __init__(self, runner, workers=16):
        super().__init__(runner)
        require(
            type(workers) is int and 1 <= workers <= 16,
            "Invalid development task concurrency",
        )
        self.workers = workers

    def evaluate(
        self,
        model,
        suite,
        *,
        attempt_id,
        evaluation_id,
        output,
        checkpoint=None,
        resume=False,
        progress=None,
    ):
        require(
            suite.backend == self.runner.backend, "Development execution domain changed"
        )
        if checkpoint is not None:
            verify_local_checkpoint(checkpoint)
        output = Path(output).resolve()
        envelope = {
            "model": model.to_dict(),
            "suite": suite.to_dict(),
            "attempt_id": attempt_id,
            "evaluation_id": evaluation_id,
            "checkpoint": None if checkpoint is None else checkpoint.to_dict(),
        }
        if output.exists():
            require(
                resume and read_json(output / "request.json") == envelope,
                "Existing parallel evaluation requires matching explicit resume",
            )
            if (output / "COMMITTED.json").exists():
                verify_directory(output)
                return load_task_evaluation(
                    output,
                    expected_model=model,
                    expected_suite=suite,
                    expected_checkpoint=checkpoint,
                )
        else:
            output.mkdir(parents=True)
            atomic_json(output / "request.json", envelope)
        requests = [
            TaskRunRequest(
                evaluation_id + "-" + digest((t.task_hash, seed))[:24],
                attempt_id,
                model,
                t,
                seed,
                suite.sampling,
                suite.protocol,
                suite.record_hash,
                suite.backend,
                checkpoint,
            )
            for t in suite.tasks
            for seed in suite.seeds
        ]

        def execute(request):
            folder = output / ("run-" + request.record_hash)
            usage = ResourceUsage()
            dispatched = False
            start = perf_counter()
            try:
                if folder.exists():
                    dispatched = (folder / "dispatched.json").exists() and not (
                        folder / "COMMITTED.json"
                    ).exists()
                    require(resume, "Existing task output requires resume")
                    verify_directory(folder)
                    require(
                        read_json(folder / "request.json") == request.to_dict(),
                        "Resumed task request changed",
                    )
                    result = parse_outer_contract(
                        "TaskRunResult", read_json(folder / "result.json")
                    )
                else:
                    folder.mkdir()
                    atomic_json(folder / "request.json", request.to_dict())
                    atomic_json(
                        folder / "dispatched.json",
                        {"request_hash": request.record_hash},
                    )
                    dispatched = True
                    result = self.runner.execute(request, folder / "execution")
                if isinstance(getattr(result, "usage", None), ResourceUsage):
                    usage = result.usage
                require(
                    isinstance(result, TaskRunResult)
                    and result.request_hash == request.record_hash,
                    "Task result identity changed",
                )
                require(
                    suite.backend != "real"
                    or not any(
                        r.uri.startswith(("mock://", "fixture://"))
                        for r in (result.actions, result.observations)
                    ),
                    "Synthetic traces cannot enter real development evaluation",
                )
                event = CostEvent(
                    "rollout-" + request.record_hash,
                    attempt_id,
                    "rollout",
                    request.run_id,
                    model.model_hash,
                    "completed" if result.reward is not None else "failed",
                    usage,
                    ArtifactRef(
                        "record://task-result/" + result.record_hash, result.record_hash
                    ),
                    suite.backend,
                )
                ledger = CostLedger(evaluation_id, (event,), suite.backend)
                rollout = RolloutRecord(
                    request.run_id,
                    model,
                    request.task.task_id,
                    request.task.task_hash,
                    request.seed,
                    suite.sampling,
                    result.actions,
                    result.observations,
                    result.tokenization_hash,
                    result.action_mask_hash,
                    result.generated_tokens,
                    result.reward,
                    result.status,
                    result.stop_reason,
                    request.task.scorer,
                    ledger.reference((event.event_id,)),
                    suite.backend,
                )
                if dispatched:
                    atomic_json(folder / "result.json", result.to_dict())
                    atomic_json(folder / "rollout.json", rollout.to_dict())
                    atomic_json(folder / "cost.json", event.to_dict())
                    commit_directory(folder)
                return request, result, event, rollout, None
            except Exception as exc:
                usage = getattr(exc, "usage", usage)
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "wall_seconds": perf_counter() - start,
                }
                event = None
                if dispatched:
                    event = CostEvent(
                        "rollout-" + request.record_hash,
                        attempt_id,
                        "rollout",
                        request.run_id,
                        model.model_hash,
                        "failed",
                        usage,
                        ArtifactRef(
                            "record://task-failure/" + digest(error), digest(error)
                        ),
                        suite.backend,
                    )
                return request, None, event, None, error

        results = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(execute, r): r for r in requests}
            for future in as_completed(futures):
                row = future.result()
                results[row[0].record_hash] = row
                if progress:
                    progress.update(
                        "development",
                        len(results),
                        len(requests),
                        evaluation_id=evaluation_id,
                    )
        ordered = [results[r.record_hash] for r in requests]
        events = [r[2] for r in ordered if r[2] is not None]
        rollouts = [r[3] for r in ordered if r[3] is not None]
        errors = [r[4] for r in ordered if r[4] is not None]
        if errors:
            ledger = CostLedger(evaluation_id, tuple(events), suite.backend)
            atomic_json(
                output / "failure.json",
                {
                    "errors": errors,
                    "ledger": ledger.to_dict(),
                    "rollouts": [r.to_dict() for r in rollouts],
                },
            )
            raise TaskValidationFailure(
                "Parallel development calls failed; all completed/failed call costs retained",
                ledger=ledger,
                rollouts=rollouts,
            )
        try:
            evaluation = self._aggregate(
                model,
                suite,
                evaluation_id,
                attempt_id,
                output,
                events,
                rollouts,
                checkpoint,
                requests,
                [r[1] for r in ordered],
            )
            commit_directory(output)
            return evaluation
        except Exception as exc:
            raise TaskValidationFailure(
                str(exc),
                ledger=CostLedger(evaluation_id, tuple(events), suite.backend),
                rollouts=rollouts,
            ) from exc
