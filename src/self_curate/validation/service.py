"""Run a fixed development suite and aggregate explicit, task-level observations."""

from pathlib import Path
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    CapabilityReport,
    ComputeCost,
    CostEvent,
    CostLedger,
    ModelRef,
    ResourceUsage,
    RolloutRecord,
    digest,
    task_set_hash,
)
from self_curate.contracts.types import require, text
from self_curate.data.pool import write_json
from self_curate.training.artifacts import verify_local_checkpoint
from .contracts import (
    TaskEvaluation,
    TaskRunRequest,
    TaskRunResult,
    TaskSuite,
    capability_metrics,
)


class TaskValidationFailure(RuntimeError):
    def __init__(self, message, *, ledger, rollouts):
        super().__init__(message)
        self.ledger = ledger
        self.rollouts = tuple(rollouts)


class TaskValidator:
    def __init__(self, runner):
        self.runner = runner

    def evaluate(
        self,
        model,
        suite: TaskSuite,
        *,
        attempt_id,
        evaluation_id,
        output,
        checkpoint=None,
    ):
        require(isinstance(suite, TaskSuite), "Expected fixed TaskSuite")
        require(self.runner.backend == suite.backend, "Task runner/domain mismatch")
        text(evaluation_id, "capability evaluation ID")
        text(attempt_id, "capability attempt ID")
        require(isinstance(model, ModelRef), "Expected task model identity")
        if checkpoint is not None:
            verify_local_checkpoint(checkpoint)
        if suite.backend == "real":
            require(
                model.tokenizer.backend != "fixture",
                "Fixture model cannot run real validation",
            )
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(
            output / "request.json",
            {
                "model": model.to_dict(),
                "suite": suite.to_dict(),
                "attempt_id": attempt_id,
                "evaluation_id": evaluation_id,
                "checkpoint": None if checkpoint is None else checkpoint.to_dict(),
            },
        )
        events, rollouts = [], []
        execution_requests, execution_results = [], []
        for task in suite.tasks:
            for seed in suite.seeds:
                request = TaskRunRequest(
                    evaluation_id + "-" + digest((task.task_hash, seed))[:24],
                    attempt_id,
                    model,
                    task,
                    seed,
                    suite.sampling,
                    suite.protocol,
                    suite.record_hash,
                    suite.backend,
                    checkpoint,
                )
                folder = output / ("run-" + request.record_hash)
                start = perf_counter()
                usage = ResourceUsage()
                dispatched = False
                try:
                    folder.mkdir()
                    write_json(folder / "request.json", request.to_dict())
                    dispatched = True
                    result = self.runner.execute(request, folder / "execution")
                    if isinstance(getattr(result, "usage", None), ResourceUsage):
                        usage = result.usage
                    require(
                        isinstance(result, TaskRunResult), "Invalid task runner result"
                    )
                    require(
                        result.request_hash == request.record_hash,
                        "Task result belongs to another model/task/seed/protocol",
                    )
                    if suite.backend == "real":
                        require(
                            not any(
                                r.uri.strip()
                                .lower()
                                .startswith(("fixture://", "mock://"))
                                for r in (result.actions, result.observations)
                            ),
                            "Known mock task traces cannot enter real capability",
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
                            "record://task-result/" + result.record_hash,
                            result.record_hash,
                        ),
                        suite.backend,
                    )
                    events.append(event)
                    ledger = CostLedger(evaluation_id, tuple(events), suite.backend)
                    rollout = RolloutRecord(
                        request.run_id,
                        model,
                        task.task_id,
                        task.task_hash,
                        seed,
                        suite.sampling,
                        result.actions,
                        result.observations,
                        result.tokenization_hash,
                        result.action_mask_hash,
                        result.generated_tokens,
                        result.reward,
                        result.status,
                        result.stop_reason,
                        task.scorer,
                        ledger.reference((event.event_id,)),
                        suite.backend,
                    )
                    rollouts.append(rollout)
                    execution_requests.append(request)
                    execution_results.append(result)
                    write_json(folder / "result.json", result.to_dict())
                    write_json(folder / "rollout.json", rollout.to_dict())
                    write_json(folder / "cost.json", event.to_dict())
                except Exception as exc:
                    if isinstance(getattr(exc, "usage", None), ResourceUsage):
                        usage = exc.usage
                    if usage.compute.wall_seconds is None:
                        from dataclasses import replace

                        usage = replace(
                            usage,
                            compute=replace(
                                usage.compute, wall_seconds=perf_counter() - start
                            ),
                        )
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    # Replace the same physical call event if publication failed after execution.
                    if dispatched:
                        events = [
                            e
                            for e in events
                            if e.event_id != "rollout-" + request.record_hash
                        ]
                        events.append(
                            CostEvent(
                                "rollout-" + request.record_hash,
                                attempt_id,
                                "rollout",
                                request.run_id,
                                model.model_hash,
                                "failed",
                                usage,
                                ArtifactRef(
                                    "record://task-failure/" + digest(error),
                                    digest(error),
                                ),
                                suite.backend,
                            )
                        )
                    ledger = CostLedger(evaluation_id, tuple(events), suite.backend)
                    try:
                        write_json(
                            output / "failure.json",
                            {
                                "error": error,
                                "ledger": ledger.to_dict(),
                                "rollouts": [r.to_dict() for r in rollouts],
                            },
                        )
                    except OSError:
                        pass  # Preserve costs on the raised failure even if the disk is unavailable.
                    raise TaskValidationFailure(
                        str(exc), ledger=ledger, rollouts=rollouts
                    ) from exc
        try:
            return self._aggregate(
                model,
                suite,
                evaluation_id,
                attempt_id,
                output,
                events,
                rollouts,
                checkpoint,
                execution_requests,
                execution_results,
            )
        except Exception as exc:
            # Task calls have already occurred, even if aggregation/publication fails.
            ledger = CostLedger(evaluation_id, tuple(events), suite.backend)
            raise TaskValidationFailure(
                str(exc), ledger=ledger, rollouts=rollouts
            ) from exc

    def _aggregate(
        self,
        model,
        suite,
        evaluation_id,
        attempt_id,
        output,
        events,
        rollouts,
        checkpoint,
        execution_requests,
        execution_results,
    ):
        start = perf_counter()
        valid = all(r.reward is not None for r in rollouts)
        metrics = capability_metrics(suite, rollouts)
        report_id = "capability-" + digest(
            (evaluation_id, model.model_hash, suite.record_hash)
        )
        event = CostEvent(
            "aggregate-" + report_id,
            attempt_id,
            "capability",
            report_id,
            model.model_hash,
            "completed",
            ResourceUsage(ComputeCost(perf_counter() - start, 0, 0)),
            ArtifactRef(
                "record://capability-metrics/" + digest([m.to_dict() for m in metrics]),
                digest([m.to_dict() for m in metrics]),
            ),
            suite.backend,
        )
        events.append(event)
        ledger = CostLedger(evaluation_id, tuple(events), suite.backend)
        report = CapabilityReport(
            report_id,
            model,
            tuple(t.task_id for t in suite.tasks),
            task_set_hash(suite.tasks),
            tuple(r.rollout_id for r in rollouts),
            "development",
            tuple(metrics),
            "valid" if valid else "invalid",
            (
                None
                if valid
                else "At least one task lacks a scored outcome; no mean was fabricated"
            ),
            ledger.reference((event.event_id,)),
            suite.backend,
        )
        evaluation = TaskEvaluation(
            suite,
            report,
            tuple(rollouts),
            ledger,
            checkpoint,
            tuple(execution_requests),
            tuple(execution_results),
        )
        write_json(output / "evaluation.json", evaluation.to_dict())
        return evaluation
