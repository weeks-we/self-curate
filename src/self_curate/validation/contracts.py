"""Task-execution evaluation, distinct from inner-loop proxy valuation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import math

from self_curate.contracts import (
    ArtifactRef,
    CapabilityMetric,
    CapabilityReport,
    CostLedger,
    ModelRef,
    ResourceUsage,
    RolloutRecord,
    TaskRecord,
    task_set_hash,
)
from self_curate.contracts.types import integer, require, sha256, text
from self_curate.contracts.workflow import WorkflowRecord, finite, unique


@dataclass(frozen=True)
class TaskSuite(WorkflowRecord):
    suite_id: str
    tasks: tuple[TaskRecord, ...]
    seeds: tuple[int, ...]
    sampling: ArtifactRef
    protocol: ArtifactRef
    aggregation: str
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        text(self.suite_id, "validation suite")
        require(self.schema_version == "1", "Unsupported task suite")
        require(
            type(self.tasks) is tuple
            and bool(self.tasks)
            and all(isinstance(t, TaskRecord) for t in self.tasks),
            "Expected task records",
        )
        task_set_hash(self.tasks)
        require(
            all(
                t.split == "development" and t.backend == self.backend
                for t in self.tasks
            ),
            "Outer optimization uses development tasks in the same backend domain",
        )
        require(self.backend in ("mock", "real"), "Invalid task execution backend")
        require(
            type(self.seeds) is tuple and bool(self.seeds),
            "Explicit rollout seeds required",
        )
        for seed in self.seeds:
            integer(seed, "rollout seed")
        require(len(set(self.seeds)) == len(self.seeds), "Duplicate rollout seed")
        require(
            isinstance(self.sampling, ArtifactRef)
            and isinstance(self.protocol, ArtifactRef),
            "Sampling and execution protocol artifacts are required",
        )
        require(
            self.aggregation in ("task_mean", "rollout_mean"), "Unknown aggregation"
        )
        if self.backend == "real":
            refs = [self.sampling, self.protocol]
            for task in self.tasks:
                refs.extend((task.payload, task.environment, task.scorer))
            require(
                not any(
                    x.uri.strip().lower().startswith(("fixture://", "mock://"))
                    for x in refs
                ),
                "Known mock task artifacts cannot be real",
            )


@dataclass(frozen=True)
class TaskRunRequest(WorkflowRecord):
    run_id: str
    attempt_id: str
    model_ref: ModelRef
    task: TaskRecord
    seed: int
    sampling: ArtifactRef
    protocol: ArtifactRef
    suite_hash: str
    backend: str
    checkpoint: ArtifactRef | None = None
    schema_version: str = "1"

    def __post_init__(self):
        text(self.run_id, "task run ID")
        text(self.attempt_id, "task run attempt")
        require(
            self.schema_version == "1"
            and isinstance(self.model_ref, ModelRef)
            and isinstance(self.task, TaskRecord),
            "Invalid task execution request",
        )
        require(
            isinstance(self.sampling, ArtifactRef)
            and isinstance(self.protocol, ArtifactRef),
            "Task execution requires a pinned protocol",
        )
        sha256(self.suite_hash, "task suite hash")
        integer(self.seed, "rollout seed")
        require(self.backend == self.task.backend, "Task execution domain mismatch")
        require(
            self.checkpoint is None or isinstance(self.checkpoint, ArtifactRef),
            "Invalid task model checkpoint",
        )


@dataclass(frozen=True)
class TaskRunResult(WorkflowRecord):
    request_hash: str
    actions: ArtifactRef
    observations: ArtifactRef
    tokenization_hash: str
    action_mask_hash: str
    generated_tokens: int
    reward: float | None
    status: str
    stop_reason: str
    usage: ResourceUsage
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported task result")
        for name in ("request_hash", "tokenization_hash", "action_mask_hash"):
            sha256(getattr(self, name), name)
        require(
            isinstance(self.actions, ArtifactRef)
            and isinstance(self.observations, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Invalid task result artifacts/costs",
        )
        integer(self.generated_tokens, "generated task tokens")
        require(
            self.status
            in (
                "completed",
                "task_failed",
                "budget_exhausted",
                "environment_failed",
                "cancelled",
            ),
            "Unknown task status",
        )
        text(self.stop_reason, "task stop reason")
        if self.reward is not None:
            finite(self.reward, "observed task reward")
        if self.status in ("completed", "task_failed"):
            require(self.reward is not None, "Scored task result needs a reward")
        if self.status in ("environment_failed", "cancelled"):
            require(
                self.reward is None, "Infrastructure failure is not zero capability"
            )


@dataclass(frozen=True)
class TaskEvaluation(WorkflowRecord):
    suite: TaskSuite
    report: CapabilityReport
    rollouts: tuple[RolloutRecord, ...]
    ledger: CostLedger
    checkpoint: ArtifactRef | None = None
    execution_requests: tuple[TaskRunRequest, ...] = ()
    execution_results: tuple[TaskRunResult, ...] = ()

    def __post_init__(self):
        require(
            isinstance(self.suite, TaskSuite)
            and isinstance(self.report, CapabilityReport)
            and isinstance(self.ledger, CostLedger),
            "Invalid task evaluation",
        )
        require(
            self.checkpoint is None or isinstance(self.checkpoint, ArtifactRef),
            "Invalid evaluated checkpoint",
        )
        require(
            type(self.rollouts) is tuple
            and all(isinstance(r, RolloutRecord) for r in self.rollouts),
            "Invalid task rollouts",
        )
        require(
            self.report.task_set_hash == task_set_hash(self.suite.tasks)
            and self.report.task_ids == tuple(t.task_id for t in self.suite.tasks)
            and self.report.rollout_ids == tuple(r.rollout_id for r in self.rollouts)
            and self.report.backend == self.ledger.backend == self.suite.backend,
            "Capability evaluation differs from exact task suite",
        )
        require(
            len(self.rollouts) == len(self.suite.tasks) * len(self.suite.seeds),
            "Capability evaluation has incomplete task/seed coverage",
        )
        unique(tuple(r.rollout_id for r in self.rollouts), "executed rollout IDs")
        by_task = {t.task_id: t for t in self.suite.tasks}
        require(
            {(r.task_id, r.seed) for r in self.rollouts}
            == {
                (t.task_id, seed) for t in self.suite.tasks for seed in self.suite.seeds
            },
            "Task/seed outcomes were duplicated or substituted",
        )
        for r in self.rollouts:
            require(
                r.model_ref == self.report.model_ref
                and r.task_hash == by_task[r.task_id].task_hash
                and r.sampling == self.suite.sampling
                and r.scorer == by_task[r.task_id].scorer
                and r.backend == self.suite.backend
                and self.ledger.reference(r.cost.event_ids) == r.cost,
                "Rollout identity or cost differs from capability evaluation",
            )
        require(
            self.ledger.reference(self.report.cost.event_ids) == self.report.cost,
            "Capability aggregation cost reference mismatch",
        )
        require(
            self.report.metrics == capability_metrics(self.suite, self.rollouts)
            and self.report.status
            == (
                "valid"
                if all(r.reward is not None for r in self.rollouts)
                else "invalid"
            ),
            "Capability metrics differ from observed task rewards",
        )
        expected_rollout_events = set()
        for r in self.rollouts:
            require(
                len(r.cost.event_ids) == 1, "Each rollout needs one physical cost event"
            )
            event = next(
                e for e in self.ledger.events if e.event_id == r.cost.event_ids[0]
            )
            require(
                event.operation == "rollout"
                and event.subject_id == r.rollout_id
                and event.model_hash == r.model_ref.model_hash,
                "Rollout physical cost belongs to another model or execution",
            )
            expected_rollout_events.update(r.cost.event_ids)
        require(
            expected_rollout_events.isdisjoint(self.report.cost.event_ids)
            and len(self.report.cost.event_ids) == 1
            and expected_rollout_events | set(self.report.cost.event_ids)
            == {e.event_id for e in self.ledger.events},
            "Duplicate or extraneous capability costs",
        )
        aggregate = next(
            e for e in self.ledger.events if e.event_id == self.report.cost.event_ids[0]
        )
        require(
            aggregate.operation == "capability"
            and aggregate.subject_id == self.report.report_id
            and aggregate.model_hash == self.report.model_ref.model_hash
            and aggregate.status == "completed",
            "Capability aggregation cost belongs to another report",
        )
        require(
            type(self.execution_requests) is tuple
            and type(self.execution_results) is tuple
            and all(isinstance(r, TaskRunRequest) for r in self.execution_requests)
            and all(isinstance(r, TaskRunResult) for r in self.execution_results),
            "Invalid task execution evidence",
        )
        require(
            bool(self.execution_requests) == bool(self.execution_results),
            "Task execution requests/results must be supplied together",
        )
        if self.execution_requests:
            require(
                len(self.execution_requests)
                == len(self.execution_results)
                == len(self.rollouts),
                "Execution evidence must cover every rollout",
            )
            event_by_id = {e.event_id: e for e in self.ledger.events}
            for request, result, rollout in zip(
                self.execution_requests, self.execution_results, self.rollouts
            ):
                event = event_by_id[rollout.cost.event_ids[0]]
                require(
                    request.run_id == rollout.rollout_id
                    and request.model_ref == self.report.model_ref
                    and request.task == by_task[rollout.task_id]
                    and request.seed == rollout.seed
                    and request.sampling == self.suite.sampling
                    and request.protocol == self.suite.protocol
                    and request.suite_hash == self.suite.record_hash
                    and request.backend == self.suite.backend
                    and request.checkpoint == self.checkpoint
                    and request.attempt_id == event.attempt_id == aggregate.attempt_id
                    and result.request_hash == request.record_hash,
                    "Task execution model/checkpoint/protocol/request binding mismatch",
                )
                require(
                    all(
                        getattr(result, name) == getattr(rollout, name)
                        for name in (
                            "actions",
                            "observations",
                            "tokenization_hash",
                            "action_mask_hash",
                            "generated_tokens",
                            "reward",
                            "status",
                            "stop_reason",
                        )
                    )
                    and result.usage == event.usage
                    and event.receipt.sha256 == result.record_hash
                    and event.status
                    == ("completed" if result.reward is not None else "failed"),
                    "Execution output differs from the scored rollout or its physical costs",
                )
        if self.suite.backend == "real":
            refs = [a for r in self.rollouts for a in (r.actions, r.observations)]
            if self.checkpoint is not None:
                refs.append(self.checkpoint)
            require(
                self.report.model_ref.tokenizer.backend != "fixture"
                and not any(
                    a.uri.strip().lower().startswith(("fixture://", "mock://"))
                    for a in refs
                ),
                "Known mock capability evidence cannot be declared real",
            )

    @property
    def protocol_verified(self):
        """Legacy reports without per-execution evidence are readable, not decision-ready."""
        return bool(self.execution_requests)


def capability_metrics(suite, rollouts):
    """Conservative complete-suite aggregation; missing observations remain missing."""
    valid = all(r.reward is not None for r in rollouts)
    metrics = []
    for skill in sorted({s for t in suite.tasks for s in t.skills}):
        tasks = tuple(t.task_id for t in suite.tasks if skill in t.skills)
        runs = tuple(r for r in rollouts if r.task_id in tasks)
        mean = None
        if valid:
            if suite.aggregation == "task_mean":
                mean = math.fsum(r.reward / len(suite.seeds) / len(tasks) for r in runs)
            else:
                mean = math.fsum(r.reward / len(runs) for r in runs)
        metrics.append(
            CapabilityMetric(
                skill,
                tasks,
                tuple(r.rollout_id for r in runs),
                len(runs),
                suite.aggregation,
                mean,
                None,
                None,
            )
        )
    return tuple(metrics)


class TaskRunner(Protocol):
    backend: str

    def execute(self, request: TaskRunRequest, output: Path) -> TaskRunResult:
        """Load the exact model, freely execute the task, score it, save full traces.

        The adapter must enforce request.task.budget and protocol. Dataset/environment
        specifics stay here; proxy scores or teacher-forced loss are not task rewards.
        """
        ...
