"""Cross-record identity checks, not numerical gradient or acceptance algorithms."""

from dataclasses import fields
import math

from .errors import ContractError, ErrorCode
from .selection import verify_batch
from .program import selection_identity
from .signals import ModelRef
from .types import ArtifactRef, Record, digest, require, sha256
from .workflow import (
    WorkflowBundle,
    WorkflowRecord,
    TaskRecord,
    RolloutRecord,
    ObjectiveSpec,
)


def task_set_hash(tasks):
    require(type(tasks) in (tuple, list), "Expected a task sequence")
    items = tuple(tasks)
    require(all(isinstance(t, TaskRecord) for t in items), "Invalid task set item")
    require(
        len({t.task_id for t in items}) == len(items), "Duplicate task in reference set"
    )
    return digest(sorted((t.task_id, t.task_hash) for t in items))


def rollout_set_hash(rollouts):
    require(type(rollouts) in (tuple, list), "Expected a rollout sequence")
    items = tuple(rollouts)
    require(
        all(isinstance(r, RolloutRecord) for r in items), "Invalid rollout set item"
    )
    require(
        len({r.rollout_id for r in items}) == len(items),
        "Duplicate rollout in reference set",
    )
    return digest(sorted((r.rollout_id, r.record_hash) for r in items))


def _index(items, field):
    result = {getattr(item, field): item for item in items}
    require(len(result) == len(items), f"Duplicate {field}")
    return result


def _same(condition, message):
    require(condition, message, ErrorCode.ARTIFACT_MISMATCH)


def _lookup(index, keys, name):
    _same(set(keys).issubset(index), f"Unknown {name} reference")
    return tuple(index[k] for k in keys)


def _sum(values):
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD, "Aggregate is not finite"
        ) from exc
    require(math.isfinite(total), "Aggregate is not finite")
    return total


def _equal_number(a, b):
    return (
        a is not None
        and b is not None
        and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    )


def _walk(value):
    yield value
    if isinstance(value, Record):
        for f in fields(value):
            if f.init:
                yield from _walk(getattr(value, f.name))
    elif isinstance(value, tuple):
        for child in value:
            yield from _walk(child)


def _coordinates(spec):
    return (
        spec.parameter_coordinates_hash,
        spec.parameter_count,
        spec.projection,
        spec.compute_dtype,
        spec.transform_id,
        spec.transform_hash,
        spec.optimizer_state_hash,
    )


def validate_workflow(
    bundle,
    view=None,
    *,
    expected_model=None,
    expected_objective=None,
    expected_selection_context_hash=None,
    expected_batch_id=None,
):
    """Validate the immutable record graph; optionally verify M1 training inputs.

    Returns only a verification receipt. It does not load referenced vectors,
    score tasks, evaluate a proxy, apply an acceptance threshold or commit models.
    """
    require(isinstance(bundle, WorkflowBundle), "Expected WorkflowBundle")
    b = bundle
    if expected_batch_id is not None:
        sha256(expected_batch_id, "expected batch ID")
        _same(
            b.batch.batch_id == expected_batch_id,
            "Workflow differs from the exact selected batch",
        )
    if expected_model is not None:
        require(isinstance(expected_model, ModelRef), "Expected trusted ModelRef")
        require(
            b.model_ref == expected_model,
            "Workflow differs from the system model",
            ErrorCode.STALE_CACHE,
        )
    if expected_objective is not None:
        require(
            isinstance(expected_objective, ObjectiveSpec),
            "Expected trusted ObjectiveSpec",
        )
        _same(
            b.objective == expected_objective,
            "Workflow differs from the fixed system objective",
        )
    if expected_selection_context_hash is not None:
        sha256(expected_selection_context_hash, "expected selection context hash")
        _same(
            b.selection_context_hash == expected_selection_context_hash,
            "Workflow differs from the trusted selection context",
        )
    for record in _walk(b):
        if isinstance(record, WorkflowRecord) and hasattr(record, "backend"):
            _same(record.backend == b.backend, "Mixed workflow execution backends")
        if b.backend == "real" and isinstance(record, ArtifactRef):
            require(
                not record.uri.strip().lower().startswith(("fixture://", "mock://")),
                "Mock artifact in real workflow",
            )
    if b.backend == "real":
        require(
            b.model_ref.tokenizer.backend != "fixture"
            and b.batch.synthetic is not True,
            "Fixture training inputs cannot be declared real",
        )
    program_hash, execution_hash = selection_identity(
        b.program, b.batch.policy_hash, b.selection_context_hash, b.batch.snapshot_hash
    )
    _same(
        b.batch.program_hash == program_hash
        and b.batch.execution_hash == execution_hash,
        "Program/context/batch receipt mismatch",
    )
    _same(
        b.batch.model_hash == b.model_ref.model_hash, "Batch belongs to another model"
    )
    if view is not None:
        verify_batch(b.batch, view, model_ref=b.model_ref)
        if b.backend == "real":
            require(
                not any(
                    view.record(data_id).known_mock_artifacts
                    for data_id in b.batch.data_ids
                ),
                "Known mock trajectory provenance in real training inputs",
                ErrorCode.ARTIFACT_MISMATCH,
            )
    tasks = _index(b.tasks, "task_id")
    rollouts = _index(b.rollouts, "rollout_id")
    targets = _index(b.target_gradients, "target_version")
    samples = _index(b.sample_gradients, "cache_key")
    _index(b.target_gradients, "target_id")
    evaluations = _index(b.evaluations, "evaluation_id")
    capabilities = _index(b.capabilities, "report_id")
    ledger = b.ledger
    events = _index(ledger.events, "event_id")
    require(
        all(t.split != "test" for t in b.tasks),
        "Final test tasks cannot enter an optimization bundle",
    )
    candidate = (
        None
        if b.round_record is None or b.round_record.training is None
        else b.round_record.training.candidate_model
    )
    models = {b.model_ref.model_hash}
    if candidate is not None:
        models.add(candidate.model_hash)
    for event in ledger.events:
        _same(
            event.model_hash is None or event.model_hash in models,
            "Cost event belongs to an unrelated model",
        )

    def cost(ref, *, model_hash=None, operations=None, subject=None):
        expected = ledger.reference(ref.event_ids)
        _same(ref.events_hash == expected.events_hash, "Cost reference content changed")
        for event in _lookup(events, ref.event_ids, "cost event"):
            if model_hash is not None:
                _same(
                    event.model_hash is None or event.model_hash == model_hash,
                    "Cost model mismatch",
                )
            if operations is not None:
                _same(
                    event.operation in operations, "Wrong operation in cost reference"
                )
            if subject is not None:
                _same(event.subject_id == subject, "Cost subject mismatch")

    for rollout in b.rollouts:
        (task,) = _lookup(tasks, (rollout.task_id,), "task")
        _same(
            rollout.task_hash == task.task_hash and rollout.scorer == task.scorer,
            "Rollout task/scorer version mismatch",
        )
        _same(
            rollout.model_ref.model_hash in models
            and rollout.model_ref.tokenizer == b.model_ref.tokenizer,
            "Rollout model/tokenizer mismatch",
        )
        require(
            rollout.generated_tokens <= task.budget.max_output_tokens,
            "Rollout exceeds task token declaration",
        )
        cost(
            rollout.cost,
            model_hash=rollout.model_ref.model_hash,
            operations=("rollout",),
            subject=rollout.rollout_id,
        )

    members = {m.data_id: m for m in b.batch.members}
    for sample in b.sample_gradients:
        _same(sample.binding.data_id in members, "Gradient is not a selected sample")
        member = members[sample.binding.data_id]
        _same(
            sample.model_ref == b.model_ref and sample.spec.scope == "assistant",
            "Training batch gradient model/scope mismatch",
        )
        _same(
            sample.binding.content_hash == member.content_hash
            and sample.binding.tokenization_hash == member.tokenization_hash
            and sample.binding.token_count == member.supervised_tokens,
            "Gradient data/token binding mismatch",
        )
        if view is not None:
            from self_curate.data.enrichment import data_binding

            _same(
                sample.binding
                == data_binding(
                    view.record(member.data_id), sample.spec.scope, sample.spec.skill
                ),
                "Sample mask selection changed",
            )

    for target in b.target_gradients:
        _same(target.model_ref == b.model_ref, "Target belongs to another checkpoint")
        runs = _lookup(rollouts, target.rollout_ids, "target rollout")
        selected_tasks = _lookup(tasks, target.task_ids, "target task")
        _same(
            all(
                t.split == "development" and target.spec.skill in t.skills
                for t in selected_tasks
            ),
            "Target requires its development skill group",
        )
        _same(
            all(
                r.task_id in target.task_ids and r.model_ref == target.model_ref
                for r in runs
            ),
            "Target rollout model/task mismatch",
        )
        _same(
            target.task_set_hash == task_set_hash(selected_tasks)
            and target.rollouts_hash == rollout_set_hash(runs),
            "Target task/rollout binding mismatch",
        )
        if target.validity in ("valid", "zero"):
            _same(
                all(r.reward is not None for r in runs)
                and {r.task_id for r in runs} == set(target.task_ids),
                "Target rollout lacks full reward/task evidence",
            )
        cost(
            target.cost,
            model_hash=target.model_ref.model_hash,
            operations=("target_gradient",),
            subject=target.target_id,
        )

    weights = {w.skill: w.weight for w in b.objective.weights}
    conditions = {c.name: c for c in b.objective.constraints}
    for report in b.evaluations:
        _same(
            report.model_ref == b.model_ref
            and report.batch_id == b.batch.batch_id
            and report.objective_hash == b.objective.objective_hash,
            "Valuation model/batch/objective mismatch",
        )
        _same(
            set(report.coverage.required_ids) == set(b.batch.data_ids),
            "Coverage differs from exact batch",
        )
        chosen_samples = _lookup(samples, report.sample_cache_keys, "sample cache key")
        _same(
            len(chosen_samples) == len(report.coverage.computed_ids)
            and {s.binding.data_id for s in chosen_samples}
            == set(report.coverage.computed_ids),
            "Computed sample coverage/cache mismatch",
        )
        require(
            not chosen_samples
            or all(s.spec == chosen_samples[0].spec for s in chosen_samples),
            "Sample loss/reduction/normalization specifications differ",
            ErrorCode.STALE_CACHE,
        )
        reference_skills = {r.skill for r in report.targets}
        value_skills = {v.skill for v in report.skill_values}
        _same(
            reference_skills.issubset(weights) and value_skills.issubset(weights),
            "Unknown skill in valuation",
        )
        if report.status == "valid":
            _same(
                reference_skills == set(weights) and value_skills == set(weights),
                "Valuation skills differ from fixed objective",
            )
        chosen_targets = []
        for ref in report.targets:
            (feature,) = _lookup(targets, (ref.target_version,), "target version")
            _same(feature.spec.skill == ref.skill, "Target reference skill mismatch")
            if report.status == "valid":
                require(
                    feature.validity == "valid",
                    "Invalid/zero target cannot produce a rankable valuation",
                    (
                        ErrorCode.ZERO_TARGET_GRADIENT
                        if feature.validity == "zero"
                        else ErrorCode.INVALID_REWARD
                    ),
                )
            chosen_targets.append(feature)
        specs = [s.spec for s in chosen_samples] + [
            t.spec.gradient for t in chosen_targets
        ]
        require(
            not specs or all(_coordinates(s) == _coordinates(specs[0]) for s in specs),
            "Sample/target gradient coordinate mismatch",
            ErrorCode.STALE_CACHE,
        )
        reported_conditions = {r.name for r in report.constraints}
        _same(reported_conditions.issubset(conditions), "Unknown constraint result")
        if report.status == "valid":
            _same(
                reported_conditions == set(conditions),
                "Constraint results differ from fixed objective",
            )
        values = {v.skill: v.value for v in report.skill_values}
        if report.status == "valid":
            expected = _sum(weights[k] * values[k] for k in weights)
            _same(
                _equal_number(expected, report.total_value),
                "Valuation total differs from fixed weighted sum",
            )
        for result in report.constraints:
            condition = conditions[result.name]
            value = values.get(condition.skill)
            if result.value is None:
                require(
                    report.status != "valid",
                    "Valid constraint result cannot be missing",
                )
            else:
                passed = (
                    (condition.minimum is None or value >= condition.minimum)
                    and (condition.maximum is None or value <= condition.maximum)
                    if value is not None
                    else None
                )
                _same(
                    _equal_number(result.value, value) and result.passed == passed,
                    "Constraint outcome disagrees with fixed bounds",
                )
        cost(
            report.cost,
            model_hash=report.model_ref.model_hash,
            operations=(
                "selection",
                "scoring",
                "sample_gradient",
                "target_gradient",
                "evaluation",
            ),
        )

    for report in b.capabilities:
        _same(
            report.model_ref.model_hash in models
            and report.model_ref.tokenizer == b.model_ref.tokenizer,
            "Capability belongs to an unrelated model",
        )
        chosen_tasks = _lookup(tasks, report.task_ids, "capability task")
        runs = _lookup(rollouts, report.rollout_ids, "capability rollout")
        runs_by_task = {}
        for run in runs:
            runs_by_task.setdefault(run.task_id, []).append(run)
        _same(
            report.task_set_hash == task_set_hash(chosen_tasks)
            and all(t.split == report.split for t in chosen_tasks),
            "Capability task set/split mismatch",
        )
        _same(
            all(
                r.task_id in report.task_ids and r.model_ref == report.model_ref
                for r in runs
            ),
            "Capability rollout model/task mismatch",
        )
        expected_skills = {s for t in chosen_tasks for s in t.skills}
        _same(
            {m.skill for m in report.metrics} == expected_skills,
            "Capability skill coverage mismatch",
        )
        for metric in report.metrics:
            skill_tasks = {t.task_id for t in chosen_tasks if metric.skill in t.skills}
            skill_runs = tuple(r for r in runs if r.task_id in skill_tasks)
            _same(
                set(metric.task_ids) == skill_tasks
                and set(metric.rollout_ids) == {r.rollout_id for r in skill_runs},
                "Capability metric members/count mismatch",
            )
            if report.status == "valid":
                require(
                    all(r.reward is not None for r in skill_runs)
                    and {r.task_id for r in skill_runs} == skill_tasks,
                    "Valid capability cannot hide missing/infrastructure outcomes",
                )
                if metric.aggregation == "rollout_mean":
                    expected = _sum(r.reward for r in skill_runs) / len(skill_runs)
                else:
                    expected = _sum(
                        _sum(r.reward for r in runs_by_task[tid])
                        / len(runs_by_task[tid])
                        for tid in sorted(skill_tasks)
                    ) / len(skill_tasks)
                _same(
                    _equal_number(metric.mean_reward, expected),
                    "Capability aggregation differs from declared task/rollout mean",
                )
        cost(
            report.cost,
            model_hash=report.model_ref.model_hash,
            operations=("rollout", "capability"),
        )

    r = b.round_record
    if r is not None:
        _same(
            r.starting_model == b.model_ref
            and r.objective_hash == b.objective.objective_hash
            and r.decision_policy_hash == b.objective.acceptance_policy.sha256,
            "Round starting model/objective/policy mismatch",
        )
        _same(
            all(e.attempt_id == r.attempt_id for e in ledger.events),
            "Foreign attempt in round cost ledger",
        )
        _same(
            set(r.cost.event_ids) == set(events),
            "Round must retain all attempt cost events",
        )
        cost(r.cost)
        if r.selected_batch_id is not None:
            _same(
                r.selected_batch_id == b.batch.batch_id, "Round selected another batch"
            )
        if r.evaluation_id is not None:
            (evaluation,) = _lookup(evaluations, (r.evaluation_id,), "round evaluation")
            require(
                evaluation.feasible, "Selected batch must have a feasible evaluation"
            )
        if r.training is not None:
            training = r.training
            _same(
                training.backend == b.backend
                and training.starting_model == b.model_ref
                and training.batch_id == b.batch.batch_id
                and training.training_data_hash == b.batch.training_data_hash,
                "Training changed the exact selected data/order",
            )
            cost(
                training.cost,
                model_hash=b.model_ref.model_hash,
                operations=("training",),
                subject=training.training_id,
            )
        if r.before_capability_id is not None:
            before, after = _lookup(
                capabilities,
                (r.before_capability_id, r.after_capability_id),
                "paired capability",
            )
            require(
                r.training is not None and r.training.status == "completed",
                "Candidate capability requires completed training",
            )
            _same(
                before.model_ref == b.model_ref
                and after.model_ref == r.training.candidate_model
                and before.task_set_hash == after.task_set_hash
                and before.split == after.split == "development",
                "Paired capability model/task identities differ",
            )

            def protocol(report):
                return sorted(
                    (rollouts[i].task_id, rollouts[i].sampling.sha256)
                    for i in report.rollout_ids
                )

            _same(
                protocol(before) == protocol(after)
                and sorted((m.skill, m.aggregation) for m in before.metrics)
                == sorted((m.skill, m.aggregation) for m in after.metrics),
                "Paired evaluation protocol differs",
            )
            if r.decision == "accept":
                require(
                    before.status == after.status == "valid",
                    "Acceptance needs valid paired capabilities",
                )
    return {
        "status": "verified",
        "validation_scope": "record_identities",
        "validation_version": "p0-v3",
        "expected_batch_verified": expected_batch_id is not None,
        "expected_model_verified": expected_model is not None,
        "expected_objective_verified": expected_objective is not None,
        "selection_context_verified": expected_selection_context_hash is not None,
        "training_inputs_verified": view is not None,
        "artifact_contents_verified": False,
        "bundle_hash": b.record_hash,
        "backend": b.backend,
        "batch_id": b.batch.batch_id,
        "model_hash": b.model_ref.model_hash,
        "tasks": len(tasks),
        "rollouts": len(rollouts),
        "evaluations": len(evaluations),
        "cost_events": len(events),
        "round_decision": None if r is None else r.decision,
    }
