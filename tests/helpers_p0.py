"""Declared mock artifacts and values; no model, gradients or training executed."""

from dataclasses import replace

from helpers import ref
from helpers_m2 import MODEL, context, fixture_program, records
from self_curate.contracts import (
    ArtifactRef,
    CapabilityMetric,
    CapabilityReport,
    ComputeCost,
    ConstraintResult,
    CostEvent,
    CostLedger,
    EvaluationReport,
    GradientCoverage,
    GradientSpec,
    ObjectiveSpec,
    ResourceUsage,
    RolloutRecord,
    RoundRecord,
    SkillConstraint,
    SkillValue,
    SkillWeight,
    TargetGradientFeature,
    TargetGradientSpec,
    TargetRef,
    TaskBudget,
    TaskRecord,
    TrainingRecord,
    WorkflowBundle,
    digest,
    rollout_set_hash,
    task_set_hash,
)
from self_curate.data import make_gradient
from self_curate.operators import execute_program


def ar(name):
    return ArtifactRef(**ref(name))


def mock_gradient_spec():
    return GradientSpec(
        digest("mock-coordinates"),
        8,
        "mock-assistant-ce",
        "token_mean",
        "none",
        digest("none"),
        "float32",
        "raw",
        digest("raw"),
        "declared-mock-gradient",
        digest("mock-code"),
        digest("mock-config"),
    )


def mock_usage(tokens=0, gradient=False, training=False):
    return ResourceUsage(
        ComputeCost(0.01, 0, tokens),
        0,
        0,
        0,
        int(gradient),
        tokens if gradient else 0,
        0,
        0,
        tokens if training else 0,
    )


def fixture_bundle(decision="accept"):
    rows = records()
    ctx = context(rows, target=64, tolerance=0, batch=8, share_tolerance=0)
    recipe = fixture_program()
    batch = execute_program(recipe, ctx).batch
    gradients = tuple(
        make_gradient(
            ctx.view.record(i),
            MODEL,
            mock_gradient_spec(),
            vector=ar("mock-vector-" + i),
            dimension=8,
            dtype="float32",
            norm=1.0,
            provenance=(ar("declared-mock-values"),),
            cost=ComputeCost(0.01, 0, 8),
        )
        for i in batch.data_ids
    )
    candidate = replace(MODEL, weights_hash=digest("mock-candidate-weights"))
    budget = TaskBudget(8, 32, 512, 30)
    tasks = tuple(
        TaskRecord(
            i,
            "1",
            (skill,),
            ar(i),
            ar("mock-environment"),
            ar("mock-scorer"),
            "development",
            budget,
            "mock",
        )
        for i, skill in (
            ("dev-a", "recovery"),
            ("dev-b", "recovery"),
            ("dev-c", "regression"),
        )
    )
    run_specs = []
    for label, m in (("before", MODEL), ("after", candidate)):
        for task, seed in (("dev-a", 0), ("dev-a", 1), ("dev-b", 0), ("dev-c", 0)):
            reward = 0.0 if label == "before" and task == "dev-a" else 1.0
            run_specs.append((f"{label}-{task}-{seed}", m, task, seed, reward))
    events = []

    def event(event_id, op, subject, m=MODEL, tokens=0):
        events.append(
            CostEvent(
                event_id,
                "mock-attempt-1",
                op,
                subject,
                m.model_hash,
                "completed",
                mock_usage(tokens, op == "sample_gradient", op == "training"),
                ar("receipt-" + event_id),
                "mock",
            )
        )

    for run_id, m, _, _, _ in run_specs:
        event("cost-" + run_id, "rollout", run_id, m, 2)
    for skill in ("recovery", "regression"):
        event("cost-target-" + skill, "target_gradient", "target-" + skill)
    for gradient in gradients:
        event(
            "cost-gradient-" + gradient.binding.data_id,
            "sample_gradient",
            gradient.cache_key,
            tokens=8,
        )
    event("cost-selection", "selection", recipe.program_id)
    event("cost-evaluation", "evaluation", "mock-evaluation-1")
    event("cost-training", "training", "mock-training-1", tokens=64)
    event("cost-before-capability", "capability", "before-capability", MODEL)
    event("cost-after-capability", "capability", "after-capability", candidate)
    ledger = CostLedger("mock-ledger", tuple(events), "mock")
    by_task = {t.task_id: t for t in tasks}
    rollouts = tuple(
        RolloutRecord(
            run_id,
            m,
            task,
            by_task[task].task_hash,
            seed,
            ar("mock-sampling"),
            ar("actions-" + run_id),
            ar("observations-" + run_id),
            digest("tokens-" + run_id),
            digest("mask-" + run_id),
            2,
            reward,
            "completed",
            "finished",
            by_task[task].scorer,
            ledger.reference(("cost-" + run_id,)),
            "mock",
        )
        for run_id, m, task, seed, reward in run_specs
    )
    targets = []
    for skill in ("recovery", "regression"):
        runs = tuple(
            r
            for r in rollouts
            if r.model_ref == MODEL and skill in by_task[r.task_id].skills
        )
        task_group = tuple(t for t in tasks if skill in t.skills)
        spec = TargetGradientSpec(
            skill,
            replace(
                mock_gradient_spec(),
                loss_id="mock-reward-policy",
                reduction="token_sum",
            ),
            ar("mock-reward-config"),
            ar("mock-no-baseline"),
            ar("mock-task-aggregation"),
        )
        targets.append(
            TargetGradientFeature(
                "target-" + skill,
                MODEL,
                spec,
                task_set_hash(task_group),
                tuple(t.task_id for t in task_group),
                tuple(r.rollout_id for r in runs),
                rollout_set_hash(runs),
                ar("mock-target-vector-" + skill),
                8,
                "float32",
                2.0,
                "valid",
                None,
                ledger.reference(("cost-target-" + skill,)),
                "mock",
            )
        )
    objective = ObjectiveSpec(
        "mock-objective",
        (SkillWeight("recovery", 0.7), SkillWeight("regression", 0.3)),
        (SkillConstraint("retain-regression", "regression", -0.01, None),),
        ar("mock-estimator"),
        ar("mock-fixed-acceptance-policy"),
    )
    eval_cost = tuple(
        e.event_id
        for e in events
        if e.operation
        in ("selection", "sample_gradient", "target_gradient", "evaluation")
    )
    evaluation = EvaluationReport(
        "mock-evaluation-1",
        MODEL,
        batch.batch_id,
        objective.objective_hash,
        tuple(TargetRef(t.spec.skill, t.target_version) for t in targets),
        tuple(g.cache_key for g in gradients),
        GradientCoverage(batch.data_ids, batch.data_ids, ()),
        (SkillValue("recovery", 0.2), SkillValue("regression", 0.1)),
        0.17,
        (ConstraintResult("retain-regression", 0.1, True),),
        ar("mock-diagnostics"),
        "valid",
        None,
        ledger.reference(eval_cost),
        "mock",
    )
    capabilities = []
    for label, m in (("before", MODEL), ("after", candidate)):
        runs = tuple(r for r in rollouts if r.model_ref == m)
        metrics = []
        for skill in ("recovery", "regression"):
            group = tuple(t for t in tasks if skill in t.skills)
            skill_runs = tuple(
                r for r in runs if r.task_id in {t.task_id for t in group}
            )
            mean = 0.5 if label == "before" and skill == "recovery" else 1.0
            metrics.append(
                CapabilityMetric(
                    skill,
                    tuple(t.task_id for t in group),
                    tuple(r.rollout_id for r in skill_runs),
                    len(skill_runs),
                    "task_mean",
                    mean,
                    None,
                    None,
                )
            )
        capabilities.append(
            CapabilityReport(
                label + "-capability",
                m,
                tuple(t.task_id for t in tasks),
                task_set_hash(tasks),
                tuple(r.rollout_id for r in runs),
                "development",
                tuple(metrics),
                "valid",
                None,
                ledger.reference(
                    tuple("cost-" + r.rollout_id for r in runs)
                    + ("cost-" + label + "-capability",)
                ),
                "mock",
            )
        )
    training = TrainingRecord(
        "mock-training-1",
        MODEL,
        batch.batch_id,
        batch.training_data_hash,
        ar("mock-training-recipe"),
        "completed",
        candidate,
        ar("mock-checkpoint"),
        64,
        ar("mock-training-receipt"),
        ledger.reference(("cost-training",)),
        "mock",
    )
    round_record = RoundRecord(
        "mock-round-1",
        "mock-attempt-1",
        MODEL,
        candidate if decision == "accept" else MODEL,
        0,
        int(decision == "accept"),
        objective.objective_hash,
        objective.acceptance_policy.sha256,
        batch.batch_id,
        evaluation.evaluation_id,
        training,
        capabilities[0].report_id,
        capabilities[1].report_id,
        decision,
        "declared mock decision; no execution",
        ar("mock-decision-evidence"),
        ledger.reference(tuple(e.event_id for e in events)),
        "mock",
    )
    return WorkflowBundle(
        "mock-p0-contract-chain",
        "mock",
        MODEL,
        recipe,
        ctx.context_hash,
        batch,
        objective,
        tasks,
        rollouts,
        tuple(targets),
        gradients,
        (evaluation,),
        tuple(capabilities),
        ledger,
        round_record,
    )
