"""CPU synthetic wiring demo. Numeric cosine runs; rewards/weights are fixtures."""

from pathlib import Path

from self_curate.agents import RoleTeam, load_prompts
from self_curate.agents.demo import FixtureRoleClient
from self_curate.contracts import (
    ArtifactRef,
    ResourceUsage,
    ValidationRequest,
    digest,
    load_workflow,
    parse_feedback_contract,
)
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require
from self_curate.data import read_pool
from self_curate.data.pool import build_pool, load_config, write_json
from self_curate.operators import prepare_to_directory, load_preparation
from self_curate.orchestration.demo import FixtureSFTBackend, FixtureTaskRunner
from self_curate.search import InnerSearch, StopDecision
from self_curate.training import TrainingExample, TrainingService, SFTRecipe
from self_curate.validation import TaskSuite, TaskValidator
from . import create_validator
from .common import artifact
from .contracts import NICERecipe, NICEReference, NICEScores, ReplayTrajectory
from .nice import fresh_adam, score_matrix


def _ref(name):
    return ArtifactRef("fixture://leval-demo/" + name, digest(name))


class FixtureNICEBackend:
    backend = "mock"
    implementation_hash = digest("fixture-vectors-actual-cosine-v1")

    def compute(self, request, data, reference, recipe, output):
        import numpy as np

        require(request.backend == "mock", "Fixture NICE only")
        raw = tuple(
            np.array([1.0, 1.0 + int(x.content_hash[:2], 16) / 255.0, 0.5])
            for x in data.examples
        )
        adam = tuple(
            fresh_adam(g, recipe.adam_epsilon, recipe.max_grad_norm) for g in raw
        )
        targets = tuple(
            np.array([1.0, 1.0, 0.5]) for _ in reference.evaluation.suite.tasks
        )
        np.savez_compressed(
            output / "fixture_vectors.npz", raw=raw, adam=adam, targets=targets
        )
        return NICEScores(
            request.request_hash,
            data.record_hash,
            reference.record_hash,
            recipe.record_hash,
            tuple(x.data_id for x in data.examples),
            tuple(t.task_id for t in reference.evaluation.suite.tasks),
            score_matrix(adam, targets),
            score_matrix(raw, targets),
            tuple(float(np.linalg.norm(t)) for t in targets),
            artifact(output / "fixture_vectors.npz"),
            ResourceUsage(new_gradient_samples=0, training_supervised_tokens=0),
            "mock",
        )


def run_valuation_demo(output, *, method, project_root="."):
    root, output = Path(project_root).resolve(), Path(output).resolve()
    require(method in ("nice", "sft"), "Explicit demo method required")
    output.mkdir(parents=True, exist_ok=False)
    build_pool(load_config(root / "configs/m2.data.fixture.json"), output / "pool")
    prepare_to_directory(
        output / "pool",
        root / "configs/programs/feedback.parent.fixture.json",
        root / "configs/m2.fixture.json",
        root / "configs/p1.objective.fixture.json",
        output / "seed",
        backend="mock",
        attempt_id="leval-demo",
        invocation_id="seed",
    )
    view = read_pool(output / "pool").metric_view()
    initial = load_preparation(output / "seed", view)
    model = initial.context.model_ref
    suite = TaskSuite(
        "leval-demo",
        load_workflow(root / "configs/p0.fixture.json").tasks,
        (11, 12),
        _ref("sampling"),
        _ref("task-protocol"),
        "task_mean",
        "mock",
    )
    validator = TaskValidator(FixtureTaskRunner())
    settings = {"model": model, "view": view, "output": output / "validations"}
    if method == "nice":
        evaluation = validator.evaluate(
            model,
            suite,
            attempt_id="reference",
            evaluation_id="reference",
            output=output / "reference",
        )
        step = TrainingExample("mock-action", digest("action"), (1, 2, 3), (-100, 2, 3))
        reference = NICEReference(
            evaluation,
            tuple(
                ReplayTrajectory(
                    r.record_hash, (step,), _ref(r.rollout_id), model.tokenizer
                )
                for r in evaluation.rollouts
            ),
        )
        settings.update(
            reference=reference,
            recipe=NICERecipe(digest("fixture-coordinates"), 3, 1e-8, 1.0),
            engine=FixtureNICEBackend(),
        )
    else:
        settings.update(
            trainer=TrainingService(FixtureSFTBackend()),
            validator=validator,
            suite=suite,
            recipe=SFTRecipe(0.01, 1, 3, 128, 7),
        )
    provider = create_validator(method, **settings)
    request = ValidationRequest(
        "leval-demo",
        model,
        initial.program,
        initial.workflow.batch,
        initial.objective,
        initial.context.context_hash,
        "predicted_gain" if method == "nice" else "reward_delta",
        provider.protocol,
        "mock",
    )
    policy = parse_feedback_contract(
        "EditPolicy",
        load_json(
            (root / "configs/feedback.policy.fixture.json").read_text(encoding="utf-8")
        ),
    )
    prompts = load_prompts(root / "configs/roles")
    client = FixtureRoleClient(model)
    team = RoleTeam(
        client,
        model=model,
        session_id="leval-demo-" + method,
        backend="mock",
        prompts=prompts,
    )
    search = InnerSearch(
        team, initial, request, policy, search_id="leval-demo-" + method
    )
    # Test/demo policy only. Production stopping conditions remain caller-owned.
    result = search.run(
        provider,
        stop_condition=lambda state: (
            StopDecision(True, "three synthetic evaluations")
            if state.evaluated_count >= 3
            else StopDecision(False)
        ),
    )
    require(
        result.status == "stopped" and result.best is not None, "Demo search failed"
    )
    search.save(output / "checkpoint")
    search.publish_best(output / "best")
    for i, cycle in enumerate(search.history):
        team.publish_cycle(cycle, output / ("cycle-" + str(i)))
    if method == "sft":
        baseline = provider.save_reference(output / "baseline-reference")
        write_json(output / "baseline-reference-anchor.json", baseline.to_dict())
    write_json(output / "role_calls.json", client.calls)
    summary = {
        "schema_version": "1",
        "backend": "mock",
        "method": method,
        "evaluated_count": result.state.evaluated_count,
        "best_value": result.best.validation.total_value,
        "history_comparisons": len(search.history[-1].evaluation.context.history),
        "shared_model_unchanged": team.runtime.model_ref == model,
        "validation_kernel_executed": True,
        "actual_gradients_computed": False,
        "actual_sft_executed": False,
        "real_task_scoring": False,
        "improvement_verified": False,
        "production_stop_condition": None,
        "production_acceptance_policy": None,
    }
    write_json(output / "summary.json", summary)
    return summary
