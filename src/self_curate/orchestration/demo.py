"""Explicit synthetic integration example; no weights trained or ability measured."""

from dataclasses import replace
from pathlib import Path

from self_curate.agents.demo import FixtureRoleClient
from self_curate.contracts import (
    ArtifactRef,
    ResourceUsage,
    digest,
    load_workflow,
    parse_feedback_contract,
)
from self_curate.contracts.parsing import load_json
from self_curate.data import read_pool
from self_curate.data.pool import write_json
from self_curate.search import InnerSearch
from self_curate.search.demo import run_inner_demo
from self_curate.training import (
    BackendTrainingResult,
    SFTRecipe,
    TrainingData,
    TrainingExample,
    TrainingService,
)
from self_curate.validation import TaskRunResult, TaskSuite, TaskValidator
from .contracts import AcceptanceDecision
from .outer_loop import OuterCycle


def _ref(label):
    return ArtifactRef("fixture://outer-demo/" + label, digest(label))


class FixtureSFTBackend:
    backend = "mock"

    def train(self, request, output):
        tokens = request.data.supervised_tokens * request.recipe.epochs
        return BackendTrainingResult(
            replace(request.starting_model, weights_hash=digest(request.record_hash)),
            _ref("declared-weights-" + request.record_hash),
            tokens,
            request.recipe.epochs
            * (
                (len(request.data.examples) + request.recipe.accumulation_steps - 1)
                // request.recipe.accumulation_steps
            ),
            None,
            ResourceUsage(training_supervised_tokens=tokens),
        )


class FixtureTaskRunner:
    backend = "mock"

    def execute(self, request, output):
        return TaskRunResult(
            request.record_hash,
            _ref("actions"),
            _ref("observations"),
            digest("tokens"),
            digest("action-mask"),
            2,
            0.5,
            "completed",
            "declared synthetic outcome",
            ResourceUsage(),
        )


class FixtureCurationProvider:
    def prepare(self, handoff):
        return TrainingData(
            "curation",
            handoff.record_hash,
            handoff.starting_model.tokenizer,
            (
                TrainingExample(
                    "synthetic-curation",
                    digest("synthetic-curation"),
                    (1, 2, 3),
                    (-100, 2, 3),
                ),
            ),
            (_ref("declared-curation-data"),),
            "mock",
        )


class FixtureDecisionProvider:
    def decide(self, context):
        return AcceptanceDecision(
            context.record_hash,
            "accept",
            "Scripted mock branch acceptance",
            _ref("scripted-decision"),
        )


def run_outer_demo(output, *, project_root="."):
    root, output = Path(project_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_inner_demo(output / "inner", project_root=root)
    request = parse_feedback_contract(
        "ValidationRequest",
        load_json(
            (output / "inner/checkpoint-final/initial_request.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    policy = parse_feedback_contract(
        "EditPolicy",
        load_json(
            (output / "inner/checkpoint-final/policy.json").read_text(encoding="utf-8")
        ),
    )
    view = read_pool(output / "inner/pool").metric_view()
    search = InnerSearch.restore(
        output / "inner/checkpoint-final",
        FixtureRoleClient(request.model_ref),
        view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id="inner-demo",
        expected_session_id="inner-demo",
        expected_attempt_id="inner-demo",
    )
    result = search.result()
    suite = TaskSuite(
        "synthetic-outer-suite",
        load_workflow(root / "configs/p0.fixture.json").tasks,
        (11,),
        _ref("sampling"),
        _ref("protocol"),
        "task_mean",
        "mock",
    )
    recipe = SFTRecipe(0.01, 1, 3, 128, 7)
    cycle = OuterCycle(
        result,
        view,
        expected_model=request.model_ref,
        expected_batch_id=result.best.request.batch.batch_id,
        trainer=TrainingService(FixtureSFTBackend()),
        validator=TaskValidator(FixtureTaskRunner()),
        suite=suite,
        subset_recipe=recipe,
        curation_recipe=recipe,
        output=output / "outer",
        curation_provider=FixtureCurationProvider(),
        acceptance_policy=_ref("scripted-policy"),
    )
    state = cycle.run(decider=FixtureDecisionProvider())
    summary = {
        **state,
        "training_count": len(cycle.trainings),
        "validation_count": len(cycle.evaluations),
        "same_task_suite": cycle.evaluations[0].suite == cycle.evaluations[1].suite,
        "sequential_weights": cycle.trainings[1].request.starting_model
        == cycle.trainings[0].result.candidate_model,
        "exact_best_training_order": tuple(
            x.data_id for x in cycle.trainings[0].request.data.examples
        )
        == result.best.request.batch.training_order,
        "training_executed": False,
        "task_execution_real": False,
        "improvement_verified": False,
        "production_acceptance_policy": None,
        "production_stop_condition": None,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "ledger.json", cycle.ledger.to_dict())
    return summary
