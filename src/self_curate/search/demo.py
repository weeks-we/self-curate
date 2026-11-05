"""Explicit mock demonstration; demo limits and values are not research policies."""

from pathlib import Path

from self_curate.agents import RoleTeam, load_prompts
from self_curate.agents.demo import FixtureRoleClient
from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ConstraintResult,
    ResourceUsage,
    SkillValue,
    ValidationRequest,
    ValidationResult,
    digest,
    parse_feedback_contract,
)
from self_curate.contracts.parsing import artifact, load_json
from self_curate.contracts.types import require
from self_curate.data import read_pool
from self_curate.data.pool import build_pool, load_config, write_json
from self_curate.operators import prepare_to_directory, load_preparation
from .contracts import StopDecision
from .inner_loop import InnerSearch
from .storage import _directory, _finish, load_best


class FixtureValidationProvider:
    """Return predeclared mock values, never model/gradient performance scores."""

    def __init__(self, values):
        self._values = iter(values)
        self.calls = []

    def evaluate(self, request):
        require(
            request.backend == "mock", "Fixture verifier cannot evaluate a real request"
        )
        amount = next(self._values)
        self.calls.append(request.request_hash)
        constraints = tuple(
            ConstraintResult(
                b.name,
                amount,
                (b.minimum is None or amount >= b.minimum)
                and (b.maximum is None or amount <= b.maximum),
            )
            for b in request.objective.constraints
        )
        return ValidationResult(
            "declared-fixture-" + str(len(self.calls)),
            request.request_hash,
            "valid",
            tuple(SkillValue(w.skill, amount) for w in request.objective.weights),
            sum(w.weight * amount for w in request.objective.weights),
            constraints,
            None,
            ArtifactRef(
                "fixture://declared-inner-values",
                digest((request.request_hash, amount)),
            ),
            ResourceUsage(ComputeCost(None, 0, 0), 0, 0, 0, 0, 0, 0, 0, 0),
            "mock",
        )


def run_inner_demo(output, *, project_root=".", prompt_directory=None):
    output, staging = _directory(output)
    root = Path(project_root).resolve()
    build_pool(load_config(root / "configs/m2.data.fixture.json"), staging / "pool")
    prepare_to_directory(
        staging / "pool",
        root / "configs/programs/feedback.parent.fixture.json",
        root / "configs/m2.fixture.json",
        root / "configs/p1.objective.fixture.json",
        staging / "seed",
        backend="mock",
        attempt_id="inner-demo",
        invocation_id="seed",
    )
    view = read_pool(staging / "pool").metric_view()
    seed = load_preparation(staging / "seed", view)
    request = ValidationRequest(
        "inner-demo-seed",
        seed.context.model_ref,
        seed.program,
        seed.workflow.batch,
        seed.objective,
        seed.context.context_hash,
        "predicted_gain",
        artifact(
            load_json(
                (root / "configs/feedback.protocol.fixture.json").read_text(
                    encoding="utf-8"
                )
            )
        ),
        "mock",
    )
    policy = parse_feedback_contract(
        "EditPolicy",
        load_json(
            (root / "configs/feedback.policy.fixture.json").read_text(encoding="utf-8")
        ),
    )
    prompts = load_prompts(prompt_directory or root / "configs/roles")
    client = FixtureRoleClient(request.model_ref)
    team = RoleTeam(
        client,
        model=request.model_ref,
        session_id="inner-demo",
        backend="mock",
        prompts=prompts,
    )
    search = InnerSearch(team, seed, request, policy, search_id="inner-demo")
    provider = FixtureValidationProvider((0.1, 0.3, 0.2))
    search.step(provider)
    search.step()
    search.save(staging / "checkpoint-1")
    search = InnerSearch.restore(
        staging / "checkpoint-1",
        client,
        view,
        expected_request=request,
        expected_policy=policy,
        expected_search_id="inner-demo",
        expected_session_id="inner-demo",
        expected_attempt_id="inner-demo",
        expected_prompts=prompts,
    )

    def demo_stop(state):
        # Only this synthetic example stops after 3 declared evaluations.
        # No production stopping rule or threshold is installed.
        return (
            StopDecision(True, "synthetic demo completed three declared values")
            if state.evaluated_count >= 3
            else StopDecision(False)
        )

    result = search.run(provider, stop_condition=demo_stop)
    search.save(staging / "checkpoint-final")
    search.publish_best(staging / "best")
    loaded = load_best(
        staging / "best", view, expected_request=result.require_best().request
    )
    require(loaded == result.best, "Best handoff differs from actual evaluated batch")
    require(
        abs(result.best.validation.total_value - 0.3) < 1e-12,
        "Demo lost historical best",
    )
    summary = {
        "schema_version": "1",
        "backend": "mock",
        "status": result.status,
        "declared_values": [0.1, 0.3, 0.2],
        "evaluated_count": result.state.evaluated_count,
        "candidate_attempts": result.state.candidate_attempts,
        "best_value": result.best.validation.total_value,
        "last_value": result.state.current.validation.total_value,
        "best_batch_id": result.best.request.batch.batch_id,
        "last_batch_id": result.state.current.request.batch.batch_id,
        "model_calls": len(client.calls),
        "validation_calls": len(provider.calls),
        "history_restored": True,
        "production_stop_condition": None,
        "real_model_called": False,
        "validation_implemented": False,
        "improvement_verified": False,
    }
    write_json(staging / "summary.json", summary)
    write_json(staging / "result.json", result.to_dict())
    write_json(staging / "client_calls.json", client.calls)
    write_json(staging / "validation_calls.json", provider.calls)
    _finish(staging, output)
    return summary
