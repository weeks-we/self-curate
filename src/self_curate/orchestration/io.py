"""Strict fixed-registry JSON adapters for external data/recipes/task suites."""

from pathlib import Path
from dataclasses import replace

from self_curate.contracts import ContractError, ErrorCode, TokenizerRef
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require
from self_curate.contracts.workflow_parsing import CONTRACT_TYPES, _decode
from self_curate.training.contracts import (
    BackendTrainingResult,
    SFTRecipe,
    TrainingData,
    TrainingExample,
    TrainingOutcome,
    TrainingRequest,
)
from self_curate.validation.contracts import (
    TaskEvaluation,
    TaskRunRequest,
    TaskRunResult,
    TaskSuite,
)
from .contracts import AcceptanceDecision, DecisionContext, OuterHandoff


OUTER_TYPES = {
    cls.__name__: cls
    for cls in (
        BackendTrainingResult,
        SFTRecipe,
        TrainingData,
        TrainingExample,
        TrainingOutcome,
        TrainingRequest,
        TaskEvaluation,
        TaskRunRequest,
        TaskRunResult,
        TaskSuite,
        AcceptanceDecision,
        DecisionContext,
        OuterHandoff,
    )
}


def parse_outer_contract(kind, value):
    require(type(kind) is str and kind in OUTER_TYPES, "Unknown outer contract type")
    try:
        return _decode(
            OUTER_TYPES[kind],
            value,
            kind,
            registry={**CONTRACT_TYPES, **OUTER_TYPES, "TokenizerRef": TokenizerRef},
        )
    except ContractError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD,
            "Malformed outer contract",
            kind=kind,
            reason=type(exc).__name__,
        ) from exc


def load_outer_contract(kind, path):
    return parse_outer_contract(kind, load_json(Path(path).read_text(encoding="utf-8")))


def load_task_evaluation(
    directory, *, expected_model, expected_suite, expected_checkpoint
):
    """Verify saved request/result evidence, including legacy 0.8.0 directories.

    Returns an enriched immutable object; never rewrites historical report files.
    Supplied anchors must come from the caller's saved model and validation plan.
    """
    root = Path(directory).resolve()
    evaluation = load_outer_contract("TaskEvaluation", root / "evaluation.json")
    require(
        evaluation.report.model_ref == expected_model
        and evaluation.suite == expected_suite
        and evaluation.checkpoint == expected_checkpoint,
        "Stored task evaluation differs from trusted model/suite/checkpoint",
    )
    by_id = {}
    for folder in root.glob("run-*"):
        require(
            folder.is_dir() and not folder.is_symlink(),
            "Invalid task execution directory",
        )
        for name in ("request.json", "result.json"):
            require(
                not (folder / name).is_symlink(),
                "Execution evidence cannot be a symlink",
            )
        request = load_outer_contract("TaskRunRequest", folder / "request.json")
        result = load_outer_contract("TaskRunResult", folder / "result.json")
        require(
            folder.name == "run-" + request.record_hash and request.run_id not in by_id,
            "Execution evidence directory/identity mismatch",
        )
        by_id[request.run_id] = (request, result)
    require(
        set(by_id) == set(evaluation.report.rollout_ids),
        "Incomplete or unexpected execution evidence",
    )
    requests = tuple(by_id[r.rollout_id][0] for r in evaluation.rollouts)
    results = tuple(by_id[r.rollout_id][1] for r in evaluation.rollouts)
    if evaluation.protocol_verified:
        require(
            requests == evaluation.execution_requests
            and results == evaluation.execution_results,
            "Stored execution files differ from published capability evidence",
        )
    return replace(evaluation, execution_requests=requests, execution_results=results)


class JsonCurationDataProvider:
    """Read already-produced trajectories. Does not invent or summarize training data."""

    def __init__(self, path):
        self.path = Path(path)

    def prepare(self, handoff):
        result = load_outer_contract("TrainingData", self.path)
        require(
            result.kind == "curation" and result.source_hash == handoff.record_hash,
            "Curation file belongs to another stopped search",
        )
        return result
