"""Strict decoder for a fixed, code-owned set of P0 record classes."""

from dataclasses import MISSING, fields as dataclass_fields
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from . import workflow as w
from .errors import ContractError, ErrorCode
from .features import ComputeCost, GradientFeature, GradientSpec
from .parsing import (
    artifact,
    fields,
    gradient_feature,
    gradient_spec,
    load_json,
    model_ref,
    sequence,
)
from .program import ProgramSpec, parse_program
from .selection import BatchManifest, parse_batch_manifest
from .signals import ModelRef
from .types import ArtifactRef, require


CONTRACT_TYPES = {
    cls.__name__: cls
    for cls in (
        w.TaskBudget,
        w.TaskRecord,
        w.ResourceUsage,
        w.CostEvent,
        w.CostRef,
        w.CostLedger,
        w.RolloutRecord,
        w.TargetGradientSpec,
        w.TargetGradientFeature,
        w.SkillWeight,
        w.SkillConstraint,
        w.ObjectiveSpec,
        w.GradientCoverage,
        w.TargetRef,
        w.SkillValue,
        w.ConstraintResult,
        w.EvaluationReport,
        w.CapabilityMetric,
        w.CapabilityReport,
        w.TrainingRecord,
        w.RoundRecord,
        w.WorkflowBundle,
    )
}

_EXISTING = {
    ArtifactRef: artifact,
    ModelRef: model_ref,
    GradientSpec: gradient_spec,
    GradientFeature: gradient_feature,
    ProgramSpec: parse_program,
    BatchManifest: parse_batch_manifest,
}


def _decode(cls, value, name, *, registry=None):
    registry = CONTRACT_TYPES if registry is None else registry
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        for option in args:
            if option is not type(None):
                return _decode(option, value, name, registry=registry)
        require(False, f"Invalid optional {name}")
    if origin is tuple:
        return tuple(
            _decode(args[0], v, name, registry=registry) for v in sequence(value, name)
        )
    if cls in _EXISTING:
        return _EXISTING[cls](value)
    if cls in (str, int, float, bool):
        require(
            (type(value) in (int, float) if cls is float else type(value) is cls),
            f"Invalid scalar type for {name}",
        )
        return value
    require(cls in registry.values() or cls is ComputeCost, "Unknown decoder type")
    definitions = tuple(f for f in dataclass_fields(cls) if f.init)
    required = [
        f.name
        for f in definitions
        if f.name == "schema_version"
        or f.default is MISSING
        and f.default_factory is MISSING
    ]
    optional = [f.name for f in definitions if f.name not in required]
    row = fields(value, required, optional)
    hints = get_type_hints(cls)
    return cls(
        **{k: _decode(hints[k], v, k, registry=registry) for k, v in row.items()}
    )


def parse_contract(kind, value):
    """No user-provided module/class names or arbitrary code are evaluated."""
    require(
        isinstance(kind, str) and kind in CONTRACT_TYPES, "Unknown P0 contract type"
    )
    try:
        return _decode(CONTRACT_TYPES[kind], value, kind)
    except ContractError:
        raise
    except (
        TypeError,
        ValueError,
        OverflowError,
        KeyError,
        AttributeError,
        RecursionError,
    ) as exc:
        raise ContractError(
            ErrorCode.INVALID_RECORD,
            "Malformed workflow contract",
            kind=kind,
            reason=type(exc).__name__,
        ) from exc


def parse_task_record(value):
    return parse_contract("TaskRecord", value)


def parse_rollout_record(value):
    return parse_contract("RolloutRecord", value)


def parse_evaluation_report(value):
    return parse_contract("EvaluationReport", value)


def parse_capability_report(value):
    return parse_contract("CapabilityReport", value)


def parse_round_record(value):
    return parse_contract("RoundRecord", value)


def parse_workflow_bundle(value):
    return parse_contract("WorkflowBundle", value)


def load_workflow(
    path,
    *,
    view=None,
    expected_model=None,
    expected_objective=None,
    expected_selection_context_hash=None,
    expected_batch_id=None,
):
    from .workflow_validation import validate_workflow

    result = parse_workflow_bundle(load_json(Path(path).read_text(encoding="utf-8")))
    validate_workflow(
        result,
        view,
        expected_model=expected_model,
        expected_objective=expected_objective,
        expected_selection_context_hash=expected_selection_context_hash,
        expected_batch_id=expected_batch_id,
    )
    return result
