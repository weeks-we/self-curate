"""Leval interpretation of imported results. No verifier or estimator implementation."""

from typing import Protocol

from self_curate.contracts.feedback import (
    ValidationRequest,
    ValidationResult,
    FeedbackFact,
    EditHint,
    LevalFeedback,
)
from self_curate.contracts import (
    ArtifactRef,
    EvaluationReport,
    ResourceUsage,
    ErrorCode,
    digest,
)
from self_curate.contracts.types import require
from self_curate.contracts.workflow_validation import _sum, _equal_number


class ValidationProvider(Protocol):
    def evaluate(self, request: ValidationRequest) -> ValidationResult: ...


def check_external_result(request, result):
    require(
        isinstance(request, ValidationRequest) and isinstance(result, ValidationResult),
        "Expected validation request/result",
    )
    require(
        result.request_hash == request.request_hash
        and result.backend == request.backend,
        "External result belongs to another request/model/batch/protocol",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        result.backend != "real"
        or not result.evidence.uri.strip()
        .lower()
        .startswith(("mock://", "fixture://")),
        "Mock evidence cannot be declared real",
    )
    weights = {w.skill: w.weight for w in request.objective.weights}
    values = {v.skill: v.value for v in result.skill_values}
    constraints = {c.name: c for c in request.objective.constraints}
    require(
        set(values).issubset(weights)
        and {c.name for c in result.constraints}.issubset(constraints),
        "External result contains unknown skills/constraints",
    )
    if result.status == "valid":
        require(
            set(values) == set(weights)
            and {c.name for c in result.constraints} == set(constraints),
            "External verification coverage is incomplete",
        )
        require(
            _equal_number(
                result.total_value, _sum(weights[s] * values[s] for s in weights)
            ),
            "Imported total disagrees with fixed aggregation",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    for outcome in result.constraints:
        bound = constraints[outcome.name]
        value = values.get(bound.skill)
        if result.status == "valid" or outcome.value is not None:
            passed = (
                value is not None
                and (bound.minimum is None or value >= bound.minimum)
                and (bound.maximum is None or value <= bound.maximum)
            )
            require(
                _equal_number(outcome.value, value) and outcome.passed == passed,
                "Imported constraint disagrees with fixed objective",
                ErrorCode.ARTIFACT_MISMATCH,
            )
    return result


def result_from_evaluation(request, report, evidence, *, usage=None):
    require(
        isinstance(report, EvaluationReport) and isinstance(evidence, ArtifactRef),
        "Expected external EvaluationReport/evidence",
    )
    require(
        request.measure == "predicted_gain"
        and report.model_ref == request.model_ref
        and report.batch_id == request.batch.batch_id
        and report.objective_hash == request.objective.objective_hash,
        "Evaluation report model/batch/measure/objective mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        set(report.coverage.required_ids) == set(request.batch.data_ids),
        "Evaluation coverage differs from the exact selected batch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    result = ValidationResult(
        report.evaluation_id,
        request.request_hash,
        report.status,
        report.skill_values,
        report.total_value,
        report.constraints,
        report.reason,
        evidence,
        usage or ResourceUsage(),
        report.backend,
    )
    return check_external_result(request, result)


def build_feedback(request, result, *, memory_namespace="leval"):
    check_external_result(request, result)
    facts = []
    hints = []
    if result.status != "valid":
        facts.append(
            FeedbackFact("EVIDENCE_UNAVAILABLE", None, None, None, result.reason)
        )
        summary = "外部验证尚不可用；保留原因，不进行基于数值的程序优化。"
    else:
        values = {v.skill: v.value for v in result.skill_values}
        for skill in sorted(values):
            facts.append(
                FeedbackFact(
                    "OBSERVED_VALUE",
                    skill,
                    values[skill],
                    None,
                    f"{request.measure}: {skill} = {values[skill]}",
                )
            )
        bounds = {c.name: c for c in request.objective.constraints}
        for constraint in result.constraints:
            if constraint.passed:
                continue
            bound = bounds[constraint.name]
            value = values[bound.skill]
            direction = (
                "increase"
                if bound.minimum is not None and value < bound.minimum
                else "decrease"
            )
            limit = bound.minimum if direction == "increase" else bound.maximum
            facts.append(
                FeedbackFact(
                    "CONSTRAINT_FAILED",
                    bound.skill,
                    value,
                    limit,
                    f"固定约束 {bound.name} 未满足。",
                )
            )
            hints.append(
                EditHint(
                    bound.skill,
                    direction,
                    "假设：调整该技能分支的样本曝光或证据条件可能改善此项；必须重新验证。",
                )
            )
        if not hints:
            positive = {w.skill for w in request.objective.weights if w.weight > 0}
            skill = min(positive, key=lambda s: (values[s], s))
            hints.append(
                EditHint(
                    skill,
                    "increase",
                    "假设：探索本轮较低指标对应技能的曝光配置；最低指标不等于已证明样本不足。",
                )
            )
        summary = f'已接收外部 {request.measure} 结果，总值 {result.total_value}；发现 {len([f for f in facts if f.code=="CONSTRAINT_FAILED"])} 项固定约束未满足。建议仅为待检验假设。'
    identity = digest(
        {
            "request": request.request_hash,
            "result": result.record_hash,
            "namespace": memory_namespace,
        }
    )
    return LevalFeedback(
        "feedback-" + identity,
        request.request_hash,
        result.record_hash,
        request.model_ref,
        request.objective.objective_hash,
        request.batch.batch_id,
        request.measure,
        request.protocol.sha256,
        result.status,
        result.total_value,
        tuple(facts),
        tuple(hints),
        summary,
        memory_namespace,
        request.backend,
    )
