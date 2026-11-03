"""Validated DAG execution with non-editable quality and budget boundaries."""

from fractions import Fraction

from self_curate.contracts import ContractError, ErrorCode, canonical_json, digest
from self_curate.contracts.program import (
    OPERATOR_NAMES,
    ProgramSpec,
    selection_identity,
)
from self_curate.contracts.selection import (
    BatchManifest,
    BatchMember,
    NodeTrace,
    SelectionResult,
)
from self_curate.contracts.types import require
from .context import ExecutionContext
from .handlers import Candidates, HANDLERS, stable_order


def _trace(node_id, operator, sources, candidates, exclusions=(), details=None):
    return NodeTrace(
        node_id,
        operator,
        tuple((name, value.ids) for name, value in sources),
        candidates.ids,
        tuple(exclusions),
        canonical_json(details or {}),
    )


def quality_gate(ctx):
    ids, kept, excluded = tuple(sorted(ctx.view.data_ids)), [], []
    excluded_ids, families = set(ctx.quality.excluded_ids), set(
        ctx.quality.excluded_families
    )
    for data_id in ids:
        row = ctx.view.record(data_id)
        reason = None
        if data_id in excluded_ids or families.intersection(row.source.family_ids):
            reason = "SYSTEM_EXCLUSION"
        elif row.quality.status == "fail" or (
            row.quality.status == "unknown" and not ctx.quality.allow_unknown
        ):
            reason = "QUALITY_" + row.quality.status.upper()
        elif any(u.decision == "pending" for u in row.usage.candidate_sft):
            reason = "PENDING_TRAINING"
        elif row.usage.accepted_sft and ctx.quality.exclude_accepted:
            reason = "ACCEPTED_TRAINING_HISTORY"
        elif ctx.quality.rejected_policy == "exclude" and any(
            u.decision == "rejected" for u in row.usage.candidate_sft
        ):
            reason = "REJECTED_TRAINING_HISTORY"
        if reason:
            excluded.append((data_id, reason, None))
        else:
            kept.append(data_id)
    candidate = Candidates(tuple(kept))
    return candidate, _trace(
        "eligible_pool",
        "QualityGate",
        (("pool", Candidates(ids)),),
        candidate,
        excluded,
        {"policy": ctx.quality.to_dict()},
    )


def _ratio_after_removal(tokens, shares, removed_branch, removed_tokens, total):
    if not total or not shares:
        return (0, 0.0)
    missing, error = 0, 0.0
    for name, share in shares.items():
        value = tokens[name] - (removed_tokens if name == removed_branch else 0)
        missing += value == 0
        error += abs(value / total - share)
    return missing, error


def _semantic_exposure(ids, ctx):
    """Count actual skill labels separately from unique allocation ownership."""
    exposure, missing_labels = {}, 0
    for data_id in ids:
        row = ctx.view.record(data_id)
        if row.annotations is None:
            missing_labels += 1
            continue
        metrics = {m.skill: m for m in row.evidence_metrics}
        for annotation in row.annotations.skills:
            stat = exposure.setdefault(
                annotation.skill,
                {
                    "sample_count": 0,
                    "trajectory_supervised_tokens": 0,
                    "known_evidence_supervised_tokens": None,
                    "evidence_missing_samples": 0,
                },
            )
            stat["sample_count"] += 1
            stat["trajectory_supervised_tokens"] += row.stats.supervised_tokens
            metric = metrics.get(annotation.skill)
            if metric is None or metric.supervised_evidence_tokens is None:
                stat["evidence_missing_samples"] += 1
            else:
                stat["known_evidence_supervised_tokens"] = (
                    stat["known_evidence_supervised_tokens"] or 0
                ) + metric.supervised_evidence_tokens
    return {
        "skills": dict(sorted(exposure.items())),
        "missing_label_samples": missing_labels,
        "overlapping_labels_count_separately": True,
    }


def _usage_exposure(ids, ctx):
    """Report historical exposure without creating any new training event."""
    fields = (
        "used_forward_scoring",
        "used_gradient_extraction",
        "used_candidate_sft",
        "used_accepted_sft",
        "used_rejected_sft",
        "pending_sft",
    )
    batch = ctx.view.static_batch(ids, fields)
    total = sum(ctx._proofs[i][3] for i in ids)
    report = {}
    for name in fields:
        selected = tuple(i for i, used in zip(ids, batch.column(name)) if used)
        tokens = sum(ctx._proofs[i][3] for i in selected)
        report[name] = {
            "data_ids": selected,
            "sample_count": len(selected),
            "sample_fraction": len(selected) / len(ids) if ids else None,
            "supervised_tokens": tokens,
            "token_fraction": tokens / total if total else None,
        }
    return report


def budget_pack(candidate, ctx):
    ids = candidate.ids
    lengths = {i: ctx.view.record(i).stats.supervised_tokens for i in ids}
    assignments, shares = dict(candidate.assignments), dict(candidate.shares)
    minimum, maximum = ctx.budget.minimum, ctx.budget.maximum
    branch_tokens = {name: 0 for name in shares}
    chosen, used, seen_content, excluded = [], 0, set(), []
    for data_id in ids:
        content = ctx._proofs[data_id][0]
        if ctx.budget.deduplicate_content and content in seen_content:
            excluded.append((data_id, "FINAL_EXACT_DUPLICATE", None))
            continue
        seen_content.add(content)
        if ctx.budget.max_samples is not None and len(chosen) >= ctx.budget.max_samples:
            excluded.append((data_id, "MAX_SAMPLES", None))
            continue
        if used + lengths[data_id] > maximum:
            excluded.append((data_id, "TOTAL_TOKEN_CAP", None))
            continue
        chosen.append(data_id)
        used += lengths[data_id]
        if shares:
            branch_tokens[assignments[data_id]] += lengths[data_id]
    while len(chosen) % ctx.budget.effective_batch:

        def removal_score(index):
            data_id = chosen[index]
            length = lengths[data_id]
            tokens = used - length
            return (
                tokens < minimum,
                *_ratio_after_removal(
                    branch_tokens, shares, assignments.get(data_id), length, tokens
                ),
                -index,
            )

        index = min(range(len(chosen)), key=removal_score)
        removed = chosen.pop(index)
        used -= lengths[removed]
        if shares:
            branch_tokens[assignments[removed]] -= lengths[removed]
        excluded.append((removed, "EFFECTIVE_BATCH_BOUNDARY", None))
    violations = []
    if not chosen:
        violations.append("EMPTY_BATCH")
    if used < minimum:
        violations.append("TOKEN_SHORTFALL")
    actual_tokens = branch_tokens
    if shares:
        for name, share in shares.items():
            if not actual_tokens[name] or abs(
                Fraction(actual_tokens[name], used) - Fraction(str(share))
            ) > Fraction(str(ctx.budget.share_tolerance)):
                violations.append("BRANCH_SHARE:" + name)
    details = {
        "algorithm": ctx.budget.algorithm,
        "token_target": ctx.budget.token_target,
        "minimum_tokens": minimum,
        "maximum_tokens": maximum,
        "supervised_tokens": used,
        "sample_count": len(chosen),
        "effective_batch": ctx.budget.effective_batch,
        "branch_tokens": actual_tokens,
        "actual_branch_shares": {
            k: v / used if used else 0 for k, v in actual_tokens.items()
        },
        "branch_share_residuals": {
            k: (v / used if used else 0) - shares[k] for k, v in actual_tokens.items()
        },
        "token_target_residual": used - ctx.budget.token_target,
        "semantic_skill_exposure": _semantic_exposure(chosen, ctx),
        "usage_exposure": _usage_exposure(chosen, ctx),
        "violations": violations,
        "global_infeasibility_proven": False,
    }
    result = Candidates(
        tuple(chosen),
        tuple((i, assignments[i]) for i in chosen) if assignments else (),
        candidate.shares,
    )
    return result, tuple(excluded), details


def execute_program(program: ProgramSpec, context: ExecutionContext) -> SelectionResult:
    require(
        isinstance(program, ProgramSpec) and isinstance(context, ExecutionContext),
        "Expected ProgramSpec and ExecutionContext",
    )
    require(
        len(program.nodes) <= context.max_nodes,
        "Program exceeds frozen node limit",
        ErrorCode.INVALID_PROGRAM,
    )
    require(
        set(HANDLERS) == set(OPERATOR_NAMES),
        "Operator registry/schema mismatch",
        ErrorCode.INVALID_PROGRAM,
    )
    ordered = program.ordered_nodes()
    program_hash, execution_hash = selection_identity(
        program, context.policy_hash, context.context_hash, context.snapshot_hash
    )
    initial, gate = quality_gate(context)
    traces, values = [gate], {"eligible_pool": initial}
    for node in ordered:
        sources = tuple((name, values[name]) for name in node.inputs)
        try:
            result = HANDLERS[node.op](
                node,
                tuple(value for _, value in sources),
                context,
                digest({"program_seed": program.seed, "node_id": node.id}),
            )
            ids = result.candidates.ids
            require(
                len(set(ids)) == len(ids)
                and set(ids).issubset({i for _, source in sources for i in source.ids}),
                "Operator invented or duplicated members",
                ErrorCode.INVALID_PROGRAM,
                node_id=node.id,
            )
            traces.append(
                _trace(
                    node.id,
                    node.op,
                    sources,
                    result.candidates,
                    result.exclusions,
                    {"params": node.params, **(result.details or {})},
                )
            )
            if result.problem is not None:
                return SelectionResult(
                    "infeasible",
                    program_hash,
                    execution_hash,
                    None,
                    tuple(traces),
                    result.problem["reason"],
                    canonical_json(
                        {"global_infeasibility_proven": False, **result.problem}
                    ),
                )
            values[node.id] = result.candidates
        except ContractError as exc:
            raise ContractError(
                exc.code,
                str(exc),
                **{
                    **exc.details,
                    "node_id": node.id,
                    "program_hash": program_hash,
                    "execution_hash": execution_hash,
                    "completed_traces": [t.to_dict() for t in traces],
                },
            ) from exc
    candidate = values[program.output]
    packed, exclusions, diagnostics = budget_pack(candidate, context)
    traces.append(
        _trace(
            "final_batch",
            "BudgetPack",
            ((program.output, candidate),),
            packed,
            exclusions,
            diagnostics,
        )
    )
    if diagnostics["violations"]:
        return SelectionResult(
            "infeasible",
            program_hash,
            execution_hash,
            None,
            tuple(traces),
            "PACKING_CONSTRAINTS",
            canonical_json(diagnostics),
        )
    branches = dict(packed.assignments)
    members = tuple(
        BatchMember(i, *context._proofs[i], branches.get(i, program.output))
        for i in sorted(packed.ids)
    )
    training_order = (
        packed.ids
        if context.training_order == "selection"
        else stable_order(tuple(sorted(packed.ids)), context.training_seed)
    )
    synthetic = (
        True
        if all(
            context.view.record(i).tokenization.identity.backend == "fixture"
            for i in packed.ids
        )
        else None
    )
    batch = BatchManifest(
        program_hash,
        execution_hash,
        context.snapshot_hash,
        context.policy_hash,
        members,
        packed.ids,
        training_order,
        context.training_seed,
        packed.shares,
        None if context.model_ref is None else context.model_ref.model_hash,
        synthetic,
    )
    return SelectionResult(
        "feasible", program_hash, execution_hash, batch, tuple(traces)
    )
