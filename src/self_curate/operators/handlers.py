"""The eight deterministic operators. Model computation is always external."""

from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import math

from self_curate.contracts import ContractError, ErrorCode, canonical_json, digest


@dataclass(frozen=True)
class Candidates:
    ids: tuple[str, ...]
    assignments: tuple[tuple[str, str], ...] = ()
    shares: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class OperatorResult:
    candidates: Candidates
    exclusions: tuple[tuple[str, str, str | None], ...] = ()
    details: object = None
    problem: object = None


def stable_order(ids, seed):
    return tuple(sorted(ids, key=lambda i: (digest({"seed": seed, "id": i}), i)))


def _result(ids, keep, reasons=None, details=None, problem=None):
    keep = tuple(keep)
    selected = set(keep)
    excluded = tuple(
        (i, (reasons or {}).get(i, "FILTERED"), None) for i in ids if i not in selected
    )
    return OperatorResult(Candidates(keep), excluded, details or {}, problem)


def _missing(policy, node, data_id, what):
    if policy == "error":
        raise ContractError(
            ErrorCode.MISSING_SIGNAL,
            "Required annotation/metadata is missing",
            node_id=node.id,
            data_id=data_id,
            field=what,
        )
    return policy == "include"


def skill_select(node, inputs, ctx, seed):
    ids = inputs[0].ids
    p = node.params
    labels = ctx.view.static_batch(ids, ("skills",)).column("skills")
    wanted, keep, reasons = set(p["skills"]), [], {}
    for data_id, skills in zip(ids, labels):
        if skills is None:
            accepted = _missing(p["missing_policy"], node, data_id, "skills")
            reasons[data_id] = "MISSING_SKILLS"
        else:
            accepted = (
                bool(wanted.intersection(skills))
                if p["match"] == "any"
                else wanted.issubset(skills)
            )
            reasons[data_id] = "SKILL_MISMATCH"
        if accepted:
            keep.append(data_id)
    return _result(ids, keep, reasons)


def evidence_select(node, inputs, ctx, seed):
    ids, p = inputs[0].ids, node.params
    names = (
        "share",
        "density",
        "supervised_evidence_tokens",
        "complete_events",
        "behaviors",
        "status",
    )
    batch = ctx.view.evidence_batch(ids, p["target_skill"], fields=names)
    keep, reasons = [], {}
    thresholds = (
        ("share", "min_share"),
        ("density", "min_density"),
        ("supervised_evidence_tokens", "min_tokens"),
        ("complete_events", "min_events"),
    )
    for index, data_id in enumerate(ids):
        missing, failed = [], []
        if batch.column("status")[index] != "annotated":
            missing.append("evidence")
        for field, param in thresholds:
            if p[param] is not None:
                value = batch.column(field)[index]
                if value is None:
                    missing.append(field)
                elif value < p[param]:
                    failed.append(field)
        if p["required_behaviors"]:
            value = batch.column("behaviors")[index]
            if value is None:
                missing.append("behaviors")
            elif not set(p["required_behaviors"]).issubset(value):
                failed.append("behaviors")
        if failed:
            reasons[data_id] = "EVIDENCE_REQUIREMENTS:" + ",".join(failed)
        elif missing and not _missing(
            p["missing_policy"], node, data_id, ",".join(missing)
        ):
            reasons[data_id] = "MISSING_EVIDENCE:" + ",".join(missing)
        else:
            keep.append(data_id)
    return _result(
        ids,
        keep,
        reasons,
        {"annotation_versions": list(zip(ids, batch.context["annotation_versions"]))},
    )


def _inside(value, interval):
    return interval is None or (
        (interval[0] is None or value >= interval[0])
        and (interval[1] is None or value <= interval[1])
    )


def score_select(node, inputs, ctx, seed):
    ids, p = inputs[0].ids, node.params
    metric = "nll" if node.op == "NLLSelect" else "entropy"
    field = metric if p["scale"] == "raw" else f"{metric}_percentile"
    spec, reference = ctx.signal_context(p["scope"], p["skill"], p["reference_group"])
    batch = ctx.view.signal_batch(
        ids,
        ctx.model_ref,
        spec,
        fields=(field,),
        percentile_reference=reference,
        missing="none",
    )
    values = dict(zip(ids, batch.column(field)))
    missing = tuple(i for i in ids if values[i] is None)
    if missing and p["missing_policy"] != "exclude":
        request = {
            "metric": metric,
            "scale": p["scale"],
            "data_ids": missing,
            "model_ref": ctx.model_ref.to_dict(),
            "spec": spec.to_dict(),
            "percentile_reference": None if reference is None else reference.to_dict(),
        }
        raise ContractError(
            ErrorCode.MISSING_SIGNAL,
            (
                "External computation required"
                if p["missing_policy"] == "request"
                else "Scores are missing in the fixed context"
            ),
            node_id=node.id,
            missing_ids=missing,
            requests=[request] if p["missing_policy"] == "request" else [],
        )
    reasons = {
        i: "MISSING_SIGNAL" if values[i] is None else "OUTSIDE_INTERVAL" for i in ids
    }
    keep = [
        i for i in ids if values[i] is not None and _inside(values[i], p["interval"])
    ]
    if p["rank_direction"] is not None:
        sign = 1 if p["rank_direction"] == "ascending" else -1
        keep.sort(key=lambda i: (sign * values[i], i))
    if p["keep_fraction"] is not None:
        count = math.ceil(len(keep) * p["keep_fraction"])
        for data_id in keep[count:]:
            reasons[data_id] = "RANK_TRUNCATION"
        keep = keep[:count]
    return _result(
        ids,
        keep,
        reasons,
        {
            "metric": field,
            "model_hash": ctx.model_ref.model_hash,
            "spec_hash": spec.spec_hash,
            "reference_hash": batch.context["reference_hash"],
            "feature_keys": list(zip(ids, batch.context["feature_keys"])),
            "signal_cells_read": len(ids),
        },
    )


def length_select(node, inputs, ctx, seed):
    ids, p = inputs[0].ids, node.params
    mapping = {
        "assistant_turn_range": "assistant_turns",
        "tool_call_range": "tool_calls",
        "supervised_token_range": "supervised_tokens",
        "context_token_range": "context_tokens",
    }
    names = tuple(mapping[k] for k, value in p.items() if value is not None)
    batch = ctx.view.static_batch(ids, names)
    keep = [
        data_id
        for index, data_id in enumerate(ids)
        if all(
            _inside(batch.column(mapping[k])[index], value)
            for k, value in p.items()
            if value is not None
        )
    ]
    return _result(ids, keep, {i: "LENGTH_REQUIREMENTS" for i in ids})


def _representatives(ids, groups, rule, ctx):
    position = {i: n for n, i in enumerate(ids)}

    def priority(i):
        if rule == "first":
            return position[i]
        if rule == "shortest":
            return ctx.view.record(i).stats.supervised_tokens, i
        return i

    reps = {min(group, key=priority) for group in groups}
    return tuple(i for i in ids if i in reps)


def dedup_select(node, inputs, ctx, seed):
    ids, p = inputs[0].ids, node.params
    if p["mode"] == "semantic":
        position = {i: n for n, i in enumerate(ids)}
        if p["representative_rule"] == "first":
            ordered = ids
        elif p["representative_rule"] == "lowest_id":
            ordered = tuple(sorted(ids))
        else:
            ordered = tuple(
                sorted(
                    ids, key=lambda i: (ctx.view.record(i).stats.supervised_tokens, i)
                )
            )
        kept, reasons, comparisons, missing_pairs = [], {}, 0, []
        for data_id in ordered:
            rejected = False
            for representative in kept:
                comparisons += 1
                score = (
                    None
                    if ctx.similarities is None
                    else ctx.similarities.lookup_bound(
                        data_id,
                        ctx._proofs[data_id][0],
                        representative,
                        ctx._proofs[representative][0],
                    )
                )
                if score is None:
                    missing_pairs.append((data_id, representative))
                    if not _missing(
                        p["missing_policy"], node, data_id, "semantic_similarity"
                    ):
                        rejected = True
                        reasons[data_id] = "MISSING_SIMILARITY"
                        break
                elif score >= p["threshold"]:
                    rejected = True
                    reasons[data_id] = "SEMANTIC_DUPLICATE:" + representative
                    break
            if not rejected:
                kept.append(data_id)
        return _result(
            ids,
            sorted(kept, key=position.get),
            reasons,
            {
                "comparisons": comparisons,
                "missing_pairs": missing_pairs,
                "similarity_identity": (
                    None
                    if ctx.similarities is None
                    else ctx.similarities.identity.to_dict()
                ),
            },
        )
    if p["mode"] == "exact":
        groups = defaultdict(list)
        for i in ids:
            groups[ctx._proofs[i][0]].append(i)
        keep = _representatives(ids, groups.values(), p["representative_rule"], ctx)
        return _result(ids, keep, {i: "EXACT_DUPLICATE" for i in ids})
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    group_owner, eligible, reasons = {}, [], {}
    labels = ctx.view.static_batch(ids, ("duplicate_groups",)).column(
        "duplicate_groups"
    )
    for data_id, memberships in zip(ids, labels):
        if memberships is None:
            if not _missing(p["missing_policy"], node, data_id, "duplicate_groups"):
                reasons[data_id] = "MISSING_DUPLICATE_GROUP"
                continue
            memberships = ()
        eligible.append(data_id)
        for group in memberships:
            if group in group_owner:
                left, right = root(data_id), root(group_owner[group])
                parent[max(left, right)] = min(left, right)
            else:
                group_owner[group] = data_id
    groups = defaultdict(list)
    for data_id in eligible:
        groups[root(data_id)].append(data_id)
    keep = _representatives(eligible, groups.values(), p["representative_rule"], ctx)
    reasons.update({i: reasons.get(i, "ANNOTATED_DUPLICATE") for i in ids})
    return _result(ids, keep, reasons)


def _strata(node, ids, ctx, seed):
    p, groups, excluded = node.params, defaultdict(list), {}
    batch = ctx.view.static_batch(ids, tuple(p["strata"]))
    for index, data_id in enumerate(ids):
        values = [batch.column(name)[index] for name in p["strata"]]
        if any(v is None for v in values) and not _missing(
            p["missing_policy"], node, data_id, "strata"
        ):
            excluded[data_id] = "MISSING_STRATUM"
            continue
        groups[canonical_json(values)].append(data_id)
    return {
        name: stable_order(items, seed) for name, items in sorted(groups.items())
    }, excluded


def _token_take(ids, target, ctx):
    chosen, used = [], 0
    for i in ids:
        length = ctx.view.record(i).stats.supervised_tokens
        if used + length <= target:
            chosen.append(i)
            used += length
    return chosen


def sample_select(node, inputs, ctx, seed):
    ids, p = inputs[0].ids, node.params
    chosen, reasons, groups = [], {}, {}
    seed = digest({"execution_seed": seed, "node_seed": p["seed"]})
    if p["method"] == "stratified":
        groups, reasons = _strata(node, ids, ctx, seed)
        available = sum(len(g) for g in groups.values())
        if p["size"] is not None and p["size"] > available:
            return _result(
                ids,
                (),
                reasons,
                problem={
                    "reason": "SAMPLE_COUNT_SHORTFALL",
                    "requested": p["size"],
                    "available": available,
                },
            )
        if p["size"] is not None:
            size = p["size"]
            quota = (
                {k: size * len(v) // available for k, v in groups.items()}
                if available
                else {}
            )
            remainder = sorted(
                groups, key=lambda k: (-(size * len(groups[k]) % available), k)
            )
            for key in remainder[: size - sum(quota.values())]:
                quota[key] += 1
            chosen = [i for key, group in groups.items() for i in group[: quota[key]]]
            chosen = list(stable_order(chosen, seed))
        else:
            queues = {k: deque(v) for k, v in groups.items()}
            weights = {
                k: sum(ctx.view.record(i).stats.supervised_tokens for i in v)
                for k, v in groups.items()
            }
            used_by_group = {k: 0 for k in groups}
            heap = [(0.0, k) for k in groups]
            heapq.heapify(heap)
            used = 0
            while heap:
                _, key = heapq.heappop(heap)
                data_id = queues[key].popleft()
                length = ctx.view.record(data_id).stats.supervised_tokens
                if used + length <= p["token_target"]:
                    chosen.append(data_id)
                    used += length
                    used_by_group[key] += length
                if queues[key]:
                    heapq.heappush(heap, (used_by_group[key] / weights[key], key))
    else:
        ordered = ids if p["method"] == "ranked" else stable_order(ids, seed)
        if p["size"] is not None:
            if len(ids) < p["size"]:
                return _result(
                    ids,
                    (),
                    problem={
                        "reason": "SAMPLE_COUNT_SHORTFALL",
                        "requested": p["size"],
                        "available": len(ids),
                    },
                )
            chosen = list(ordered[: p["size"]])
        else:
            chosen = _token_take(ordered, p["token_target"], ctx)
    reasons.update({i: reasons.get(i, "NOT_SAMPLED") for i in ids})
    selected_tokens = sum(ctx.view.record(i).stats.supervised_tokens for i in chosen)
    chosen_set = set(chosen)
    return _result(
        ids,
        chosen,
        reasons,
        {
            "supervised_tokens": selected_tokens,
            "token_target_shortfall": (
                None
                if p["token_target"] is None
                else p["token_target"] - selected_tokens
            ),
            "strata": [
                (key, len(group), sum(i in chosen_set for i in group))
                for key, group in groups.items()
            ],
        },
    )


def mix_allocate(node, inputs, ctx, seed):
    p, input_by_name = node.params, dict(zip(node.inputs, inputs))
    priority = (
        node.inputs
        if p["overlap_rule"] == "fixed_priority"
        else stable_order(node.inputs, digest({"seed": seed, "mix_seed": p["seed"]}))
    )
    owners, branches, overlap = {}, {}, []
    for name in priority:
        branch = []
        for data_id in input_by_name[name].ids:
            if data_id in owners:
                overlap.append((data_id, name, owners[data_id]))
                continue
            owners[data_id] = name
            branch.append(data_id)
        branches[name] = branch
    selected, selected_owners, deficits, branch_stats = [], {}, [], []
    for name in node.inputs:
        target, lower, upper = ctx.budget.branch_window(p["branch_token_shares"][name])
        chosen, used = [], 0
        for data_id in branches[name]:
            if used >= target:
                break
            length = ctx.view.record(data_id).stats.supervised_tokens
            if used + length <= upper:
                chosen.append(data_id)
                used += length
        if used < lower or not chosen:
            deficits.append(
                {
                    "branch": name,
                    "selected_tokens": used,
                    "minimum_tokens": lower,
                    "available_tokens": sum(
                        ctx.view.record(i).stats.supervised_tokens
                        for i in branches[name]
                    ),
                }
            )
        selected.extend(chosen)
        selected_owners.update({i: name for i in chosen})
        branch_stats.append((name, float(target), used))
    all_ids = tuple(dict.fromkeys(i for values in inputs for i in values.ids))
    selected_set = set(selected)
    exclusions = tuple(
        (i, "BRANCH_BUDGET", owners.get(i)) for i in all_ids if i not in selected_set
    )
    candidate = Candidates(
        tuple(selected),
        tuple((i, selected_owners[i]) for i in selected),
        tuple(sorted(p["branch_token_shares"].items())),
    )
    return OperatorResult(
        candidate,
        exclusions,
        {"overlaps": overlap, "branch_tokens": branch_stats},
        (
            {"reason": "BRANCH_QUOTA_SHORTFALL", "branches": deficits}
            if deficits
            else None
        ),
    )


HANDLERS = {
    "SkillSelect": skill_select,
    "EvidenceSelect": evidence_select,
    "NLLSelect": score_select,
    "EntropySelect": score_select,
    "LengthSelect": length_select,
    "DedupSelect": dedup_select,
    "SampleSelect": sample_select,
    "MixAllocate": mix_allocate,
}
