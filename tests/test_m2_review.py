"""Independent reference and boundary regression cases from the second review."""

from dataclasses import replace
from types import SimpleNamespace
import json
import random

import pytest

from helpers import ref
from helpers_m2 import MODEL, context, fixture_program, node, program, records
from self_curate.contracts import (
    ArtifactRef,
    ContractError,
    ErrorCode,
    TrainingUse,
    canonical_json,
    digest,
    verify_batch,
)
from self_curate.data import MetricView, attach_annotations, record_candidate_training
from self_curate.operators import (
    BudgetPolicy,
    ExecutionContext,
    QualityPolicy,
    execute_program,
    parse_context,
)
from self_curate.operators.handlers import Candidates
from self_curate.operators.runtime import budget_pack


class SyntheticView:
    """Explicit cheap rows for a packer-only numerical reference (no real scores)."""

    def __init__(self, lengths):
        self.rows = {
            i: SimpleNamespace(
                stats=SimpleNamespace(supervised_tokens=n), annotations=None
            )
            for i, n in lengths.items()
        }

    def record(self, data_id):
        return self.rows[data_id]

    def static_batch(self, ids, fields):
        return SimpleNamespace(column=lambda _: (False,) * len(ids))


def synthetic_context(lengths, policy):
    return SimpleNamespace(
        view=SyntheticView(lengths),
        budget=policy,
        _proofs={i: (digest(i), "", "", n) for i, n in lengths.items()},
    )


def reference_pack(ids, lengths, assignments, shares, policy):
    """Slow independent enumeration of each possible one-row removal."""
    selected = []
    for data_id in ids:
        if policy.max_samples is not None and len(selected) >= policy.max_samples:
            continue
        if sum(lengths[i] for i in selected) + lengths[data_id] <= policy.maximum:
            selected.append(data_id)
    while len(selected) % policy.effective_batch:
        alternatives = []
        for index in range(len(selected)):
            remaining = selected[:index] + selected[index + 1 :]
            total = sum(lengths[i] for i in remaining)
            totals = {
                b: sum(lengths[i] for i in remaining if assignments[i] == b)
                for b in shares
            }
            missing = sum(n == 0 for n in totals.values()) if total else 0
            error = (
                sum(abs(totals[b] / total - share) for b, share in shares.items())
                if total
                else 0
            )
            alternatives.append(
                ((total < policy.minimum, missing, error, -index), remaining)
            )
        selected = min(alternatives, key=lambda item: item[0])[1]
    return tuple(selected)


@pytest.mark.parametrize("branches", [0, 2, 3])
def test_packer_matches_independent_reference_for_variable_lengths_and_trimming(
    branches,
):
    rng = random.Random(147)
    shares = (
        {}
        if not branches
        else ({"a": 0.5, "b": 0.5} if branches == 2 else {"a": 0.2, "b": 0.3, "c": 0.5})
    )
    for _ in range(80):
        lengths = {
            f"r{i:03}": rng.randrange(1, 30) for i in range(rng.randrange(1, 36))
        }
        ids = tuple(lengths)
        assignments = (
            {i: tuple(shares)[k % branches] for k, i in enumerate(ids)}
            if branches
            else {}
        )
        policy = BudgetPolicy(
            rng.randrange(1, sum(lengths.values()) + 1),
            0.25,
            rng.randrange(1, 9),
            0.2,
            deduplicate_content=False,
        )
        candidate = Candidates(ids, tuple(assignments.items()), tuple(shares.items()))
        actual, _, report = budget_pack(candidate, synthetic_context(lengths, policy))
        assert actual.ids == reference_pack(ids, lengths, assignments, shares, policy)
        assert report["supervised_tokens"] == sum(lengths[i] for i in actual.ids)
        assert len(actual.ids) % policy.effective_batch == 0
        if not report["violations"]:
            assert policy.minimum <= report["supervised_tokens"] <= policy.maximum


def test_reuse_report_counts_final_rows_and_keeps_scoring_separate_from_training():
    rows = list(records(4))
    receipt = ArtifactRef(**ref("review-training"))
    rows[0] = record_candidate_training(
        rows[0], TrainingUse("reject", "m", receipt, "rejected")
    )
    rows[1] = record_candidate_training(
        rows[1], TrainingUse("accept", "m", receipt, "accepted")
    )
    rows[2] = replace(
        rows[2],
        usage=replace(
            rows[2].usage, forward_scoring=(receipt,), gradient_extraction=(receipt,)
        ),
    )
    initial = tuple(row.record_hash for row in rows)
    ctx = context(
        rows,
        target=32,
        tolerance=0,
        quality=QualityPolicy(exclude_accepted=False, rejected_policy="allow"),
    )
    recipe = program(
        [node("all", "LengthSelect", {"supervised_token_range": [1, None]})]
    )
    result = execute_program(recipe, ctx)
    report = result.traces[-1].details["usage_exposure"]
    rejected = report["used_rejected_sft"]
    assert rejected == {
        "data_ids": ["data-00"],
        "sample_count": 1,
        "sample_fraction": 0.25,
        "supervised_tokens": 8,
        "token_fraction": 0.25,
    }
    assert report["used_accepted_sft"]["data_ids"] == ["data-01"]
    assert report["used_candidate_sft"]["sample_count"] == 2
    assert report["used_forward_scoring"]["sample_count"] == 4
    assert report["used_gradient_extraction"]["data_ids"] == ["data-02"]
    assert report["pending_sft"]["sample_count"] == 0
    assert tuple(row.record_hash for row in rows) == initial
    # Only final packed members count, not all gate-eligible rows.
    trimmed = execute_program(recipe, replace(ctx, budget=BudgetPolicy(16, 0, 2)))
    assert (
        trimmed.traces[-1].details["usage_exposure"]["used_gradient_extraction"][
            "sample_count"
        ]
        == 0
    )


def test_one_context_cannot_mix_tokenizers_even_without_model_scores():
    rows = list(records(2))
    rows = [attach_annotations(replace(r, signals=()), None) for r in rows]
    changed = replace(
        rows[1].tokenization.identity, tokenizer_hash=digest("other-tokenizer")
    )
    rows[1] = replace(
        rows[1], tokenization=replace(rows[1].tokenization, identity=changed)
    )
    with pytest.raises(ContractError) as exc:
        ExecutionContext(MetricView(rows), QualityPolicy(), BudgetPolicy(16, 0, 2))
    assert exc.value.code == ErrorCode.TOKENIZATION_MISMATCH


def test_handoff_checks_expected_model_and_old_operator_context_is_rejected():
    ctx = context(records(), target=64, tolerance=0, batch=8, share_tolerance=0)
    batch = execute_program(fixture_program(), ctx).batch
    assert verify_batch(batch, ctx.view, model_ref=MODEL) == batch.training_order
    with pytest.raises(ContractError) as exc:
        verify_batch(
            batch,
            ctx.view,
            model_ref=replace(MODEL, weights_hash=digest("next-checkpoint")),
        )
    assert exc.value.code == ErrorCode.STALE_CACHE
    value = json.loads(canonical_json(ctx.to_dict()))
    value["operator_version"] = "m2-v1"
    with pytest.raises(ContractError) as exc:
        parse_context(value, ctx.view)
    assert exc.value.code == ErrorCode.STALE_CACHE


def test_budget_rounding_uses_declared_decimal_boundaries_without_hidden_slack():
    ordinary = BudgetPolicy(100, 0.07, 1, 0)
    assert (ordinary.minimum, ordinary.maximum) == (93, 107)
    narrow = BudgetPolicy(100, 0.00999999999999999, 1, 0)
    assert (narrow.minimum, narrow.maximum) == (100, 100)
    assert narrow.branch_window(0.5)[1:] == (50, 50)
    large = BudgetPolicy(2**53 - 1, 0, 1)
    assert large.minimum == large.maximum == 2**53 - 1
    with pytest.raises(ContractError):
        BudgetPolicy(10**400, 0, 1)
    policy = BudgetPolicy(16, 0, 1, 0, deduplicate_content=False)
    lengths = {"x": 8, "y": 8}
    candidate = Candidates(
        ("x", "y"),
        (("x", "a"), ("y", "b")),
        (("a", 0.5000000000000001), ("b", 0.4999999999999999)),
    )
    _, _, report = budget_pack(candidate, synthetic_context(lengths, policy))
    assert report["violations"] == ["BRANCH_SHARE:a", "BRANCH_SHARE:b"]


def test_ranking_truncation_has_real_order_effects():
    rows = records(6)
    scores = {"keep_fraction": 0.5, "rank_direction": "ascending"}
    skill = {"skills": ["regression"]}
    first = program(
        [
            node("score", "NLLSelect", scores),
            node("skill", "SkillSelect", skill, ("score",)),
        ]
    )
    second = program(
        [
            node("skill", "SkillSelect", skill),
            node("score", "NLLSelect", scores, ("skill",)),
        ]
    )
    ctx = context(rows)
    assert execute_program(first, ctx).batch.data_ids == ("data-01",)
    assert execute_program(second, ctx).batch.data_ids == ("data-01", "data-03")


@pytest.mark.parametrize(
    "operator,metric", [("NLLSelect", "nll"), ("EntropySelect", "entropy")]
)
def test_skill_scope_reads_only_its_bound_scores_not_assistant_scores(operator, metric):
    from self_curate.contracts import SignalSpec
    from self_curate.data import make_signal, attach_signals
    from self_curate.operators import SignalContext

    rows = records(6)[::2]
    skill_spec = SignalSpec(
        "skill_evidence",
        "token_mean",
        "e",
        "review-fixture",
        digest("skill-score"),
        "recovery",
    )
    for i in (0, 1):
        row = rows[i]
        value = make_signal(
            row,
            MODEL,
            skill_spec,
            provenance=(ArtifactRef(**ref("external-skill-fixture")),),
            **{metric: 20 - i},
        )
        rows = rows[:i] + (attach_signals(row, [value]),) + rows[i + 1 :]
    ctx = context(rows, signals=(SignalContext(skill_spec),))
    params = {
        "scope": "skill_evidence",
        "skill": "recovery",
        "rank_direction": "ascending",
        "missing_policy": "exclude",
    }
    recipe = program([node("score", operator, params)])
    result = execute_program(recipe, ctx)
    assert result.batch.selection_order == ("data-02", "data-00")
    score_trace = result.traces[1]
    assert score_trace.details["spec_hash"] == skill_spec.spec_hash
    assert score_trace.exclusions == (("data-04", "MISSING_SIGNAL", None),)


def test_seeded_mix_ownership_replays_and_cannot_silently_refill_short_branch():
    ctx = context(records(8), target=32, tolerance=0, batch=2, share_tolerance=0)
    nodes = [
        node("a", "SampleSelect", {"method": "ranked", "size": 6}),
        node("b", "SampleSelect", {"method": "ranked", "size": 8}),
    ]
    outcomes = set()
    for seed in range(8):
        mix = node(
            "mix",
            "MixAllocate",
            {
                "branch_token_shares": {"a": 0.5, "b": 0.5},
                "overlap_rule": "seeded_priority",
                "seed": seed,
            },
            ("a", "b"),
        )
        recipe = program([*nodes, mix])
        result = execute_program(recipe, ctx)
        assert execute_program(recipe, ctx) == result
        outcomes.add(result.status)
        if result.batch:
            assert len(result.batch.data_ids) == 4
            assert result.batch.actual_branch_tokens == {"a": 16, "b": 16}
        else:
            assert result.reason == "BRANCH_QUOTA_SHORTFALL"
            assert (
                json.loads(result.diagnostics_json)["global_infeasibility_proven"]
                is False
            )
    assert outcomes == {"feasible", "infeasible"}
