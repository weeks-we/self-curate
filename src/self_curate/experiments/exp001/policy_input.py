"""Initial policy input grounded in current code contracts and actual M0 labels."""

import math

from self_curate.contracts import parse_program
from self_curate.contracts.program import normalize_params, OPERATOR_VERSION


def policy_catalog(*, skills, count, seed):
    raw = {
        "SkillSelect": {"skills": [skills[0]], "match": "any"},
        "EvidenceSelect": {"target_skill": skills[0], "min_tokens": 1},
        "NLLSelect": {
            "scope": "assistant",
            "scale": "raw",
            "keep_fraction": 1.0,
            "rank_direction": "ascending",
        },
        "EntropySelect": {
            "scope": "assistant",
            "scale": "raw",
            "keep_fraction": 1.0,
            "rank_direction": "descending",
        },
        "LengthSelect": {"supervised_token_range": [1, None]},
        "DedupSelect": {"mode": "exact", "representative_rule": "lowest_id"},
        "SampleSelect": {"method": "ranked", "size": count, "seed": seed},
        "MixAllocate": {
            "branch_token_shares": {"left": 0.5, "right": 0.5},
            "seed": seed,
        },
    }
    params = {op: normalize_params(op, value) for op, value in raw.items()}
    catalog = {
        op: {
            "inputs": ["left", "right"] if op == "MixAllocate" else ["eligible_pool"],
            "params_template": value,
        }
        for op, value in params.items()
    }
    examples = []
    for op in ("NLLSelect", "EntropySelect"):
        example = {
            "schema_version": "1",
            "program_id": "initial-schema-example-" + op,
            "parent_id": None,
            "seed": seed,
            "nodes": [
                {
                    "id": "rank",
                    "op": op,
                    "inputs": ["eligible_pool"],
                    "params": params[op],
                },
                {
                    "id": "sample",
                    "op": "SampleSelect",
                    "inputs": ["rank"],
                    "params": params["SampleSelect"],
                },
            ],
            "output": "sample",
        }
        examples.append(parse_program(example).to_dict())
    return {
        "operator_version": OPERATOR_VERSION,
        "operators": catalog,
        "program_examples": examples,
        "rules": [
            "JSON only; every node has id/op/inputs/params; source is eligible_pool.",
            "Unknown keys/operators are rejected. No Python, arbitrary code or scoring tools.",
            "Use only supplied skill IDs and raw assistant metrics; no registered CDF or semantic similarity table.",
            "Exactly one of interval/keep_fraction for NLLSelect/EntropySelect.",
            "SampleSelect method is uniform/ranked/stratified; exactly one of size/token_target.",
            "End with ranked SampleSelect size=count, and use the single supplied seed everywhere.",
            "If unsure a hard filter retains count candidates, prefer a broad ranking without that filter.",
            "Examples specify valid shape, not a measured winning policy.",
        ],
    }


def pool_summary(records, signals, skills):
    def quantiles(values):
        ordered = sorted(values)
        if not ordered:
            return None
        return {
            str(q): ordered[int((len(ordered) - 1) * q)]
            for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
        }

    def summarize(indices):
        return {
            "rows": len(indices),
            "supervised_tokens": sum(
                records[i].stats.supervised_tokens for i in indices
            ),
            **{
                name: {
                    "mean": (
                        None
                        if not indices
                        else math.fsum(
                            getattr(signals[i], name) / len(indices) for i in indices
                        )
                    ),
                    "empirical_quantiles": quantiles(
                        [getattr(signals[i], name) for i in indices]
                    ),
                }
                for name in ("nll", "entropy")
            },
        }

    labels = [
        {s.skill for s in r.annotations.skills} if r.annotations is not None else set()
        for r in records
    ]
    return {
        "all_candidates": summarize(list(range(len(records)))),
        "by_skill": {
            skill: summarize([i for i, tags in enumerate(labels) if skill in tags])
            for skill in skills
        },
        "multi_label_counts_overlap": True,
        "quantiles_are_descriptive_not_operator_CDF": True,
    }
