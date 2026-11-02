from helpers import ref
from self_curate.contracts import (
    ArtifactRef,
    ModelRef,
    PercentileReference,
    SignalSpec,
    canonical_json,
    digest,
    parse_program,
)
from self_curate.contracts.parsing import message
from self_curate.data import (
    FixtureTokenizer,
    MetricView,
    make_signal,
    normalize_trajectory,
)
from self_curate.operators import (
    BudgetPolicy,
    ExecutionContext,
    QualityPolicy,
    SignalContext,
)


MODEL = ModelRef(
    "fixture-only",
    "v1",
    digest("fixture-weights"),
    FixtureTokenizer.identity,
    "fixture",
)
SCORE = SignalSpec(
    "assistant", "token_mean", "e", "fixture-only", digest("fixture-forward")
)
CDF = PercentileReference(
    "coding",
    digest(["fixed-reference"]),
    "v1",
    ArtifactRef(**ref("cdf")),
    MODEL.model_hash,
    SCORE.spec_hash,
)


def raw_rows(n=12):
    rows = []
    for i in range(n):
        data_id = f"data-{i:02d}"
        raw = {
            "schema_version": "1",
            "data_id": data_id,
            "source": {
                "source_id": "synthetic-coding",
                "revision": "1",
                "record_id": data_id,
                "artifact": ref(data_id),
                "family_ids": [f"family-{i%3}"],
            },
            "messages": [
                {"role": "user", "content": "Fix this synthetic task"},
                {"role": "assistant", "content": f"fix-{i:04d}"},
            ],
            "quality": {
                "status": "pass",
                "method": "executable",
                "verifier_id": "fixture-only",
                "evidence": ref("fixture-verdict"),
            },
            "metadata": {
                "task_type": "bugfix",
                "repository": f"repo-{i%2}",
                "duplicate_groups": [],
            },
        }
        encoded = FixtureTokenizer().encode(tuple(message(m) for m in raw["messages"]))
        start = encoded.message_spans[1].start
        skill = "recovery" if i % 2 == 0 else "regression"
        raw["annotations"] = {
            "tokenizer_hash": encoded.identity.tokenizer_hash,
            "template_hash": encoded.identity.template_hash,
            "skills": [
                {
                    "skill": skill,
                    "annotator_id": "fixture-only",
                    "spans": [{"message_index": 1, "start": start, "end": start + 4}],
                    "complete_events": 1,
                    "behaviors": ["test_after_final_edit"],
                }
            ],
        }
        record = normalize_trajectory(
            raw, FixtureTokenizer(), ("recovery", "regression")
        )
        receipt = ArtifactRef(**ref("external-fixture-score"))
        score = make_signal(
            record,
            MODEL,
            SCORE,
            provenance=(receipt,),
            nll=float(i),
            entropy=float(n - i),
        )
        percentile = make_signal(
            record,
            MODEL,
            SCORE,
            provenance=(receipt,),
            nll=float(i),
            entropy=float(n - i),
            nll_percentile=i / max(1, n - 1),
            entropy_percentile=(n - 1 - i) / max(1, n - 1),
            percentile_reference=CDF,
        )
        raw["signals"] = [score.to_dict(), percentile.to_dict()]
        rows.append(raw)
    return rows


def records(n=12):
    import json

    return tuple(
        normalize_trajectory(
            json.loads(canonical_json(raw)),
            FixtureTokenizer(),
            ("recovery", "regression"),
        )
        for raw in raw_rows(n)
    )


def context(
    rows,
    *,
    target=None,
    tolerance=0.99,
    batch=1,
    share_tolerance=0.1,
    quality=None,
    model=MODEL,
    signals=None,
    similarities=None,
    **extra,
):
    return ExecutionContext(
        MetricView(rows),
        quality or QualityPolicy(),
        BudgetPolicy(
            sum(r.stats.supervised_tokens for r in rows) if target is None else target,
            tolerance,
            batch,
            share_tolerance,
        ),
        model,
        (SignalContext(SCORE, (CDF,)),) if signals is None else signals,
        similarities,
        training_seed=41,
        **extra,
    )


def node(node_id, op, params, inputs=("eligible_pool",)):
    return {"id": node_id, "op": op, "params": params, "inputs": list(inputs)}


def program(nodes, output=None, seed=11):
    return parse_program(
        {
            "schema_version": "1",
            "program_id": "fixture",
            "nodes": nodes,
            "output": output or nodes[-1]["id"],
            "seed": seed,
        }
    )


def fixture_program():
    return program(
        [
            node(
                "dedup",
                "DedupSelect",
                {"mode": "exact", "representative_rule": "lowest_id"},
            ),
            node(
                "length", "LengthSelect", {"supervised_token_range": [8, 8]}, ("dedup",)
            ),
            node("recovery", "SkillSelect", {"skills": ["recovery"]}, ("length",)),
            node(
                "evidence",
                "EvidenceSelect",
                {
                    "target_skill": "recovery",
                    "min_tokens": 4,
                    "min_events": 1,
                    "required_behaviors": ["test_after_final_edit"],
                },
                ("recovery",),
            ),
            node(
                "nll",
                "NLLSelect",
                {"interval": [0, 100], "rank_direction": "ascending"},
                ("evidence",),
            ),
            node("a", "SampleSelect", {"method": "ranked", "size": 4}, ("nll",)),
            node("regression", "SkillSelect", {"skills": ["regression"]}, ("length",)),
            node(
                "entropy",
                "EntropySelect",
                {
                    "scale": "percentile",
                    "reference_group": "coding",
                    "interval": [0, 1],
                    "rank_direction": "ascending",
                },
                ("regression",),
            ),
            node(
                "b",
                "SampleSelect",
                {
                    "method": "stratified",
                    "size": 4,
                    "strata": ["repository"],
                    "seed": 22,
                },
                ("entropy",),
            ),
            node(
                "mix",
                "MixAllocate",
                {"branch_token_shares": {"a": 0.5, "b": 0.5}},
                ("a", "b"),
            ),
        ]
    )
