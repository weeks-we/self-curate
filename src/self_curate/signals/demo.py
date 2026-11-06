"""Explicit synthetic logits -> actual statistics -> M1 pool -> raw operators."""

from pathlib import Path

from self_curate.contracts import (
    ArtifactRef,
    ModelRef,
    ResourceUsage,
    ComputeCost,
    TokenizerRef,
    canonical_json,
    digest,
    parse_program,
)
from self_curate.contracts.types import require
from self_curate.data import (
    ImportConfig,
    build_pool,
    read_pool,
    publish_signal_enrichment,
)
from self_curate.data.pool import file_sha256, write_json
from self_curate.operators import (
    ExecutionContext,
    QualityPolicy,
    BudgetPolicy,
    SignalContext,
    execute_program,
)
from .contracts import LogitsOutput
from .backends import ProvidedLogitsBackend, local_artifact
from .service import SignalService
from .io import load_scoring_batch


def _ref(name):
    return ArtifactRef("fixture://signals-demo/" + name, digest(name))


class FixtureLogitsProvider:
    implementation_hash = digest("explicit-synthetic-raw-logits-v1")

    def __init__(self, model):
        self.model_ref, self.calls = model, 0

    def forward(self, request):
        import numpy as np

        require(request.backend == "mock", "Fixture logits must remain mock")
        self.calls += 1
        values = np.zeros((len(request.positions), 16), dtype=np.float64)
        for i, p in enumerate(request.positions):
            values[i, request.input_ids[p]] = 1 + request.input_ids[p] / 8
        return LogitsOutput(
            request.record_hash,
            self.model_ref,
            request.positions,
            values,
            _ref("declared-logits-" + request.record_hash),
            ResourceUsage(
                ComputeCost(0, 0, 0),
                scoring_tokens=0,
                new_gradient_samples=0,
                new_gradient_tokens=0,
                training_supervised_tokens=0,
            ),
        )


def run_signals_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = TokenizerRef(
        "explicit-toy-tokens",
        "1",
        digest("toy-tokenizer"),
        "toy-template",
        digest("toy-template"),
        "precomputed",
    )
    rows = []
    for i in range(3):
        rows.append(
            {
                "schema_version": "1",
                "data_id": "demo-" + str(i),
                "source": {
                    "source_id": "explicit-synthetic",
                    "revision": "1",
                    "record_id": str(i),
                    "artifact": _ref(str(i)).to_dict(),
                },
                "messages": [
                    {"role": "user", "content": "prompt-" + str(i)},
                    {"role": "assistant", "content": "answer-" + str(i)},
                ],
                "quality": {
                    "status": "pass",
                    "method": "manual",
                    "verifier_id": "fixture",
                    "evidence": _ref("manual").to_dict(),
                },
                "tokenization": {
                    "identity": tokenizer.to_dict(),
                    "input_ids": [0, 1, 2, 3, 4 + i, 7 + i, 0],
                    "assistant_mask": [False, False, False, False, True, True, False],
                    "message_spans": [{"start": 1, "end": 3}, {"start": 4, "end": 6}],
                },
                "annotations": {
                    "tokenizer_hash": tokenizer.tokenizer_hash,
                    "template_hash": tokenizer.template_hash,
                    "skills": [
                        {
                            "skill": "recovery",
                            "annotator_id": "fixture",
                            "spans": [{"message_index": 1, "start": 4, "end": 5}],
                        }
                    ],
                },
            }
        )
    input_path = output / "input.jsonl"
    input_path.write_text(
        "".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    build_pool(
        ImportConfig(
            input_path,
            "precomputed",
            ("recovery",),
            100,
            "error",
            "allow",
            tokenizer=tokenizer,
        ),
        output / "original",
    )
    original = read_pool(output / "original")
    before = {
        p.name: file_sha256(p) for p in (output / "original").iterdir() if p.is_file()
    }
    model = ModelRef(
        "synthetic-logits-only",
        "demo-v1",
        digest("no-real-model-weights"),
        tokenizer,
        "fixture",
    )
    provider = FixtureLogitsProvider(model)
    scorer = SignalService(
        model=model,
        backend=ProvidedLogitsBackend(provider, backend="mock", vocab_size=16),
        output=output / "scores",
    )
    batch = scorer.score(
        original.records,
        specs=(scorer.spec(), scorer.spec(scope="skill_evidence", skill="recovery")),
    )
    folder = next((output / "scores").glob("score-*"))
    restored = load_scoring_batch(
        folder, expected_model=model, expected_computation_hash=scorer.computation_hash
    )
    require(restored == batch, "Signal handoff did not roundtrip")
    publish_signal_enrichment(
        output / "original",
        batch.records,
        output / "enriched",
        receipt=local_artifact(folder / "result.json"),
    )
    pool = read_pool(output / "enriched")
    ctx = ExecutionContext(
        pool.metric_view(),
        QualityPolicy(),
        BudgetPolicy(6, 0, 1, 1),
        model,
        (SignalContext(scorer.spec()),),
    )
    program = parse_program(
        {
            "schema_version": "1",
            "program_id": "signal-consumption",
            "seed": 1,
            "nodes": [
                {
                    "id": "nll",
                    "op": "NLLSelect",
                    "inputs": ["eligible_pool"],
                    "params": {"scale": "raw", "interval": [0, 20]},
                },
                {
                    "id": "entropy",
                    "op": "EntropySelect",
                    "inputs": ["nll"],
                    "params": {"scale": "raw", "interval": [0, 20]},
                },
            ],
            "output": "entropy",
        }
    )
    selected = execute_program(program, ctx)
    require(
        selected.status == "feasible" and len(selected.batch.data_ids) == 3,
        "Raw operators did not consume M1 signal labels",
    )
    require(
        before
        == {
            p.name: file_sha256(p)
            for p in (output / "original").iterdir()
            if p.is_file()
        },
        "Original pool was modified",
    )
    summary = {
        "schema_version": "1",
        "backend": "mock",
        "records": len(batch.records),
        "signal_records": len(batch.outcomes),
        "producer_calls": provider.calls,
        "model_hash": model.model_hash,
        "computation_hash": scorer.computation_hash,
        "raw_operator_consumption": True,
        "roundtrip_verified": True,
        "original_pool_preserved": True,
        "actual_model_forward": False,
        "actual_statistic_computation": True,
        "gpu_used": False,
        "improvement_verified": False,
        "values": [
            {"data_id": o.data_id, "nll": o.signal.nll, "entropy": o.signal.entropy}
            for o in batch.outcomes
            if o.spec.scope == "assistant"
        ],
    }
    write_json(output / "summary.json", summary)
    return summary
