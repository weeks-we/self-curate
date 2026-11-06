"""Independent numeric oracles plus actual forward and M1/operator handoffs."""

from copy import deepcopy
from dataclasses import replace
import json
import math
import random
from types import SimpleNamespace

import pytest

from helpers import golden_record, record as raw_record, ref
from helpers_m2 import context, node, program
from test_valuation import compact_setup
from self_curate import contracts as c
from self_curate.data import (
    PrecomputedTokenizer,
    normalize_trajectory,
    get_signal,
    publish_signal_enrichment,
    read_pool,
    build_pool,
    ImportConfig,
)
from self_curate.operators import execute_program, SignalContext
from self_curate.signals import (
    SignalService,
    ProvidedLogitsBackend,
    LogitsOutput,
    ScoringFailure,
    load_scoring_batch,
    parse_forward_contract,
)
from self_curate.signals.math import token_statistics, aggregate
from self_curate.signals.torch_backend import TorchSignalBackend, _state_hash


def golden(data_id="golden"):
    raw, tokenizer = golden_record()
    raw["data_id"] = data_id
    row = normalize_trajectory(
        raw,
        PrecomputedTokenizer(tokenizer),
        ("recovery", "testing", "empty", "missing"),
    )
    model = c.ModelRef(
        "explicit-forward-fixture", "1", c.digest("weights"), tokenizer, "fixture"
    )
    return row, model


class MatrixProvider:
    implementation_hash = c.digest("explicit-full-vocab-logits-fixture-v1")

    def __init__(self, model_ref):
        self.model_ref, self.requests = model_ref, []

    def forward(self, request):
        import numpy as np

        self.requests.append(request)
        logits = np.zeros((len(request.positions), 20), dtype=np.float64)
        for j, p in enumerate(request.positions):
            logits[j, request.input_ids[p]] = (p % 4) / 2
        return LogitsOutput(
            request.record_hash,
            self.model_ref,
            request.positions,
            logits,
            c.ArtifactRef(
                "fixture://raw-logits", c.digest((request.record_hash, logits.tolist()))
            ),
            c.ResourceUsage(
                c.ComputeCost(0, 0, len(request.input_ids)),
                scoring_tokens=len(request.input_ids),
                new_gradient_samples=0,
                new_gradient_tokens=0,
                training_supervised_tokens=0,
            ),
        )


def service(tmp_path, model=None, *, provider=None, **kwargs):
    row, default_model = golden()
    provider = provider or MatrixProvider(model or default_model)
    backend = ProvidedLogitsBackend(
        provider, backend="mock", vocab_size=20, chunk_size=2
    )
    return (
        SignalService(
            model=model or default_model,
            backend=backend,
            output=tmp_path / "signals",
            **kwargs,
        ),
        provider,
        row,
    )


def independent(p, v=20):
    a = math.exp((p % 4) / 2)
    z = a + v - 1
    q = a / z
    return -math.log(q), -q * math.log(q) - (v - 1) / z * math.log(1 / z)


def test_nll_is_target_surprisal_and_entropy_is_full_distribution():
    logits = [[math.log(0.8), math.log(0.2)], [math.log(0.8), math.log(0.2)]]
    nll, entropy = token_statistics(logits, (0, 1), chunk_size=1)
    assert nll == pytest.approx((-math.log(0.8), -math.log(0.2)))
    assert entropy == pytest.approx((-0.8 * math.log(0.8) - 0.2 * math.log(0.2),) * 2)
    assert nll[0] != entropy[0] and nll[1] != entropy[1]


def test_logsumexp_extremes_are_stable_without_clipping_target_probability():
    nll, h = token_statistics([[10000.0, -10000.0], [-10000.0, 10000.0]], (0, 0))
    assert nll == (0.0, 20000.0) and h == (0.0, 0.0)


def test_logits_row_mapping_cannot_silently_drop_extra_input_rows():
    with pytest.raises(c.ContractError):
        token_statistics([[0.0, 0.0], [1.0, 0.0]], (0,))
    nll, _ = token_statistics([[0.0, 0.0], [1.0, 0.0]], (0,), rows=(1,))
    assert nll[0] == pytest.approx(math.log1p(math.exp(-1)))


@pytest.mark.parametrize(
    "logits",
    [
        [[float("nan"), 1]],
        [[float("inf"), 1]],
        [[-float("inf"), 1]],
        [[True, False]],
        [[1 + 2j, 3]],
    ],
)
def test_nonfinite_and_unsupported_logits_are_not_silently_converted(logits):
    with pytest.raises(c.ContractError):
        token_statistics(logits, (0,))


def test_mean_reduction_does_not_overflow_an_intermediate_sum(tmp_path):
    producer, _, _ = service(tmp_path)
    assert aggregate((1e308, 1e308), producer.spec()) == 1e308
    with pytest.raises(c.ContractError):
        aggregate((1e308, 1e308), producer.spec(reduction="token_sum"))


def test_global_skill_positions_union_and_one_forward_for_multiple_specs(tmp_path):
    scorer, provider, row = service(tmp_path)
    specs = (
        scorer.spec(),
        scorer.spec(scope="skill_evidence", skill="recovery"),
        scorer.spec(
            scope="skill_evidence", skill="testing", reduction="token_sum", log_base="2"
        ),
        scorer.spec(scope="skill_evidence", skill="empty"),
        scorer.spec(scope="skill_evidence", skill="missing"),
    )
    original = row.record_hash
    result = scorer.score((row,), specs=specs)
    assert len(provider.requests) == 1
    assert provider.requests[0].positions == (4, 5, 6, 7, 8, 14, 15)
    assert provider.requests[0].input_ids == row.tokenization.input_ids
    all_signal, recovery, testing = (o.signal for o in result.outcomes[:3])
    assert all_signal.nll == pytest.approx(
        sum(independent(p)[0] for p in (4, 5, 6, 7, 8, 14, 15)) / 7
    )
    assert recovery.binding.token_count == 4
    assert recovery.nll == pytest.approx(
        sum(independent(p)[0] for p in (4, 5, 6, 7)) / 4
    )
    assert testing.nll == pytest.approx(
        sum(independent(p)[0] for p in (6, 7, 8)) / math.log(2)
    )
    assert all_signal.entropy == pytest.approx(
        sum(independent(p)[1] for p in (4, 5, 6, 7, 8, 14, 15)) / 7
    )
    assert all_signal.nll_percentile is None and all_signal.entropy_percentile is None
    assert [o.status for o in result.outcomes[-2:]] == ["missing", "missing"]
    assert all(o.signal is None for o in result.outcomes[-2:])
    assert (
        result.updated_records[0].signals
        and result.updated_records[0].usage.forward_scoring
    )
    assert (
        not result.updated_records[0].usage.candidate_sft
        and not result.updated_records[0].usage.gradient_extraction
    )
    assert row.record_hash == original and not row.signals
    assert result.ledger.summary()["scoring_tokens"]["total"] == 17


def test_scope_extension_only_computes_new_positions_and_cache_rebinds_aggregates(
    tmp_path,
):
    scorer, provider, row = service(tmp_path)
    a = scorer.spec(scope="skill_evidence", skill="recovery")
    first = scorer.score((row,), specs=(a,))
    assert provider.requests[0].positions == (4, 5, 6, 7)
    second = scorer.score(first.records)
    assert provider.requests[1].positions == (8, 14, 15)
    third = scorer.score(
        second.records, specs=(scorer.spec(reduction="token_sum", log_base="2"),)
    )
    assert len(provider.requests) == 2 and not third.ledger.events
    assert third.outcomes[0].status == "cached"
    assert third.outcomes[0].signal.nll == pytest.approx(
        second.outcomes[0].signal.nll * 7 / math.log(2)
    )


def test_attached_metrics_are_reusable_by_a_new_service_without_recharging(tmp_path):
    scorer, _, row = service(tmp_path)
    result = scorer.score((row,))
    again, provider, _ = service(tmp_path / "new")
    second = again.score(result.records)
    assert not provider.requests and not second.ledger.events
    assert second.outcomes[0].signal == result.outcomes[0].signal
    assert second.outcomes[0].status == "cached"


@pytest.mark.parametrize("field", ["revision", "weights_hash", "adapter_hash"])
def test_model_versions_coexist_without_overwriting_old_signals(tmp_path, field):
    scorer, _, row = service(tmp_path)
    first = scorer.score((row,))
    model = replace(
        scorer.model_ref, **{field: "next" if field == "revision" else c.digest(field)}
    )
    next_scorer, provider, _ = service(tmp_path / "next", model)
    second = next_scorer.score(first.records)
    assert len(provider.requests) == 1 and len(second.records[0].signals) == 2
    assert (
        get_signal(second.records[0], scorer.model_ref, scorer.spec())
        == first.outcomes[0].signal
    )
    assert (
        get_signal(second.records[0], model, next_scorer.spec())
        == second.outcomes[0].signal
    )


def test_input_order_stays_id_aligned_and_operator_raw_filters_consume_tags(tmp_path):
    scorer, _, a = service(tmp_path)
    b, _ = golden("b")
    a = replace(
        a,
        quality=c.QualityCheck(
            "pass", "manual", "fixture", c.ArtifactRef(**ref("manual"))
        ),
    )
    b = replace(b, quality=a.quality)
    result = scorer.score((b, a))
    assert tuple(r.data_id for r in result.records) == ("b", "golden")
    values = result.metric_view().signal_batch(
        ("golden", "b"), scorer.model_ref, scorer.spec()
    )
    assert values.data_ids == ("golden", "b") and all(
        v is not None for v in values.column("nll")
    )
    ctx = context(
        result.records,
        model=scorer.model_ref,
        batch=1,
        signals=(SignalContext(scorer.spec()),),
    )
    spec = program(
        [
            node("nll", "NLLSelect", {"scale": "raw", "interval": [0.0, 100.0]}),
            node(
                "entropy",
                "EntropySelect",
                {"scale": "raw", "interval": [0.0, 100.0]},
                ("nll",),
            ),
        ]
    )
    selected = execute_program(spec, ctx)
    # The fixed packer deduplicates the two identical trajectory payloads.
    assert selected.status == "feasible" and set(selected.batch.data_ids) == {"b"}


def test_signal_provider_and_data_provider_interfaces_are_directly_usable(tmp_path):
    scorer, _, row = service(tmp_path)

    class Data:
        def records(self):
            yield row

    batch = scorer.score_provider(Data())
    signal = scorer.compute(batch.records[0], scorer.model_ref, scorer.spec())
    assert signal == batch.outcomes[0].signal
    with pytest.raises(c.ContractError):
        scorer.compute(row, replace(scorer.model_ref, revision="other"), scorer.spec())


def test_empty_data_and_missing_scopes_do_not_create_fake_zero_metrics(tmp_path):
    scorer, provider, row = service(tmp_path)
    assert not scorer.score(()).outcomes
    missing = scorer.spec(scope="skill_evidence", skill="missing")
    with pytest.raises(c.ContractError) as exc:
        scorer.compute(row, scorer.model_ref, missing)
    assert exc.value.code == c.ErrorCode.MISSING_SIGNAL
    assert not provider.requests


def test_input_budget_and_duplicate_ids_reject_before_any_forward(tmp_path):
    scorer, provider, row = service(tmp_path, max_input_tokens=16)
    with pytest.raises(c.ContractError):
        scorer.score((row,))
    assert not provider.requests
    other, provider2, _ = service(tmp_path / "duplicates")
    with pytest.raises(c.ContractError):
        other.score((row, row))
    assert not provider2.requests


def test_unpredictable_first_token_is_rejected_instead_of_silently_dropped(tmp_path):
    raw, tokenizer = golden_record()
    raw["messages"] = [{"role": "assistant", "content": "a"}]
    raw.pop("annotations")
    raw["tokenization"].update(
        input_ids=[1, 2],
        assistant_mask=[True, False],
        message_spans=[{"start": 0, "end": 1}],
    )
    row = normalize_trajectory(raw, PrecomputedTokenizer(tokenizer), ("recovery",))
    scorer, provider, _ = service(tmp_path)
    with pytest.raises(c.ContractError):
        scorer.score((row,))
    assert not provider.requests


@pytest.mark.parametrize("kind", ["ForwardRequest", "TokenMetricResult"])
def test_new_forward_contracts_have_strict_json_roundtrip(tmp_path, kind):
    scorer, provider, row = service(tmp_path)
    scorer.score((row,))
    path = next(
        (tmp_path / "signals").glob(
            "*/fwd-0/" + ("request.json" if kind == "ForwardRequest" else "result.json")
        )
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    obj = parse_forward_contract(kind, raw)
    assert json.loads(c.canonical_json(obj.to_dict())) == raw
    with pytest.raises(c.ContractError):
        parse_forward_contract(kind, {**raw, "unknown": 1})


def test_enriched_records_roundtrip_and_artifact_checksums(tmp_path):
    scorer, _, row = service(tmp_path)
    batch = scorer.score((row,))
    folder = next((tmp_path / "signals").glob("score-*"))
    assert (
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )
        == batch
    )
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=replace(scorer.model_ref, revision="other"),
            expected_computation_hash=scorer.computation_hash,
        )
    (folder / "records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


def test_signal_enrichment_publishes_new_m1_pool_and_preserves_original(tmp_path):
    from self_curate.data.pool import file_sha256

    raw, identity = golden_record()
    path = tmp_path / "input.jsonl"
    path.write_text(c.canonical_json(raw) + "\n", encoding="utf-8")
    build_pool(
        ImportConfig(
            path,
            "precomputed",
            ("recovery", "testing", "empty", "missing"),
            100,
            "error",
            "allow",
            tokenizer=identity,
        ),
        tmp_path / "pool",
    )
    pool = read_pool(tmp_path / "pool")
    before = {
        p.name: file_sha256(p) for p in (tmp_path / "pool").iterdir() if p.is_file()
    }
    row, model = golden()
    scorer, _, _ = service(tmp_path, model)
    batch = scorer.score(pool.records)
    folder = next((tmp_path / "signals").glob("score-*"))
    receipt = c.ArtifactRef(
        (folder / "result.json").resolve().as_uri(), file_sha256(folder / "result.json")
    )
    published = publish_signal_enrichment(
        tmp_path / "pool", batch.records, tmp_path / "enriched", receipt=receipt
    )
    enriched = read_pool(tmp_path / "enriched")
    assert (
        published["parent_pool_id"] == pool.manifest["pool_id"]
        and enriched.records == batch.records
    )
    assert enriched.metric_view().signal_batch(
        tuple(r.data_id for r in enriched.records), model, scorer.spec()
    ).column("nll") == (batch.outcomes[0].signal.nll,)
    assert before == {
        p.name: file_sha256(p) for p in (tmp_path / "pool").iterdir() if p.is_file()
    }
    with pytest.raises(c.ContractError):
        publish_signal_enrichment(
            tmp_path / "pool", batch.records, tmp_path / "enriched", receipt=receipt
        )


def test_failed_later_sample_keeps_partial_labels_and_forward_costs(tmp_path):
    row, model = golden("a")
    other, _ = golden("b")

    class FailingProvider(MatrixProvider):
        def forward(self, request):
            if request.data_id == "b":
                raise ScoringFailure(
                    "reported remote failure", usage=c.ResourceUsage(scoring_tokens=11)
                )
            return super().forward(request)

    scorer, _, _ = service(tmp_path, model, provider=FailingProvider(model))
    with pytest.raises(ScoringFailure) as error:
        scorer.score((row, other))
    assert (
        len(error.value.ledger.events) == 2
        and error.value.ledger.events[-1].usage.scoring_tokens == 11
    )
    assert error.value.records[0].signals and not error.value.records[1].signals
    assert len(error.value.outcomes) == 1


@pytest.mark.parametrize(
    "change",
    [
        "wrong_hash",
        "wrong_rows",
        "wrong_dtype",
        "wrong_model",
        "changed_live_model",
        "bad_return",
    ],
)
def test_bad_backend_or_logits_results_never_attach_labels_and_keep_usage(
    tmp_path, change
):
    row, model = golden()

    class BadProvider(MatrixProvider):
        def forward(self, request):
            response = super().forward(request)
            if change == "wrong_hash":
                return replace(response, request_hash=c.digest("wrong"))
            if change == "wrong_rows":
                return replace(response, positions=tuple(reversed(response.positions)))
            if change == "wrong_dtype":
                return replace(response, logits=response.logits.astype("float32"))
            if change == "wrong_model":
                return replace(
                    response, model_ref=replace(response.model_ref, revision="other")
                )
            if change == "changed_live_model":
                self.model_ref = replace(self.model_ref, revision="other")
                return response
            return SimpleNamespace(usage=response.usage)

    scorer, _, _ = service(tmp_path, model, provider=BadProvider(model))
    with pytest.raises(ScoringFailure) as error:
        scorer.score((row,))
    assert not error.value.records[0].signals
    assert error.value.ledger.events[0].usage.scoring_tokens == 17


@pytest.mark.parametrize("layer", ["logits_producer", "token_backend"])
def test_external_implementations_require_real_version_hashes_before_work(
    tmp_path, layer
):
    row, model = golden()
    provider = MatrixProvider(model)
    with pytest.raises(c.ContractError):
        if layer == "logits_producer":
            provider.implementation_hash = "v1-is-not-a-code-hash"
            ProvidedLogitsBackend(provider, backend="mock", vocab_size=20)
        else:
            backend = SimpleNamespace(
                backend="mock", implementation_hash="v1-is-not-a-code-hash"
            )
            SignalService(
                model=model, backend=backend, output=tmp_path / "invalid-version"
            )
    assert not provider.requests and not row.signals


def test_changed_backend_version_rejects_before_cached_values_are_used(tmp_path):
    scorer, provider, row = service(tmp_path)
    first = scorer.score((row,))
    provider.implementation_hash = c.digest("changed")
    with pytest.raises(c.ContractError):
        scorer.score(first.records)
    assert len(provider.requests) == 1


def test_actual_torch_forward_mask_shift_entropy_and_frozen_state(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    row = ctx.view.record(ctx.view.data_ids[0])
    scorer = SignalService(
        model=request.model_ref,
        backend=TorchSignalBackend(store, backend="mock", chunk_size=2),
        output=tmp_path / "torch",
    )
    original, rng, py_rng = (
        _state_hash(store.initial),
        torch.get_rng_state().clone(),
        random.getstate(),
    )
    result = scorer.score((row,))
    signal = result.outcomes[0].signal
    independent = deepcopy(store.initial)
    ids = torch.tensor([row.tokenization.input_ids])
    with torch.no_grad():
        log_p = torch.log_softmax(
            independent(ids, torch.ones_like(ids)).logits[0].double(), dim=-1
        )
    positions = [i for i, yes in enumerate(row.tokenization.assistant_mask) if yes]
    expected_nll = sum(
        -log_p[p - 1, row.tokenization.input_ids[p]].item() for p in positions
    ) / len(positions)
    expected_h = sum(
        (-log_p[p - 1].exp() * log_p[p - 1]).sum().item() for p in positions
    ) / len(positions)
    assert signal.nll == pytest.approx(expected_nll, abs=1e-10)
    assert signal.entropy == pytest.approx(expected_h, abs=1e-10)
    assert _state_hash(store.initial) == original and all(
        p.grad is None for p in store.models[0].parameters()
    )
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py_rng
    assert not store.models[0].training
    assert result.ledger.summary()["new_gradient_samples"]["total"] == 0
    assert result.ledger.summary()["training_supervised_tokens"]["total"] == 0


def test_actual_torch_scoring_ignores_ambient_autocast(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    row = ctx.view.record(ctx.view.data_ids[0])
    values = []
    for enabled in (False, True):
        scorer = SignalService(
            model=request.model_ref,
            backend=TorchSignalBackend(store, backend="mock"),
            output=tmp_path / str(enabled),
        )
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            values.append(scorer.score((row,)).outcomes[0].signal)
    assert values[0].nll == values[1].nll and values[0].entropy == values[1].entropy


def test_actual_torch_detects_weight_mutation_during_forward(tmp_path):
    torch, store, request, ctx, _, _ = compact_setup(tmp_path)
    load = store.load_branch

    def broken(model_ref):
        loaded = load(model_ref)
        forward = loaded.model.forward

        def mutate(*args, **kwargs):
            result = forward(*args, **kwargs)
            loaded.model.projection.weight.add_(0.1)
            return result

        loaded.model.forward = mutate
        return loaded

    store.load_branch = broken
    scorer = SignalService(
        model=request.model_ref,
        backend=TorchSignalBackend(store, backend="mock"),
        output=tmp_path / "mutated",
    )
    with pytest.raises(ScoringFailure):
        scorer.score((ctx.view.record(ctx.view.data_ids[0]),))


def test_tool_and_user_tokens_stay_context_and_never_enter_metric_denominators(
    tmp_path,
):
    from self_curate.data import FixtureTokenizer

    raw = raw_record(tool_output="observation" * 400)
    identity = replace(
        FixtureTokenizer.identity,
        tokenizer_id="compact-tools",
        tokenizer_hash=c.digest("compact-tools"),
    )

    class Compact:
        def encode(self, messages, provided=None):
            encoded = FixtureTokenizer().encode(messages)
            return replace(
                encoded,
                identity=identity,
                input_ids=tuple(p % 20 for p in encoded.input_ids),
            )

    Compact.identity = identity
    row = normalize_trajectory(raw, Compact(), ("recovery",))
    model = c.ModelRef("tool-fixture", "1", c.digest("weights"), identity, "fixture")
    scorer, provider, _ = service(tmp_path, model)
    result = scorer.score((row,))
    request = provider.requests[0]
    assert request.input_ids == row.tokenization.input_ids
    assert request.positions == tuple(
        i for i, value in enumerate(row.tokenization.assistant_mask) if value
    )
    tool = row.tokenization.message_spans[3]
    assert not set(range(tool.start, tool.end)).intersection(request.positions)
    assert result.outcomes[0].signal.binding.token_count == row.stats.supervised_tokens
    assert len(request.input_ids) > 10 * len(request.positions)


def test_changed_skill_mask_reuses_tokens_but_gets_a_new_exact_binding(tmp_path):
    from self_curate.data import attach_annotations

    scorer, provider, row = service(tmp_path)
    spec = scorer.spec(scope="skill_evidence", skill="recovery")
    first = scorer.score((row,), specs=(spec,))
    changed_skill = replace(row.annotations.skills[0], spans=(c.EvidenceSpan(1, 6, 9),))
    updated = attach_annotations(
        row,
        replace(row.annotations, skills=(changed_skill, *row.annotations.skills[1:])),
        declared_skills=("recovery", "testing", "empty", "missing"),
    )
    second = scorer.score((updated,), specs=(spec,))
    assert provider.requests[1].positions == (8,)
    assert second.outcomes[0].signal.binding.token_count == 3
    assert first.outcomes[0].signal.binding != second.outcomes[0].signal.binding
    assert second.outcomes[0].signal.nll == pytest.approx(
        sum(independent(p)[0] for p in (6, 7, 8)) / 3
    )


def test_handshake_loader_rejects_dropped_outcomes_even_with_recomputed_checksums(
    tmp_path,
):
    from self_curate.data.pool import file_sha256, write_json

    scorer, _, row = service(tmp_path)
    scorer.score((row,))
    folder = next((tmp_path / "signals").glob("score-*"))
    raw = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    raw["outcomes"] = []
    write_json(folder / "result.json", raw)
    checks = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
    checks["result.json"] = file_sha256(folder / "result.json")
    write_json(folder / "checksums.json", checks)
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


def test_cli_demo_covers_labels_pool_reload_and_both_raw_operators(tmp_path, capsys):
    from self_curate.cli import main

    assert main(["signals-demo", "--output", str(tmp_path / "demo")]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["records"] == value["producer_calls"] == 3
    assert (
        value["raw_operator_consumption"]
        and value["original_pool_preserved"]
        and value["roundtrip_verified"]
    )
    assert not value["actual_model_forward"] and value["actual_statistic_computation"]


def test_known_mock_source_cannot_be_scored_as_real(tmp_path):
    row, model = golden()
    provider = MatrixProvider(model)
    scorer = SignalService(
        model=model,
        backend=ProvidedLogitsBackend(provider, backend="real", vocab_size=20),
        output=tmp_path / "not-real",
    )
    with pytest.raises(c.ContractError):
        scorer.score((row,))
    assert not provider.requests


def test_changed_tokenizer_version_rejects_old_token_ids_before_forward(tmp_path):
    row, model = golden()
    model = replace(
        model,
        tokenizer=replace(
            model.tokenizer, tokenizer_hash=c.digest("changed-tokenizer")
        ),
    )
    scorer, provider, _ = service(tmp_path, model)
    with pytest.raises(c.ContractError) as error:
        scorer.score((row,))
    assert error.value.code == c.ErrorCode.STALE_CACHE and not provider.requests


@pytest.mark.parametrize("kind", ["untyped", "foreign_request"])
def test_token_metric_backend_boundary_preserves_costs_before_rejecting_bad_results(
    tmp_path, kind
):
    from self_curate.signals import TokenMetricResult

    row, model = golden()

    class BadTokenBackend:
        backend = "mock"
        implementation_hash = c.digest("bad-token-backend-boundary-test")

        def score(self, request, output):
            usage = c.ResourceUsage(scoring_tokens=7)
            if kind == "untyped":
                return SimpleNamespace(usage=usage)
            return TokenMetricResult(
                c.digest("foreign-request"),
                request.positions,
                (0.0,) * len(request.positions),
                (0.0,) * len(request.positions),
                c.ArtifactRef(**ref("synthetic-invalid-result")),
                usage,
                "mock",
            )

    scorer = SignalService(
        model=model, backend=BadTokenBackend(), output=tmp_path / "bad-backend"
    )
    with pytest.raises(ScoringFailure) as error:
        scorer.score((row,))
    assert error.value.ledger.events[0].usage.scoring_tokens == 7
    assert not error.value.records[0].signals


def rewrite_batch(folder, batch):
    """A self-consistent serializer mistake must fail semantic validation too."""
    from self_curate.data.pool import file_sha256, write_json

    (folder / "records.jsonl").write_text(
        "".join(c.canonical_json(r.to_dict()) + "\n" for r in batch.records),
        encoding="utf-8",
    )
    write_json(folder / "result.json", batch.to_dict())
    checks = {
        p.relative_to(folder).as_posix(): file_sha256(p)
        for p in folder.rglob("*")
        if p.is_file() and p.name != "checksums.json"
    }
    write_json(folder / "checksums.json", checks)


@pytest.mark.parametrize("metric", ["nll", "entropy"])
@pytest.mark.parametrize("cache", [False, True])
def test_reload_rechecks_aggregates_against_tokens_including_cached_scope(
    tmp_path, metric, cache
):
    scorer, _, row = service(tmp_path)
    if cache:
        scorer.score((row,))
    result = scorer.score((row,), specs=(scorer.spec(reduction="token_sum"),))
    outcome = result.outcomes[0]
    changed = replace(outcome.signal, **{metric: getattr(outcome.signal, metric) + 1.0})
    wrong = replace(
        result,
        records=(replace(result.records[0], signals=(changed,)),),
        outcomes=(replace(outcome, signal=changed),),
    )
    folder = scorer.output / ("score-" + result.batch_id)
    rewrite_batch(folder, wrong)
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


def test_reload_rejects_unrelated_label_mutation_during_scoring(tmp_path):
    scorer, _, row = service(tmp_path)
    result = scorer.score((row,))
    changed = replace(
        result.records[0],
        quality=replace(result.records[0].quality, verifier_id="different-verifier"),
    )
    folder = scorer.output / ("score-" + result.batch_id)
    rewrite_batch(folder, replace(result, records=(changed,)))
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


def test_failure_persists_reloadable_partial_m1_records(tmp_path):
    from self_curate.signals import read_records
    from self_curate.data.pool import file_sha256

    row, model = golden("first")
    other, _ = golden("second")

    class Failure(MatrixProvider):
        def forward(self, request):
            if request.data_id == "second":
                raise ScoringFailure(
                    "remote-failed", usage=c.ResourceUsage(scoring_tokens=9)
                )
            return super().forward(request)

    scorer, _, _ = service(tmp_path, model, provider=Failure(model))
    with pytest.raises(ScoringFailure) as error:
        scorer.score((row, other))
    folder = next(scorer.output.glob("score-*"))
    assert read_records(folder / "partial-records.jsonl") == error.value.records
    failure = json.loads((folder / "failure.json").read_text(encoding="utf-8"))
    assert failure["partial_records"] == {
        "uri": "partial-records.jsonl",
        "sha256": file_sha256(folder / "partial-records.jsonl"),
    }
    assert not (folder / "checksums.json").exists()


@pytest.mark.parametrize(
    "counter",
    ["new_gradient_samples", "new_gradient_tokens", "training_supervised_tokens"],
)
def test_forward_service_rejects_reported_gradient_or_training_work(tmp_path, counter):
    row, model = golden()

    class TrainingProvider(MatrixProvider):
        def forward(self, request):
            value = super().forward(request)
            return replace(value, usage=replace(value.usage, **{counter: 3}))

    scorer, _, _ = service(tmp_path, model, provider=TrainingProvider(model))
    with pytest.raises(ScoringFailure) as error:
        scorer.score((row,))
    assert getattr(error.value.ledger.events[0].usage, counter) == 3
    assert not error.value.records[0].signals


def test_reload_supports_direct_metric_and_new_overlapping_skill_in_same_batch(
    tmp_path,
):
    scorer, _, row = service(tmp_path)
    first = scorer.score((row,))
    other, provider, _ = service(tmp_path / "other")
    result = other.score(
        first.records,
        specs=(other.spec(), other.spec(scope="skill_evidence", skill="recovery")),
    )
    assert [o.status for o in result.outcomes] == ["cached", "computed"]
    assert provider.requests[0].positions == (4, 5, 6, 7)
    assert (
        load_scoring_batch(
            other.output / ("score-" + result.batch_id),
            expected_model=other.model_ref,
            expected_computation_hash=other.computation_hash,
        )
        == result
    )


def test_reload_reconstructs_aggregates_across_two_cached_forwards_and_log_bases(
    tmp_path,
):
    scorer, provider, row = service(tmp_path)
    skill = scorer.spec(scope="skill_evidence", skill="recovery")
    first = scorer.score((row,), specs=(skill,))
    second = scorer.score(first.records)
    third = scorer.score(
        second.records,
        specs=(scorer.spec(reduction="token_sum", log_base="2"),),
    )
    assert len(provider.requests) == 2 and not third.ledger.events
    for result in (first, second, third):
        assert (
            load_scoring_batch(
                scorer.output / ("score-" + result.batch_id),
                expected_model=scorer.model_ref,
                expected_computation_hash=scorer.computation_hash,
            )
            == result
        )


@pytest.mark.parametrize("artifact", ["inputs.jsonl", "token-statistics.json"])
def test_reload_requires_declared_snapshots_even_after_checksum_file_rebuild(
    tmp_path, artifact
):
    scorer, _, row = service(tmp_path)
    result = scorer.score((row,))
    folder = scorer.output / ("score-" + result.batch_id)
    (folder / artifact).unlink()
    rewrite_batch(folder, result)
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


def test_reload_rejects_computed_outcome_mislabeled_as_cached(tmp_path):
    scorer, _, row = service(tmp_path)
    result = scorer.score((row,))
    wrong = replace(result, outcomes=(replace(result.outcomes[0], status="cached"),))
    folder = scorer.output / ("score-" + result.batch_id)
    rewrite_batch(folder, wrong)
    with pytest.raises(c.ContractError):
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )


@pytest.mark.parametrize("kind", ["empty", "missing", "direct"])
def test_snapshot_roundtrip_with_no_physical_forward(tmp_path, kind):
    scorer, provider, row = service(tmp_path)
    if kind == "empty":
        result = scorer.score(())
    elif kind == "missing":
        result = scorer.score(
            (row,), specs=(scorer.spec(scope="skill_evidence", skill="missing"),)
        )
    else:
        first = scorer.score((row,))
        result = scorer.score(first.records)
    calls = len(provider.requests)
    assert not result.ledger.events
    assert (
        load_scoring_batch(
            scorer.output / ("score-" + result.batch_id),
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )
        == result
    )
    assert len(provider.requests) == calls


def test_legacy_root_request_schema_one_is_readable_without_rewriting(tmp_path):
    from self_curate.data.pool import file_sha256, write_json

    scorer, _, row = service(tmp_path)
    result = scorer.score((row,))
    folder = scorer.output / ("score-" + result.batch_id)
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    request["schema_version"] = "1"
    for field in ("input_records_file", "token_statistics_file"):
        (folder / request.pop(field)).unlink()
    write_json(folder / "request.json", request)
    rewrite_batch(folder, result)
    before = {
        p.relative_to(folder).as_posix(): file_sha256(p)
        for p in folder.rglob("*")
        if p.is_file()
    }
    assert (
        load_scoring_batch(
            folder,
            expected_model=scorer.model_ref,
            expected_computation_hash=scorer.computation_hash,
        )
        == result
    )
    assert all(file_sha256(folder / name) == sha for name, sha in before.items())
