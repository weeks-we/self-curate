"""Cross-experiment reuse is checked independently of a producer process."""

from dataclasses import replace
import json
import shutil

import pytest

from helpers_feedback import inputs
from helpers_m2 import MODEL, fixture_program
from helpers_p0 import fixture_bundle
from test_outer_cycle import FixtureRunner, ref
from test_signal_computation import service
from self_curate import contracts as c
from self_curate.agents.prompts import PromptTemplate
from self_curate.assets import (
    AssetSpec,
    AssetStore,
    Producer,
    register_signals,
    reuse_signals,
    register_capability,
    reuse_capability,
    strategy_spec,
    register_strategy,
    reuse_strategy,
    freeze_run_inputs,
    load_run_inputs,
    register_lopt_call,
)
from self_curate.data.pool import file_sha256, write_json
from self_curate.validation import TaskSuite, TaskValidator


def producer(run="first"):
    return Producer(
        run, c.digest("source"), c.digest("environment"), c.digest("config")
    )


def generic(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    (folder / "one.json").write_text('{"value":7}\n', encoding="utf-8")
    key = AssetSpec.create(
        "data_preparation",
        "mock",
        {"source_lock": c.digest("raw"), "ontology": "seven-skills-v1"},
    )
    store = AssetStore(tmp_path / "a")
    return store, key, folder


def test_shared_snapshot_survives_source_changes_and_storage_relocation(tmp_path):
    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    (folder / "one.json").write_text("replaced source", encoding="utf-8")
    moved = tmp_path / "m"
    shutil.copytree(store.root, moved)
    loaded = AssetStore(moved).get(key, reference=asset.reference)
    assert loaded.file("one.json").read_text(encoding="utf-8") == '{"value":7}\n'
    assert loaded.reference == asset.reference


def test_identity_is_canonical_but_backend_and_dependencies_are_separate(tmp_path):
    store, key, folder = generic(tmp_path)
    assert (
        AssetSpec.create(
            key.kind, key.backend, dict(reversed(list(key.identity.items())))
        )
        == key
    )
    parent = store.publish(key, folder, producer=producer())
    child = AssetSpec.create(
        "report", "mock", {"inputs": c.digest("x")}, dependencies=(parent.reference,)
    )
    value = store.publish(child, folder, producer=producer("second"))
    assert store.inspect(value.spec.key_hash).spec.dependencies == (parent.reference,)
    assert store.get(replace(key, backend="real")) is None


def test_idempotent_registration_rejects_conflicting_output_or_producer(tmp_path):
    store, key, folder = generic(tmp_path)
    first = store.publish(key, folder, producer=producer())
    assert store.publish(key, folder, producer=producer()) == first
    with pytest.raises(c.ContractError):
        store.publish(key, folder, producer=producer("another-physical-call"))
    (folder / "one.json").write_text("other values", encoding="utf-8")
    with pytest.raises(c.ContractError):
        store.publish(key, folder, producer=producer())
    assert store.get(key) == first


@pytest.mark.parametrize("mutation", ["payload", "extra", "marker", "manifest"])
def test_incomplete_or_corrupt_asset_never_becomes_a_silent_cache_miss(
    tmp_path, mutation
):
    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    if mutation == "payload":
        (asset.payload / "one.json").write_text("changed", encoding="utf-8")
    elif mutation == "extra":
        (asset.payload / "extra").write_text("extra", encoding="utf-8")
    elif mutation == "marker":
        (asset.path / "COMPLETE.json").unlink()
    else:
        (asset.path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(c.ContractError):
        store.get(key)


def test_pinned_reference_is_required_to_match_and_missing_pins_do_not_recompute(
    tmp_path,
):
    store, key, folder = generic(tmp_path)
    assert store.get(key) is None
    missing = c.ArtifactRef(
        "asset://" + key.key_hash + "/manifest.json", c.digest("missing")
    )
    with pytest.raises(c.ContractError):
        store.get(key, reference=missing)
    asset = store.publish(key, folder, producer=producer())
    with pytest.raises(c.ContractError):
        store.get(key, reference=missing)
    with pytest.raises(c.ContractError):
        store.get(replace(key, backend="real"), reference=asset.reference)


def test_dependency_hash_and_domain_are_checked(tmp_path):
    store, key, folder = generic(tmp_path)
    parent = store.publish(key, folder, producer=producer())
    wrong = replace(parent.reference, sha256=c.digest("different-output"))
    for reference, backend in ((wrong, "mock"), (parent.reference, "real")):
        child = AssetSpec.create("report", backend, {"x": 1}, dependencies=(reference,))
        with pytest.raises(c.ContractError):
            store.publish(child, folder, producer=producer())


def test_copy_race_does_not_publish_a_half_updated_snapshot(tmp_path, monkeypatch):
    store, key, folder = generic(tmp_path)
    original = shutil.copytree

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (folder / "one.json").write_text("changed during snapshot", encoding="utf-8")
        return result

    monkeypatch.setattr(shutil, "copytree", mutate)
    with pytest.raises(c.ContractError):
        store.publish(key, folder, producer=producer())
    assert store.get(key) is None


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/file", "a\\b"])
def test_manifest_paths_are_rejected_even_after_marker_rehash(tmp_path, name):
    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    row = asset.manifest
    row["files"] = {name: row["files"]["one.json"]}
    write_json(asset.path / "manifest.json", row)
    write_json(
        asset.path / "COMPLETE.json",
        {
            "schema_version": "1",
            "key_hash": key.key_hash,
            "manifest_hash": file_sha256(asset.path / "manifest.json"),
        },
    )
    with pytest.raises(c.ContractError):
        store.get(key)


def test_reuse_receipt_is_separate_exclusive_and_does_not_recharge_cost(tmp_path):
    store, key, folder = generic(tmp_path)
    event = c.CostEvent(
        "original",
        "first",
        "scoring",
        "data",
        c.digest("model"),
        "completed",
        c.ResourceUsage(scoring_tokens=123),
        ref("cost"),
        "mock",
    )
    cost = c.CostLedger("first-ledger", (event,), "mock")
    asset = store.publish(key, folder, producer=producer(), origin_ledger=cost)
    before = file_sha256(asset.path / "manifest.json")
    for run in ("v0", "v1"):
        reused, receipt = store.reuse(
            key,
            consumer_run=run,
            receipt_path=tmp_path / (run + ".json"),
            reference=asset.reference,
        )
        assert reused.origin_ledger == cost
        assert (
            receipt["new_model_calls"] == receipt["new_training_supervised_tokens"] == 0
        )
        assert (
            not receipt["origin_cost_recharged"] and receipt["producer_run"] == "first"
        )
        assert receipt["io_wall_seconds"] >= 0
    assert file_sha256(asset.path / "manifest.json") == before
    with pytest.raises(c.ContractError):
        store.reuse(key, consumer_run="again", receipt_path=tmp_path / "v0.json")
    with pytest.raises(c.ContractError):
        store.reuse(key, consumer_run="bad", receipt_path=asset.path / "new.json")


def test_signal_reuse_matches_ids_and_keeps_current_quality_metadata_and_usage(
    tmp_path,
):
    scorer, calls, row = service(tmp_path)
    batch = scorer.score((row,))
    store = AssetStore(tmp_path / "a")
    saved = register_signals(
        store,
        scorer.output / ("score-" + batch.batch_id),
        model=scorer.model_ref,
        specs=(scorer.spec(),),
        producer=producer(),
    )
    current = replace(
        row,
        quality=replace(row.quality, verifier_id="new-current-label"),
        metadata=c.DataLabels(task_type="coding"),
    )
    reused, receipt = reuse_signals(
        store,
        (current,),
        model=scorer.model_ref,
        specs=(scorer.spec(),),
        backend="mock",
        consumer_run="v1",
        reference=saved.reference,
    )
    assert len(calls.requests) == 1
    assert (
        reused[0].quality == current.quality and reused[0].metadata == current.metadata
    )
    assert reused[0].signals[0].nll == batch.outcomes[0].signal.nll
    assert saved.reference in reused[0].usage.forward_scoring
    assert not reused[0].usage.candidate_sft and receipt["new_model_calls"] == 0
    proof = batch.outcomes[0].signal.provenance[0]
    assert file_sha256(saved.resolve_evidence(proof)) == proof.sha256


@pytest.mark.parametrize("change", ["model", "input", "source", "scope", "computation"])
def test_signal_version_changes_are_cache_misses(tmp_path, change):
    scorer, provider, row = service(tmp_path)
    batch = scorer.score((row,))
    store = AssetStore(tmp_path / "a")
    register_signals(
        store,
        scorer.output / ("score-" + batch.batch_id),
        model=scorer.model_ref,
        specs=(scorer.spec(),),
        producer=producer(),
    )
    model, spec = scorer.model_ref, scorer.spec()
    if change == "model":
        model = replace(model, adapter_hash=c.digest("new-adapter"))
    elif change == "input":
        row = replace(
            row,
            tokenization=replace(
                row.tokenization, input_ids=(19, *row.tokenization.input_ids[1:])
            ),
        )
    elif change == "source":
        row = replace(row, source=replace(row.source, revision="new-source"))
    elif change == "scope":
        spec = scorer.spec(scope="skill_evidence", skill="recovery")
    else:
        spec = replace(spec, computation_hash=c.digest("new-computation"))
    assert (
        reuse_signals(
            store,
            (row,),
            model=model,
            specs=(spec,),
            backend="mock",
            consumer_run="next",
        )
        is None
    )
    assert len(provider.requests) == 1


def test_registering_cached_signals_requires_their_original_asset_provenance(tmp_path):
    scorer, _, row = service(tmp_path)
    batch = scorer.score((row,))
    cached = scorer.score(batch.records)
    with pytest.raises(c.ContractError):
        register_signals(
            AssetStore(tmp_path / "a"),
            scorer.output / ("score-" + cached.batch_id),
            model=scorer.model_ref,
            specs=(scorer.spec(),),
            producer=producer(),
        )


def baseline(tmp_path, *, fail=False):
    suite = TaskSuite(
        "dev",
        fixture_bundle().tasks,
        (1,),
        ref("sampling"),
        ref("protocol"),
        "task_mean",
        "mock",
    )
    runner = FixtureRunner(fail=fail)
    folder = tmp_path / "ev"
    value = TaskValidator(runner).evaluate(
        MODEL, suite, attempt_id="m0", evaluation_id="baseline", output=folder
    )
    return suite, runner, value, folder


def test_base_capability_is_reused_without_tasks_or_model_calls(tmp_path):
    suite, runner, value, folder = baseline(tmp_path)
    store = AssetStore(tmp_path / "a")
    execution = c.digest("frozen-task-runner-and-inference-config")
    asset = register_capability(
        store,
        folder,
        model=MODEL,
        suite=suite,
        execution_hash=execution,
        producer=producer(),
    )
    count = len(runner.requests)
    for name in ("v0", "v1", "v2"):
        reused, receipt = reuse_capability(
            store,
            model=MODEL,
            suite=suite,
            execution_hash=execution,
            consumer_run=name,
            reference=asset.reference,
        )
        assert reused == value and receipt["new_model_calls"] == 0
    assert len(runner.requests) == count


@pytest.mark.parametrize(
    "change", ["model", "seeds", "sampling", "protocol", "execution"]
)
def test_base_capability_reuse_is_bound_to_protocol_and_execution(tmp_path, change):
    suite, _, _, folder = baseline(tmp_path)
    store = AssetStore(tmp_path / "a")
    execution, model = c.digest("execution"), MODEL
    register_capability(
        store,
        folder,
        model=model,
        suite=suite,
        execution_hash=execution,
        producer=producer(),
    )
    if change == "model":
        model = replace(model, weights_hash=c.digest("other"))
    elif change == "execution":
        execution = c.digest("changed-runner")
    else:
        suite = replace(
            suite, **{change: (2,) if change == "seeds" else ref("new-" + change)}
        )
    assert (
        reuse_capability(
            store,
            model=model,
            suite=suite,
            execution_hash=execution,
            consumer_run="next",
        )
        is None
    )


def test_invalid_task_execution_cannot_be_registered_as_a_shared_baseline(tmp_path):
    suite, _, _, folder = baseline(tmp_path, fail=True)
    with pytest.raises(c.ContractError):
        register_capability(
            AssetStore(tmp_path / "a"),
            folder,
            model=MODEL,
            suite=suite,
            execution_hash=c.digest("execution"),
            producer=producer(),
        )


def strategy():
    request, _, _, policy = inputs()
    prompt = PromptTemplate("lopt", "frozen-lopt", "1", "fixed instructions")
    payload = {
        "schema": "role-work-v1",
        "role": "lopt",
        "session_id": "shared-initial",
        "memory_namespace": "initial/lopt",
        "model_ref": request.model_ref.to_dict(),
        "prompt_hash": prompt.prompt_hash,
        "work": {
            "parent": request.program.to_dict(),
            "request_hash": request.request_hash,
        },
        "history": [],
    }
    sampling = ref("sampling-parameters-and-seed")
    key = strategy_spec(
        request.model_ref,
        prompt,
        payload,
        policy,
        sampling,
        backend="mock",
        implementation_hash=c.digest("client"),
        timeout_seconds=30,
    )
    program = fixture_program()
    response = {
        "program": program.to_dict(),
        "rationale": "actual saved fixture reply",
        "hypothesis": "requires validation",
        "cited_feedback_id": "saved-feedback",
    }
    return request.model_ref, prompt, payload, policy, sampling, key, program, response


def test_initial_lopt_recipe_reuse_does_not_replay_agent_or_reuse_selection(tmp_path):
    *_, key, program, response = strategy()
    store = AssetStore(tmp_path / "a")
    asset = register_strategy(
        store,
        key,
        program=program,
        response=response,
        snapshot_output=tmp_path / "proposal",
        producer=producer(),
    )
    reused, receipt = reuse_strategy(
        store, key, consumer_run="v3", reference=asset.reference
    )
    assert reused == program and receipt["new_model_calls"] == 0
    assert not hasattr(reused, "batch") and not hasattr(reused, "history")


@pytest.mark.parametrize(
    "change", ["model", "prompt", "history", "policy", "sampling", "client", "timeout"]
)
def test_lopt_first_strategy_is_not_universal_across_generation_contexts(
    tmp_path, change
):
    model, prompt, payload, policy, sampling, key, program, response = strategy()
    store = AssetStore(tmp_path / "a")
    register_strategy(
        store,
        key,
        program=program,
        response=response,
        snapshot_output=tmp_path / "proposal",
        producer=producer(),
    )
    implementation = c.digest("client")
    if change == "model":
        model = replace(model, revision="other")
        payload = {**payload, "model_ref": model.to_dict()}
    elif change == "prompt":
        prompt = replace(prompt, instructions="different")
        payload = {**payload, "prompt_hash": prompt.prompt_hash}
    elif change == "history":
        payload = {**payload, "history": [{"actual_prior_input": "different"}]}
    elif change == "policy":
        policy = replace(policy, max_changed_nodes=1)
    elif change == "sampling":
        sampling = ref("different-temperature-or-seed")
    elif change == "client":
        implementation = c.digest("new-client-code")
    different = strategy_spec(
        model,
        prompt,
        payload,
        policy,
        sampling,
        backend="mock",
        implementation_hash=implementation,
        timeout_seconds=10 if change == "timeout" else 30,
    )
    assert (
        different != key
        and reuse_strategy(store, different, consumer_run="next") is None
    )


def test_strategy_response_must_match_saved_program(tmp_path):
    *_, key, program, response = strategy()
    wrong = {**response, "program": replace(program, program_id="another").to_dict()}
    store = AssetStore(tmp_path / "a")
    with pytest.raises(c.ContractError):
        register_strategy(
            store,
            key,
            program=program,
            response=wrong,
            snapshot_output=tmp_path / "bad",
            producer=producer(),
        )
    assert store.get(key) is None


def test_strategy_snapshot_cannot_write_into_an_existing_shared_asset(tmp_path):
    store, base, folder = generic(tmp_path)
    original = store.publish(base, folder, producer=producer())
    *_, key, program, response = strategy()
    with pytest.raises(c.ContractError):
        register_strategy(
            store,
            key,
            program=program,
            response=response,
            snapshot_output=original.payload / "unexpected",
            producer=producer(),
        )
    assert (
        store.get(base) == original and not (original.payload / "unexpected").exists()
    )


def test_frozen_run_inputs_survive_main_checkout_updates_but_refuse_wrong_resume_code(
    tmp_path,
):
    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    config = {"variant": "V0", "provider": {"api_key_env": "EXTERNAL_LABEL_API_KEY"}}
    stamp = replace(producer("v0"), config_hash=c.digest(config))
    path = tmp_path / "run/frozen.json"
    frozen = freeze_run_inputs(
        store, path, producer=stamp, config=config, inputs={"common": asset.reference}
    )
    loaded = load_run_inputs(
        store,
        path,
        expected_lock_hash=frozen["lock_hash"],
        expected_source_tree_hash=stamp.source_tree_hash,
        expected_environment_hash=stamp.environment_hash,
    )
    assert loaded == frozen
    for name in ("expected_source_tree_hash", "expected_environment_hash"):
        expected = dict(
            expected_lock_hash=frozen["lock_hash"],
            expected_source_tree_hash=stamp.source_tree_hash,
            expected_environment_hash=stamp.environment_hash,
        )
        expected[name] = c.digest("new-version")
        with pytest.raises(c.ContractError):
            load_run_inputs(store, path, **expected)
    with pytest.raises(c.ContractError):
        freeze_run_inputs(
            store,
            path,
            producer=stamp,
            config=config,
            inputs={"common": asset.reference},
        )


@pytest.mark.parametrize("change", ["config_stamp", "input_pin", "credential"])
def test_run_lock_rejects_unpinned_inputs_credentials_and_wrong_config(
    tmp_path, change
):
    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    config = {"variant": "V0"}
    stamp = replace(producer(), config_hash=c.digest(config))
    reference = asset.reference
    if change == "config_stamp":
        config = {"variant": "V1"}
    elif change == "input_pin":
        reference = replace(reference, sha256=c.digest("different"))
    else:
        config = {"provider": {"api_key": "synthetic-secret-must-not-be-written"}}
        stamp = replace(stamp, config_hash=c.digest(config))
    target = tmp_path / "run.json"
    with pytest.raises(c.ContractError):
        freeze_run_inputs(
            store, target, producer=stamp, config=config, inputs={"shared": reference}
        )
    assert not target.exists()


def test_inspect_asset_cli_is_read_only(tmp_path, capsys):
    from self_curate.cli import main

    store, key, folder = generic(tmp_path)
    asset = store.publish(key, folder, producer=producer())
    before = file_sha256(asset.path / "manifest.json")
    assert (
        main(["inspect-asset", "--store", str(store.root), "--key", key.key_hash]) == 0
    )
    assert json.loads(capsys.readouterr().out)["key_hash"] == key.key_hash
    assert file_sha256(asset.path / "manifest.json") == before


@pytest.mark.parametrize("which", [0, 1])
def test_actual_saved_lopt_call_exports_only_its_own_past_and_cost(tmp_path, which):
    from self_curate.agents.runtime import RoleRuntime

    *_, policy, sampling, _, program, response = strategy()

    class Client:
        model_ref = MODEL

        def __init__(self):
            self.payloads = []

        def generate_json(self, **kwargs):
            self.payloads.append(kwargs)
            return {
                "content": response,
                "usage": c.ResourceUsage(
                    agent_input_tokens=5, agent_output_tokens=2
                ).to_dict(),
            }

    client = Client()
    runtime = RoleRuntime(client, model=MODEL, session_id="initial", backend="mock")
    replies = [runtime.generate("lopt", {"visible_at_call": i}) for i in (0, 1)]
    store = AssetStore(tmp_path / "a")
    saved = register_lopt_call(
        store,
        runtime,
        replies[which].call_id,
        policy=policy,
        sampling=sampling,
        implementation_hash=c.digest("frozen-client"),
        snapshot_output=tmp_path / "proposal",
        producer=producer(),
    )
    actual = saved.spec.identity["request"]
    assert actual["payload"] == client.payloads[which]["payload"]
    assert actual["system_prompt"] == client.payloads[which]["system_prompt"]
    assert len(actual["payload"]["history"]) == which
    assert len(saved.origin_ledger.events) == 1
    assert saved.origin_ledger.events[0].event_id == replies[which].call_id
    reused, receipt = reuse_strategy(
        store, saved.spec, consumer_run="next", reference=saved.reference
    )
    assert (
        reused == program
        and receipt["new_model_calls"] == 0
        and len(client.payloads) == 2
    )


@pytest.mark.parametrize("kind", ["signals", "capability", "strategy"])
def test_typed_reuse_resolves_store_once_and_times_the_initial_lookup(
    tmp_path, monkeypatch, kind
):
    from self_curate.assets import adapters, store as storage

    shared = AssetStore(tmp_path / "a")
    if kind == "signals":
        scorer, _, row = service(tmp_path)
        batch = scorer.score((row,))
        register_signals(
            shared,
            scorer.output / ("score-" + batch.batch_id),
            model=scorer.model_ref,
            specs=(scorer.spec(),),
            producer=producer(),
        )
        action = lambda: reuse_signals(
            shared,
            (row,),
            model=scorer.model_ref,
            specs=(scorer.spec(),),
            backend="mock",
            consumer_run="next",
        )
    elif kind == "capability":
        suite, _, _, folder = baseline(tmp_path)
        execution = c.digest("execution")
        register_capability(
            shared,
            folder,
            model=MODEL,
            suite=suite,
            execution_hash=execution,
            producer=producer(),
        )
        action = lambda: reuse_capability(
            shared,
            model=MODEL,
            suite=suite,
            execution_hash=execution,
            consumer_run="next",
        )
    else:
        *_, key, program, response = strategy()
        register_strategy(
            shared,
            key,
            program=program,
            response=response,
            snapshot_output=tmp_path / "proposal",
            producer=producer(),
        )
        action = lambda: reuse_strategy(shared, key, consumer_run="next")
    original = shared.get
    clock, calls = [0.0], []

    def timed(*args, **kwargs):
        calls.append(1)
        clock[0] += 10.0
        return original(*args, **kwargs)

    monkeypatch.setattr(shared, "get", timed)
    monkeypatch.setattr(storage, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(adapters, "perf_counter", lambda: clock[0])
    _, receipt = action()
    assert len(calls) == 1 and receipt["io_wall_seconds"] == 10.0
