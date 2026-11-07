"""Checked bridges for M0 signals, development capability, and frozen Lopt recipes."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from self_curate.contracts import (
    ContractError,
    ErrorCode,
    ModelRef,
    SignalSpec,
    digest,
    parse_program,
)
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import require, sha256, text
from self_curate.contracts.workflow import finite
from self_curate.data import MetricView, attach_signals, signal_binding
from self_curate.data.pool import _publish, _staging_directory, write_json, file_sha256
from self_curate.signals import load_scoring_batch
from self_curate.orchestration.io import load_task_evaluation
from self_curate.validation import TaskSuite
from .contracts import AssetSpec
from .store import structured


def signal_spec(records, model, specs, *, backend, dependencies=()):
    records, specs = tuple(records), tuple(specs)
    MetricView(records)
    require(
        isinstance(model, ModelRef)
        and bool(specs)
        and all(isinstance(s, SignalSpec) for s in specs),
        "Typed model/signal scopes required",
    )
    require(
        len({s.spec_hash for s in specs}) == len(specs)
        and len({s.computation_hash for s in specs}) == 1,
        "Duplicate or mixed signal implementations",
    )
    inputs = []
    for record in records:
        require(
            record.tokenization.identity == model.tokenizer,
            "Signal input uses another tokenizer",
        )
        require(
            backend != "real" or not record.known_mock_artifacts,
            "Mock signals cannot be registered as real",
        )
        bindings = []
        for spec in specs:
            try:
                bindings.append(signal_binding(record, spec).to_dict())
            except ContractError as exc:
                if exc.code != ErrorCode.MISSING_SIGNAL:
                    raise
                bindings.append(None)
        inputs.append(
            {
                "data_id": record.data_id,
                "content_hash": record.content_hash,
                "source_hash": digest(record.source.to_dict()),
                "tokenization_hash": digest(record.tokenization.to_dict()),
                "bindings": bindings,
            }
        )
    return AssetSpec.create(
        "signals",
        backend,
        {
            "model_ref": model.to_dict(),
            "specs": [s.to_dict() for s in specs],
            "inputs": inputs,
        },
        dependencies=dependencies,
    )


@structured
def register_signals(store, source, *, model, specs, producer, dependencies=()):
    specs = tuple(specs)
    require(bool(specs), "Signal scopes required")
    batch = load_scoring_batch(
        source,
        expected_model=model,
        expected_computation_hash=specs[0].computation_hash,
    )
    key = signal_spec(
        batch.records, model, specs, backend=batch.backend, dependencies=dependencies
    )
    require(
        tuple(o.spec for o in batch.outcomes)
        == tuple(s for _ in batch.records for s in specs),
        "Signal scopes differ from publication",
    )
    require(
        dependencies
        or not any(
            o.status == "cached" and o.signal is not None for o in batch.outcomes
        ),
        "Cached-input publication requires pinned origin assets; register the original fresh scoring output first",
    )

    def validate(path):
        copied = load_scoring_batch(
            path,
            expected_model=model,
            expected_computation_hash=specs[0].computation_hash,
        )
        require(copied == batch, "Scoring source changed during registration")

    return store.publish(
        key, source, producer=producer, origin_ledger=batch.ledger, validator=validate
    )


@structured
def reuse_signals(
    store,
    records,
    *,
    model,
    specs,
    backend,
    consumer_run,
    receipt_path=None,
    dependencies=(),
    reference=None,
):
    started = perf_counter()
    records, specs = tuple(records), tuple(specs)
    key = signal_spec(records, model, specs, backend=backend, dependencies=dependencies)
    asset = store.get(key, reference=reference)
    if asset is None:
        return None
    batch = load_scoring_batch(
        asset.payload,
        expected_model=model,
        expected_computation_hash=specs[0].computation_hash,
    )
    require(
        signal_spec(
            batch.records, model, specs, backend=backend, dependencies=dependencies
        )
        == key,
        "Saved signal input binding mismatch",
    )
    additions = {r.data_id: [] for r in records}
    for outcome in batch.outcomes:
        if outcome.signal is not None:
            signal = outcome.signal
            signal = replace(
                signal,
                provenance=tuple(
                    sorted(
                        set((*signal.provenance, asset.reference)),
                        key=lambda r: (r.uri, r.sha256),
                    )
                ),
            )
            additions[outcome.data_id].append(signal)
    updated = tuple(attach_signals(r, additions[r.data_id]) for r in records)
    receipt = store._receipt(
        asset,
        consumer_run=consumer_run,
        receipt_path=receipt_path,
        started=started,
    )
    return updated, receipt


def capability_spec(model, suite, checkpoint=None, *, execution_hash, dependencies=()):
    from self_curate.validation import service, contracts

    sha256(execution_hash, "task execution implementation/config hash")
    require(
        isinstance(model, ModelRef) and isinstance(suite, TaskSuite),
        "Pinned capability model/suite required",
    )
    return AssetSpec.create(
        "capability",
        suite.backend,
        {
            "model_ref": model.to_dict(),
            "suite": suite.to_dict(),
            "checkpoint": None if checkpoint is None else checkpoint.to_dict(),
            "execution_hash": execution_hash,
            "kernel_hash": digest(
                (
                    file_sha256(Path(service.__file__)),
                    file_sha256(Path(contracts.__file__)),
                )
            ),
        },
        dependencies=dependencies,
    )


@structured
def register_capability(
    store,
    source,
    *,
    model,
    suite,
    producer,
    execution_hash,
    checkpoint=None,
    dependencies=(),
):
    key = capability_spec(
        model,
        suite,
        checkpoint,
        execution_hash=execution_hash,
        dependencies=dependencies,
    )
    value = load_task_evaluation(
        source,
        expected_model=model,
        expected_suite=suite,
        expected_checkpoint=checkpoint,
    )
    require(
        value.protocol_verified and value.report.status == "valid",
        "Incomplete capability cannot be a shared baseline",
    )

    def validate(path):
        copied = load_task_evaluation(
            path,
            expected_model=model,
            expected_suite=suite,
            expected_checkpoint=checkpoint,
        )
        require(copied == value, "Capability source changed during registration")

    return store.publish(
        key, source, producer=producer, origin_ledger=value.ledger, validator=validate
    )


@structured
def reuse_capability(
    store,
    *,
    model,
    suite,
    consumer_run,
    execution_hash,
    checkpoint=None,
    receipt_path=None,
    dependencies=(),
    reference=None,
):
    started = perf_counter()
    key = capability_spec(
        model,
        suite,
        checkpoint,
        execution_hash=execution_hash,
        dependencies=dependencies,
    )
    asset = store.get(key, reference=reference)
    if asset is None:
        return None
    value = load_task_evaluation(
        asset.payload,
        expected_model=model,
        expected_suite=suite,
        expected_checkpoint=checkpoint,
    )
    require(
        value.protocol_verified and value.report.status == "valid",
        "Saved baseline is invalid",
    )
    receipt = store._receipt(
        asset,
        consumer_run=consumer_run,
        receipt_path=receipt_path,
        started=started,
    )
    return value, receipt


def strategy_spec(
    model,
    prompt,
    payload,
    policy,
    sampling,
    *,
    backend,
    implementation_hash,
    timeout_seconds,
    dependencies=(),
):
    from self_curate.agents.prompts import PromptTemplate
    from self_curate.contracts import ArtifactRef, EditPolicy

    sha256(implementation_hash, "generation implementation hash")
    finite(timeout_seconds, "generation timeout", 0)
    require(timeout_seconds > 0, "Positive generation timeout required")
    require(
        isinstance(model, ModelRef)
        and isinstance(prompt, PromptTemplate)
        and prompt.role == "lopt"
        and isinstance(policy, EditPolicy)
        and isinstance(sampling, ArtifactRef),
        "Typed Lopt generation inputs required",
    )
    fields(
        payload,
        (
            "schema",
            "role",
            "session_id",
            "model_ref",
            "memory_namespace",
            "prompt_hash",
            "work",
            "history",
        ),
    )
    text(payload["session_id"], "generation session")
    text(payload["memory_namespace"], "generation memory namespace")
    require(
        payload["schema"] == "role-work-v1"
        and payload["role"] == "lopt"
        and payload["model_ref"] == model.to_dict()
        and payload["prompt_hash"] == prompt.prompt_hash
        and type(payload["history"]) is list
        and type(payload["work"]) is dict,
        "Save the full actual Lopt payload, including history and model",
    )
    require(
        backend != "real"
        or (
            model.tokenizer.backend != "fixture"
            and not sampling.uri.strip().lower().startswith(("mock://", "fixture://"))
        ),
        "Mock generation cannot be real",
    )
    return AssetSpec.create(
        "strategy",
        backend,
        {
            "request": {
                "system_prompt": prompt.render(),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            },
            "policy": policy.to_dict(),
            "sampling": sampling.to_dict(),
            "implementation_hash": implementation_hash,
        },
        dependencies=dependencies,
    )


def _strategy_program(path, key):
    require(key.kind == "strategy", "Expected Lopt generation asset")
    request = load_json((path / "request.json").read_text(encoding="utf-8"))
    require(request == key.identity, "Lopt request/context changed")
    program = parse_program(
        load_json((path / "program.json").read_text(encoding="utf-8"))
    )
    reply = fields(
        load_json((path / "response.json").read_text(encoding="utf-8")),
        ("request_hash", "content"),
    )
    content = fields(
        reply["content"], ("program", "rationale", "hypothesis", "cited_feedback_id")
    )
    for name in ("rationale", "hypothesis", "cited_feedback_id"):
        text(content[name], name)
    require(
        reply["request_hash"] == digest(request)
        and parse_program(content["program"]) == program,
        "Lopt output does not match saved request/program",
    )
    return program


@structured
def register_strategy(
    store, key, *, program, response, snapshot_output, producer, origin_ledger=None
):
    from self_curate.contracts import ProgramSpec

    require(
        isinstance(program, ProgramSpec) and key.kind == "strategy",
        "Expected generated ProgramSpec and generation key",
    )
    output = Path(snapshot_output).resolve()
    require(
        not output.is_relative_to(store.root),
        "Original strategy snapshot must not modify shared assets",
    )
    require(
        not output.exists(),
        "Strategy snapshot output already exists",
        ErrorCode.OUTPUT_EXISTS,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = _staging_directory(output)
    write_json(stage / "request.json", key.identity)
    write_json(stage / "program.json", program.to_dict())
    write_json(
        stage / "response.json",
        {"request_hash": digest(key.identity), "content": response},
    )
    _strategy_program(stage, key)
    _publish(stage, output)
    return store.publish(
        key,
        output,
        producer=producer,
        origin_ledger=origin_ledger,
        validator=lambda p: _strategy_program(p, key),
    )


@structured
def reuse_strategy(store, key, *, consumer_run, receipt_path=None, reference=None):
    started = perf_counter()
    asset = store.get(key, reference=reference)
    if asset is None:
        return None
    program = _strategy_program(asset.payload, key)
    receipt = store._receipt(
        asset,
        consumer_run=consumer_run,
        receipt_path=receipt_path,
        started=started,
    )
    return program, receipt


@structured
def register_lopt_call(
    store,
    runtime,
    call_id,
    *,
    policy,
    sampling,
    implementation_hash,
    snapshot_output,
    producer,
    dependencies=(),
):
    """Export a completed call's actual before-context, never its later history.

    The generation implementation/sampling declaration comes from the producer's
    frozen run configuration; an external service must attest those settings.
    """
    from self_curate.agents.runtime import RoleRuntime
    from self_curate.agents.history import RoleHistory
    from self_curate.contracts import CostLedger

    require(
        isinstance(runtime, RoleRuntime),
        "Expected original or checked restored RoleRuntime",
    )
    events = runtime.history("lopt")
    starts = [
        (index, event)
        for index, event in enumerate(events)
        if event.call_id == call_id
        and event.phase == "before"
        and event.operation == "model"
    ]
    ends = [
        event
        for event in events
        if event.call_id == call_id
        and event.phase == "after"
        and event.operation == "model"
    ]
    require(
        len(starts) == len(ends) == 1 and ends[0].payload["status"] == "completed",
        "Only completed Lopt model calls are reusable strategies",
    )
    index, before = starts[0]
    prior = RoleHistory(
        "lopt",
        before.namespace,
        before.model_hash,
        before.prompt_hash,
        before.backend,
        events[:index],
    )
    context = prior.context(len(before.payload["history_call_ids"]))
    require(
        digest(context) == before.payload["history_hash"],
        "Saved Lopt before-history changed",
    )
    payload = {
        "schema": "role-work-v1",
        "role": "lopt",
        "session_id": runtime.session_id,
        "model_ref": runtime.model_ref.to_dict(),
        "memory_namespace": before.namespace,
        "prompt_hash": before.prompt_hash,
        "work": before.payload["work"],
        "history": context,
    }
    key = strategy_spec(
        runtime.model_ref,
        runtime.prompts["lopt"],
        payload,
        policy,
        sampling,
        backend=runtime.backend,
        implementation_hash=implementation_hash,
        timeout_seconds=before.payload["timeout_seconds"],
        dependencies=dependencies,
    )
    require(
        key.identity["request"]["system_prompt"] == before.payload["system_prompt"],
        "Call prompt differs from restored runtime",
    )
    response = ends[0].payload["result"]
    program = parse_program(response["program"])
    original = runtime.ledger()
    events_cost = tuple(e for e in original.events if e.event_id == call_id)
    require(len(events_cost) == 1, "Missing physical Lopt cost event")
    cost = CostLedger("lopt-origin-" + digest(call_id), events_cost, original.backend)
    return register_strategy(
        store,
        key,
        program=program,
        response=response,
        snapshot_output=snapshot_output,
        producer=producer,
        origin_ledger=cost,
    )
