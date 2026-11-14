"""Reusable M0 capability, full-pool metrics, and one frozen Lopt proposal."""

from dataclasses import replace
from pathlib import Path
import math

from self_curate.contracts import CostLedger, digest, parse_program
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import require
from self_curate.assets import (
    register_signals,
    reuse_signals,
    register_capability,
    reuse_capability,
    register_lopt_call,
    reuse_strategy,
    strategy_spec,
)
from self_curate.agents.runtime import RoleRuntime
from self_curate.operators import ExecutionContext, SignalContext
from self_curate.data import MetricView
from self_curate.data.pool import write_json
from .policy_input import policy_catalog, pool_summary


def prepare_initial(
    inputs, ports, config, store, output, producer, *, event_sink=None, progress=None
):
    output = Path(output)
    output.mkdir()
    model = inputs.model
    new_events = [] if event_sink is None else event_sink
    if progress:
        progress.update("initial_development", 0, 1)
    capability = reuse_capability(
        store,
        model=model,
        suite=inputs.development,
        execution_hash=ports.task_execution_hash,
        consumer_run=producer.run_id,
        receipt_path=output / "capability-reuse.json",
    )
    if capability is None:
        if hasattr(ports, "enter_phase"):
            ports.enter_phase("development")
        evaluated = ports.validator.evaluate(
            model,
            inputs.development,
            attempt_id=producer.run_id,
            evaluation_id=producer.run_id + "-common-m0",
            output=output / "capability-origin",
        )
        require(
            evaluated.report.status == "valid" and evaluated.protocol_verified,
            "M0 task measurement incomplete",
        )
        new_events.extend(evaluated.ledger.events)
        register_capability(
            store,
            output / "capability-origin",
            model=model,
            suite=inputs.development,
            producer=producer,
            execution_hash=ports.task_execution_hash,
        )
        capability = reuse_capability(
            store,
            model=model,
            suite=inputs.development,
            execution_hash=ports.task_execution_hash,
            consumer_run=producer.run_id,
            receipt_path=output / "capability-reuse.json",
        )
    if progress:
        progress.update("initial_development", 1, 1)
    if "static_m0" not in config.modes:
        context = ports.context(inputs.records, config)
        write_json(output / "context.json", context.to_dict())
        return (
            inputs.records,
            context,
            capability[0],
            None,
            {"m0_capability": capability[1]["asset"]},
            CostLedger("initial-new-work", tuple(new_events), config.backend),
        )
    scorer = ports.scorer
    require(
        scorer.model_ref == model and scorer.backend.backend == config.backend,
        "M0 scorer identity mismatch",
    )
    specs = (scorer.spec(),)
    if progress:
        progress.update("initial_signals", 0, len(inputs.records))
    signals = reuse_signals(
        store,
        inputs.records,
        model=model,
        specs=specs,
        backend=config.backend,
        consumer_run=producer.run_id,
        receipt_path=output / "signals-reuse.json",
    )
    if signals is None:
        if hasattr(ports, "enter_phase"):
            ports.enter_phase("scoring")
        # Initialization intentionally includes EVERY candidate, across all skills.
        fresh = scorer.score(inputs.records, specs=specs)
        new_events.extend(fresh.ledger.events)
        require(
            all(o.signal is not None for o in fresh.outcomes),
            "Initial whole-pool metrics are incomplete",
        )
        register_signals(
            store,
            scorer.output / ("score-" + fresh.batch_id),
            model=model,
            specs=specs,
            producer=producer,
        )
        signals = reuse_signals(
            store,
            inputs.records,
            model=model,
            specs=specs,
            backend=config.backend,
            consumer_run=producer.run_id,
            receipt_path=output / "signals-reuse.json",
        )
    if progress:
        progress.update("initial_signals", len(inputs.records), len(inputs.records))
        progress.update("initial_policy", 0, 1)
    if hasattr(ports, "enter_phase"):
        ports.enter_phase("policy")
    records = signals[0]
    baseline = capability[0]
    # Context and objective are externally fixed; no Leval proxy is fabricated.
    template = ports.context(records, config)
    require(
        isinstance(template, ExecutionContext) and template.model_ref == model,
        "Initial selector context mismatch",
    )
    context = replace(
        template, view=MetricView(records), signals=(SignalContext(specs[0]),)
    )
    values = [
        next(
            s
            for s in r.signals
            if s.model_ref == model
            and s.spec == specs[0]
            and s.percentile_reference is None
        )
        for r in records
    ]
    work = {
        "kind": "initial-policy-v1",
        "initial_capability": baseline.report.to_dict(),
        "selection_seed": config.seed,
        "selection_context": context.role_summary(),
        "context_hash": context.context_hash,
        "objective": ports.objective.to_dict(),
        "count": config.selected_count(len(records)),
        "pool_hash": digest([r.record_hash for r in records]),
        "signal_spec": specs[0].to_dict(),
        "operator_contracts": policy_catalog(
            skills=inputs.skills,
            count=config.selected_count(len(records)),
            seed=config.seed,
        ),
        "candidate_summary": pool_summary(records, values, inputs.skills),
        "metrics": {
            name: {
                "min": min(getattr(s, name) for s in values),
                "max": max(getattr(s, name) for s in values),
                "mean": math.fsum(getattr(s, name) / len(values) for s in values),
            }
            for name in ("nll", "entropy")
        },
        "instruction": "Propose one complete feasible selection program. Use fixed M0 labels and scores. Do not request Leval, training, test data, or later revisions.",
    }
    feedback_ref = None
    from .initial_feedback import initial_feedback, TEMPLATE
    if ports.prompts["leval"].template_id == TEMPLATE:
        interpretation, feedback_events, feedback_ref = initial_feedback(
            baseline.report, ports, config, store, output, producer, capability[1]["asset"]
        )
        new_events.extend(feedback_events)
        work["initial_feedback"] = interpretation
        work["initial_feedback_asset"] = feedback_ref.to_dict()
    # Stable session/history identity makes the common prefix reusable across V0/V1/etc.
    session = "common-initial-" + digest(work)[:20]
    runtime = RoleRuntime(
        ports.client,
        model=model,
        session_id=session,
        backend=config.backend,
        prompts=ports.prompts,
    )
    prompt = runtime.prompts["lopt"]
    payload = {
        "schema": "role-work-v1",
        "role": "lopt",
        "session_id": session,
        "model_ref": model.to_dict(),
        "memory_namespace": runtime.namespace("lopt"),
        "prompt_hash": prompt.prompt_hash,
        "work": work,
        "history": [],
    }
    key = strategy_spec(
        model,
        prompt,
        payload,
        ports.edit_policy,
        ports.agent_sampling,
        backend=config.backend,
        implementation_hash=ports.agent_implementation_hash,
        timeout_seconds=config.lopt_timeout_seconds,
    )
    strategy = reuse_strategy(
        store,
        key,
        consumer_run=producer.run_id,
        receipt_path=output / "strategy-reuse.json",
    )

    def validate(content):
        fields(content, ("program", "rationale", "hypothesis", "cited_feedback_id"))
        require(
            type(content["rationale"]) is str
            and content["rationale"].strip()
            and type(content["hypothesis"]) is str
            and content["hypothesis"].strip(),
            "Lopt rationale/hypothesis required",
        )
        require(
            content["cited_feedback_id"] == baseline.report.report_id,
            "Initial policy must cite the actual M0 development feedback",
        )
        program = parse_program(content["program"])
        require(
            program.parent_id is None
            and program.seed == config.seed
            and all(
                n.params.get("seed", config.seed) == config.seed for n in program.nodes
            ),
            "Initial policy must use the one frozen selection seed",
        )

    if strategy is None:
        try:
            reply = runtime.generate(
                "lopt",
                work,
                timeout_seconds=config.lopt_timeout_seconds,
                validator=validate,
            )
        finally:
            new_events.extend(runtime.ledger().events)
            runtime.save(output / "roles")
        register_lopt_call(
            store,
            runtime,
            reply.call_id,
            policy=ports.edit_policy,
            sampling=ports.agent_sampling,
            implementation_hash=ports.agent_implementation_hash,
            snapshot_output=output / "strategy-origin",
            producer=producer,
        )
        strategy = reuse_strategy(
            store,
            key,
            consumer_run=producer.run_id,
            receipt_path=output / "strategy-reuse.json",
        )
    validate(
        load_json(
            (store.get(key).payload / "response.json").read_text(encoding="utf-8")
        )["content"]
    )
    write_json(output / "context.json", context.to_dict())
    write_json(output / "work.json", work)
    write_json(output / "program.json", strategy[0].to_dict())
    # Lexec starts with isolated memory; optional initial Leval has a separate saved session.
    runtime_snapshot = output / "roles"
    if not runtime_snapshot.exists():
        runtime.save(runtime_snapshot)
    refs = {
        "m0_capability": capability[1]["asset"],
        "m0_signals": signals[1]["asset"],
        "initial_strategy": strategy[1]["asset"],
    }
    if feedback_ref is not None:
        refs["initial_feedback"] = feedback_ref.to_dict()
    if progress:
        progress.update("initial_policy", 1, 1)
    return (
        records,
        context,
        baseline,
        strategy[0],
        refs,
        CostLedger("initial-new-work", tuple(new_events), config.backend),
    )
