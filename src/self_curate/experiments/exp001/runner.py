"""EXP001 control flow. No inner search, Leval, refresh or control-trajectory SFT."""

from pathlib import Path
import csv
import importlib
import math
import platform
import shutil
import subprocess
import importlib.metadata
import uuid
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    CostLedger,
    digest,
    verify_batch,
    parse_program,
)
from self_curate.contracts.parsing import artifact, load_json
from self_curate.contracts.types import require
from self_curate.assets import AssetStore, Producer, freeze_run_inputs
from self_curate.agents.roles import ExecutionAgent
from self_curate.agents.runtime import RoleRuntime
from self_curate.data import MetricView
from self_curate.data.pool import file_sha256, write_json
from self_curate.operators import prepare_selection, ExecutionContext
from self_curate.operators.preparation import publish_preparation
from self_curate.training import TrainingData, TrainingExample, TrainingRequest
from self_curate.training.continuous import wave_boundaries
from .config import ExperimentConfig
from .initial import prepare_initial


def source_hash():
    root = Path(__file__).parents[2]
    return digest(
        {
            p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(root.rglob("*.py"))
        }
    )


def execution_identity():
    root = Path(__file__).parents[4]
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "numpy", "transformers", "peft"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    commit = None
    try:
        top = subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(top).resolve() == root.resolve():
            dirty = subprocess.run(
                ["git", "--no-optional-locks", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            if not dirty.strip():
                commit = subprocess.run(
                    ["git", "--no-optional-locks", "rev-parse", "HEAD"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return environment, commit


def random_program(count, seed):
    return parse_program(
        {
            "schema_version": "1",
            "program_id": "exp001-random",
            "parent_id": None,
            "seed": seed,
            "nodes": [
                {
                    "id": "sample",
                    "op": "SampleSelect",
                    "inputs": ["eligible_pool"],
                    "params": {
                        "method": "uniform",
                        "size": count,
                        "token_target": None,
                        "seed": seed,
                        "strata": [],
                        "missing_policy": "error",
                    },
                }
            ],
            "output": "sample",
        }
    )


def _publish_complete(output, result):
    write_json(output / "summary.json", result)
    checks = {
        p.relative_to(output).as_posix(): file_sha256(p)
        for p in output.rglob("*")
        if p.is_file() and p != output / "completion.json"
    }
    write_json(
        output / "completion.json",
        {"status": "completed", "checksums": checks, "summary_hash": digest(result)},
    )


def inspect_completed(output, config):
    root = Path(output).resolve()
    marker = load_json((root / "completion.json").read_text(encoding="utf-8"))
    lock = load_json((root / "run.json").read_text(encoding="utf-8"))
    require(
        lock["config_hash"] == config.config_hash
        and lock["source_hash"] == source_hash(),
        "Completed run uses another config/source",
    )
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "completion.json"
    }
    require(actual == set(marker["checksums"]), "Run file set changed")
    for name, checksum in marker["checksums"].items():
        path = root / name
        require(
            path.resolve().is_relative_to(root)
            and not path.is_symlink()
            and file_sha256(path) == checksum,
            "Run evidence changed",
        )
    result = load_json((root / "summary.json").read_text(encoding="utf-8"))
    require(
        marker["status"] == "completed" and digest(result) == marker["summary_hash"],
        "Run summary changed",
    )
    return result


def _run_experiment(
    config,
    *,
    output,
    asset_root,
    ports=None,
    resume=False,
    handoff=None,
    parallel_training=False,
):
    require(isinstance(config, ExperimentConfig), "Typed experiment config required")
    output, asset_root = Path(output).resolve(), Path(asset_root).resolve()
    require(
        not output.is_relative_to(asset_root) and not asset_root.is_relative_to(output),
        "Run output and shared assets must be separate",
    )
    if output.exists():
        require(resume, "Existing run requires explicit --resume")
        if (output / "completion.json").is_file():
            return inspect_completed(output, config)
        return _resume(
            config, output, asset_root, ports=ports, parallel_training=parallel_training
        )
    # Validate source pins BEFORE importing a server factory or touching models.
    if config.backend == "real":
        from .sources import load_source_registry

        binding = load_json(config.adapter_settings_json).get("source_registry")
        require(
            type(binding) is dict
            and set(binding) == {"path", "sha256"}
            and type(binding["path"]) is str
            and type(binding["sha256"]) is str,
            "Real experiment needs a pinned reviewed source registry",
        )
        load_source_registry(binding["path"], expected_sha256=binding["sha256"])
    if ports is None:
        module, factory = config.adapter.split(":")
        ports = getattr(importlib.import_module(module), factory)(
            config, output, asset_root
        )
    require(ports.backend == config.backend, "Adapter/domain mismatch")
    proof = ports.preflight(config)
    require(
        type(proof) is dict
        and proof.get("passed") is True
        and proof.get("backend") == config.backend,
        "Adapter preflight did not pass",
    )
    if config.backend == "real":
        require(
            proof.get("gpu_uuids") == list(config.gpu_uuids)
            and proof.get("max_context_tokens") == config.max_context_tokens,
            "Server capacity/resource protocol not verified",
        )
    output.mkdir(parents=True, exist_ok=False)
    from .progress import Progress

    progress = Progress(output / "progress")
    progress.update("initialization", 0, None)
    if config.backend == "real":
        shutil.copyfile(binding["path"], output / "source-registry.json")
    run_id = config.experiment_id + "-" + uuid.uuid4().hex[:16]
    started = perf_counter()
    stamp = source_hash()
    environment, commit = execution_identity()
    source_root = Path(__file__).parents[2]
    write_json(
        output / "source-checksums.json",
        {
            p.relative_to(source_root).as_posix(): file_sha256(p)
            for p in source_root.rglob("*.py")
        },
    )
    if handoff is not None:
        shutil.copyfile(handoff, output / "handoff.md")
    write_json(
        output / "run.json",
        {
            "schema_version": "1",
            "config": config.to_dict(),
            "config_hash": config.config_hash,
            "run_id": run_id,
            "source_hash": stamp,
            "code_commit": commit,
            "environment": environment,
            "preflight": proof,
            "handoff_hash": (
                None if handoff is None else file_sha256(output / "handoff.md")
            ),
            "training_rule": "three_continuous_waves_no_acceptance",
            "benchmark_rule": "all_checkpoints_after_all_training",
        },
    )
    producer = Producer(
        run_id,
        stamp,
        digest({"environment": environment, "preflight": proof}),
        config.config_hash,
        code_commit=commit,
    )
    store = AssetStore(asset_root)
    events = []
    initial_events = []
    runtime = None
    try:
        inputs = ports.prepare(config, output / "preparation")
        inputs.validate(config)
        write_json(output / "development-suite.json", inputs.development.to_dict())
        k = math.floor(len(inputs.records) * config.selection_fraction)
        require(k >= 3, "10% pool is too small for three nonempty waves")
        prepared_asset = inputs.register(
            store, output / "prepared-input-origin", producer=producer
        )
        store._receipt(
            prepared_asset,
            consumer_run=run_id,
            receipt_path=output / "preparation-reuse.json",
            started=perf_counter(),
        )
        records, context, baseline, static_program, refs, initial_ledger = (
            prepare_initial(
                inputs,
                ports,
                config,
                store,
                output / "initial",
                producer,
                event_sink=initial_events,
                progress=progress,
            )
        )
        require(
            isinstance(context, ExecutionContext)
            and context.model_ref == inputs.model
            and context.view.data_ids == tuple(r.data_id for r in records)
            and all(context.view.record(r.data_id) == r for r in records)
            and context.budget.effective_batch == 1
            and context.budget.max_samples == k
            and context.training_seed == config.seed
            and context.training_order == "seeded",
            "Use the frozen full-pool count selection context, without batch-tail sample removal",
        )
        events.extend(initial_ledger.events)
        if "static_m0" in config.modes:
            runtime = RoleRuntime(
                ports.client,
                model=inputs.model,
                session_id=run_id + "-selection",
                backend=config.backend,
                prompts=ports.prompts,
                attempt_id=run_id,
            )
        refs["coding_preparation"] = prepared_asset.reference.to_dict()
        freeze_run_inputs(
            store,
            output / "frozen-inputs.json",
            producer=producer,
            config=config.to_dict(),
            inputs={n: artifact(r) for n, r in refs.items()},
        )
        view = MetricView(records)
        plans, manifests = [], {}
        for mode in config.modes:
            program = (
                random_program(k, config.seed) if mode == "random" else static_program
            )
            require(source_hash() == stamp, "Source changed during experiment")
            if mode == "random":
                selected = prepare_selection(
                    program,
                    context,
                    ports.objective,
                    backend=config.backend,
                    attempt_id=run_id,
                    invocation_id="random-once",
                )
            else:
                selected = ExecutionAgent(runtime).execute(
                    program,
                    context,
                    ports.objective,
                    attempt_id=run_id,
                    invocation_id="static-once",
                    use_model=config.lexec_use_model,
                )
            folder = output / "selections" / mode
            publish_preparation(selected, folder)
            events.extend(selected.ledger.events)
            require(
                selected.status == "feasible" and selected.workflow is not None,
                "One-shot selection infeasible; no silent relaxation or policy search",
            )
            selected.validate_workflow()
            batch = selected.workflow.batch
            order = verify_batch(batch, view, model_ref=inputs.model)
            require(
                len(order) == k, "Selection must produce exactly the frozen 10% count"
            )
            examples = tuple(
                TrainingExample(
                    i,
                    view.record(i).content_hash,
                    view.record(i).tokenization.input_ids,
                    view.record(i).labels,
                )
                for i in order
            )
            data = TrainingData(
                "subset",
                batch.batch_id,
                inputs.model.tokenizer,
                examples,
                (
                    ArtifactRef(
                        "record://selected-batch/" + batch.batch_id, batch.batch_id
                    ),
                ),
                config.backend,
            )
            manifests[mode] = {
                "batch_id": batch.batch_id,
                "training_order": list(order),
                "wave_boundaries": list(wave_boundaries(k)),
                "supervised_tokens": data.supervised_tokens,
            }
            for index, rate in enumerate(config.learning_rates):
                branch = mode + "-lr" + str(index + 1)
                plans.append(
                    (
                        branch,
                        TrainingRequest(
                            branch,
                            run_id,
                            inputs.model,
                            data,
                            batch.batch_id,
                            batch.training_data_hash,
                            config.recipe(rate),
                            config.backend,
                        ),
                    )
                )
        # Both choices and every recipe are frozen before ANY training or final-test execution.
        write_json(
            output / "training-plans.json",
            {"manifests": manifests, "requests": [p.to_dict() for _, p in plans]},
        )
        if runtime is not None:
            runtime.save(output / "selection-roles")
        # Includes optional Lexec inference. Tool wrapper and selected execution
        # are separate records; shared event IDs are merged exactly once.
        known = {e.event_id for e in events}
        if runtime is not None:
            events.extend(e for e in runtime.ledger().events if e.event_id not in known)
        from .recovery import freeze_phase

        phase = {
            "model": inputs.model.to_dict(),
            "development": inputs.development.to_dict(),
            "benchmarks": [v.to_dict() for v in inputs.benchmarks],
            "baseline_report": baseline.report.to_dict(),
            "refs": refs,
            "manifests": manifests,
            "pool_size": len(records),
            "selected": k,
            "asset_root": str(asset_root),
            "elapsed_before_training": perf_counter() - started,
            "ledger": CostLedger(run_id, tuple(events), config.backend).to_dict(),
        }
        freeze_phase(output, phase)
        return _finish(
            config,
            output,
            asset_root,
            ports,
            plans,
            inputs,
            manifests,
            phase["baseline_report"],
            refs,
            len(records),
            k,
            run_id,
            stamp,
            events,
            started,
            progress,
            resume=False,
            parallel_training=parallel_training,
        )
    except Exception as exc:
        if hasattr(ports, "close"):
            ports.close()
        known = {e.event_id for e in events}
        events.extend(e for e in initial_events if e.event_id not in known)
        if runtime is not None:
            known = {e.event_id for e in events}
            events.extend(e for e in runtime.ledger().events if e.event_id not in known)
            try:
                runtime.save(output / "failed-selection-roles")
            except OSError:
                pass
        ledger = getattr(exc, "ledger", None)
        write_json(
            output / "failure.json",
            {
                "status": "blocked",
                "error": str(exc),
                "source_hash": stamp,
                "controller_ledger": CostLedger(
                    run_id, tuple(events), config.backend
                ).to_dict(),
                "failed_operation_ledger": None if ledger is None else ledger.to_dict(),
                "automatic_retry": False,
            },
        )
        raise


def _finish(
    config,
    output,
    asset_root,
    ports,
    plans,
    inputs,
    manifests,
    baseline_report,
    refs,
    pool_size,
    k,
    run_id,
    stamp,
    events,
    started,
    progress,
    *,
    resume,
    parallel_training,
):
    checkpoints_by_branch, dev_reports = {}, {}
    progress.update("training_matrix", 0, len(plans))
    from .training import execute_branch, run_processes

    if hasattr(ports, "enter_phase"):
        ports.enter_phase("training")

    if config.backend == "real":
        # All eight ranks belong to ONE branch; four branches run successively.
        # Published adapters are evaluated after the distributed worker exits.
        ports.release_initial()
        branch_results = {}
        for branch, request in plans:
            require(source_hash() == stamp, "Source changed before FSDP branch")
            ports.enter_phase("training")
            branch_results[branch] = execute_branch(
                ports, config, request, inputs.development, output / "branches" / branch, resume=resume
            )
    elif parallel_training:
        ports.release_initial()
        branch_results = run_processes(config, plans, output, asset_root, resume=resume)
    else:
        branch_results = {}
        for branch, request in plans:
            require(source_hash() == stamp, "Source changed during experiment")
            branch_results[branch] = execute_branch(
                ports,
                config,
                request,
                inputs.development,
                output / "branches" / branch,
                resume=resume,
            )
            events.extend(branch_results[branch][2].events)
    for branch, _ in plans:
        checkpoints, reports, ledger = branch_results[branch]
        known = {e.event_id for e in events}
        events.extend(e for e in ledger.events if e.event_id not in known)
        checkpoints_by_branch[branch] = checkpoints
        dev_reports[branch] = reports
        progress.update("training_matrix", len(dev_reports), len(plans), branch=branch)
    sealed = {
        "config_hash": config.config_hash,
        "source_hash": stamp,
        "plans_hash": file_sha256(output / "training-plans.json"),
        "checkpoints": {
            b: [v.to_dict() for v in values]
            for b, values in checkpoints_by_branch.items()
        },
    }
    from .recovery import atomic_json, read_json

    seal_file = output / "training-sealed.json"
    if seal_file.exists():
        require(resume and read_json(seal_file) == sealed, "Training seal changed")
    else:
        atomic_json(seal_file, sealed)
    seal_hash = digest(sealed)
    # Benchmark tasks are passed only now, never to Lopt/Lexec or checkpoint decisions.
    from .benchmark import evaluate_benchmark

    benchmark_reports = {}
    models = [("initial", inputs.model, None)] + [
        (b + "-stage" + str(v.stage), v.model_ref, v.checkpoint)
        for b, values in checkpoints_by_branch.items()
        for v in values
    ]
    final_caches = {name: {} for name, _, _ in models}
    for suite in inputs.benchmarks:
        if hasattr(ports, "enter_phase"):
            ports.enter_phase("benchmark", suite=suite)
        for name, model, checkpoint in models:
            require(source_hash() == stamp, "Source changed before final benchmark")
            evaluation_id = run_id + "-" + name + "-" + digest(suite.suite_id)[:12]
            report = evaluate_benchmark(
                ports.benchmark_runner,
                model,
                suite,
                checkpoint=checkpoint,
                output=output / "benchmark" / name / suite.suite_id,
                evaluation_id=evaluation_id,
                sealed_training_hash=seal_hash,
                workers=getattr(
                    ports,
                    "benchmark_workers",
                    16 if config.backend == "real" else 1,
                ),
                cache=final_caches[name],
                resume=resume,
                progress=progress,
            )
            benchmark_reports[evaluation_id] = report
    with (output / "checkpoints.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "branch",
                "stage",
                "learning_rate",
                "examples_seen",
                "supervised_tokens",
                "optimizer_steps",
                "model_hash",
                "checkpoint_uri",
            )
        )
        writer.writerow(("initial", 0, "", 0, 0, 0, inputs.model.model_hash, ""))
        for branch, request in plans:
            for v in checkpoints_by_branch[branch]:
                writer.writerow(
                    (
                        branch,
                        v.stage,
                        request.recipe.learning_rate,
                        v.examples_seen,
                        v.supervised_tokens,
                        v.optimizer_steps,
                        v.model_ref.model_hash,
                        v.checkpoint.uri,
                    )
                )
    write_json(
        output / "controller-ledger.json",
        CostLedger(run_id, tuple(events), config.backend).to_dict(),
    )
    result = {
        "schema_version": "1",
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "backend": config.backend,
        "status": "completed",
        "pool_size": pool_size,
        "selected_per_mode": k,
        "seed": config.seed,
        "learning_rates": list(config.learning_rates),
        "observations_per_branch": 4,
        "training_waves": 3,
        "manifests": manifests,
        "initial_development": baseline_report,
        "development": dev_reports,
        "benchmark": benchmark_reports,
        "shared_assets": refs,
        "leval_calls": 0,
        "signal_refreshes_after_initial": 0,
        "control_sft_executed": False,
        "checkpoint_acceptance_applied": False,
        "training_seal_hash": seal_hash,
        "seconds": perf_counter() - started,
        "target_execution": config.backend == "real",
        "all_reports_valid": all(
            r["status"] == "valid" for reports in dev_reports.values() for r in reports
        )
        and all(r["status"] == "valid" for r in benchmark_reports.values()),
        "improvement_verified": False,
        "unreconciled_training_interruptions": [
            e.event_id for e in events if e.event_id.startswith("interrupted-")
        ],
    }
    progress.update("completed", 1, 1)
    if hasattr(ports, "close"):
        ports.close()
    _publish_complete(output, result)
    return result


def _resume(config, output, asset_root, *, ports, parallel_training):
    from types import SimpleNamespace
    from .recovery import load_phase, read_json, atomic_json
    from .progress import Progress
    from .inputs import BenchmarkSuite
    from self_curate.contracts import parse_contract, parse_task_record
    from self_curate.contracts.parsing import model_ref
    from self_curate.orchestration.io import parse_outer_contract

    require(
        (output / "run.json").is_file()
        and (output / "resume-plan.sha256.json").is_file(),
        "Incomplete initialization/selection: reconcile one-shot work before recovery",
    )
    lock = read_json(output / "run.json")
    require(
        lock["config_hash"] == config.config_hash
        and lock["source_hash"] == source_hash(),
        "Recovery config/source changed; use the frozen source revision",
    )
    environment, _ = execution_identity()
    require(environment == lock["environment"], "Recovery environment changed")
    phase = load_phase(output)
    require(phase["asset_root"] == str(asset_root), "Recovery asset root changed")
    if ports is None:
        module, factory = config.adapter.split(":")
        ports = getattr(importlib.import_module(module), factory)(
            config, output, asset_root
        )
    require(ports.backend == config.backend, "Recovery backend changed")
    proof = ports.preflight(config)
    require(proof == lock["preflight"], "Recovery preflight/resources changed")
    suites = []
    for v in phase["benchmarks"]:
        suites.append(
            BenchmarkSuite(
                v["suite_id"],
                tuple(parse_task_record(t) for t in v["tasks"]),
                tuple(v["seeds"]),
                artifact(v["sampling"]),
                artifact(v["protocol"]),
                v["backend"],
            )
        )
    inputs = SimpleNamespace(
        model=model_ref(phase["model"]),
        development=parse_outer_contract("TaskSuite", phase["development"]),
        benchmarks=tuple(suites),
    )
    ports.attach_training(config, inputs.model)
    plans = [
        (v["training_id"], parse_outer_contract("TrainingRequest", v))
        for v in read_json(output / "training-plans.json")["requests"]
    ]
    events = list(parse_contract("CostLedger", phase["ledger"]).events)
    recovery_id = uuid.uuid4().hex
    atomic_json(
        output / "recovery" / (recovery_id + ".json"),
        {
            "source_hash": source_hash(),
            "run_id": lock["run_id"],
            "previous_failure": (
                read_json(output / "failure.json")
                if (output / "failure.json").exists()
                else None
            ),
        },
    )
    progress = Progress(output / "progress")
    progress.update("recovering", 0, len(plans))
    return _finish(
        config,
        output,
        asset_root,
        ports,
        plans,
        inputs,
        phase["manifests"],
        phase["baseline_report"],
        phase["refs"],
        phase["pool_size"],
        phase["selected"],
        lock["run_id"],
        lock["source_hash"],
        events,
        perf_counter() - phase["elapsed_before_training"],
        progress,
        resume=True,
        parallel_training=parallel_training,
    )


def run_experiment(
    config,
    *,
    output,
    asset_root,
    ports=None,
    resume=False,
    handoff=None,
    parallel_training=False,
):
    from self_curate.training.recovery import lease

    root = Path(output).resolve()
    with lease(root.parent / ("." + root.name + ".lease")):
        return _run_experiment(
            config,
            output=root,
            asset_root=asset_root,
            ports=ports,
            resume=resume,
            handoff=handoff,
            parallel_training=parallel_training,
        )
