"""Isolated branch workers: one optimizer/process/GPU, no threaded Torch RNG."""

from pathlib import Path
from dataclasses import replace
import os
import subprocess
import sys
import time
import uuid

from self_curate.contracts import (
    ArtifactRef,
    CostLedger,
    CostEvent,
    ResourceUsage,
    digest,
    parse_contract,
)
from self_curate.contracts.parsing import artifact, load_json, model_ref
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256, write_json
from self_curate.orchestration.io import parse_outer_contract
from self_curate.training.continuous import ContinuousTrainingService, StageCheckpoint
from self_curate.training import TrainingFailure
from self_curate.data import OuterExposureHistory, OuterTrainingExposure


def _execute_branch(ports, config, request, suite, output, *, resume=False):
    output = Path(output)
    if (output / "branch-complete.json").exists():
        require(resume, "Existing branch requires resume")
        return load_branch(request, output)
    output.mkdir(parents=True, exist_ok=resume)
    from .progress import Progress
    from self_curate.training.recovery import atomic_json, read_json

    progress = Progress(output / "progress", label=request.training_id)
    reports, dev_events = [], []
    previous_failures = []
    if (output / "branch-failure.json").exists():
        previous = read_json(output / "branch-failure.json")
        require(
            previous["request_hash"] == request.record_hash,
            "Failed branch changed request",
        )
        previous_failures = [
            e
            for e in parse_contract("CostLedger", previous["ledger"]).events
            if e.status != "completed"
        ]
        atomic_json(
            output / "recovery-attempts" / (uuid.uuid4().hex + ".json"), previous
        )
    history = OuterExposureHistory(
        request.attempt_id + "/" + request.training_id, config.backend
    )
    if (output / "exposure-index.json").exists():
        from self_curate.data.outer_usage import load_outer_exposures

        index = read_json(output / "exposure-index.json")
        history = load_outer_exposures(
            output / "exposure-history" / index["history_hash"],
            expected_hash=index["history_hash"],
            expected_branch=history.branch_id,
        )
    previous_model, previous_end, previous_steps = request.starting_model, 0, 0
    # A killed process cannot report its last uncommitted work. Preserve this
    # uncertainty separately from verified successful data exposure.
    recovery_costs = output / "recovery-costs.json"
    interrupted = []
    if recovery_costs.exists():
        saved = read_json(recovery_costs)
        require(
            saved["request_hash"] == request.record_hash,
            "Recovery costs belong to another request",
        )
        interrupted = list(parse_contract("CostLedger", saved["ledger"]).events)
    if (
        resume
        and (output / "training").exists()
        and not (output / "training/complete.json").exists()
        and not (output / "branch-failure.json").exists()
    ):
        attempt = "interrupted-" + uuid.uuid4().hex
        receipt_hash = digest({"request_hash": request.record_hash, "attempt": attempt})
        receipt = ArtifactRef(
            "record://unaccounted-interrupted-training/" + receipt_hash, receipt_hash
        )
        interrupted.append(
            CostEvent(
                attempt,
                request.attempt_id,
                "training",
                request.training_id,
                request.starting_model.model_hash,
                "failed",
                ResourceUsage(),
                receipt,
                config.backend,
            )
        )
        history = history.mark_unresolved(attempt, receipt)
        history.persist(output / "exposure-history")
        index_file = output / "exposure-index.json"
        index = (
            read_json(index_file)
            if index_file.exists()
            else {
                "branch_id": history.branch_id,
                "stage": 0,
                "model_ref": request.starting_model.to_dict(),
            }
        )
        atomic_json(
            index_file,
            {
                **index,
                "history_hash": history.history_hash,
                "blocked_outer_exposure": True,
            },
        )
        atomic_json(
            recovery_costs,
            {
                "request_hash": request.record_hash,
                "ledger": CostLedger(
                    request.training_id, tuple(interrupted), config.backend
                ).to_dict(),
            },
        )
    physical = {e.event_id: e for e in previous_failures}
    for event in interrupted:
        require(
            event.event_id not in physical or physical[event.event_id] == event,
            "Conflicting interrupted-work costs",
        )
        physical[event.event_id] = event
    previous_failures = list(physical.values())

    def checkpoint_ready(value):
        nonlocal history, previous_model, previous_end, previous_steps
        if config.backend == "real":
            # FSDP has finished and released all eight devices before emit.
            ports.enter_phase("development")
        event = OuterTrainingExposure(
            request.training_id + "-wave" + str(value.stage),
            request.attempt_id,
            request.record_hash,
            request.batch_id,
            tuple(
                (x.data_id, x.content_hash)
                for x in request.data.examples[previous_end : value.examples_seen]
            ),
            previous_model.model_hash,
            value.model_ref.model_hash,
            value.usage.training_supervised_tokens,
            value.optimizer_steps - previous_steps,
            value.checkpoint,
            config.backend,
        )
        if value.stage <= len(history.events):
            require(
                history.events[value.stage - 1] == event,
                "Recovery changed outer training exposure",
            )
        else:
            history = replace(history, events=(*history.events, event))
        history.persist(output / "exposure-history")
        if (
            not (output / "exposure-index.json").exists()
            or value.stage >= read_json(output / "exposure-index.json")["stage"]
        ):
            atomic_json(
                output / "exposure-index.json",
                {
                    "branch_id": history.branch_id,
                    "history_hash": history.history_hash,
                    "model_ref": value.model_ref.to_dict(),
                    "stage": value.stage,
                    "blocked_outer_exposure": bool(history.unresolved),
                },
            )
        previous_model, previous_end, previous_steps = (
            value.model_ref,
            value.examples_seen,
            value.optimizer_steps,
        )
        evaluation = ports.validator.evaluate(
            value.model_ref,
            suite,
            attempt_id=request.attempt_id,
            evaluation_id=request.attempt_id
            + "-"
            + request.training_id
            + "-dev-"
            + str(value.stage),
            checkpoint=value.checkpoint,
            output=output / ("development-" + str(value.stage)),
            **({"resume": True} if resume else {}),
            **(
                {"progress": progress}
                if isinstance(
                    ports.validator,
                    __import__(
                        "self_curate.validation", fromlist=["TaskValidator"]
                    ).TaskValidator,
                )
                else {}
            ),
        )
        dev_events.extend(evaluation.ledger.events)
        reports.append(evaluation.report.to_dict())

    try:
        backend = ports.training_backend(request.training_id)
        if hasattr(backend, "progress"):
            backend.progress = progress
        checkpoints, ledger = ContinuousTrainingService(backend).train(
            request, output / "training", on_checkpoint=checkpoint_ready, resume=resume
        )
    except Exception as exc:
        known = {e.event_id: e for e in (*dev_events, *previous_failures)}
        if isinstance(getattr(exc, "ledger", None), CostLedger):
            for e in exc.ledger.events:
                require(
                    e.event_id not in known or known[e.event_id] == e,
                    "Conflicting branch physical costs",
                )
                known[e.event_id] = e
        ledger = CostLedger(request.training_id, tuple(known.values()), config.backend)
        if any(
            e.operation == "training"
            and e.status != "completed"
            and e.usage.training_supervised_tokens != 0
            for e in ledger.events
        ):
            failure_hash = digest({"request": request.record_hash, "error": str(exc)})
            history = history.mark_unresolved(
                request.training_id + "-unfinished-wave",
                ArtifactRef(
                    "record://continuous-exposure-unknown/" + failure_hash, failure_hash
                ),
            )
            history.persist(output / "exposure-history")
            write_json(
                output / "exposure-index.json",
                {
                    "branch_id": history.branch_id,
                    "history_hash": history.history_hash,
                    "model_ref": previous_model.to_dict(),
                    "stage": len(history.events),
                    "blocked_outer_exposure": True,
                },
            )
        write_json(
            output / "branch-failure.json",
            {
                "request_hash": request.record_hash,
                "error": str(exc),
                "ledger": ledger.to_dict(),
            },
        )
        raise TrainingFailure(str(exc), ledger=ledger) from exc
    result = {
        "request_hash": request.record_hash,
        "checkpoints": [v.to_dict() for v in checkpoints],
        "development": reports,
        "ledger": CostLedger(
            request.training_id,
            (*ledger.events, *dev_events, *previous_failures),
            config.backend,
        ).to_dict(),
    }
    progress.update("completed", 3, 3)
    write_json(output / "branch-result.json", result)
    write_json(
        output / "branch-complete.json",
        {
            "checksums": {
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            }
        },
    )
    return checkpoints, reports, parse_contract("CostLedger", result["ledger"])


def load_branch(request, output):
    root = Path(output)
    checks = load_json((root / "branch-complete.json").read_text(encoding="utf-8"))[
        "checksums"
    ]
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "branch-complete.json"
    }
    require(set(checks) == actual, "Incomplete branch output")
    for name, checksum in checks.items():
        path = root / name
        require(
            path.resolve().is_relative_to(root.resolve())
            and not path.is_symlink()
            and file_sha256(path) == checksum,
            "Branch output changed",
        )
    value = load_json((root / "branch-result.json").read_text(encoding="utf-8"))
    require(
        value["request_hash"] == request.record_hash, "Branch trained another request"
    )
    checkpoints = tuple(
        StageCheckpoint(
            **{
                **v,
                "model_ref": model_ref(v["model_ref"]),
                "checkpoint": artifact(v["checkpoint"]),
                "usage": parse_contract("ResourceUsage", v["usage"]),
            }
        )
        for v in value["checkpoints"]
    )
    require(
        len(checkpoints) == 3
        and tuple(v.stage for v in checkpoints) == (1, 2, 3)
        and all(v.request_hash == request.record_hash for v in checkpoints),
        "Branch checkpoint prefix mismatch",
    )
    return (
        checkpoints,
        value["development"],
        parse_contract("CostLedger", value["ledger"]),
    )


def run_worker(config, output, asset_root, branch, *, resume=False):
    from .runner import source_hash
    import importlib

    root = Path(output).resolve()
    lock = load_json((root / "run.json").read_text(encoding="utf-8"))
    require(
        lock["source_hash"] == source_hash()
        and lock["config_hash"] == config.config_hash,
        "Worker source/config changed",
    )
    plans = load_json((root / "training-plans.json").read_text(encoding="utf-8"))
    matches = [p for p in plans["requests"] if p["training_id"] == branch]
    require(len(matches) == 1, "Unknown training branch")
    request = parse_outer_contract("TrainingRequest", matches[0])
    suite = parse_outer_contract(
        "TaskSuite",
        load_json((root / "development-suite.json").read_text(encoding="utf-8")),
    )
    module, factory = config.adapter.split(":")
    ports = getattr(importlib.import_module(module), factory)(
        config, root, Path(asset_root)
    )
    require(ports.backend == config.backend, "Worker backend changed")
    ports.attach_training(config, request.starting_model)
    return execute_branch(
        ports, config, request, suite, root / "branches" / branch, resume=resume
    )


def run_processes(config, plans, output, asset_root, *, resume=False):
    """Four fixed independent jobs; CUDA visibility is an authorized UUID, not rank math."""
    output = Path(output).resolve()
    config_file = output / "resolved-config.json"
    write_json(config_file, config.to_dict())
    logs = output / "worker-logs"
    logs.mkdir(exist_ok=resume)
    children = []
    try:
        for index, (branch, request) in enumerate(plans):
            if (
                resume
                and (output / "branches" / branch / "branch-complete.json").exists()
            ):
                load_branch(request, output / "branches" / branch)
                continue
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            environment["PYTHONPATH"] = (
                str(Path(__file__).parents[3])
                + os.pathsep
                + environment.get("PYTHONPATH", "")
            )
            if config.backend == "real":
                environment["CUDA_VISIBLE_DEVICES"] = config.gpu_uuids[index]
                environment["EXP001_TRAIN_GPU_UUID"] = config.gpu_uuids[index]
            stream = (
                logs
                / (branch + ("-resume-" + uuid.uuid4().hex if resume else "") + ".log")
            ).open("w", encoding="utf-8")
            args = [
                sys.executable,
                "-m",
                "self_curate.experiments.exp001",
                "--config",
                str(config_file),
                "--output",
                str(output),
                "--assets",
                str(Path(asset_root).resolve()),
                "--train-branch",
                branch,
            ]
            if resume:
                args.append("--resume")
            process = subprocess.Popen(
                args,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            children.append((branch, process, stream))
        while any(p.poll() is None for _, p, _ in children):
            time.sleep(0.2)
        if not all(p.returncode == 0 for _, p, _ in children):
            known, unresolved = {}, []
            for branch, request in plans:
                branch_dir = output / "branches" / branch
                if (branch_dir / "branch-complete.json").exists():
                    ledger = load_branch(request, branch_dir)[2]
                elif (branch_dir / "branch-failure.json").exists():
                    ledger = parse_contract(
                        "CostLedger",
                        load_json(
                            (branch_dir / "branch-failure.json").read_text(
                                encoding="utf-8"
                            )
                        )["ledger"],
                    )
                else:
                    unresolved.append(branch)
                    continue
                for event in ledger.events:
                    require(
                        event.event_id not in known or known[event.event_id] == event,
                        "Conflicting matrix costs",
                    )
                    known[event.event_id] = event
            ledger = CostLedger(
                config.experiment_id, tuple(known.values()), config.backend
            )
            write_json(
                output / "matrix-failure.json",
                {"ledger": ledger.to_dict(), "unresolved_branches": unresolved},
            )
            raise TrainingFailure(
                "A branch failed; preserve outputs and audit matrix-failure.json before recovery",
                ledger=ledger,
            )
    finally:
        for _, process, stream in children:
            # Do not orphan remaining fixed-budget jobs after a launch failure.
            process.wait()
            stream.close()
    return {
        branch: load_branch(request, output / "branches" / branch)
        for branch, request in plans
    }


def execute_branch(ports, config, request, suite, output, *, resume=False):
    from self_curate.training.recovery import lease

    root = Path(output).resolve()
    with lease(root.parent / ("." + root.name + ".lease")):
        return _execute_branch(ports, config, request, suite, root, resume=resume)
