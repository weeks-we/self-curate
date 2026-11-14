"""One server entry for download, preparation, capacity, smoke, execution and reuse."""

from dataclasses import replace
from pathlib import Path
import os, secrets, random, re
from self_curate.contracts import (
    ModelRef,
    TokenizerRef,
    ObjectiveSpec,
    SkillWeight,
    EditPolicy,
    digest,
)
from self_curate.contracts.types import require
from self_curate.contracts.parsing import artifact, model_ref
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, read_json, lease
from self_curate.execution.artifacts import publish_json, read_artifact
from self_curate.execution.runtime import lock_runtime
from self_curate.preparation.sources import download_sources, read_dataset
from self_curate.preparation.provider import AnnotationClient
from self_curate.preparation.rendering import ChatTemplateTokenizer
from self_curate.preparation.pipeline import prepare_experiment
from self_curate.preparation.skills import SKILLS


def prepare_model(registry, receipt, lora, output):
    import torch, transformers
    from self_curate.training.peft_store import LoRASettings, initialize_adapter
    from self_curate.training.artifacts import checkpoint_fingerprint

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    base = Path(receipt["root"])
    cfg = LoRASettings.from_dict(lora)
    import importlib.metadata

    backend = (
        "transformers-fast-payload-offsets-v2/"
        + str(transformers.__version__)
        + "/"
        + importlib.metadata.version("tokenizers")
    )
    identity = TokenizerRef(
        registry["model"]["repo_id"],
        registry["model"]["revision"],
        file_sha256(base / "tokenizer.json"),
        "native-chat-template",
        file_sha256(base / "chat_template.jinja"),
        backend,
    )
    model = ModelRef(
        registry["model"]["repo_id"],
        registry["model"]["revision"],
        checkpoint_fingerprint(base)[0],
        identity,
        "lora:" + cfg.config_hash,
    )
    adapter = root / "initial-adapter"
    file = root / "initial-model.json"
    if file.exists():
        existing = model_ref(read_json(file))
        require(
            replace(existing, adapter_hash=None) == model,
            "Initial model source/tokenizer/config changed",
        )
        require(
            checkpoint_fingerprint(adapter)[0] == existing.adapter_hash,
            "Common initial adapter changed",
        )
        return existing
    require(
        not adapter.exists(),
        "Interrupted adapter initialization needs reconciliation; never overwrite it",
    )
    network = transformers.AutoModelForImageTextToText.from_pretrained(
        str(base),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    checksum = initialize_adapter(
        network,
        adapter,
        cfg,
        base_model_id=model.base_model_id,
        revision=model.revision,
    )
    model = replace(model, adapter_hash=checksum)
    atomic_json(file, model.to_dict())
    del network
    import gc

    gc.collect()
    return model


def bootstrap(plan_path, *, gpu_uuids, stage, execution, resume=False):
    require(
        execution == "server",
        "Server bootstrap must be explicitly authorized as server execution",
    )
    plan = read_json(plan_path)
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", plan["run_id"]) is not None,
        "Run ID must be a single safe directory name",
    )
    require(Path(plan["storage_root"]).is_absolute(), "Absolute storage root required")
    root = Path(plan["storage_root"]).resolve()
    require(
        root.is_absolute()
        and len(gpu_uuids) == 8
        and len(set(gpu_uuids)) == 8
        and all(x.startswith("GPU-") for x in gpu_uuids),
        "Absolute storage root and eight explicitly authorized GPU UUIDs required",
    )
    project = Path(__file__).parents[3]
    registry_path = project / "configs/experiments/exp001/sources.json"
    from self_curate.experiments.exp001.sources import load_source_registry

    registry = load_source_registry(registry_path)
    root.mkdir(parents=True, exist_ok=True)
    execution_root = root / "executions" / plan["run_id"]
    execution_root.mkdir(parents=True, exist_ok=True)
    from .environment import inspect_server

    environment = inspect_server(gpu_uuids)
    atomic_json(execution_root / "environment.json", environment)
    from self_curate.experiments.exp001.progress import Progress

    progress = Progress(execution_root / "progress", label="server-bootstrap")
    keyfile = root / "private/inference.key"
    keyfile.parent.mkdir(exist_ok=True)
    if not keyfile.exists():
        fd = os.open(keyfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32))
    os.environ["SELF_CURATE_INFERENCE_KEY"] = keyfile.read_text().strip()
    with lease(execution_root / ".server-bootstrap.lease"):
        frozen = {
            "plan": plan,
            "gpu_uuids": list(gpu_uuids),
            "environment_hash": digest(environment),
            "source_hash": __import__(
                "self_curate.experiments.exp001.runner", fromlist=["source_hash"]
            ).source_hash(),
        }
        lock = execution_root / "bootstrap.lock.json"
        if lock.exists():
            require(
                read_json(lock) == frozen,
                "Bootstrap plan/source/GPU assignment changed; use a new storage execution root and reuse only pinned assets",
            )
        else:
            atomic_json(lock, frozen)
        progress.update("source_download", 0, len(registry["required_sources"]))
        receipts = download_sources(
            registry, root / "sources", execution=execution, allow_download=False
        )
        from .assets import verify_assets

        verify_assets(root / "offline-assets.json", registry)
        progress.update("source_download", len(receipts), len(receipts))
        if stage == "download":
            return {
                "status": "downloaded",
                "sources": str(root / "sources/sources.lock.json"),
            }
        setup = execution_root / "setup"
        setup.mkdir(exist_ok=True)
        runtime_file = setup / "runtime-lock.json"
        if runtime_file.exists():
            from self_curate.execution.runtime import verify_runtime

            runtime = publish_json(
                setup / "runtime-reference.json", read_json(runtime_file)
            )
            verify_runtime(
                read_json(runtime_file),
                kinds=("claude_sdk", "swe_smith", "swe_bench", "harbor"),
            )
            runtime = artifact(
                {"uri": runtime_file.as_uri(), "sha256": file_sha256(runtime_file)}
            )
        else:
            runtime = lock_runtime(
                runtime_file, kinds=("claude_sdk", "swe_smith", "swe_bench", "harbor")
            )
        common_lock = (
            root
            / "environments"
            / digest(read_json(runtime_file))
            / "runtime-lock.json"
        )
        if common_lock.exists():
            require(
                read_json(common_lock) == read_json(runtime_file),
                "Shared runtime lock changed",
            )
            runtime = artifact(
                {"uri": common_lock.as_uri(), "sha256": file_sha256(common_lock)}
            )
        else:
            runtime = publish_json(common_lock, read_json(runtime_file))
        template = read_json(
            project / "configs/experiments/exp001/server.template.json"
        )
        settings = template["adapter_settings"]
        import importlib.metadata

        libraries = {
            n: importlib.metadata.version(n)
            for n in ("transformers", "peft", "tokenizers")
        }
        model_root = (
            root
            / "models"
            / digest(
                {
                    "model": registry["model"],
                    "lora": settings["lora"],
                    "libraries": libraries,
                }
            )[:24]
        )
        progress.update("initial_adapter", 0, 1)
        model = prepare_model(registry, receipts["model"], settings["lora"], model_root)
        progress.update("initial_adapter", 1, 1)
        template["gpu_uuids"] = list(gpu_uuids)
        settings["initial_model_file"] = str(model_root / "initial-model.json")
        settings["model_store"] = {
            "base_path": receipts["model"]["root"],
            "initial_adapter": str(model_root / "initial-adapter"),
            "base_fingerprint": model.weights_hash,
            "loader": "AutoModelForImageTextToText",
        }
        settings["runtime_lock"] = runtime.to_dict()
        settings["source_registry"] = {
            "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        }
        settings["server_environment_hash"] = digest(environment)
        raw = read_dataset(receipts["author_history"])
        tasks = read_dataset(receipts["development_tasks"])
        benchmarks = {
            name: read_dataset(receipts[name])
            for name in ("benchmark_full", "benchmark_lite", "benchmark_verified")
        }
        terminal_root = Path(receipts["terminal_bench"]["root"])
        terminal_tasks = sorted(
            p.parent for p in terminal_root.glob(plan["terminal_task_glob"])
        )
        require(
            bool(terminal_tasks),
            "Pinned Terminal source contains no tasks at the configured task glob",
        )
        from self_curate.execution.native import ImagePins

        native_tasks = [*tasks, *(t for values in benchmarks.values() for t in values)]
        harnesses = {t["instance_id"]: "swe_smith" for t in tasks}
        harnesses.update(
            {
                t["instance_id"]: "swe_bench"
                for values in benchmarks.values()
                for t in values
            }
        )
        images = ImagePins(
            native_tasks,
            root / "task-images/image-pins.json",
            harnesses=harnesses,
            allow_pull=False,
        )
        from transformers import AutoTokenizer

        renderer = ChatTemplateTokenizer(
            AutoTokenizer.from_pretrained(
                receipts["model"]["root"],
                local_files_only=True,
                trust_remote_code=False,
            ),
            model.tokenizer,
        )
        teacher = {**plan["teacher"], "backend": "real", "execution": "server"}
        client = AnnotationClient(teacher, root / "annotations")
        review = None
        if plan.get("review_teacher"):
            review = AnnotationClient(
                {**plan["review_teacher"], "backend": "real", "execution": "server"},
                root / "review-annotations",
            )
        prep = {
            **plan["preparation"],
            "backend": "real",
            "images": images,
            "seed": template["seed"],
            "max_context_tokens": template["max_context_tokens"],
            "source_registry_sha256": file_sha256(registry_path),
            "task_budget": read_json(
                project / "configs/experiments/exp001/swe-task-budget.json"
            ),
            "task_protocol": read_json(
                project / "configs/experiments/exp001/swe-task-protocol.json"
            ),
        }
        prep_hash = digest(
            {
                "preparation": dict(prep),
                "model": model.to_dict(),
                "teacher": teacher,
                "review": plan.get("review_teacher"),
                "runtime": runtime.sha256,
                "code": {
                    p.name: file_sha256(p)
                    for p in Path(__file__)
                    .parents[1]
                    .joinpath("preparation")
                    .glob("*.py")
                },
            }
        )
        prepared_root = root / "prepared" / prep_hash[:24]
        manifest = prepare_experiment(
            prep,
            rows=raw,
            tasks=tasks,
            benchmarks=benchmarks,
            terminal_tasks=terminal_tasks,
            renderer=renderer,
            client=client,
            model=model,
            receipts=receipts,
            runtime_lock=runtime,
            output=prepared_root,
            review_client=review,
            progress=progress,
        )
        progress.update("data_preparation", 1, 1)
        settings["prepared_manifest"] = str(manifest)
        from self_curate.preparation.sanity import check_tasks
        from self_curate.experiments.exp001.inputs import load_prepared_manifest

        prepared = load_prepared_manifest(manifest)
        require(
            int(len(prepared.records) * template["selection_fraction"]) >= 3,
            "Too few admitted trajectories for three nonempty training waves; inspect the admission report before GPU execution",
        )
        dev_instances = [
            read_artifact(t.payload)["instance"] for t in prepared.development.tasks
        ]
        sanity_root = (
            root
            / "sanity"
            / digest(
                {
                    "prepared": file_sha256(manifest),
                    "runtime": runtime.sha256,
                    "timeout": prep["grader_timeout_seconds"],
                }
            )[:24]
        )
        check_tasks(
            dev_instances,
            images,
            read_json(runtime_file),
            sanity_root / "development",
            execution=execution,
            timeout=prep["grader_timeout_seconds"],
        )
        # A source audit is a separate capability claim, not a fabricated pass on every training row.
        selected = sorted({r["instance_id"] for r in raw})
        random.Random(template["seed"]).shuffle(selected)
        lookup = {t["instance_id"]: t for t in tasks}
        audit_tasks = [
            lookup[i]
            for i in selected[: plan["preparation"]["training_audit_size"]]
            if i in lookup
        ]
        check_tasks(
            audit_tasks,
            images,
            read_json(runtime_file),
            sanity_root / "training-audit",
            execution=execution,
            timeout=prep["grader_timeout_seconds"],
        )
        atomic_json(
            setup / "training-audit-coverage.json",
            {
                "sampled": len(audit_tasks),
                "raw_rows": len(raw),
                "sampling_seed": template["seed"],
                "policy": "uniform task-ID audit; no claim of full trajectory command replay",
                "unreplayed_training_trajectories": len(raw),
            },
        )
        shared_protocols = root / "protocols"
        shared_protocols.mkdir(exist_ok=True)
        value = {
            "enabled": False,
            "reason": "EXP001 uses neither NICE nor checkpoint acceptance",
        }
        protocol = publish_json(
            shared_protocols / digest(value) / "protocol.json", value
        )
        objective = ObjectiveSpec(
            "exp001-initial-selection",
            tuple(SkillWeight(s, 1 / len(SKILLS)) for s in SKILLS),
            (),
            protocol,
            protocol,
        )
        settings["objective"] = str(setup / "objective.json")
        atomic_json(Path(settings["objective"]), objective.to_dict())
        settings["edit_policy"] = str(setup / "edit-policy.json")
        atomic_json(
            Path(settings["edit_policy"]),
            EditPolicy(0.1, 0.01, 0.99, 4, 128, 128, 1, 600).to_dict(),
        )
        value = {"temperature": 0, "seed": template["seed"], "max_tokens": 8192}
        settings["agent_sampling"] = publish_json(
            shared_protocols / digest(value) / "sampling.json", value
        ).to_dict()
        settings["prompts"] = str(project / "configs/experiments/exp001/prompts")
        import subprocess

        bridge = json_module_load(
            subprocess.run(
                [
                    "docker",
                    "network",
                    "inspect",
                    "bridge",
                    "--format",
                    "{{json .IPAM.Config}}",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            ).stdout
        )[0]["Gateway"]
        entries = [
            {
                "base_url": "http://127.0.0.1:" + str(plan["base_port"] + i),
                "port": plan["base_port"] + i,
                "gpu_uuid": uuid,
                "slots": 2,
                "base_weights_hash": model.weights_hash,
                "tokenizer_hash": model.tokenizer.tokenizer_hash,
                "api_key_env": "SELF_CURATE_INFERENCE_KEY",
                "container_base_url": "http://"
                + bridge
                + ":"
                + str(plan["base_port"] + 100 + i),
                "native_capture_root": str(execution_root / "services/native-tokens" / uuid),
                "relay_host": bridge,
                "relay_port": plan["base_port"] + 100 + i,
                "relay_grant_root": str(execution_root / "private/relay-grants"),
            }
            for i, uuid in enumerate(gpu_uuids)
        ]
        pool = publish_json(
            setup / "serving-pool.json", {"schema_version": "1", "entries": entries}
        )
        settings["task_execution"].update(
            serving_pool=pool.to_dict(),
            initial_adapter=settings["model_store"]["initial_adapter"],
            tokenizer_path=receipts["model"]["root"],
            slot_root=str(root / "resource-slots"),
            harbor_gpu_slot_root=str(root / "task-gpu-slots"),
            harbor_installed_root=str(root / "task-images/harbor-installed"),
            offline_assets={
                "uri": (root / "offline-assets.json").as_uri(),
                "sha256": file_sha256(root / "offline-assets.json"),
            },
        )
        from self_curate.execution.harbor_resources import gpu_layout

        demands = [sum(gpu_layout(p).values()) for p in terminal_tasks]
        demand = max(demands, default=0)
        require(
            type(demand) is int and demand < 8,
            "Terminal native GPU demand leaves no model-serving GPU; provide a separately provisioned environment",
        )
        reserved = list(gpu_uuids[-demand:]) if demand else []
        settings["terminal_reserved_gpu_uuids"] = reserved
        settings["task_execution"]["harbor_gpu_pool"] = reserved
        settings["services"] = {
            **plan["services"],
            "base_path": receipts["model"]["root"],
            "runtime_root": str(execution_root / "services"),
            "max_context_tokens": template["max_context_tokens"],
            "lora_rank": settings["lora"]["rank"],
        }
        atomic_json(setup / "exp001.pending.json", template)
        from self_curate.experiments.exp001.config import load_config

        prepared.validate(load_config(setup / "exp001.pending.json"))
        if stage == "prepare":
            return {
                "status": "prepared",
                "manifest": str(manifest),
                "pending_config": str(setup / "exp001.pending.json"),
            }
        progress.update("capacity_and_smoke", 0, 1)
        capacity = run_preflight(template, prepared, model, execution_root, plan)
        progress.update("capacity_and_smoke", 1, 1)
        settings["capacity_receipt"] = capacity.to_dict()
        atomic_json(setup / "exp001.server.json", template)
        if stage == "preflight":
            return {"status": "ready", "config": str(setup / "exp001.server.json")}
        from self_curate.experiments.exp001.runner import run_experiment
        from self_curate.experiments.exp001.delivery import export_bundle, verify_bundle

        output = root / "runs" / plan["run_id"]
        progress.update("experiment", 0, 1)
        result = run_experiment(
            load_config(setup / "exp001.server.json"),
            output=output,
            asset_root=root / "shared-assets",
            resume=resume,
            handoff=project / "docs/handoffs/EXP001_SERVER_V3.md",
            parallel_training=True,
        )
        bundle = output.parent / "deliverables" / output.name
        if bundle.exists():
            verify_bundle(bundle)
        else:
            export_bundle(output, bundle)
        progress.update("completed", 1, 1)
        return {
            "status": result["status"],
            "all_reports_valid": result["all_reports_valid"],
            "results_zip": str(bundle / "results.zip"),
        }


def run_preflight(config, prepared, model, root, plan):
    from .capacity import check_backward, check_inference
    from self_curate.execution.services import ServiceManager
    from self_curate.execution.runner import build_task_runner
    from self_curate.validation import TaskRunRequest

    settings = config["adapter_settings"]
    receipt = root / "setup/capacity-receipt.json"
    identity = digest(
        {
            "model": model.to_dict(),
            "gpus": config["gpu_uuids"],
            "lora": settings["lora"],
            "context": config["max_context_tokens"],
            "services": settings["services"],
            "runtime": settings["runtime_lock"],
            "prepared": settings["prepared_manifest"],
        }
    )
    if receipt.exists():
        saved = read_json(receipt)
        require(saved["identity"] == identity, "Preflight identity changed")
        return artifact({"uri": receipt.as_uri(), "sha256": file_sha256(receipt)})
    runner = build_task_runner(settings["task_execution"], model, execution="server")
    manager = ServiceManager(
        settings["services"],
        runner.router.entries,
        config["gpu_uuids"],
        execution="server",
    )
    try:
        manager.transition("scoring")
        rows = check_backward(config, root / "capacity/backward")
        manager.transition("smoke")
        runner.router.set_active(config["gpu_uuids"])
        infer = check_inference(
            runner.router,
            model,
            max_context=config["max_context_tokens"],
            max_output=32768,
            output=root / "capacity/inference",
        )
        # The real M0 development evaluation is produced ONCE and published as
        # the same shared asset consumed by the experiment's common prefix.
        from self_curate.assets import (
            AssetStore,
            Producer,
            register_capability,
            reuse_capability,
        )
        from self_curate.experiments.exp001.server import task_execution_identity
        from self_curate.validation.parallel import ParallelTaskValidator
        from self_curate.experiments.exp001.runner import source_hash

        store = AssetStore(root.parents[1] / "shared-assets")
        execution_hash = task_execution_identity(settings)
        found = reuse_capability(
            store,
            model=model,
            suite=prepared.development,
            execution_hash=execution_hash,
            consumer_run="server-readiness",
        )
        if found is None:
            output = root / "capacity/initial-development"
            evaluated = ParallelTaskValidator(runner, 16).evaluate(
                model,
                prepared.development,
                attempt_id="server-readiness",
                evaluation_id="common-m0-"
                + digest((model.model_hash, prepared.development.record_hash))[:20],
                output=output,
                resume=output.exists(),
            )
            require(
                evaluated.report.status == "valid",
                "M0 development evaluation has unscored tasks",
            )
            producer = Producer(
                "server-initial-development",
                source_hash(),
                digest(
                    {
                        "capacity": rows,
                        "runtime": read_artifact(artifact(settings["runtime_lock"])),
                    }
                ),
                digest(config),
            )
            register_capability(
                store,
                output,
                model=model,
                suite=prepared.development,
                producer=producer,
                execution_hash=execution_hash,
            )
        # Official benchmark grader sanity uses a fixed reference patch only;
        # no model is run on a final benchmark before all training is sealed.
        from self_curate.execution.graders import SWEJudge
        from types import SimpleNamespace

        kinds = {"swe_smith": True}
        for suite in prepared.benchmarks:
            if suite.suite_id == "terminal_bench":
                continue
            task = suite.tasks[0]
            instance = read_artifact(task.payload)["instance"]
            work = root / "capacity/harness" / suite.suite_id
            file = work / "official-result.json"
            if file.exists():
                judged = read_json(file)
            else:
                judged = SWEJudge().grade(
                    SimpleNamespace(
                        model_ref=model,
                        record_hash=digest(
                            ("engineering-grader-sanity", task.to_dict())
                        ),
                        task=task,
                    ),
                    instance,
                    instance["patch"],
                    "swe_bench",
                    work,
                    timeout=plan["preparation"]["grader_timeout_seconds"],
                )
            require(
                judged["reward"] == 1,
                "Official benchmark reference-patch sanity failed",
            )
        kinds["swe_bench"] = True
        active = manager.transition(
            "terminal", reserved=settings["terminal_reserved_gpu_uuids"]
        )
        runner.router.set_active(active)
        from .smoke_task import harbor_smoke_task

        smoke_root = root / "capacity/harbor-engineering"
        if (smoke_root / "task-record.json").exists():
            from self_curate.contracts import parse_task_record

            task = parse_task_record(read_json(smoke_root / "task-record.json"))
        else:
            task = harbor_smoke_task(
                smoke_root,
                artifact(settings["runtime_lock"]),
                gpu_count=len(settings["terminal_reserved_gpu_uuids"]),
            )
            atomic_json(smoke_root / "task-record.json", task.to_dict())
        suite = prepared.development
        request = TaskRunRequest(
            "engineering-harbor-smoke",
            "server-readiness",
            model,
            task,
            config["seed"],
            suite.sampling,
            suite.protocol,
            digest(task.to_dict()),
            "real",
        )
        work = smoke_root / "execution"
        if (work / "task-result.json").exists():
            reward = read_json(work / "task-result.json")["reward"]
        else:
            reward = runner.execute(request, work).reward
        require(
            reward == 1
            and read_json(work / "task-result.json")["generated_tokens"] > 0,
            "Independent engineering Harbor agent/environment/verifier smoke failed",
        )
        kinds["harbor"] = True
    finally:
        manager.close()
    return publish_json(
        receipt,
        {
            "identity": identity,
            "passed": True,
            "model_hash": model.model_hash,
            "lora_config_hash": __import__(
                "self_curate.training.peft_store", fromlist=["LoRASettings"]
            )
            .LoRASettings.from_dict(settings["lora"])
            .config_hash,
            "gpu_uuids": config["gpu_uuids"],
            "max_context_tokens": config["max_context_tokens"],
            "sft_backward_passed": True,
            "checkpoint_reload_passed": True,
            "swe_smith_smoke_passed": True,
            "swe_bench_smoke_passed": True,
            "harbor_smoke_passed": True,
            "backward_evidence": rows,
            "inference_evidence": infer,
            "harness_evidence_root": str(root / "capacity/harness"),
        },
    )


def json_module_load(text):
    import json

    return json.loads(text)
