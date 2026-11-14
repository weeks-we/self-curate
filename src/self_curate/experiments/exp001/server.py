"""Concrete server bindings. Prepared data and real smoke receipts are required."""

from pathlib import Path
import json
import gc
from self_curate.contracts import digest, parse_contract, ResourceUsage
from self_curate.contracts.parsing import artifact
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json
from self_curate.training.peft_store import PeftModelStore, LoRASettings
from self_curate.execution.artifacts import read_artifact
from self_curate.execution.runner import build_task_runner
from self_curate.execution.model import CompletionClient
from .inputs import load_prepared_manifest
from .selection import count_context
from .ports import load_experiment_prompts


def task_execution_identity(settings):
    from self_curate.data.pool import file_sha256

    execution = dict(settings["task_execution"])
    if execution.get("offline_assets"):
        execution["offline_assets"] = read_artifact(
            artifact(execution["offline_assets"])
        )
    execution["serving_pool"] = read_artifact(artifact(execution["serving_pool"]))
    execution["serving_pool"] = {
        **execution["serving_pool"],
        "entries": [
            {k: v for k, v in entry.items() if k != "relay_grant_root"}
            for entry in execution["serving_pool"]["entries"]
        ],
    }
    value = digest(
        {
            "task_execution": execution,
            "runtime_lock": read_artifact(artifact(settings["runtime_lock"])),
            "environment": settings.get("server_environment_hash"),
        }
    )
    return digest(
        {
            "settings": value,
            "code": {
                p.name: file_sha256(p)
                for p in (Path(__file__).parents[2] / "execution").glob("*.py")
            },
        }
    )


class SharedRoleClient:
    def __init__(self, router, model, output, seed):
        self.router, self.model_ref, self.output = router, model, Path(output)
        self.calls = 0
        self.seed = seed

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        from types import SimpleNamespace

        work = self.output / (role + "-" + str(self.calls))
        self.calls += 1
        work.mkdir(parents=True, exist_ok=False)
        with self.router.bind(
            SimpleNamespace(model_ref=self.model_ref, checkpoint=None), work
        ) as binding:
            client = CompletionClient(binding, work / "calls")
            response = client.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                max_tokens=8192,
                seed=self.seed,
                timeout=timeout_seconds,
                response_format={"type": "json_object"},
            )
            value = response["choices"][0]["message"]["content"]
            try:
                content = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                content = value
            return {
                "content": content,
                "usage": ResourceUsage(
                    agent_input_tokens=response["usage"]["prompt_tokens"],
                    agent_output_tokens=response["usage"]["completion_tokens"],
                ).to_dict(),
            }


class ServerPorts:
    backend = "real"
    benchmark_workers = 8

    def __init__(self, config, output, asset_root):
        self.config, self.output = config, Path(output)
        self.settings = json.loads(config.adapter_settings_json)
        self.inputs = load_prepared_manifest(self.settings["prepared_manifest"])
        self.model = self.inputs.model
        self.lora = LoRASettings.from_dict(self.settings["lora"])
        self.prompts = load_experiment_prompts(self.settings["prompts"])
        self.objective = parse_contract(
            "ObjectiveSpec", read_json(self.settings["objective"])
        )
        self.edit_policy = parse_contract(
            "EditPolicy", read_json(self.settings["edit_policy"])
        )
        self.agent_sampling = artifact(self.settings["agent_sampling"])
        self.task_runner = build_task_runner(
            self.settings["task_execution"], self.model, execution=config.execution
        )
        from self_curate.validation.parallel import ParallelTaskValidator

        self.validator = ParallelTaskValidator(self.task_runner, workers=16)
        self.benchmark_runner = self.task_runner
        self.client = SharedRoleClient(
            self.task_runner.router, self.model, self.output / "role-calls", config.seed
        )
        self.task_execution_hash = task_execution_identity(self.settings)
        from self_curate.data.pool import file_sha256

        self.agent_implementation_hash = digest(
            {
                "format": "local-chat-json-v1",
                "sampling": self.agent_sampling.to_dict(),
                "code": file_sha256(Path(__file__)),
            }
        )
        self._scorer = None
        self._store = None
        self.manager = None
        self.closed = False
        self.benchmark_workers = 16
        import os, atexit

        if self.settings.get("services") and not os.environ.get(
            "EXP001_TRAIN_GPU_UUID"
        ):
            from self_curate.execution.services import ServiceManager

            self.manager = ServiceManager(
                {
                    **self.settings["services"],
                    "training_processes": 8,
                },
                self.task_runner.router.entries,
                config.gpu_uuids,
                execution=config.execution,
            )
        if os.environ.get("EXP001_TRAIN_GPU_UUID"):
            self.task_runner.router.set_active(
                config.gpu_uuids[len(config.modes) * len(config.learning_rates) :]
            )
        atexit.register(self.close)

    def enter_phase(self, phase, *, suite=None):
        reserved = ()
        if suite is not None and suite.suite_id == "terminal_bench":
            phase = "terminal"
            reserved = tuple(self.settings.get("terminal_reserved_gpu_uuids", ()))
        from self_curate.execution.services import phase_uuids

        active = phase_uuids(
            self.config.gpu_uuids,
            phase,
            reserved=reserved,
            training_count=8,
        )
        if self.manager:
            active = self.manager.transition(phase, reserved=reserved)
        if active:
            self.task_runner.router.set_active(active)
        self.benchmark_workers = min(16, 2 * len(active)) if active else 1
        if phase != "scoring":
            self.release_initial()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.release_initial()
        if self.manager:
            self.manager.close()
            if self.output.exists():
                import shutil

                destination = self.output / "server-telemetry"
                if not destination.exists():
                    shutil.copytree(self.manager.root, destination)
                publish = self.output / "server-readiness"
                publish.mkdir(exist_ok=True)
                from self_curate.training.recovery import atomic_json

                atomic_json(
                    publish / "shared-preparation-costs.json",
                    {
                        "cost_role": "original shared asset production, not a new-run charge",
                        "ledger": self.inputs.origin_ledger.to_dict(),
                    },
                )
                for name, key in (
                    ("capacity-receipt.json", "capacity_receipt"),
                    ("runtime-lock.json", "runtime_lock"),
                ):
                    if self.settings.get(key):
                        atomic_json(
                            publish / name, read_artifact(artifact(self.settings[key]))
                        )
                prepared_root = Path(self.settings["prepared_manifest"]).parent
                for name in (
                    "isolation.json",
                    "development-coverage.json",
                    "pool/admission-report.json",
                ):
                    source = prepared_root / name
                    if source.exists():
                        shutil.copyfile(source, publish / Path(name).name)

    def preflight(self, config):
        from self_curate.execution.runtime import verify_runtime

        verify_runtime(
            read_artifact(artifact(self.settings["runtime_lock"])),
            kinds=(("mini", "swe_bench", "harbor") if self.settings.get("selection_budget") else ("mini", "swe_smith", "swe_bench", "harbor")),
        )
        receipt = read_artifact(artifact(self.settings["capacity_receipt"]))
        require(
            receipt.get("passed") is True
            and receipt.get("model_hash") == self.model.model_hash
            and receipt.get("lora_config_hash") == self.lora.config_hash
            and receipt.get("gpu_uuids") == list(config.gpu_uuids)
            and receipt.get("max_context_tokens") == config.max_context_tokens
            and receipt.get("sft_backward_passed") is True
            and receipt.get("checkpoint_reload_passed") is True
            and ((receipt.get("rebench_smoke_passed") is True and receipt.get("pro_v2_regrade_smoke_passed") is True and receipt.get("fsdp_resume_passed") is True and receipt.get("offline_bundle_verified") is True) if self.settings.get("selection_budget") else receipt.get("swe_smith_smoke_passed") is True)
            and receipt.get("swe_bench_smoke_passed") is True
            and receipt.get("harbor_smoke_passed") is True,
            "Real capacity, adapter reload and all three harness smokes must pass on server",
        )
        require(
            all(
                e["gpu_uuid"] in config.gpu_uuids
                for e in self.task_runner.router.entries
            ),
            "Inference pool contains an unauthorized GPU",
        )
        return {
            "passed": True,
            "backend": "real",
            "gpu_uuids": list(config.gpu_uuids),
            "max_context_tokens": config.max_context_tokens,
            "capacity_receipt": self.settings["capacity_receipt"],
        }

    def prepare(self, config, output):
        return self.inputs

    def context(self, records, config):
        import math

        return count_context(
            records,
            model=self.model,
            count=config.selected_count(len(records)),
            seed=config.seed,
        )

    @property
    def store(self):
        if self._store is None:
            value = self.settings["model_store"]
            self._store = PeftModelStore(
                base_path=value["base_path"],
                initial_model=self.model,
                initial_adapter=value["initial_adapter"],
                settings=self.lora,
                base_fingerprint=value["base_fingerprint"],
                loader=value.get("loader", "AutoModelForImageTextToText"),
            )
        return self._store

    @property
    def scorer(self):
        if self._scorer is None:
            from self_curate.signals.sharded import ShardedSignalService

            self._scorer = ShardedSignalService(
                model=self.model,
                store_settings={
                    **self.settings["model_store"],
                    "lora": self.settings["lora"],
                    "environment_hash": self.settings.get("server_environment_hash"),
                },
                gpu_uuids=self.config.gpu_uuids,
                output=self.output / "scoring-origin",
                max_input_tokens=self.config.max_context_tokens,
                execution=self.config.execution,
            )
        return self._scorer

    def training_backend(self, branch):
        from self_curate.training.fsdp import AccelerateFSDPBackend
        require(self.settings.get("training_backend", "accelerate_fsdp") == "accelerate_fsdp",
                "Real server SFT must use the configured Accelerate FSDP backend")
        require(self.manager is not None, "FSDP requires managed inference services to release all authorized GPUs")
        return AccelerateFSDPBackend(self.store, gpu_uuids=self.config.gpu_uuids,
                                      config=self.settings.get("fsdp", {}))

    def attach_training(self, config, model):
        require(model == self.model, "Worker M0 changed")
        self.enter_phase("training")

    def release_initial(self):
        if self._scorer is not None:
            self._scorer.backend.close()
        self._scorer = None
        gc.collect()
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def build(config, output, asset_root):
    return ServerPorts(config, output, asset_root)
