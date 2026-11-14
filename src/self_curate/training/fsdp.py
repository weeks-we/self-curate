"""Process-isolated Accelerate FSDP adapter for existing training contracts."""

from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

from self_curate.contracts import ArtifactRef, ResourceUsage, digest, parse_contract
from self_curate.contracts.parsing import artifact, model_ref
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from .contracts import BackendTrainingResult
from .continuous import StageCheckpoint
from .recovery import atomic_json, read_json, verify_directory, lease
from .runlog import new_run_id, RunLog


DEFAULTS = {"world_size": 8, "micro_batch_size": 1, "save_every_steps": 4,
            "mixed_precision": "bf16", "fsdp_version": 1, "keep_recovery_checkpoints": 2,
            "cpu_ram_efficient_loading": True}


def settings(value):
    value = {**DEFAULTS, **value}
    require(set(value) == set(DEFAULTS), "Unknown FSDP settings")
    require(type(value["world_size"]) is int and value["world_size"] == 8,
            "Production FSDP requires eight authorized GPUs")
    require(value["micro_batch_size"] == 1 and value["mixed_precision"] == "bf16" and
            value["fsdp_version"] == 1, "Reviewed FSDP1 bf16 recipe uses per-device micro batch 1")
    require(type(value["save_every_steps"]) is int and value["save_every_steps"] > 0,
            "Positive checkpoint interval required")
    require(type(value["keep_recovery_checkpoints"]) is int and value["keep_recovery_checkpoints"] >= 2,
            "Keep at least two committed recovery checkpoints")
    require(type(value["cpu_ram_efficient_loading"]) is bool, "Explicit CPU RAM loading policy required")
    return value


def stages_from_disk(output, request):
    root = Path(output)
    verify_directory(root / "complete")
    result = read_json(root / "complete" / "result.json")
    require(result["request_hash"] == request.record_hash, "Completed FSDP request changed")
    stages = []
    for item in result["stages"]:
        checkpoint = root / item["folder"]
        verify_directory(checkpoint)
        stages.append(StageCheckpoint(request.record_hash, item["stage"], model_ref(item["model_ref"]),
                      ArtifactRef(checkpoint.resolve().as_uri(), __import__(
                          "self_curate.training.artifacts", fromlist=["checkpoint_fingerprint"]
                      ).checkpoint_fingerprint(checkpoint)[0]), item["examples_seen"], item["supervised_tokens"],
                      item["optimizer_steps"], parse_contract("ResourceUsage", item["usage"])))
    return stages


class AccelerateFSDPBackend:
    backend = "real"

    def __init__(self, store, *, gpu_uuids, config=None, starting_checkpoint=None, preflight_stop_after_steps=None):
        self.store, self.gpu_uuids, self.config = store, tuple(gpu_uuids), settings(config or {})
        require(len(self.gpu_uuids) == 8 and len(set(self.gpu_uuids)) == 8 and
                all(u.startswith("GPU-") for u in self.gpu_uuids), "Eight explicit GPU UUIDs required")
        self.progress = None
        self.starting_checkpoint = starting_checkpoint
        require(preflight_stop_after_steps is None or
                (type(preflight_stop_after_steps) is int and preflight_stop_after_steps>0),
                "Invalid isolated preflight interruption")
        self.preflight_stop_after_steps = preflight_stop_after_steps


    @property
    def implementation_hash(self):
        return digest({"backend": file_sha256(Path(__file__)),
                       "worker": file_sha256(Path(__file__).with_name("fsdp_worker.py")),
                       "store": self.store.implementation_hash, "settings": self.config,
                       "gpu_uuids": self.gpu_uuids})

    def bind_starting_checkpoint(self, reference, checkpoint):
        """OuterCycle supplies its accepted parameters, including after process recovery."""
        if reference == self.store.model_ref:
            self.starting_checkpoint = None
            return
        checkpoint = checkpoint or self.starting_checkpoint
        from .artifacts import verify_local_checkpoint
        from urllib.parse import urlsplit
        from urllib.request import url2pathname
        require(checkpoint is not None and verify_local_checkpoint(checkpoint),
                "Accepted model requires a verified local checkpoint")
        folder = Path(url2pathname(urlsplit(checkpoint.uri).path))
        require(model_ref(read_json(folder / "model.json")) == reference,
                "Starting checkpoint does not contain the accepted model")
        self.starting_checkpoint = checkpoint

    def _execute(self, request, output, *, resume, continuous):
        require(request.backend == "real" and request.recipe.epochs == 1,
                "FSDP currently implements explicit one-epoch real SFT")
        require(request.recipe.accumulation_steps % 8 == 0,
                "Global effective batch (recipe.accumulation_steps) must be divisible by eight")
        root = Path(output).resolve()
        require(not root.exists() or resume, "FSDP output exists without explicit resume")
        root.mkdir(parents=True, exist_ok=True)
        job = {"request": request.to_dict(), "settings": self.config, "gpu_uuids": list(self.gpu_uuids),
               "continuous": continuous, "implementation_hash": self.implementation_hash,
               "store": {"base_path": str(self.store.base_path), "initial_adapter": str(self.store.initial_adapter),
                         "initial_model": self.store.model_ref.to_dict(), "lora": asdict(self.store.settings),
                         "base_fingerprint": self.store.model_ref.weights_hash, "loader": self.store.loader}}
        if self.preflight_stop_after_steps is not None:
            require(request.training_id.startswith("preflight-"), "Interruption hooks are preflight-only")
            job["preflight_stop_after_steps"] = self.preflight_stop_after_steps
        if request.starting_model != self.store.model_ref:
            from .artifacts import verify_local_checkpoint
            from urllib.parse import urlsplit
            from urllib.request import url2pathname
            require(self.starting_checkpoint is not None and verify_local_checkpoint(self.starting_checkpoint),
                    "Sequential SFT requires the exact previously accepted local checkpoint")
            job["starting_checkpoint"] = str(Path(url2pathname(urlsplit(self.starting_checkpoint.uri).path)).resolve())
        with lease(root / ".launcher.lease"):
            if (root / "job.json").exists():
                require(read_json(root / "job.json") == job, "FSDP model/data/source/config/GPU identities changed")
            else:
                atomic_json(root / "job.json", job)
            if not (root / "complete/COMMITTED.json").exists():
                attempt = new_run_id()
                launch_events = RunLog(root / "launcher-events", identity=attempt)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(self.gpu_uuids),
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
                log = root / "launch-logs" / (attempt + ".log")
                log.parent.mkdir(exist_ok=True)
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                           "--max_restarts=0", "-m", "self_curate.training.fsdp_worker", "--output", str(root)]
                if resume:
                    command.append("--resume")
                launch_events.event("distributed_launch", request_hash=request.record_hash, resume=resume,
                                    gpu_uuids=self.gpu_uuids, settings=self.config, console_log=str(log))
                with log.open("w", encoding="utf-8") as stream:
                    process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                launch_events.event("distributed_exit", returncode=process.returncode)
                require(process.returncode == 0, f"FSDP failed; resume same output after audit; log: {log}")
            return stages_from_disk(root, request)

    def train(self, request, output, *, emit=None, resume=False):
        stages = self._execute(request, output, resume=resume, continuous=emit is not None)
        if emit is not None:
            for stage in stages:
                emit(stage)
            return
        last = stages[-1]
        return BackendTrainingResult(last.model_ref, last.checkpoint, last.supervised_tokens,
                                     last.optimizer_steps, None, last.usage)
