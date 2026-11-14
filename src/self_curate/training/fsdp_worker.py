"""Accelerate owns FSDP wrapping/backward/state; project owns exact data windows."""

import argparse
from dataclasses import replace
import importlib.metadata
import math
import os
import shutil
from pathlib import Path
import time

from self_curate.contracts import ComputeCost, ResourceUsage, digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.orchestration.io import parse_outer_contract
from .artifacts import checkpoint_fingerprint
from .continuous import wave_boundaries
from .recovery import atomic_json, read_json, commit_directory, verify_directory, publish_directory, lease
from .runlog import RunLog, new_run_id


def windows(count, batch, boundaries):
    start = 0
    for stage, end in enumerate(boundaries, 1):
        while start < end:
            following = min(start + batch, end)
            yield stage, start, following
            start = following


def rank_examples(start, end, rank, world):
    """No sampler padding, duplicates or changes to the manifest's global order."""
    return tuple(range(start + rank, end, world))


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "accelerate", "transformers", "peft")}


def execute(root, *, resume=False, cpu_smoke=False):
    import torch
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from accelerate.utils import set_seed, broadcast_object_list

    job = read_json(root / "job.json")
    request = parse_outer_contract("TrainingRequest", job["request"])
    require(request.recipe.epochs == 1, "One-epoch exact exposure required")
    if cpu_smoke:
        require(request.backend == "mock" and not torch.cuda.is_available(), "CPU smoke must be explicit and GPU-free")
        plugin = None
    else:
        require(request.backend == "real" and torch.cuda.is_available(), "No CPU fallback for production FSDP")
        require(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") == job["gpu_uuids"], "GPU authorization changed")
        # The plugin captures its parameter-init device at construction time.
        # Bind LOCAL_RANK first; otherwise empty-weight ranks can all target GPU 0.
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sharding_strategy="FULL_SHARD",
                     state_dict_type="SHARDED_STATE_DICT", use_orig_params=False, sync_module_states=True,
                     limit_all_gathers=True, cpu_ram_efficient_loading=job["settings"]["cpu_ram_efficient_loading"])
        os.environ["ACCELERATE_USE_FSDP"] = "true"
        if job["settings"]["cpu_ram_efficient_loading"]:
            from accelerate.utils import enable_fsdp_ram_efficient_loading
            enable_fsdp_ram_efficient_loading()
    accelerator = Accelerator(cpu=cpu_smoke, mixed_precision="no" if cpu_smoke else "bf16", fsdp_plugin=plugin)
    require(cpu_smoke or (accelerator.num_processes == 8 and accelerator.distributed_type.name == "FSDP"),
            "Eight-rank FSDP is required, never silently DDP/single GPU")
    world, rank = accelerator.num_processes, accelerator.process_index
    require(request.recipe.accumulation_steps % world == 0, "Effective batch must divide across ranks")
    identity = broadcast_object_list([new_run_id() if accelerator.is_main_process else None])[0]
    log = RunLog(root / "logs", rank=rank, identity=identity)
    set_seed(request.recipe.seed, device_specific=True)
    if cpu_smoke:
        # A tiny random CPU model exercises the very same Accelerate state path.
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 4)
                self.embedding.weight.requires_grad_(False)
                self.head = torch.nn.Linear(4, 16, bias=False)
            def forward(self, input_ids, attention_mask, labels):
                logits = self.head(self.embedding(input_ids))
                return torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 16),
                           labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        model, store = Tiny(), None
        runtime = {"torch": torch.__version__, "accelerate": importlib.metadata.version("accelerate")}
    else:
        from .peft_store import PeftModelStore, LoRASettings
        from self_curate.contracts.parsing import model_ref
        spec = job["store"]
        store = PeftModelStore(base_path=spec["base_path"], initial_model=model_ref(spec["initial_model"]),
                    initial_adapter=spec["initial_adapter"], settings=LoRASettings.from_dict(spec["lora"]),
                    base_fingerprint=spec["base_fingerprint"], loader=spec["loader"])
        if request.starting_model == store.model_ref:
            inner = store.load_branch(request.starting_model).model
        else:
            require("starting_checkpoint" in job, "Noninitial model requires an explicit verified checkpoint")
            inner = store.load_checkpoint(request.starting_model, job["starting_checkpoint"]).model
        require(all("lora_" in n for n, p in inner.named_parameters() if p.requires_grad), "Only LoRA may train")
        class LossModule(torch.nn.Module):
            def __init__(self, peft_model):
                super().__init__()
                self.inner = peft_model
            def forward(self, input_ids, attention_mask, labels):
                return self.inner.self_curate_loss_sum(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        model = LossModule(inner)
        from peft.utils.other import fsdp_auto_wrap_policy
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(inner)
        runtime = versions()
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # With FSDP1 use_orig_params=False, wrapping replaces parameter coordinates.
    # Build the optimizer AFTER wrapping, as required by Accelerate's FSDP guide.
    model = accelerator.prepare_model(model)
    optimizer_type = torch.optim.AdamW if request.recipe.optimizer == "adamw" else torch.optim.SGD
    optimizer = optimizer_type((p for p in model.parameters() if p.requires_grad),
                               lr=request.recipe.learning_rate, weight_decay=request.recipe.weight_decay)
    optimizer = accelerator.prepare_optimizer(optimizer)
    bounds = wave_boundaries(len(request.data.examples)) if job["continuous"] else (len(request.data.examples),)
    plan = list(windows(len(request.data.examples), request.recipe.accumulation_steps, bounds))
    producer = digest({"job": job, "runtime": runtime,
                       "worker": file_sha256(Path(__file__)), "world_size": world})
    state = {"cursor": 0, "steps": 0, "supervised_tokens": 0, "input_tokens": 0,
             "wall_seconds": 0.0, "loss_sum": 0.0}
    latest = root / "latest.json"
    checkpoint_seconds = 0.0
    if latest.exists():
        require(resume, "Existing FSDP state requires --resume")
        pointer = read_json(latest)
        saved = root / pointer["folder"]
        require(saved.resolve().is_relative_to(root.resolve()), "Recovery escaped output root")
        verify_directory(saved)
        envelope = read_json(saved / "cursor.json")
        require(envelope["producer"] == producer, "Model/data/source/environment/world size changed")
        state = envelope["state"]
        require(any(end == state["cursor"] for _, _, end in plan) or state["cursor"] == 0, "Invalid data cursor")
        accelerator.load_state(str(saved / "accelerate"))
        log.event("resumed", cursor=state, checkpoint=str(saved),
                  uncommitted_work="May be replayed from last committed optimizer boundary; abrupt-gap cost unknown")
    elif resume:
        log.event("resume_without_committed_state", uncommitted_work="Restart from initial state; retain earlier attempt logs")
    log.event("started", request_hash=request.record_hash, runtime=runtime, world_size=world,
              global_batch=request.recipe.accumulation_steps, micro_batch=1,
              accumulation=request.recipe.accumulation_steps // world, trainable_parameters=trainable_count,
              device=str(accelerator.device), cuda_name=None if cpu_smoke else torch.cuda.get_device_name(),
              cuda_capacity_bytes=None if cpu_smoke else torch.cuda.get_device_properties(accelerator.device).total_memory)
    stages = []
    previous_tokens = 0
    previous_wall = 0.0
    attempt_start = time.perf_counter()
    restored_wall = state["wall_seconds"]

    def save_state():
        nonlocal checkpoint_seconds
        tick = time.perf_counter()
        label = broadcast_object_list([new_run_id() if accelerator.is_main_process else None])[0]
        folder = root / "recovery" / label
        if not cpu_smoke:
            base_bytes = sum(p.stat().st_size for p in Path(job["store"]["base_path"]).rglob("*") if p.is_file())
            available = broadcast_object_list([shutil.disk_usage(root).free if accelerator.is_main_process else None])[0]
            require(available > base_bytes * 1.25, "Insufficient free space for a complete sharded recovery checkpoint")
            log.event("checkpoint_disk_budget", free_bytes=available, base_bytes=base_bytes)
        accelerator.save_state(str(folder / "accelerate"), safe_serialization=True)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            atomic_json(folder / "cursor.json", {"producer": producer, "state": state})
            commit_directory(folder)
            atomic_json(latest, {"folder": folder.relative_to(root).as_posix(), "producer": producer})
            committed = sorted((p for p in (root / "recovery").iterdir()
                                if p != folder and (p / "COMMITTED.json").is_file()),
                               key=lambda p: (read_json(p / "cursor.json")["state"]["steps"],
                                              p.stat().st_mtime_ns, p.name))
            keep = job["settings"].get("keep_recovery_checkpoints", 2)
            # Hash suffixes are random: lexical filename order is NOT recency.
            for old in committed[:max(0, len(committed) - (keep - 1))]:
                require(old.resolve().is_relative_to((root / "recovery").resolve()) and old != folder,
                        "Unsafe recovery pruning target")
                # Preserve the checkpoint manifest before removing old large shards.
                atomic_json(root / "pruned-recovery" / (old.name + ".json"), read_json(old / "COMMITTED.json"))
                shutil.rmtree(old)
        accelerator.wait_for_everyone()
        checkpoint_seconds += time.perf_counter() - tick
        log.event("checkpoint_committed", step=state["steps"], cursor=state["cursor"],
                  seconds=time.perf_counter() - tick, directory=str(folder))

    try:
        for stage, end in enumerate(bounds, 1):
            stage_folder = root / "stages" / str(stage)
            if (stage_folder / "COMMITTED.json").exists():
                verify_directory(stage_folder)
                item = read_json(stage_folder / "stage.json")
                require(item["examples_seen"] == end and state["cursor"] >= end, "Stage ahead of committed recovery cursor")
                stages.append(item)
                previous_tokens, previous_wall = item["supervised_tokens"], item["cumulative_wall_seconds"]
                continue
            require(state["cursor"] <= end, "Missing export before a later saved cursor; manual audit required")
            model.train()
            for planned_stage, start, stop in plan:
                if planned_stage != stage or stop <= state["cursor"]:
                    continue
                require(start == state["cursor"], "Data exposure order changed")
                tick = time.perf_counter()
                examples = request.data.examples[start:stop]
                total_tokens = sum(e.supervised_tokens for e in examples)
                slots = math.ceil((stop - start) / world)
                optimizer.zero_grad(set_to_none=True)
                local_loss = 0.0
                local_forward_tokens = 0
                local_supervised_tokens = 0
                for slot in range(slots):
                    index = start + slot * world + rank
                    valid = index < stop
                    example = request.data.examples[index if valid else start]
                    local_forward_tokens += len(example.input_ids)
                    local_supervised_tokens += example.supervised_tokens if valid else 0
                    ids = torch.tensor([example.input_ids], device=accelerator.device)
                    labels = torch.tensor([example.labels], device=accelerator.device)
                    with accelerator.autocast():
                        loss_sum = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
                        loss = loss_sum * (world / total_tokens) * int(valid)
                    accelerator.backward(loss)
                    local_loss += float(loss_sum.detach().float().cpu()) * int(valid)
                if request.recipe.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), request.recipe.max_grad_norm)
                optimizer.step()
                reduced = accelerator.reduce(torch.tensor(local_loss, dtype=torch.float64, device=accelerator.device), reduction="sum")
                require(not accelerator.optimizer_step_was_skipped, "Optimizer skipped; do not claim data exposure")
                state.update(cursor=stop, steps=state["steps"] + 1,
                             supervised_tokens=state["supervised_tokens"] + total_tokens,
                             input_tokens=state["input_tokens"] + sum(len(e.input_ids) for e in examples),
                             wall_seconds=restored_wall + time.perf_counter() - attempt_start,
                             loss_sum=state["loss_sum"] + float(reduced.cpu()))
                seconds = time.perf_counter() - tick
                log.event("optimizer_step", stage=stage, global_step=state["steps"], examples_seen=stop,
                          total_examples=len(request.data.examples), global_supervised_tokens=total_tokens,
                          global_input_tokens=sum(len(e.input_ids) for e in examples), cumulative=dict(state),
                          rank_forward_input_tokens=local_forward_tokens, rank_supervised_tokens=local_supervised_tokens,
                          seconds=seconds, supervised_tokens_per_second=total_tokens / max(seconds, 1e-9),
                          loss=float(reduced.cpu()) / total_tokens, learning_rate=optimizer.param_groups[0]["lr"],
                          eta_seconds=(len(plan) - state["steps"]) * seconds,
                          eta_scope="optimizer_work_only_excludes_checkpoint_export_and_validation",
                          cuda_allocated_bytes=None if cpu_smoke else torch.cuda.memory_allocated(),
                          cuda_reserved_bytes=None if cpu_smoke else torch.cuda.memory_reserved(),
                          cuda_peak_allocated_bytes=None if cpu_smoke else torch.cuda.max_memory_allocated(),
                          **gpu_telemetry(torch, accelerator.device, cpu_smoke))
                if accelerator.is_main_process:
                    print(f"[FSDP] step {state['steps']}/{len(plan)} samples {stop}/{len(request.data.examples)} "
                          f"supervised_tokens {state['supervised_tokens']} loss {float(reduced.cpu()) / total_tokens:.6f} "
                          f"step_seconds {seconds:.2f}", flush=True)
                if state["steps"] % job["settings"]["save_every_steps"] == 0 or stop == end:
                    save_state()
                if not resume and job.get("preflight_stop_after_steps") == state["steps"]:
                    require(request.training_id.startswith("preflight-"), "Unexpected interruption hook")
                    if state["steps"] % job["settings"]["save_every_steps"] != 0 and stop != end:
                        save_state()
                    raise RuntimeError("Explicit FSDP preflight interruption after committed optimizer step")
                if cpu_smoke and os.environ.get("SELF_CURATE_SMOKE_INTERRUPT_STEP") == str(state["steps"]):
                    raise RuntimeError("Explicit CPU interruption after committed optimizer step")
            # All ranks participate in full state gathering; only rank 0 writes adapter export.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
            if cpu_smoke:
                full = accelerator.get_state_dict(model)
            else:
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    full = model.state_dict()
            if accelerator.is_main_process:
                target = stage_folder
                stage_folder = root / "stages" / (f".stage-{stage}-" + new_run_id())
                stage_folder.mkdir(parents=True)
                weights = stage_folder / "weights"
                unwrapped = accelerator.unwrap_model(model)
                if cpu_smoke:
                    weights.mkdir()
                    torch.save(full, weights / "model.pt")
                else:
                    inner_state = {k.removeprefix("inner."): v for k, v in full.items()}
                    unwrapped.inner.save_pretrained(weights, state_dict=inner_state, safe_serialization=True)
                    atomic_json(weights / "self-curate-lora.json", {"config_hash": store.settings.config_hash,
                        "settings": job["store"]["lora"], "base_model": store.model_ref.base_model_id,
                        "revision": store.model_ref.revision, "resolved_targets": list(store.resolved_targets)})
                candidate = replace(request.starting_model, **{
                    "weights_hash" if cpu_smoke else "adapter_hash": checkpoint_fingerprint(weights)[0]})
                atomic_json(stage_folder / "model.json", candidate.to_dict())
                wall = max(0.0, state["wall_seconds"] - previous_wall)
                amount = state["supervised_tokens"] - previous_tokens
                usage = ResourceUsage(ComputeCost(wall, 0.0 if cpu_smoke else None,
                                       sum(len(e.input_ids) for e in request.data.examples[(bounds[stage-2] if stage > 1 else 0):end])),
                                      training_supervised_tokens=amount)
                item = {"folder": target.relative_to(root).as_posix(), "stage": stage,
                        "examples_seen": end, "supervised_tokens": state["supervised_tokens"],
                        "optimizer_steps": state["steps"], "model_ref": candidate.to_dict(),
                        "usage": usage.to_dict(), "cumulative_wall_seconds": state["wall_seconds"]}
                atomic_json(stage_folder / "stage.json", item)
                commit_directory(stage_folder)
                publish_directory(stage_folder, target)
                stage_folder = target
            accelerator.wait_for_everyone()
            item = read_json(stage_folder / "stage.json")
            stages.append(item)
            previous_tokens, previous_wall = item["supervised_tokens"], item["cumulative_wall_seconds"]
            del full
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            completed = root / (".complete-" + new_run_id())
            completed.mkdir(exist_ok=False)
            atomic_json(completed / "result.json", {"request_hash": request.record_hash,
                        "producer": producer, "stages": stages, "runtime": runtime})
            commit_directory(completed)
            publish_directory(completed, root / "complete")
        log.event("completed", state=state, checkpoint_seconds=checkpoint_seconds)
    except BaseException as exc:
        log.event("failed", error_type=type(exc).__name__, message=str(exc),
                  last_observed_state=state, last_committed=read_json(latest) if latest.exists() else None)
        raise
    finally:
        accelerator.end_training()


def gpu_telemetry(torch, device, cpu_smoke):
    if cpu_smoke:
        return {"gpu_utilization_pct": None, "gpu_telemetry_status": "cpu"}
    try:
        return {"gpu_utilization_pct": torch.cuda.utilization(device), "gpu_telemetry_status": "measured_nvml"}
    except Exception as exc:
        return {"gpu_utilization_pct": None, "gpu_telemetry_status": type(exc).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if int(os.environ.get("RANK", "0")) == 0:
        # A dead controller must not launch a second job over an orphaned worker.
        # The OS releases this lock when rank 0 actually exits.
        with lease(root / ".worker.lease"):
            execute(root, resume=args.resume, cpu_smoke=args.cpu_smoke)
    else:
        execute(root, resume=args.resume, cpu_smoke=args.cpu_smoke)


if __name__ == "__main__":
    main()
