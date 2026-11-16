"""Real-model isolated capacity/backward/reload checks. Never run on the workstation."""

from dataclasses import replace
from pathlib import Path
import os, subprocess, sys, time
from self_curate.contracts.types import require
from self_curate.contracts.parsing import model_ref
from self_curate.training.recovery import atomic_json, read_json
from self_curate.execution.model import http_json


def capacity_worker(path):
    import torch
    from self_curate.training.peft_store import PeftModelStore, LoRASettings
    from self_curate.training.artifacts import checkpoint_fingerprint

    root = Path(path)
    job = read_json(root / "job.json")
    model = model_ref(job["model"])
    cfg = job["model_store"]
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == job["gpu_uuid"]
        and torch.cuda.device_count() == 1,
        "Capacity worker must see exactly its authorized GPU",
    )
    settings = LoRASettings.from_dict(job["lora"])
    store = PeftModelStore(
        base_path=cfg["base_path"],
        initial_model=model,
        initial_adapter=cfg["initial_adapter"],
        settings=settings,
        base_fingerprint=cfg["base_fingerprint"],
        loader=cfg["loader"],
    )
    loaded = store.load_branch(model)
    network = loaded.model.to("cuda:0").train()
    parameters = [p for p in network.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    token = int(network.get_output_embeddings().weight.shape[0] // 2)
    ids = torch.full(
        (1, job["max_context_tokens"]), token, device="cuda:0", dtype=torch.long
    )
    labels = ids.clone()
    labels[:, 0] = -100
    optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0)
    loss = network.self_curate_loss_sum(ids, torch.ones_like(ids), labels) / (
        ids.numel() - 1
    )
    require(torch.isfinite(loss).item(), "Non-finite full-context capacity loss")
    loss.backward()
    require(
        all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in parameters
        ),
        "Invalid LoRA capacity gradients",
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    duration = time.perf_counter() - started
    folder = root / "reload-checkpoint"
    folder.mkdir()
    store.save_branch(network, folder / "weights")
    weights = checkpoint_fingerprint(folder / "weights")[0]
    reference = replace(model, adapter_hash=weights)
    atomic_json(folder / "model.json", reference.to_dict())
    network.eval()
    probe = ids[:, :32]
    with torch.inference_mode():
        expected = (
            next(
                network.self_curate_signal_logits(
                    probe, torch.ones_like(probe), (31,), chunk_size=1
                )
            )[1]
            .float()
            .cpu()
        )
        native = (
            network(
                input_ids=probe, attention_mask=torch.ones_like(probe), use_cache=False
            )
            .logits[:, 30, :]
            .float()
            .cpu()
        )
        native_error = (native - expected).abs().max().item()
        require(
            torch.allclose(native, expected, atol=0.02, rtol=0.01),
            "Bounded text-forward kernel differs from the official model forward",
        )
    del loss, optimizer, network, loaded, ids, labels, parameters
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    reloaded = store.load_checkpoint(reference, folder).model.to("cuda:0").eval()
    with torch.inference_mode():
        actual = (
            next(
                reloaded.self_curate_signal_logits(
                    probe, torch.ones_like(probe), (31,), chunk_size=1
                )
            )[1]
            .float()
            .cpu()
        )
    require(
        torch.equal(actual, expected),
        "Reloaded adapter logits differ under the identical capacity probe",
    )
    require(
        checkpoint_fingerprint(Path(cfg["initial_adapter"]))[0] == model.adapter_hash,
        "Capacity test mutated the common initial adapter",
    )
    atomic_json(
        root / "result.json",
        {
            "passed": True,
            "gpu_uuid": job["gpu_uuid"],
            "max_context_tokens": job["max_context_tokens"],
            "sft_backward_passed": True,
            "optimizer_state_allocation_passed": True,
            "checkpoint_reload_passed": True,
            "native_forward_alignment_passed": True,
            "native_forward_max_abs_error": native_error,
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": reserved,
            "seconds": duration,
            "probe": "synthetic capacity-only IDs, one isolated optimizer step; not an experiment branch",
            "device_name": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
        },
    )


def host_budget(config):
    """Bound pinned activation RAM before launching simultaneous capacity workers."""
    settings = config["adapter_settings"]
    native = read_json(Path(settings["model_store"]["base_path"]) / "config.json")
    text = native.get("text_config", native)
    offload = settings["lora"].get("cpu_activation_offload", False)
    count = config["max_context_tokens"]
    saved = (
        int(text["hidden_size"]) * int(text["num_hidden_layers"]) * count * 2
        if offload and count >= settings["lora"].get("offload_min_tokens", 65536)
        else 0
    )
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))
    }
    branches = len(config["modes"]) * len(config["learning_rates"])
    required = branches * saved + 64 * 1024**3
    require(
        required <= memory["MemTotal"] * 0.9,
        "Host RAM is insufficient for four independent full-context offloaded branches plus task services; do not silently truncate",
    )
    per_worker = max(saved, 60 * 1024**3) + 12 * 1024**3
    require(
        memory["MemAvailable"] >= per_worker * 1.15,
        "Available host RAM is insufficient even for one capacity worker; wait for other jobs rather than risking host OOM",
    )
    concurrency = min(8, max(1, int(memory["MemAvailable"] * 0.7 / per_worker)))
    return {
        "activation_checkpoint_ram_bytes_per_branch": saved,
        "estimated_training_host_ram_bytes": required,
        "capacity_worker_limit": concurrency,
        "memory": memory,
    }


def check_backward(config, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    settings = config["adapter_settings"]
    children = []
    budget = host_budget(config)
    atomic_json(root / "host-budget.json", budget)
    try:
        for uuid in config["gpu_uuids"]:
            work = root / uuid
            work.mkdir(exist_ok=True)
            job = {
                "model": read_json(settings["initial_model_file"]),
                "model_store": settings["model_store"],
                "lora": settings["lora"],
                "gpu_uuid": uuid,
                "max_context_tokens": config["max_context_tokens"],
            }
            if (work / "job.json").exists():
                require(read_json(work / "job.json") == job, "Capacity job changed")
            else:
                atomic_json(work / "job.json", job)
            if (work / "result.json").exists():
                continue
            while (
                sum(p.poll() is None for p, _ in children)
                >= budget["capacity_worker_limit"]
            ):
                next(p for p, _ in children if p.poll() is None).wait()
            require(
                not (work / "dispatched.json").exists(),
                "Interrupted capacity test requires explicit reconciliation",
            )
            atomic_json(
                work / "dispatched.json",
                {"probe": "isolated full-context backward/reload"},
            )
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=uuid,
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            env["PYTHONPATH"] = (
                str(Path(__file__).parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            )
            stream = (work / "worker.log").open("w", encoding="utf-8")
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "self_curate.server.capacity",
                    "--worker",
                    str(work.resolve()),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            children.append((p, stream))
        for p, stream in children:
            p.wait()
            stream.close()
            require(
                p.returncode == 0,
                "Capacity worker failed; no context or batch fallback is permitted",
            )
    finally:
        for p, stream in children:
            if p.poll() is None:
                p.terminate()
            p.wait()
            stream.close()
    rows = [read_json(root / u / "result.json") for u in config["gpu_uuids"]]
    require(
        all(
            r["passed"] and r["max_context_tokens"] == config["max_context_tokens"]
            for r in rows
        ),
        "Missing full-context capacity evidence",
    )
    require(
        len({r["device_name"] for r in rows}) == 1,
        "Mixed GPU architectures require a separately validated numerical profile",
    )
    return rows


def check_inference(router, model, *, max_context, max_output, output, service_settings=None):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    from self_curate.contracts import digest
    from self_curate.training.recovery import read_json
    from collections import Counter
    import threading
    count=sum(e["slots"] for e in router.entries)
    intent={"model_hash":model.model_hash,"max_context":max_context,"max_output":max_output,
            "prefix_policy":"independent-slot-token-v1",
            "slots":{e["gpu_uuid"]:e["slots"] for e in router.entries},
            "service_profile_hash":digest(service_settings) if service_settings is not None else None}
    frozen=root/"intent.json"
    if frozen.exists():require(read_json(frozen)==intent,"Inference capacity inputs changed")
    else:atomic_json(frozen,intent)
    completed=[(root/("slot-"+str(i))/"result.json").exists() for i in range(count)]
    require(not any(completed) or all(completed),"Partial capacity probe cannot prove concurrent capacity; reconcile then use a new probe root")
    barrier=threading.Barrier(count) if not all(completed) else None

    def run(index):
        work = root / ("slot-" + str(index))
        work.mkdir(exist_ok=True)
        if (work / "result.json").exists():
            return read_json(work / "result.json")
        require(
            not (work / "dispatched.json").exists(),
            "Unknown in-flight capacity generation requires reconciliation",
        )
        with router.bind(
            SimpleNamespace(model_ref=model, checkpoint=None), work
        ) as binding:
            # All declared slots are held before any request is sent.
            # A failed worker breaks the barrier and prevents a false capacity pass.
            barrier.wait(timeout=300)
            atomic_json(
                work / "dispatched.json",
                {
                    "model_hash": model.model_hash,
                    "input_tokens": max_context - max_output,
                    "max_output_tokens": max_output,
                },
            )
            start = time.perf_counter()
            response = http_json(
                binding["base_url"],
                "/v1/completions",
                {
                    "model": binding["model_name"],
                    "prompt": [100+index] * (max_context - max_output),
                    "max_tokens": max_output,
                    "ignore_eos": True,
                    "temperature": 0,
                    "seed": 42,
                    "stream": False,
                },
                key_env=binding.get("api_key_env"),
                timeout=7200,
            )
            atomic_json(work / "response.json", response)
            require(
                response["usage"]["prompt_tokens"] == max_context - max_output
                and response["usage"]["completion_tokens"] == max_output,
                "Inference did not execute the exact frozen capacity budget",
            )
            result = {
                "passed": True,
                "gpu_uuid": binding["gpu_uuid"],
                "prefix_token_id":100+index,
                "usage": response["usage"],
                "wall_seconds": time.perf_counter() - start,
            }
            atomic_json(work / "result.json", result)
            return result

    with ThreadPoolExecutor(max_workers=count) as executor:
        rows = list(executor.map(run, range(count)))
    require(Counter(r["gpu_uuid"] for r in rows)==Counter(intent["slots"]),
            "Inference capacity did not fill every declared slot")
    require(
        {r["gpu_uuid"] for r in rows} == {e["gpu_uuid"] for e in router.entries},
        "Inference capacity did not cover every authorized GPU",
    )
    atomic_json(
        root / "report.json",
        {
            "passed": True,
            "slots": rows,
            "protocol": "all declared slots dispatched after a common barrier; synthetic tokens, not task scores",
            "intent":intent,
            "intent_hash":digest(intent),
        },
    )
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    capacity_worker(parser.parse_args().worker)
