"""Durable optimizer-boundary continuation for the continuous Torch backend."""

from dataclasses import replace
from pathlib import Path
import random
from time import perf_counter
import uuid

from self_curate.contracts import ArtifactRef, ComputeCost, ResourceUsage, digest
from self_curate.contracts.parsing import model_ref
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256, write_json
from .artifacts import checkpoint_fingerprint
from .execution import code_identity, torch_execution
from .torch_backend import LoadedModel
from .service import TrainingFailure
from .recovery import (
    atomic_json,
    read_json,
    commit_directory,
    verify_directory,
    publish_directory,
)


class TorchContinuousBackend:
    """One optimizer; a committed recovery snapshot after every optimizer step.

    In-flight accumulation is rolled back to the last committed step. Partial
    snapshot directories remain as evidence. No snapshot ever replaces another.
    Production adapter stores should save only trainable weights, not the base.
    """

    def __init__(self, store, *, backend, device="cpu", progress=None):
        require(
            backend in ("mock", "real")
            and store.checkpoint_kind in ("full", "adapter"),
            "Explicit model store/domain required",
        )
        self.store, self.backend, self.device, self.progress = (
            store,
            backend,
            device,
            progress,
        )

    @property
    def implementation_hash(self):
        return digest(
            {
                "code": file_sha256(Path(__file__)),
                "store": code_identity(self.store),
                "execution": torch_execution(self.device),
            }
        )

    def train(self, request, output, *, emit, resume=False):
        import torch
        import torch.nn.functional as F
        from .continuous import StageCheckpoint, wave_boundaries, _frozen_hash

        require(
            request.recipe.epochs == 1 and request.backend == self.backend,
            "Continuous recipe/domain mismatch",
        )
        require(
            (self.store.checkpoint_kind == "adapter")
            == (request.starting_model.adapter_hash is not None),
            "Store/adapter mismatch",
        )
        device, output = torch.device(self.device), Path(output)
        require(device.type in ("cpu", "cuda"), "Unsupported training device")
        output.mkdir(parents=True, exist_ok=resume)
        recovery = output / "recovery"
        recovery.mkdir(exist_ok=resume)
        implementation = self.implementation_hash
        snapshots = sorted(p for p in recovery.glob("step-*") if p.is_dir())
        latest = None
        for folder in snapshots:
            verify_directory(folder)
            meta = read_json(folder / "training.json")
            require(
                meta["request_hash"] == request.record_hash
                and meta["implementation_hash"] == implementation,
                "Recovery request/backend/environment identity changed",
            )
            latest = folder
        require(resume or latest is None, "Training recovery requires explicit resume")
        python_rng = random.getstate()
        devices = (
            list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        )
        callback_active, model = False, None
        stage_inputs = stage_tokens = processed = steps = cursor = 0
        uncommitted_tokens = uncommitted_inputs = 0
        uncommitted_started = perf_counter()
        started = perf_counter()
        try:
            with (
                torch.random.fork_rng(devices=devices),
                torch.autocast(device.type, enabled=False),
            ):
                random.seed(request.recipe.seed)
                torch.manual_seed(request.recipe.seed)
                restored = None
                if latest is None:
                    loaded = self.store.load_branch(request.starting_model)
                    require(
                        loaded.model_ref == request.starting_model,
                        "Wrong initial model",
                    )
                else:
                    meta = read_json(latest / "training.json")
                    loaded = self.store.load_checkpoint(
                        model_ref(read_json(latest / "model.json")), latest
                    )
                    # Only deserialize after all checkpoint bytes and producer identities verify.
                    restored = torch.load(
                        latest / "training_state.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    require(
                        restored["request_hash"] == request.record_hash,
                        "Wrong optimizer request",
                    )
                    cursor, steps, processed = (
                        meta["examples_seen"],
                        meta["optimizer_steps"],
                        meta["processed_supervised_tokens"],
                    )
                    require(
                        all(
                            restored[k] == meta[k]
                            for k in (
                                "examples_seen",
                                "optimizer_steps",
                                "processed_supervised_tokens",
                            )
                        ),
                        "Recovery state/metadata counters disagree",
                    )
                require(
                    isinstance(loaded, LoadedModel),
                    "ModelStore must return LoadedModel",
                )
                model = loaded.model.to(device)
                model.train()
                named = tuple(
                    (n, p) for n, p in model.named_parameters() if p.requires_grad
                )
                require(bool(named), "No trainable parameters")
                if restored is not None:
                    require(
                        list(n for n, _ in named) == meta["trainable_parameters"],
                        "Recovery trainable parameter coordinates changed",
                    )
                parameters = [p for _, p in named]
                frozen = _frozen_hash(model)
                if restored is not None:
                    require(
                        frozen == meta["frozen_parameters_hash"],
                        "Recovery frozen base changed",
                    )
                optimizer_class = (
                    torch.optim.AdamW
                    if request.recipe.optimizer == "adamw"
                    else torch.optim.SGD
                )
                optimizer = optimizer_class(
                    parameters,
                    lr=request.recipe.learning_rate,
                    weight_decay=request.recipe.weight_decay,
                )
                if restored is not None:
                    optimizer.load_state_dict(restored["optimizer"])
                    torch.set_rng_state(restored["torch_rng"])
                    random.setstate(restored["python_rng"])
                    require(
                        len(restored["cuda_rng"]) == len(devices),
                        "Recovery CUDA layout changed",
                    )
                    if devices:
                        torch.cuda.set_rng_state_all(restored["cuda_rng"])
                    atomic_json(
                        recovery / ("resume-" + uuid.uuid4().hex + ".json"),
                        {
                            "restored_step": steps,
                            "examples_seen": cursor,
                            "checkpoint_hash": checkpoint_fingerprint(latest)[0],
                        },
                    )
                restored_meta = dict(meta) if restored is not None else None
                examples, bounds = request.data.examples, wave_boundaries(
                    len(request.data.examples)
                )
                require(
                    cursor <= len(examples)
                    and processed
                    == sum(x.supervised_tokens for x in examples[:cursor]),
                    "Recovery cursor/token prefix mismatch",
                )
                expected_steps = sum(
                    (min(cursor, b) - a + request.recipe.accumulation_steps - 1)
                    // request.recipe.accumulation_steps
                    for a, b in zip((0, *bounds[:-1]), bounds)
                    if cursor > a
                )
                require(
                    steps == expected_steps
                    and (
                        cursor in (0, *bounds)
                        or any(
                            a < cursor < b
                            and (cursor - a) % request.recipe.accumulation_steps == 0
                            for a, b in zip((0, *bounds[:-1]), bounds)
                        )
                    ),
                    "Recovery cursor is not an optimizer boundary",
                )
                for stage, (start, end) in enumerate(zip((0, *bounds[:-1]), bounds), 1):
                    require(
                        self.implementation_hash == implementation,
                        "Training implementation changed",
                    )
                    if cursor < end:
                        offset = max(cursor, start)
                        stage_tokens = sum(
                            x.supervised_tokens for x in examples[start:offset]
                        )
                        stage_inputs = sum(
                            len(x.input_ids) for x in examples[start:offset]
                        )
                        carried_seconds = (
                            restored_meta["stage_seconds"]
                            if restored_meta is not None
                            and restored_meta["stage"] == stage
                            else 0
                        )
                        started = perf_counter() - carried_seconds
                        for offset in range(
                            offset, end, request.recipe.accumulation_steps
                        ):
                            window = examples[
                                offset : min(
                                    offset + request.recipe.accumulation_steps, end
                                )
                            ]
                            optimizer.zero_grad(set_to_none=True)
                            uncommitted_started = perf_counter()
                            denominator = sum(x.supervised_tokens for x in window)
                            window_loss = 0.0
                            step_started = perf_counter()
                            for example in window:
                                ids = torch.tensor(
                                    [example.input_ids], dtype=torch.long, device=device
                                )
                                labels = torch.tensor([example.labels], device=device)
                                custom_loss = getattr(
                                    model, "self_curate_loss_sum", None
                                )
                                if custom_loss is not None:
                                    loss = custom_loss(
                                        input_ids=ids,
                                        attention_mask=torch.ones_like(ids),
                                        labels=labels,
                                    )
                                else:
                                    logits = model(
                                        input_ids=ids,
                                        attention_mask=torch.ones_like(ids),
                                    ).logits
                                    require(
                                        logits.ndim == 3
                                        and logits.shape[:2] == ids.shape,
                                        "Misaligned causal logits",
                                    )
                                    loss = F.cross_entropy(
                                        logits[0, :-1].float(),
                                        torch.tensor(example.labels[1:], device=device),
                                        ignore_index=-100,
                                        reduction="sum",
                                    )
                                require(
                                    bool(torch.isfinite(loss).item()),
                                    "Non-finite continuous SFT loss",
                                )
                                window_loss += float(loss.detach().item()) / denominator
                                (loss / denominator).backward()
                                uncommitted_tokens += example.supervised_tokens
                                uncommitted_inputs += len(example.input_ids)
                            require(
                                all(
                                    p.grad is None
                                    or bool(torch.isfinite(p.grad).all().item())
                                    for p in parameters
                                ),
                                "Non-finite continuous gradient",
                            )
                            if request.recipe.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    parameters, request.recipe.max_grad_norm
                                )
                            optimizer.step()
                            steps += 1
                            cursor = offset + len(window)
                            processed = sum(
                                x.supervised_tokens for x in examples[:cursor]
                            )
                            stage_tokens = sum(
                                x.supervised_tokens for x in examples[start:cursor]
                            )
                            stage_inputs = sum(
                                len(x.input_ids) for x in examples[start:cursor]
                            )
                            folder = recovery / (".pending-" + uuid.uuid4().hex)
                            folder.mkdir()
                            rng, py = torch.get_rng_state(), random.getstate()
                            cuda = torch.cuda.get_rng_state_all() if devices else []
                            self.store.save_branch(model, folder / "weights")
                            weights_hash, weight_checks = checkpoint_fingerprint(
                                folder / "weights"
                            )
                            candidate = replace(
                                request.starting_model,
                                **(
                                    {"adapter_hash": weights_hash}
                                    if self.store.checkpoint_kind == "adapter"
                                    else {"weights_hash": weights_hash}
                                ),
                            )
                            torch.save(
                                {
                                    "optimizer": optimizer.state_dict(),
                                    "torch_rng": rng,
                                    "python_rng": py,
                                    "cuda_rng": cuda,
                                    "optimizer_steps": steps,
                                    "processed_supervised_tokens": processed,
                                    "examples_seen": cursor,
                                    "request_hash": request.record_hash,
                                },
                                folder / "training_state.pt",
                            )
                            write_json(folder / "model.json", candidate.to_dict())
                            write_json(folder / "weights_checksums.json", weight_checks)
                            write_json(
                                folder / "training.json",
                                {
                                    "request_hash": request.record_hash,
                                    "trainable_parameters": [n for n, _ in named],
                                    "recipe": request.recipe.to_dict(),
                                    "optimizer_state_policy": "continuous_across_waves",
                                    "micro_batch": 1,
                                    "window_loss": window_loss,
                                    "learning_rate": optimizer.param_groups[0]["lr"],
                                    "step_seconds": perf_counter() - step_started,
                                    "window_supervised_tokens": denominator,
                                    "window_input_tokens": sum(
                                        len(x.input_ids) for x in window
                                    ),
                                    "examples_seen": cursor,
                                    "optimizer_steps": steps,
                                    "processed_supervised_tokens": processed,
                                    "implementation_hash": implementation,
                                    "stage": stage,
                                    "stage_tokens": stage_tokens,
                                    "stage_inputs": stage_inputs,
                                    "stage_seconds": perf_counter() - started,
                                    "frozen_parameters_hash": frozen,
                                },
                            )
                            commit_directory(folder)
                            final = recovery / ("step-" + str(steps).zfill(8))
                            require(
                                not final.exists(),
                                "Recovery refuses to overwrite committed optimizer step",
                            )
                            publish_directory(folder, final)
                            latest = final
                            uncommitted_tokens = uncommitted_inputs = 0
                            uncommitted_started = perf_counter()
                            torch.set_rng_state(rng)
                            random.setstate(py)
                            if devices:
                                torch.cuda.set_rng_state_all(cuda)
                            if self.progress:
                                self.progress.update(
                                    "training",
                                    cursor,
                                    len(examples),
                                    optimizer_steps=steps,
                                    stage=stage,
                                )
                    import csv

                    with (output / "metrics.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as metrics:
                        columns = [
                            "optimizer_steps",
                            "examples_seen",
                            "stage",
                            "window_loss",
                            "learning_rate",
                            "window_supervised_tokens",
                            "window_input_tokens",
                            "step_seconds",
                        ]
                        writer = csv.DictWriter(metrics, fieldnames=columns)
                        writer.writeheader()
                        for snapshot in sorted(recovery.glob("step-*")):
                            row = read_json(snapshot / "training.json")
                            writer.writerow({n: row.get(n) for n in columns})
                    # An older stage may already have executed validation; emit exact evidence.
                    folder = output / ("stage-" + str(stage))
                    if not folder.exists():
                        import shutil

                        source = recovery / (
                            "step-"
                            + str(
                                sum(
                                    (b - a + request.recipe.accumulation_steps - 1)
                                    // request.recipe.accumulation_steps
                                    for a, b in zip((0, *bounds[:-1]), bounds[:stage])
                                )
                            ).zfill(8)
                        )
                        verify_directory(source)
                        pending = output / (".stage-pending-" + uuid.uuid4().hex)
                        shutil.copytree(source, pending)
                        publish_directory(pending, folder)
                    verify_directory(folder)
                    meta = read_json(folder / "training.json")
                    require(
                        meta["examples_seen"] == end and meta["stage"] == stage,
                        "Stage checkpoint prefix mismatch",
                    )
                    require(_frozen_hash(model) == frozen, "Frozen parameters changed")
                    value = StageCheckpoint(
                        request.record_hash,
                        stage,
                        model_ref(read_json(folder / "model.json")),
                        ArtifactRef(
                            folder.resolve().as_uri(), checkpoint_fingerprint(folder)[0]
                        ),
                        end,
                        meta["processed_supervised_tokens"],
                        meta["optimizer_steps"],
                        ResourceUsage(
                            ComputeCost(
                                meta["stage_seconds"],
                                0 if device.type == "cpu" else None,
                                meta["stage_inputs"],
                            ),
                            training_supervised_tokens=meta["stage_tokens"],
                        ),
                    )
                    rng, py = torch.get_rng_state(), random.getstate()
                    cuda = torch.cuda.get_rng_state_all() if devices else []
                    callback_active = True
                    emit(value)
                    callback_active = False
                    torch.set_rng_state(rng)
                    random.setstate(py)
                    if devices:
                        torch.cuda.set_rng_state_all(cuda)
                    model.train()
                    stage_tokens = stage_inputs = 0
        except Exception as exc:
            if callback_active or isinstance(exc, TrainingFailure):
                raise
            raise TrainingFailure(
                str(exc),
                usage=ResourceUsage(
                    ComputeCost(
                        perf_counter() - uncommitted_started,
                        0 if device.type == "cpu" else None,
                        uncommitted_inputs,
                    ),
                    training_supervised_tokens=uncommitted_tokens,
                ),
            ) from exc
        finally:
            random.setstate(python_rng)
            if model is not None:
                model.zero_grad(set_to_none=True)
