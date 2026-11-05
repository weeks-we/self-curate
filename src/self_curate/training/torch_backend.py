"""Optional PyTorch causal SFT; no downloads, model names or machine paths.

ModelStore.load_branch must load an isolated model with the exact requested weights
and trainable flags. Adapter stores also own saving/loading frozen-base identities.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import random
from time import perf_counter
from typing import Protocol
import weakref

from self_curate.contracts import ArtifactRef, ComputeCost, ModelRef, ResourceUsage
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256, write_json
from .contracts import BackendTrainingResult
from .artifacts import checkpoint_fingerprint
from .service import TrainingFailure


@dataclass(frozen=True)
class LoadedModel:
    model: object
    model_ref: ModelRef


class ModelStore(Protocol):
    checkpoint_kind: str  # "full" or "adapter"

    def load_branch(self, model_ref: ModelRef) -> LoadedModel:
        """Return a fresh model, sharing no mutable parameters with accepted state."""
        ...

    def save_branch(self, model, output: Path) -> None:
        """Save actual weights; adapter stores must save the adapter and its config."""
        ...

    def load_checkpoint(self, model_ref: ModelRef, checkpoint: Path) -> LoadedModel:
        """Load a previously saved candidate, including its exact adapter/base config."""
        ...


class StateDictModelStore:
    """Concrete full-state store for an injected torch architecture factory.

    Initial weights are an existing local state_dict (never downloaded). The
    starting ModelRef.weights_hash must be its actual file SHA256. Production
    transformer/PEFT deployments can implement ModelStore without this serializer.
    """

    checkpoint_kind = "full"

    def __init__(self, model_factory, *, initial_model, initial_state):
        require(
            isinstance(initial_model, ModelRef) and initial_model.adapter_hash is None,
            "Full state_dict store needs an unambiguous full-model reference",
        )
        self._factory = model_factory
        self._initial_model = initial_model
        self._initial_state = Path(initial_state).resolve()
        require(
            file_sha256(self._initial_state) == initial_model.weights_hash,
            "Initial state_dict differs from the declared weight hash",
        )

    def _load(self, model_ref, path):
        import torch

        model = self._factory()
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        return LoadedModel(model, model_ref)

    def load_branch(self, model_ref):
        require(model_ref == self._initial_model, "Unknown initial model checkpoint")
        require(
            file_sha256(self._initial_state) == model_ref.weights_hash,
            "Initial state_dict changed before training",
        )
        return self._load(model_ref, self._initial_state)

    def save_branch(self, model, output):
        import torch

        output.mkdir(parents=True, exist_ok=False)
        torch.save(model.state_dict(), output / "state.pt")

    def load_checkpoint(self, model_ref, checkpoint):
        from self_curate.contracts.parsing import load_json, model_ref as parse_model

        checkpoint = Path(checkpoint).resolve()
        require(
            parse_model(
                load_json((checkpoint / "model.json").read_text(encoding="utf-8"))
            )
            == model_ref,
            "Checkpoint metadata differs from requested weights",
        )
        require(
            checkpoint_fingerprint(checkpoint / "weights")[0] == model_ref.weights_hash,
            "Saved state_dict differs from the candidate weight hash",
        )
        return self._load(model_ref, checkpoint / "weights/state.pt")


class TorchSFTBackend:
    """Micro-batch=1, exact stored order, token-weighted accumulation windows.

    Each stage has a fresh optimizer. Only parameters marked requires_grad update.
    No truncation/packing/shuffling/scheduler/AMP or silent fallback is performed.
    """

    def __init__(self, store: ModelStore, *, backend, device="cpu"):
        require(backend in ("mock", "real"), "Specify execution domain")
        require(
            store.checkpoint_kind in ("full", "adapter"), "Unknown checkpoint format"
        )
        self.store = store
        self.backend = backend
        self.device = device
        self._models = weakref.WeakSet()
        self._produced = {}

    def train(self, request, output):
        import torch
        import torch.nn.functional as F

        require(request.backend == self.backend, "Torch backend/domain mismatch")
        require(
            (self.store.checkpoint_kind == "adapter")
            == (request.starting_model.adapter_hash is not None),
            "Checkpoint store format differs from full/adapter parameter identity",
        )
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        device = torch.device(self.device)
        require(device.type in ("cpu", "cuda"), "Supported devices are CPU and CUDA")
        cuda_devices = (
            list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        )
        start = perf_counter()
        python_rng = random.getstate()
        processed, steps, loss_sum = 0, 0, 0.0
        model = None
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(request.recipe.seed)
                torch.random.default_generator.manual_seed(request.recipe.seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(request.recipe.seed)
                saved = self._produced.get(request.starting_model.model_hash)
                if saved is None:
                    loaded = self.store.load_branch(request.starting_model)
                else:
                    saved_path, saved_hash = saved
                    require(
                        checkpoint_fingerprint(saved_path)[0] == saved_hash,
                        "Stored candidate checkpoint changed before continuation",
                    )
                    loaded = self.store.load_checkpoint(
                        request.starting_model, saved_path
                    )
                require(
                    isinstance(loaded, LoadedModel)
                    and loaded.model_ref == request.starting_model,
                    "ModelStore loaded another checkpoint identity",
                )
                model = loaded.model
                require(
                    model not in self._models,
                    "ModelStore reused a mutable training branch",
                )
                self._models.add(model)
                model.to(device)
                model.train()
                named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                require(bool(named), "No trainable model parameters")
                if saved is not None:
                    from self_curate.contracts.parsing import load_json

                    previous_training = load_json(
                        (saved_path / "training.json").read_text(encoding="utf-8")
                    )
                    require(
                        previous_training["trainable_parameters"]
                        == [n for n, _ in named],
                        "Checkpoint continuation changed the trainable parameter scope",
                    )
                parameters = [p for _, p in named]
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
                examples = request.data.examples
                window_size = request.recipe.accumulation_steps
                for _epoch in range(request.recipe.epochs):
                    for offset in range(0, len(examples), window_size):
                        window = examples[offset : offset + window_size]
                        denominator = sum(x.supervised_tokens for x in window)
                        optimizer.zero_grad(set_to_none=True)
                        for example in window:
                            ids = torch.tensor(
                                [example.input_ids], dtype=torch.long, device=device
                            )
                            labels = torch.tensor(
                                example.labels[1:], dtype=torch.long, device=device
                            )
                            output_value = model(
                                input_ids=ids, attention_mask=torch.ones_like(ids)
                            )
                            logits = output_value.logits
                            require(
                                logits.ndim == 3 and logits.shape[:2] == ids.shape,
                                "Causal model logits do not match complete input sequence",
                            )
                            loss = F.cross_entropy(
                                logits[0, :-1].float(),
                                labels,
                                ignore_index=-100,
                                reduction="sum",
                            )
                            require(
                                bool(torch.isfinite(loss).item()), "Non-finite SFT loss"
                            )
                            (loss / denominator).backward()
                            processed += example.supervised_tokens
                            loss_sum += float(loss.detach().item())
                        require(
                            all(
                                p.grad is None
                                or bool(torch.isfinite(p.grad).all().item())
                                for p in parameters
                            ),
                            "Non-finite SFT gradient",
                        )
                        if request.recipe.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                parameters,
                                request.recipe.max_grad_norm,
                                error_if_nonfinite=True,
                            )
                        optimizer.step()
                        steps += 1
                optimizer.zero_grad(set_to_none=True)
                self.store.save_branch(model, output / "weights")
                weights_hash, files = checkpoint_fingerprint(output / "weights")
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
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": (
                            torch.cuda.get_rng_state_all() if cuda_devices else []
                        ),
                        "python_rng": random.getstate(),
                        "optimizer_steps": steps,
                        "processed_supervised_tokens": processed,
                    },
                    output / "training_state.pt",
                )
                write_json(output / "model.json", candidate.to_dict())
                write_json(output / "weights_checksums.json", files)
                write_json(
                    output / "training.json",
                    {
                        "request_hash": request.record_hash,
                        "recipe": request.recipe.to_dict(),
                        "trainable_parameters": [n for n, _ in named],
                        "checkpoint_kind": self.store.checkpoint_kind,
                        "optimizer_state_policy": "fresh_per_stage",
                        "micro_batch": 1,
                        "loss_reduction": request.recipe.loss_reduction,
                    },
                )
                checkpoint_hash, _ = checkpoint_fingerprint(output)
                self._produced[candidate.model_hash] = (output, checkpoint_hash)
                usage = ResourceUsage(
                    ComputeCost(
                        perf_counter() - start,
                        0 if device.type == "cpu" else None,
                        sum(len(x.input_ids) for x in examples) * request.recipe.epochs,
                    ),
                    training_supervised_tokens=processed,
                )
                return BackendTrainingResult(
                    candidate,
                    ArtifactRef(output.as_uri(), checkpoint_hash),
                    processed,
                    steps,
                    loss_sum / processed,
                    usage,
                )
        except Exception as exc:
            raise TrainingFailure(
                str(exc),
                usage=ResourceUsage(
                    ComputeCost(
                        perf_counter() - start,
                        0 if device.type == "cpu" else None,
                        None,
                    ),
                    training_supervised_tokens=processed,
                ),
            ) from exc
        finally:
            random.setstate(python_rng)
            if model is not None:
                model.zero_grad(set_to_none=True)
