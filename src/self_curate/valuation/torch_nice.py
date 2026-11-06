"""Actual masked backward computation on an isolated, frozen-weight model branch."""

from pathlib import Path
from time import perf_counter
import hashlib
import random
import weakref

from self_curate.contracts import ComputeCost, ResourceUsage, digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256, write_json
from self_curate.training.torch_backend import LoadedModel
from self_curate.training.execution import (
    code_identity,
    torch_execution,
    execution_source_hash,
)
from .common import ValuationFailure, artifact
from .contracts import NICEScores
from .nice import fresh_adam, count_sketch, score_matrix


def parameter_coordinates(model):
    parameters = tuple(
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    )
    return digest([(n, list(p.shape), str(p.dtype)) for n, p in parameters]), sum(
        p.numel() for _, p in parameters
    )


def state_hash(model):
    import torch

    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return result.hexdigest()


class TorchNICEBackend:
    """No optimizer is constructed. Cached values bind all inputs and parameter IDs.

    Reference NLL is token SUM * reward / fixed rollout count. Candidate CE is
    per-trajectory token MEAN. Dropout is disabled with eval(), autograd stays on.
    """

    def __init__(self, store, *, backend, device="cpu"):
        require(backend in ("mock", "real"), "Explicit NICE backend required")
        self.store, self.backend, self.device = store, backend, device
        self._samples, self._targets = {}, {}
        self._loaded = weakref.WeakSet()

    @property
    def implementation_hash(self):
        # Exact source is part of the cache/protocol identity.
        from . import nice
        import numpy as np

        return digest(
            (
                file_sha256(Path(__file__)),
                file_sha256(Path(nice.__file__)),
                str(self.device),
                torch_execution(self.device),
                execution_source_hash(),
                code_identity(self.store),
                np.__version__,
            )
        )

    def compute(self, request, data, reference, recipe, output):
        import numpy as np
        import torch
        import torch.nn.functional as F

        require(request.backend == self.backend, "NICE backend mismatch")
        start, tokens, hits, new_samples, new_tokens = perf_counter(), 0, 0, 0, 0
        python_rng = random.getstate()
        device = torch.device(self.device)
        execution = torch_execution(device)
        cuda_devices = (
            []
            if device.type == "cpu"
            else [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        )
        output = Path(output)
        model = None
        before = None

        def usage():
            return ResourceUsage(
                ComputeCost(
                    perf_counter() - start, 0 if device.type == "cpu" else None, tokens
                ),
                scoring_tokens=0,
                new_gradient_samples=new_samples,
                new_gradient_tokens=new_tokens,
                cache_hits=hits,
                training_supervised_tokens=0,
            )

        try:
            with (
                torch.random.fork_rng(devices=cuda_devices),
                torch.autocast(device.type, enabled=False),
            ):
                loaded = self.store.load_branch(request.model_ref)
                require(
                    isinstance(loaded, LoadedModel)
                    and loaded.model_ref == request.model_ref,
                    "Gradient loader changed model identity",
                )
                model = loaded.model
                require(
                    model not in self._loaded,
                    "Gradient model store reused a mutable branch",
                )
                self._loaded.add(model)
                model.to(device)
                model.eval()
                require(
                    parameter_coordinates(model)
                    == (recipe.parameter_coordinates_hash, recipe.parameter_count),
                    "Gradient parameter coordinates changed",
                )
                before = state_hash(model)
                parameters = tuple(p for p in model.parameters() if p.requires_grad)

                def gradient(example, reduction):
                    nonlocal tokens, new_tokens
                    ids = torch.tensor(
                        [example.input_ids], dtype=torch.long, device=device
                    )
                    labels = torch.tensor(
                        [example.labels], dtype=torch.long, device=device
                    )
                    tokens += len(example.input_ids)
                    new_tokens += len(example.input_ids)
                    with torch.enable_grad():
                        logits = model(
                            input_ids=ids, attention_mask=torch.ones_like(ids)
                        ).logits
                        loss = F.cross_entropy(
                            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                            labels[:, 1:].reshape(-1),
                            ignore_index=-100,
                            reduction=reduction,
                        )
                        require(bool(torch.isfinite(loss)), "Nonfinite replay loss")
                        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
                    array = np.concatenate(
                        [
                            (
                                np.zeros(p.numel(), dtype=np.float64)
                                if g is None
                                else g.detach()
                                .float()
                                .cpu()
                                .numpy()
                                .reshape(-1)
                                .astype(np.float64)
                            )
                            for p, g in zip(parameters, grads)
                        ]
                    )
                    require(np.isfinite(array).all(), "Nonfinite NICE gradient")
                    return array

                def project(vector):
                    return (
                        vector.copy()
                        if recipe.projection_dimension is None
                        else count_sketch(
                            vector, recipe.projection_dimension, recipe.projection_seed
                        )
                    )

                base_key = digest(
                    (
                        request.model_ref.to_dict(),
                        recipe.to_dict(),
                        self.implementation_hash,
                    )
                )
                target_key = digest((base_key, reference.record_hash))
                if target_key in self._targets:
                    targets = self._targets[target_key]
                    hits += len(targets)
                else:
                    replays = {r.rollout_hash: r for r in reference.replays}
                    targets = []
                    suite = reference.evaluation.suite
                    for task in suite.tasks:
                        direction = np.zeros(recipe.parameter_count, dtype=np.float64)
                        runs = [
                            r
                            for r in reference.evaluation.rollouts
                            if r.task_id == task.task_id
                        ]
                        require(
                            len(runs) == len(suite.seeds),
                            "Missing reference rollout denominator",
                        )
                        for rollout in runs:
                            if rollout.reward:
                                for step in replays[rollout.record_hash].steps:
                                    direction += gradient(step, "sum") * (
                                        rollout.reward / len(suite.seeds)
                                    )
                        targets.append(project(direction))
                    targets = tuple(targets)
                    # Publish caches only after the frozen-state check below.
                raw_vectors, adam_vectors, pending = [], [], {}
                for example in data.examples:
                    key = digest(
                        (base_key, example.to_dict(), "assistant-ce-token-mean-v1")
                    )
                    if key in self._samples:
                        raw, adam = self._samples[key]
                        hits += 1
                    else:
                        g = gradient(example, "mean")
                        new_samples += 1
                        raw, adam = project(g), project(
                            fresh_adam(g, recipe.adam_epsilon, recipe.max_grad_norm)
                        )
                        pending[key] = (raw, adam)
                    raw_vectors.append(raw)
                    adam_vectors.append(adam)
                require(
                    state_hash(model) == before
                    and torch_execution(device) == execution
                    and parameter_coordinates(model)
                    == (recipe.parameter_coordinates_hash, recipe.parameter_count),
                    "Frozen NICE parameters/buffers/trainable scope changed",
                )
                self._samples.update(pending)
                self._targets[target_key] = targets
                np.savez_compressed(
                    output / "vectors.npz",
                    raw=np.asarray(raw_vectors),
                    adam=np.asarray(adam_vectors),
                    targets=np.asarray(targets),
                )
                write_json(
                    output / "gradient_identity.json",
                    {
                        "model": request.model_ref.to_dict(),
                        "recipe": recipe.to_dict(),
                        "reference_hash": reference.record_hash,
                        "data_hash": data.record_hash,
                        "state_hash_before": before,
                        "state_hash_after": state_hash(model),
                        "parameter_coordinates_hash": parameter_coordinates(model)[0],
                        "candidate_reduction": "assistant_token_mean",
                        "target_reduction": "reward_weighted_nll_token_sum/fixed_rollout_count",
                        "transformation_before_projection": True,
                        "optimizer_step_called": False,
                        "execution": execution,
                        "implementation_hash": self.implementation_hash,
                    },
                )
                return NICEScores(
                    request.request_hash,
                    data.record_hash,
                    reference.record_hash,
                    recipe.record_hash,
                    tuple(x.data_id for x in data.examples),
                    tuple(t.task_id for t in reference.evaluation.suite.tasks),
                    score_matrix(adam_vectors, targets, recipe.zero_epsilon),
                    score_matrix(raw_vectors, targets, recipe.zero_epsilon),
                    tuple(float(np.linalg.norm(t)) for t in targets),
                    artifact(output / "vectors.npz"),
                    usage(),
                    self.backend,
                )
        except Exception as exc:
            details = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "usage": usage().to_dict(),
                "frozen_state_before": before,
                "candidate_model_not_promoted": True,
            }
            try:
                write_json(output / "gradient_failure.json", details)
            except Exception:
                pass
            raise ValuationFailure(str(exc), usage=usage()) from exc
        finally:
            random.setstate(python_rng)
