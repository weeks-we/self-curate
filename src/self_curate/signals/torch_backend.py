"""Optional actual causal-model forward, frozen and isolated; no training."""

from pathlib import Path
from time import perf_counter
import hashlib
import random
import weakref

from self_curate.contracts import ComputeCost, ResourceUsage, digest
from self_curate.contracts.types import require, integer
from self_curate.data.pool import file_sha256, write_json
from self_curate.training.execution import (
    code_identity,
    torch_execution,
    execution_source_hash,
)
from self_curate.training.torch_backend import LoadedModel
from .backends import ScoringFailure, local_artifact
from .contracts import TokenMetricResult
from .math import token_statistics


def _state_hash(model):
    import torch

    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode("utf-8"))
        value.update(str((tuple(tensor.shape), tensor.dtype)).encode("utf-8"))
        value.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return value.hexdigest()


def _execution(device):
    result = torch_execution(device)
    result["policy"] = (
        "native-parameter-dtype/no-autocast/full-vocab-token-statistics-float64-v1"
    )
    return result


class TorchSignalBackend:
    def __init__(self, store, *, backend, device="cpu", chunk_size=128):
        require(backend in ("mock", "real"), "Explicit forward scoring domain required")
        integer(chunk_size, "statistics chunk size", 1)
        self.store, self.backend, self.device, self.chunk_size = (
            store,
            backend,
            device,
            chunk_size,
        )
        self._models = weakref.WeakSet()

    @property
    def implementation_hash(self):
        import numpy as np

        return digest(
            {
                "code": file_sha256(Path(__file__)),
                "execution_code": execution_source_hash(),
                "store": code_identity(self.store),
                "execution": _execution(self.device),
                "chunk_size": self.chunk_size,
                "numpy": np.__version__,
                "mode": "eval/inference",
            }
        )

    def score(self, request, output):
        import torch

        require(request.backend == self.backend, "Forward backend/domain mismatch")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        execution = _execution(self.device)
        device = torch.device(execution["device"])
        cuda_devices = (
            list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        )
        started, processed, model = perf_counter(), 0, None
        python_rng = random.getstate()

        def usage():
            return ResourceUsage(
                ComputeCost(
                    perf_counter() - started,
                    0 if device.type == "cpu" else None,
                    processed,
                ),
                scoring_tokens=processed,
                new_gradient_samples=0,
                new_gradient_tokens=0,
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
                    "ModelStore returned another scoring model",
                )
                model = loaded.model
                require(
                    model not in self._models,
                    "ModelStore reused a mutable scoring branch",
                )
                self._models.add(model)
                model.to(device)
                model.eval()
                before = _state_hash(model)
                flags = tuple(
                    (n, p.requires_grad, str(p.dtype))
                    for n, p in model.named_parameters()
                )
                with torch.inference_mode():
                    ids = torch.tensor(
                        [request.input_ids], dtype=torch.long, device=device
                    )
                    processed = (
                        None  # A failed forward may have consumed unknown partial work.
                    )
                    logits = model(
                        input_ids=ids, attention_mask=torch.ones_like(ids)
                    ).logits
                    processed = len(request.input_ids)
                    require(
                        logits.ndim == 3
                        and tuple(logits.shape[:2]) == (1, len(request.input_ids)),
                        "Causal logits do not match the complete input",
                    )
                    nll, entropy = token_statistics(
                        logits[0],
                        tuple(request.input_ids[p] for p in request.positions),
                        rows=tuple(p - 1 for p in request.positions),
                        chunk_size=self.chunk_size,
                    )
                after = _state_hash(model)
                require(
                    before == after
                    and flags
                    == tuple(
                        (n, p.requires_grad, str(p.dtype))
                        for n, p in model.named_parameters()
                    ),
                    "Scoring modified model weights/buffers/trainable scope",
                )
                require(
                    execution == _execution(device),
                    "Numeric execution policy changed during scoring",
                )
                write_json(
                    output / "token_statistics.json",
                    {
                        "schema_version": "1",
                        "request": request.to_dict(),
                        "model_ref": request.model_ref.to_dict(),
                        "positions": request.positions,
                        "token_nll_nats": nll,
                        "token_entropy_nats": entropy,
                        "parameter_state_before": before,
                        "parameter_state_after": after,
                        "parameter_dtypes": sorted({dtype for _, _, dtype in flags}),
                        "implementation_hash": self.implementation_hash,
                        "execution": execution,
                        "full_vocabulary_size": logits.shape[-1],
                        "mode": "eval/inference",
                        "weights_updated": False,
                        "backward_called": False,
                        "optimizer_step_called": False,
                    },
                )
                return TokenMetricResult(
                    request.record_hash,
                    request.positions,
                    nll,
                    entropy,
                    local_artifact(output / "token_statistics.json"),
                    usage(),
                    self.backend,
                )
        except Exception as exc:
            try:
                write_json(
                    output / "failure.json",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "usage": usage().to_dict(),
                    },
                )
            except OSError:
                pass
            raise ScoringFailure(str(exc), usage=usage()) from exc
        finally:
            random.setstate(python_rng)
