"""One owned frozen model and bounded output-head chunks for initial pool scoring."""

from pathlib import Path
from time import perf_counter
from self_curate.contracts import ResourceUsage, ComputeCost, digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256, write_json
from self_curate.training.execution import code_identity, torch_execution
from .contracts import TokenMetricResult
from .backends import local_artifact, ScoringFailure


class ResidentSignalBackend:
    def __init__(self, store, *, backend="real", device="cuda:0", chunk_size=32):
        self.store, self.backend, self.device, self.chunk_size = (
            store,
            backend,
            device,
            chunk_size,
        )
        self.model = None
        self.model_ref = None
        require(backend in ("real", "mock"), "Unknown scoring execution domain")
        require(
            type(chunk_size) is int and chunk_size > 0,
            "Positive scoring chunk size required",
        )

    @property
    def implementation_hash(self):
        return digest(
            {
                "code": file_sha256(Path(__file__)),
                "math": file_sha256(Path(__file__).with_name("math.py")),
                "store": code_identity(self.store),
                "execution": torch_execution(self.device),
                "chunk_size": self.chunk_size,
                "policy": "owned-frozen-model/full-vocabulary-float64-stats-v1",
            }
        )

    def score(self, request, output):
        import torch

        root = Path(output)
        root.mkdir(parents=True, exist_ok=False)
        start = perf_counter()
        processed = None

        def usage():
            return ResourceUsage(
                ComputeCost(perf_counter() - start, None, processed),
                scoring_tokens=processed,
                training_supervised_tokens=0,
                new_gradient_samples=0,
                new_gradient_tokens=0,
            )

        try:
            require(request.backend == self.backend, "Scoring backend changed")
            if self.model is None:
                loaded = self.store.load_branch(request.model_ref)
                require(loaded.model_ref == request.model_ref, "Wrong scoring model")
                self.model = loaded.model.to(self.device).eval()
                self.model.requires_grad_(False)
                self.model_ref = loaded.model_ref
            require(
                self.model_ref == request.model_ref,
                "Resident scorer cannot switch model versions",
            )
            require(
                callable(getattr(self.model, "self_curate_signal_logits", None)),
                "Store must provide bounded selected-position logits",
            )
            state = lambda: tuple(
                (n, id(t), t._version)
                for n, t in self.model.state_dict(keep_vars=True).items()
            )
            before = state()
            nll = []
            entropy = []
            observed = []
            with torch.inference_mode():
                ids = torch.tensor([request.input_ids], device=self.device)
                for positions, logits in self.model.self_curate_signal_logits(
                    ids,
                    torch.ones_like(ids),
                    request.positions,
                    chunk_size=self.chunk_size,
                ):
                    values = logits.double()
                    require(
                        values.ndim == 2
                        and values.shape[0] == len(positions)
                        and torch.isfinite(values).all().item(),
                        "Invalid full-vocabulary logits",
                    )
                    logp = values - torch.logsumexp(values, dim=-1, keepdim=True)
                    targets = torch.tensor(
                        [request.input_ids[p] for p in positions], device=values.device
                    )
                    losses = -logp.gather(1, targets[:, None]).flatten()
                    entropies = -(logp.exp() * logp).sum(dim=-1)
                    require(
                        torch.isfinite(losses).all().item()
                        and torch.isfinite(entropies).all().item(),
                        "Non-finite GPU token statistics",
                    )
                    nll.extend(losses.tolist())
                    entropy.extend(entropies.tolist())
                    observed.extend(positions)
            require(
                tuple(observed) == request.positions and state() == before,
                "Scoring mutated weights or substituted token positions",
            )
            processed = len(request.input_ids)
            write_json(
                root / "token_statistics.json",
                {
                    "request": request.to_dict(),
                    "token_nll_nats": nll,
                    "token_entropy_nats": entropy,
                    "full_vocabulary": True,
                    "weights_updated": False,
                    "model_loaded_once": True,
                    "computation_hash": self.implementation_hash,
                },
            )
            return TokenMetricResult(
                request.record_hash,
                request.positions,
                tuple(nll),
                tuple(entropy),
                local_artifact(root / "token_statistics.json"),
                usage(),
                self.backend,
            )
        except Exception as exc:
            raise ScoringFailure(str(exc), usage=usage()) from exc

    def close(self):
        self.model = None
        self.model_ref = None
