"""Adapter for caller-supplied raw logits. Does not load any model or dataset."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from self_curate.contracts import ArtifactRef, ResourceUsage, ModelRef, digest
from self_curate.contracts.types import require, integer, sha256
from self_curate.data.pool import file_sha256, write_json
from self_curate.training.execution import code_identity
from .contracts import ForwardRequest, LogitsOutput, TokenMetricResult
from .math import token_statistics


class ScoringFailure(RuntimeError):
    def __init__(self, message, *, usage=None, ledger=None, records=(), outcomes=()):
        super().__init__(message)
        self.usage = usage if isinstance(usage, ResourceUsage) else ResourceUsage()
        self.ledger = ledger
        self.records, self.outcomes = tuple(records), tuple(outcomes)


def local_artifact(path):
    return ArtifactRef(Path(path).resolve().as_uri(), file_sha256(Path(path)))


class ProvidedLogitsBackend:
    def __init__(
        self, provider, *, backend, vocab_size, logits_dtype="float64", chunk_size=128
    ):
        require(backend in ("mock", "real"), "Explicit logits scoring domain required")
        require(
            isinstance(getattr(provider, "model_ref", None), ModelRef),
            "Logits provider must declare ModelRef",
        )
        sha256(
            getattr(provider, "implementation_hash", None),
            "logits producer implementation_hash",
        )
        integer(vocab_size, "full vocabulary size", 1)
        integer(chunk_size, "statistics chunk size", 1)
        require(
            logits_dtype in ("float16", "bfloat16", "float32", "float64"),
            "Explicit raw-logit dtype required",
        )
        self.provider, self.backend, self.vocab_size = provider, backend, vocab_size
        self.logits_dtype, self.chunk_size = logits_dtype, chunk_size

    @property
    def implementation_hash(self):
        import numpy as np

        return digest(
            {
                "code": file_sha256(Path(__file__)),
                "producer": code_identity(self.provider),
                "vocab_size": self.vocab_size,
                "raw_logits_dtype": self.logits_dtype,
                "chunk_size": self.chunk_size,
                "statistics_dtype": "float64",
                "numpy": np.__version__,
            }
        )

    def score(self, request: ForwardRequest, output):
        import numpy as np

        require(
            request.backend == self.backend
            and self.provider.model_ref == request.model_ref,
            "Logits producer model/domain mismatch",
        )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        usage, response, numeric_start = ResourceUsage(), None, None
        try:
            response = self.provider.forward(request)
            if isinstance(getattr(response, "usage", None), ResourceUsage):
                usage = response.usage
            require(
                isinstance(response, LogitsOutput), "Producer must return LogitsOutput"
            )
            require(
                response.request_hash == request.record_hash
                and response.positions == request.positions
                and response.model_ref == request.model_ref
                and self.provider.model_ref == request.model_ref,
                "Logits belong to another request/model/input/position order",
            )
            require(
                self.backend != "real"
                or not response.evidence.uri.strip()
                .lower()
                .startswith(("fixture://", "mock://")),
                "Mock logits cannot produce real signals",
            )
            values = (
                response.logits
                if hasattr(response.logits, "shape")
                else np.asarray(response.logits)
            )
            require(
                tuple(values.shape) == (len(request.positions), self.vocab_size),
                "Logits must cover every requested row and the full configured vocabulary",
            )
            require(
                str(values.dtype).removeprefix("torch.") == self.logits_dtype,
                "Producer changed raw-logit dtype",
            )
            numeric_start = perf_counter()
            nll, entropy = token_statistics(
                values,
                tuple(request.input_ids[p] for p in request.positions),
                chunk_size=self.chunk_size,
            )
            numeric_wall = perf_counter() - numeric_start
            if usage.compute.wall_seconds is not None:
                usage = replace(
                    usage,
                    compute=replace(
                        usage.compute,
                        wall_seconds=usage.compute.wall_seconds + numeric_wall,
                    ),
                )
            write_json(
                output / "token_statistics.json",
                {
                    "schema_version": "1",
                    "request": request.to_dict(),
                    "model_ref": request.model_ref.to_dict(),
                    "implementation_hash": self.implementation_hash,
                    "positions": request.positions,
                    "token_nll_nats": nll,
                    "token_entropy_nats": entropy,
                    "producer_evidence": response.evidence.to_dict(),
                    "producer_usage": response.usage.to_dict(),
                    "local_numeric_wall_seconds": numeric_wall,
                    "full_vocabulary_size": self.vocab_size,
                    "raw_logits_dtype": self.logits_dtype,
                    "statistics_dtype": "float64",
                    "weights_updated": False,
                },
            )
            return TokenMetricResult(
                request.record_hash,
                request.positions,
                nll,
                entropy,
                local_artifact(output / "token_statistics.json"),
                usage,
                self.backend,
            )
        except Exception as exc:
            if response is None and isinstance(
                getattr(exc, "usage", None), ResourceUsage
            ):
                usage = exc.usage
            try:
                write_json(
                    output / "failure.json",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "usage": usage.to_dict(),
                    },
                )
            except OSError:
                pass
            raise ScoringFailure(str(exc), usage=usage) from exc
