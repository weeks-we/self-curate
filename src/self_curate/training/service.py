"""Execute one isolated SFT job and produce existing P0 TrainingRecord/costs."""

from pathlib import Path
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ResourceUsage,
    TrainingRecord,
)
from self_curate.contracts.types import require
from self_curate.data.pool import write_json
from .contracts import BackendTrainingResult, TrainingOutcome, TrainingRequest
from .artifacts import verify_local_checkpoint


class TrainingFailure(RuntimeError):
    """A failed physical operation; all reported usage is preserved, never invented."""

    def __init__(self, message, *, usage=None, ledger=None):
        super().__init__(message)
        self.usage = usage or ResourceUsage()
        self.ledger = ledger


class TrainingService:
    def __init__(self, backend):
        self.backend = backend

    def train(self, request: TrainingRequest, output) -> TrainingOutcome:
        require(isinstance(request, TrainingRequest), "Expected TrainingRequest")
        require(
            self.backend.backend == request.backend, "Training backend/domain mismatch"
        )
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "request.json", request.to_dict())
        start = perf_counter()
        usage = ResourceUsage()
        try:
            result = self.backend.train(request, output / "checkpoint")
            if isinstance(getattr(result, "usage", None), ResourceUsage):
                usage = result.usage
            require(isinstance(result, BackendTrainingResult), "Invalid trainer result")
            verify_local_checkpoint(result.checkpoint)
            require(
                result.processed_supervised_tokens
                == request.data.supervised_tokens * request.recipe.epochs,
                "Backend changed the declared training exposure budget",
            )
            require(
                result.optimizer_steps
                == request.recipe.epochs
                * (
                    (len(request.data.examples) + request.recipe.accumulation_steps - 1)
                    // request.recipe.accumulation_steps
                ),
                "Unexpected optimizer step count",
            )
            require(
                result.candidate_model.model_hash != request.starting_model.model_hash,
                "Training did not produce a distinct candidate identity",
            )
            if request.backend == "real":
                require(
                    not result.checkpoint.uri.strip()
                    .lower()
                    .startswith(("fixture://", "mock://")),
                    "Mock checkpoint cannot enter real training",
                )
            event = CostEvent(
                "training-" + request.record_hash,
                request.attempt_id,
                "training",
                request.training_id,
                request.starting_model.model_hash,
                "completed",
                usage,
                ArtifactRef(
                    "record://sft-result/" + result.record_hash, result.record_hash
                ),
                request.backend,
            )
            ledger = CostLedger("sft-" + request.record_hash, (event,), request.backend)
            record = TrainingRecord(
                request.training_id,
                request.starting_model,
                request.batch_id,
                request.training_data_hash,
                ArtifactRef(
                    "record://sft-recipe/" + request.recipe.record_hash,
                    request.recipe.record_hash,
                ),
                "completed",
                result.candidate_model,
                result.checkpoint,
                result.processed_supervised_tokens,
                event.receipt,
                ledger.reference((event.event_id,)),
                request.backend,
            )
            outcome = TrainingOutcome(request, record, ledger, result)
            write_json(output / "outcome.json", outcome.to_dict())
            return outcome
        except Exception as exc:
            if isinstance(exc, TrainingFailure):
                usage = exc.usage
            # Preserve measured elapsed time if the backend reported no usage at all.
            if usage.compute.wall_seconds is None:
                from dataclasses import replace

                usage = replace(
                    usage,
                    compute=replace(usage.compute, wall_seconds=perf_counter() - start),
                )
            error = {"type": type(exc).__name__, "message": str(exc)}
            from self_curate.contracts import digest

            event = CostEvent(
                "training-" + request.record_hash,
                request.attempt_id,
                "training",
                request.training_id,
                request.starting_model.model_hash,
                "failed",
                usage,
                ArtifactRef("record://sft-failure/" + digest(error), digest(error)),
                request.backend,
            )
            ledger = CostLedger("sft-" + request.record_hash, (event,), request.backend)
            try:
                write_json(
                    output / "failure.json",
                    {"error": error, "ledger": ledger.to_dict()},
                )
            except OSError:
                pass  # The typed exception still carries incurred costs if storage also failed.
            raise TrainingFailure(str(exc), usage=usage, ledger=ledger) from exc
