"""One optimizer across three fixed data waves; no selection or acceptance here."""

from dataclasses import dataclass
import hashlib
import uuid
from pathlib import Path

from self_curate.contracts import (
    ArtifactRef,
    CostEvent,
    CostLedger,
    ResourceUsage,
    ModelRef,
    digest,
)
from self_curate.contracts.types import integer, require, sha256
from self_curate.contracts.workflow import WorkflowRecord
from self_curate.data.pool import write_json
from .contracts import TrainingRequest
from .artifacts import verify_local_checkpoint
from .service import TrainingFailure


def wave_boundaries(count):
    integer(count, "selected examples", 3)
    q, remainder = divmod(count, 3)
    sizes = tuple(q + (i < remainder) for i in range(3))
    return (sizes[0], sizes[0] + sizes[1], count)


@dataclass(frozen=True)
class StageCheckpoint(WorkflowRecord):
    request_hash: str
    stage: int
    model_ref: ModelRef
    checkpoint: ArtifactRef
    examples_seen: int
    supervised_tokens: int
    optimizer_steps: int
    usage: ResourceUsage
    schema_version: str = "1"

    def __post_init__(self):
        sha256(self.request_hash, "continuous request hash")
        for name in ("stage", "examples_seen", "supervised_tokens", "optimizer_steps"):
            integer(getattr(self, name), name, 1)
        require(
            self.schema_version == "1"
            and self.stage <= 3
            and isinstance(self.model_ref, ModelRef)
            and isinstance(self.checkpoint, ArtifactRef)
            and isinstance(self.usage, ResourceUsage),
            "Invalid stage checkpoint",
        )


class ContinuousTrainingService:
    def __init__(self, backend):
        self.backend = backend

    def train(self, request, output, *, on_checkpoint, resume=False):
        require(
            isinstance(request, TrainingRequest) and request.recipe.epochs == 1,
            "EXP001 consumes each selected trajectory once",
        )
        require(
            request.backend == self.backend.backend,
            "Continuous backend/domain mismatch",
        )
        boundaries = wave_boundaries(len(request.data.examples))
        output = Path(output).resolve()
        if output.exists():
            require(resume, "Existing training requires explicit resume")
            from self_curate.training.recovery import read_json

            require(
                read_json(output / "request.json") == request.to_dict()
                and read_json(output / "boundaries.json") == list(boundaries),
                "Recovery changed training request/boundaries",
            )
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "request.json", request.to_dict())
            write_json(output / "boundaries.json", list(boundaries))
        checkpoints, events = [], []
        callback_failed = False

        def emit(value):
            nonlocal callback_failed
            from self_curate.contracts import ModelRef

            stage = len(checkpoints) + 1
            require(
                isinstance(value, StageCheckpoint)
                and value.stage == stage
                and stage <= 3,
                "Wrong or repeated training checkpoint",
            )
            end = boundaries[stage - 1]
            start = boundaries[stage - 2] if stage > 1 else 0
            expected_tokens = sum(
                x.supervised_tokens for x in request.data.examples[:end]
            )
            steps = sum(
                (b - a + request.recipe.accumulation_steps - 1)
                // request.recipe.accumulation_steps
                for a, b in zip((0, *boundaries[:-1]), boundaries[:stage])
            )
            require(
                value.request_hash == request.record_hash
                and value.examples_seen == end
                and value.supervised_tokens == expected_tokens
                and value.optimizer_steps == steps,
                "Checkpoint changed the fixed data prefix or optimizer windows",
            )
            require(
                isinstance(value.model_ref, ModelRef)
                and value.model_ref.tokenizer == request.starting_model.tokenizer
                and value.model_ref.base_model_id
                == request.starting_model.base_model_id
                and value.model_ref.revision == request.starting_model.revision
                and value.model_ref.parameterization
                == request.starting_model.parameterization,
                "Checkpoint changed model family or tokenization",
            )
            require(
                (value.model_ref.adapter_hash is None)
                == (request.starting_model.adapter_hash is None),
                "Checkpoint parameterization changed",
            )
            if value.model_ref.adapter_hash is not None:
                require(
                    value.model_ref.weights_hash == request.starting_model.weights_hash,
                    "LoRA changed frozen base identity",
                )
            require(
                value.model_ref
                != (
                    checkpoints[-1].model_ref if checkpoints else request.starting_model
                ),
                "Training checkpoint identity did not change",
            )
            require(
                isinstance(value.usage, ResourceUsage)
                and value.usage.training_supervised_tokens
                == sum(x.supervised_tokens for x in request.data.examples[start:end]),
                "Checkpoint incremental exposure mismatch",
            )
            verified = verify_local_checkpoint(value.checkpoint)
            require(
                request.backend != "real" or verified,
                "Real continuous checkpoints must be local and byte-verified",
            )
            checkpoints.append(value)
            events.append(
                CostEvent(
                    "continuous-" + request.record_hash + "-" + str(stage),
                    request.attempt_id,
                    "training",
                    request.training_id + "-" + str(stage),
                    request.starting_model.model_hash,
                    "completed",
                    value.usage,
                    value.checkpoint,
                    request.backend,
                )
            )
            from self_curate.training.recovery import atomic_json, read_json

            stage_file = output / ("stage-" + str(stage) + ".json")
            if stage_file.exists():
                require(
                    resume and read_json(stage_file) == value.to_dict(),
                    "Saved stage evidence changed",
                )
            else:
                atomic_json(stage_file, value.to_dict())
            try:
                on_checkpoint(value)
            except BaseException:
                callback_failed = True
                raise

        try:
            kwargs = {"resume": True} if resume else {}
            self.backend.train(request, output / "checkpoints", emit=emit, **kwargs)
            require(len(checkpoints) == 3, "Continuous training omitted checkpoints")
            ledger = CostLedger(request.training_id, tuple(events), request.backend)
            result = {
                "request_hash": request.record_hash,
                "checkpoints": [v.to_dict() for v in checkpoints],
                "ledger": ledger.to_dict(),
                "optimizer_policy": "continuous_across_waves",
            }
            write_json(output / "complete.json", result)
            return tuple(checkpoints), ledger
        except Exception as exc:
            # A callback/validation failure must not erase completed training work.
            partial = getattr(exc, "usage", None)
            extra = getattr(exc, "ledger", None)
            if partial is None and extra is None and not callback_failed:
                partial = (
                    ResourceUsage()
                )  # Dispatched work with unknown exposure, not zero.
            if isinstance(extra, CostLedger):
                require(
                    extra.backend == request.backend,
                    "Failed operation changed execution domain",
                )
                known = {e.event_id for e in events}
                events.extend(e for e in extra.events if e.event_id not in known)
            if isinstance(partial, ResourceUsage):
                events.append(
                    CostEvent(
                        "continuous-failure-"
                        + request.record_hash
                        + "-"
                        + uuid.uuid4().hex,
                        request.attempt_id,
                        "training",
                        request.training_id,
                        request.starting_model.model_hash,
                        "failed",
                        partial,
                        ArtifactRef(
                            "record://continuous-failure/" + digest(str(exc)),
                            digest(str(exc)),
                        ),
                        request.backend,
                    )
                )
            ledger = CostLedger(request.training_id, tuple(events), request.backend)
            try:
                from .recovery import atomic_json, read_json

                failure_file = output / "failure.json"
                if failure_file.exists():
                    atomic_json(
                        output / "failures" / (uuid.uuid4().hex + ".json"),
                        read_json(failure_file),
                    )
                write_json(
                    failure_file,
                    {
                        "error": str(exc),
                        "ledger": ledger.to_dict(),
                        "completed_stages": len(checkpoints),
                    },
                )
            except OSError:
                pass
            raise TrainingFailure(str(exc), ledger=ledger) from exc


from .resumable import TorchContinuousBackend

__all__ = [
    "ContinuousTrainingService",
    "StageCheckpoint",
    "TorchContinuousBackend",
    "wave_boundaries",
]


def _frozen_hash(model):
    """Bounded host-memory check; never clone a 27B frozen base into a CPU dict."""
    import torch

    value = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad:
            continue
        value.update(
            str((name, str(tensor.dtype), tuple(tensor.shape))).encode("utf-8")
        )
        flat = tensor.detach().reshape(-1)
        chunk = max(1, 64 * 1024 * 1024 // tensor.element_size())
        for offset in range(0, flat.numel(), chunk):
            value.update(
                flat[offset : offset + chunk]
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    return value.hexdigest()
