"""Versioned scientific choices, separate from server paths and adapter settings."""

from dataclasses import dataclass
from pathlib import Path

from self_curate.contracts import canonical_json, digest
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import integer, require, text
from self_curate.contracts.workflow import finite
from self_curate.training import SFTRecipe


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    backend: str
    seed: int
    selection_fraction: float
    learning_rates: tuple[float, ...]
    modes: tuple[str, ...]
    effective_batch: int
    max_context_tokens: int
    lexec_use_model: bool
    lopt_timeout_seconds: float
    adapter: str
    adapter_settings_json: str
    execution: str
    gpu_uuids: tuple[str, ...]
    schema_version: str = "1"

    def __post_init__(self):
        require(
            self.schema_version == "1" and self.backend in ("mock", "real"),
            "Invalid experiment schema/domain",
        )
        text(self.experiment_id, "experiment ID")
        integer(self.seed, "experiment seed")
        finite(self.selection_fraction, "selection fraction", 0)
        require(0 < self.selection_fraction <= 1, "Selection fraction must be in (0,1]")
        require(
            type(self.learning_rates) is tuple
            and len(self.learning_rates) == 2
            and len(set(self.learning_rates)) == 2,
            "Exactly two distinct learning rates",
        )
        for rate in self.learning_rates:
            finite(rate, "learning rate", 0)
            require(rate > 0, "Positive learning rate required")
        require(
            type(self.modes) is tuple
            and bool(self.modes)
            and len(set(self.modes)) == len(self.modes)
            and all(m in ("random", "static_m0") for m in self.modes),
            "Choose Random/static_m0 modes explicitly",
        )
        integer(self.effective_batch, "effective batch", 1)
        integer(self.max_context_tokens, "maximum complete context", 2)
        require(type(self.lexec_use_model) is bool, "Explicit Lexec mode required")
        finite(self.lopt_timeout_seconds, "Lopt timeout", 0)
        require(self.lopt_timeout_seconds > 0, "Positive timeout required")
        text(self.adapter, "adapter factory")
        require(self.adapter.count(":") == 1, "Adapter format is module:factory")
        require(
            type(load_json(self.adapter_settings_json)) is dict,
            "Adapter settings must be JSON object",
        )
        require(
            type(self.gpu_uuids) is tuple
            and len(set(self.gpu_uuids)) == len(self.gpu_uuids),
            "Unique immutable GPU assignment required",
        )
        require(
            (
                self.backend == "mock"
                and self.execution == "local_mock"
                and not self.gpu_uuids
            )
            or (
                self.backend == "real"
                and self.execution == "server"
                and len(self.gpu_uuids) == 8
                and all(type(v) is str and v.startswith("GPU-") for v in self.gpu_uuids)
            ),
            "Real execution requires explicit server and eight authorized GPU UUIDs",
        )

    @property
    def config_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "backend": self.backend,
            "seed": self.seed,
            "selection_fraction": self.selection_fraction,
            "learning_rates": list(self.learning_rates),
            "modes": list(self.modes),
            "effective_batch": self.effective_batch,
            "max_context_tokens": self.max_context_tokens,
            "lexec_use_model": self.lexec_use_model,
            "lopt_timeout_seconds": self.lopt_timeout_seconds,
            "adapter": self.adapter,
            "adapter_settings": load_json(self.adapter_settings_json),
            "execution": self.execution,
            "gpu_uuids": list(self.gpu_uuids),
        }

    def recipe(self, rate):
        require(rate in self.learning_rates, "Unregistered learning rate")
        return SFTRecipe(
            rate,
            1,
            self.effective_batch,
            self.max_context_tokens,
            self.seed,
            optimizer="adamw",
            weight_decay=0.0,
            max_grad_norm=1.0,
        )


def load_config(path):
    raw = fields(
        load_json(Path(path).read_text(encoding="utf-8")),
        (
            "schema_version",
            "experiment_id",
            "backend",
            "seed",
            "selection_fraction",
            "learning_rates",
            "modes",
            "effective_batch",
            "max_context_tokens",
            "lexec_use_model",
            "lopt_timeout_seconds",
            "adapter",
            "adapter_settings",
            "execution",
            "gpu_uuids",
        ),
    )
    require(
        type(raw["learning_rates"]) is list
        and type(raw["gpu_uuids"]) is list
        and type(raw["modes"]) is list,
        "Expected JSON arrays",
    )
    raw["modes"] = tuple(raw["modes"])
    raw["learning_rates"], raw["gpu_uuids"] = tuple(raw["learning_rates"]), tuple(
        raw["gpu_uuids"]
    )
    raw["adapter_settings_json"] = canonical_json(raw.pop("adapter_settings"))
    return ExperimentConfig(**raw)
