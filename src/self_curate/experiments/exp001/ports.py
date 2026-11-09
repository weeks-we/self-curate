"""Server bindings are implemented once and shared by later experiments."""

from pathlib import Path
from typing import Protocol

from self_curate.contracts import ModelRef
from self_curate.agents.prompts import PromptTemplate, default_prompts
from self_curate.contracts.parsing import load_json
from .inputs import PreparedInputs


class ExperimentPorts(Protocol):
    backend: str
    scorer: object  # Existing SignalService protocol: spec/model_ref/score/output.
    validator: object  # Existing TaskValidator protocol; returns TaskEvaluation.
    benchmark_runner: object  # Thread-safe TaskRunner; every request pins model.
    benchmark_workers: int
    client: object  # Existing SharedModelClient pinned to M0.
    prompts: dict
    objective: object
    edit_policy: object
    agent_sampling: object
    agent_implementation_hash: str
    task_execution_hash: str

    def preflight(self, config) -> dict: ...
    def prepare(self, config, output: Path) -> PreparedInputs: ...
    def context(self, records, config): ...
    def training_backend(self, branch): ...
    def attach_training(self, config, model: ModelRef) -> None: ...
    def release_initial(self) -> None: ...


def load_experiment_prompts(directory):
    prompts = default_prompts()
    for role in ("lopt", "lexec"):
        row = load_json(
            (Path(directory) / (role + ".json")).read_text(encoding="utf-8")
        )
        prompt = PromptTemplate(**row)
        from self_curate.contracts.types import require

        require(prompt.role == role, "EXP001 prompt role mismatch")
        prompts[role] = prompt
    return prompts
