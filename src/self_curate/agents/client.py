"""Stateless role-generation interface for one shared, frozen parameter identity."""

from typing import Protocol
from self_curate.contracts import ModelRef


class SharedModelClient(Protocol):
    """All roles use this same instance and ModelRef, with complete per-call input.

    Implementations must honor timeout_seconds and must not reuse a global chat
    transcript across roles. Remote implementations must verify loaded weights.
    """

    model_ref: ModelRef

    def generate_json(
        self, *, role: str, system_prompt: str, payload: dict, timeout_seconds: float
    ) -> dict: ...
