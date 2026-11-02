"""Lossless, explicit usage transitions; recording never starts computation."""

from dataclasses import replace

from self_curate.contracts import ErrorCode, TrainingUse, UsageHistory
from self_curate.contracts.types import require


def attach_usage(record, history: UsageHistory):
    require(isinstance(history, UsageHistory), "Expected UsageHistory")
    old = record.usage
    require(
        set(old.forward_scoring).issubset(history.forward_scoring)
        and set(old.gradient_extraction).issubset(history.gradient_extraction)
        and set(old.accepted_sft).issubset(history.accepted_sft),
        "Usage evidence cannot be erased",
        ErrorCode.USAGE_CONFLICT,
    )
    incoming = {v.attempt_id: v for v in history.candidate_sft}
    for previous in old.candidate_sft:
        current = incoming.get(previous.attempt_id)
        require(
            current is not None
            and (previous.model_id, previous.receipt)
            == (current.model_id, current.receipt),
            "Training attempt identity cannot change",
            ErrorCode.USAGE_CONFLICT,
        )
        require(
            previous.decision == "pending" or previous.decision == current.decision,
            "Terminal training decision cannot change",
            ErrorCode.USAGE_CONFLICT,
        )
    return replace(record, usage=history, normalization_version="m1-v3")


def record_candidate_training(record, use: TrainingUse):
    require(isinstance(use, TrainingUse), "Expected TrainingUse")
    attempts = {v.attempt_id: v for v in record.usage.candidate_sft}
    attempts[use.attempt_id] = use
    history = replace(
        record.usage,
        candidate_sft=tuple(attempts[k] for k in sorted(attempts)),
        accepted_sft=tuple(
            sorted(k for k, v in attempts.items() if v.decision == "accepted")
        ),
    )
    return attach_usage(record, history)


def resolve_training(record, attempt_id: str, decision: str):
    require(
        decision in ("accepted", "rejected"),
        "Training resolution must accept or reject",
    )
    found = next(
        (u for u in record.usage.candidate_sft if u.attempt_id == attempt_id), None
    )
    require(
        found is not None,
        "Cannot resolve an unrecorded training attempt",
        ErrorCode.USAGE_CONFLICT,
    )
    return record_candidate_training(record, replace(found, decision=decision))
