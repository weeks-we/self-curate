"""Structured errors shared with future operators and model backends."""

from enum import Enum


class ErrorCode(str, Enum):
    INVALID_RECORD = "INVALID_RECORD"
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_PROGRAM = "INVALID_PROGRAM"
    INSUFFICIENT_POOL = "INSUFFICIENT_POOL"
    MISSING_SIGNAL = "MISSING_SIGNAL"
    SIGNAL_CONFLICT = "SIGNAL_CONFLICT"
    GRADIENT_CONFLICT = "GRADIENT_CONFLICT"
    USAGE_CONFLICT = "USAGE_CONFLICT"
    UNKNOWN_DATA_ID = "UNKNOWN_DATA_ID"
    STALE_CACHE = "STALE_CACHE"
    ZERO_TARGET_GRADIENT = "ZERO_TARGET_GRADIENT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    INVALID_REWARD = "INVALID_REWARD"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    TOKENIZATION_MISMATCH = "TOKENIZATION_MISMATCH"
    ZERO_SUPERVISION = "ZERO_SUPERVISION"
    DUPLICATE_ID = "DUPLICATE_ID"
    ARTIFACT_MISMATCH = "ARTIFACT_MISMATCH"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"


class ContractError(ValueError):
    def __init__(self, code: ErrorCode, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self):
        return {"code": self.code.value, "message": str(self), "details": self.details}
