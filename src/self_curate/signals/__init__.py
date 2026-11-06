"""Forward-only NLL/entropy with exact model/data/scope versions and M1 labels."""

from .contracts import (
    ForwardRequest,
    LogitsOutput,
    LogitsProvider,
    ScoringDataProvider,
    TokenMetricBackend,
    TokenMetricResult,
    SignalOutcome,
    ScoringBatch,
)
from .backends import ProvidedLogitsBackend, ScoringFailure
from .service import SignalService
from .io import load_scoring_batch, parse_forward_contract, read_records

__all__ = [
    "ForwardRequest",
    "LogitsOutput",
    "LogitsProvider",
    "ScoringDataProvider",
    "TokenMetricBackend",
    "TokenMetricResult",
    "SignalOutcome",
    "ScoringBatch",
    "ProvidedLogitsBackend",
    "ScoringFailure",
    "SignalService",
    "load_scoring_batch",
    "parse_forward_contract",
    "read_records",
]
