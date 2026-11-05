"""SFT interfaces and exact-batch materialization; torch is an optional backend."""

from .contracts import (
    BackendTrainingResult,
    CurationDataProvider,
    SFTRecipe,
    TrainingBackend,
    TrainingData,
    TrainingExample,
    TrainingOutcome,
    TrainingRequest,
)
from .data import prepare_subset
from .service import TrainingFailure, TrainingService

__all__ = [
    "BackendTrainingResult",
    "CurationDataProvider",
    "SFTRecipe",
    "TrainingBackend",
    "TrainingData",
    "TrainingExample",
    "TrainingOutcome",
    "TrainingRequest",
    "prepare_subset",
    "TrainingFailure",
    "TrainingService",
]
