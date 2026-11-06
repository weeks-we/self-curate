"""Switchable Leval kernels; real model/task loading remains injectable."""

from self_curate.contracts.types import require
from .contracts import NICERecipe, NICEReference, NICEScores, ReplayTrajectory
from .nice import NICEValidator
from .sft import SFTValidator
from .common import ValuationFailure


def create_validator(method, **settings):
    """Choose once before building ValidationRequest/RoleTeam/InnerSearch."""
    require(method in ("nice", "sft"), "Choose nice or sft explicitly")
    return {"nice": NICEValidator, "sft": SFTValidator}[method](**settings)


__all__ = [
    "NICERecipe",
    "NICEReference",
    "NICEScores",
    "ReplayTrajectory",
    "NICEValidator",
    "SFTValidator",
    "ValuationFailure",
    "create_validator",
]
