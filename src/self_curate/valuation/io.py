"""Strict JSON decoding and the single method switch; no dynamic code loading."""

from pathlib import Path

from self_curate.contracts import ContractError, ErrorCode, TokenizerRef
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import require
from self_curate.contracts.workflow_parsing import CONTRACT_TYPES, _decode
from self_curate.orchestration.io import OUTER_TYPES
from .contracts import NICERecipe, NICEReference, NICEScores, ReplayTrajectory

VALUATION_TYPES = {
    cls.__name__: cls
    for cls in (NICERecipe, NICEReference, NICEScores, ReplayTrajectory)
}


def parse_valuation_contract(kind, value):
    require(type(kind) is str and kind in VALUATION_TYPES, "Unknown valuation contract")
    try:
        return _decode(
            VALUATION_TYPES[kind],
            value,
            kind,
            registry={
                **CONTRACT_TYPES,
                **OUTER_TYPES,
                **VALUATION_TYPES,
                "TokenizerRef": TokenizerRef,
            },
        )
    except ContractError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ContractError(ErrorCode.INVALID_RECORD, "Invalid valuation JSON") from exc


def load_valuation_contract(kind, path):
    return parse_valuation_contract(
        kind, load_json(Path(path).read_text(encoding="utf-8"))
    )


def load_method(path):
    raw = fields(
        load_json(Path(path).read_text(encoding="utf-8")), ("schema_version", "method")
    )
    require(
        raw["schema_version"] == "1" and raw["method"] in ("nice", "sft"),
        "Choose nice or sft",
    )
    return raw["method"]
