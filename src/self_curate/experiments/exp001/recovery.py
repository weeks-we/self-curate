"""Strict frozen-phase experiment recovery."""

from pathlib import Path
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, read_json


def freeze_phase(output, value):
    """Publish only after one-shot initialization and all selection plans finish."""
    output = Path(output)
    paths = [
        "run.json",
        "source-checksums.json",
        "development-suite.json",
        "training-plans.json",
        "frozen-inputs.json",
    ]
    for name in ("handoff.md", "source-registry.json"):
        if (output / name).is_file():
            paths.append(name)
    for prefix in ("initial", "selections", "selection-roles"):
        paths.extend(
            p.relative_to(output).as_posix()
            for p in (output / prefix).rglob("*")
            if p.is_file()
        )
    value = {**value, "frozen_checksums": {n: file_sha256(output / n) for n in paths}}
    atomic_json(output / "resume-plan.json", value)
    atomic_json(output / "resume-plan.sha256.json", {"sha256": digest(value)})


def load_phase(output):
    output = Path(output).resolve()
    value = read_json(output / "resume-plan.json")
    require(
        digest(value) == read_json(output / "resume-plan.sha256.json")["sha256"],
        "Frozen resume plan changed",
    )
    for name, checksum in value["frozen_checksums"].items():
        path = output / name
        require(
            path.resolve().is_relative_to(output)
            and not path.is_symlink()
            and file_sha256(path) == checksum,
            "Frozen experiment input changed",
        )
    return value
