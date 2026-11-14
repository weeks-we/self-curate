"""Validate shared public source pins and server-provided preparation receipts.

No network or dataset/model download is performed here. Public revision validity
is audited in docs; local bytes still need independent server verification.
"""

from pathlib import Path
import re
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from .recovery import read_json


def load_source_registry(path, *, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        require(
            type(expected_sha256) is str
            and re.fullmatch("[0-9a-f]{64}", expected_sha256),
            "Source registry needs a complete SHA256 pin",
        )
        require(file_sha256(path) == expected_sha256, "Source registry file changed")
    value = read_json(path)
    require(value["schema_version"] == "1", "Unknown source registry version")
    for row in [
        value["model"],
        *value["datasets"].values(),
        *([value["author_ids"]] if "author_ids" in value else []),
        value["terminal_bench"],
    ]:
        require(
            re.fullmatch("[0-9a-f]{40}", row["revision"]) is not None,
            "Source revisions must be immutable full commits",
        )
    require(
        value["model"]["repo_id"] == "Qwen/Qwen3.8-27B"
        and value["model"]["tokenizer_revision"] == value["model"]["revision"],
        "EXP001 model/tokenizer identity changed",
    )
    for name, row in value["datasets"].items():
        require(
            row["url"].startswith(
                "https://huggingface.co/datasets/" + row["repo_id"] + "/"
            )
            and row["revision"] in row["url"],
            "Source URL and dataset revision disagree",
        )
        require(
            not name.startswith("benchmark_") or row["split"] == "test",
            "Final benchmark must retain test split",
        )
    mode = value.get("training_selection", {}).get("mode", "historical_exact")
    if mode == "rebench_success":
        require(value["datasets"]["rebench_trajectories"]["repo_id"] == "nebius/SWE-rebench-openhands-trajectories"
                and value["datasets"]["rebench_trajectories"]["split"] == "train"
                and value["datasets"]["development_tasks"]["repo_id"] == "nebius/SWE-rebench",
                "Rebench V1 task/trajectory source mismatch")
        require(isinstance(value.get("skills"), list) and bool(value["skills"])
                and len(set(value["skills"])) == len(value["skills"]), "Frozen external skill ontology required")
        require(value["terminal_bench"].get("release") == "4.1", "Expected reviewed Terminal 4.1 release")
    else:
        require(mode == "historical_exact", "Unsupported training source mode")
        require(value["datasets"]["tool_trajectories"]["split"] == "tool"
                and value["datasets"]["author_history"]["split"] == "train",
                "Do not interchange historical selected rows and tool trajectories")
        require(value["terminal_bench"]["dataset"] == "terminal-bench/terminal-bench@4.0.0",
                "Legacy terminal benchmark release changed")
    return value


def validate_source_receipts(registry, lock):
    """Require explicit server receipts; a reachable URL is not a prepared asset."""
    receipts = lock.get("source_receipts")
    require(
        type(receipts) is dict
        and set(
            registry.get(
                "required_sources",
                [*registry["datasets"], "model", "author_ids", "terminal_bench"],
            )
        )
        <= set(receipts)
        and set(receipts)
        <= set(registry["datasets"]) | {"model", "author_ids", "terminal_bench"},
        "Prepared source lock needs receipts for all pinned sources",
    )
    for name, receipt in receipts.items():
        expected = (
            registry["datasets"][name]
            if name in registry["datasets"]
            else registry[name]
        )
        require(
            receipt.get("revision") == expected["revision"]
            and receipt.get("repo_id") == expected["repo_id"],
            "Prepared source receipt changed repository/revision",
        )
        files = receipt.get("files")
        require(
            type(files) is dict
            and bool(files)
            and all(
                type(n) is str
                and n
                and type(h) is str
                and re.fullmatch("[0-9a-f]{64}", h)
                for n, h in files.items()
            ),
            "Prepared source receipt needs file checksums",
        )
        if name in registry["datasets"]:
            require(
                receipt.get("split") == expected["split"]
                and receipt.get("config") == expected["config"],
                "Prepared dataset receipt changed split/config",
            )
        root = Path(receipt.get("root", ""))
        require(
            root.is_absolute() and root.is_dir(),
            "Source receipt needs an existing absolute cache root",
        )
        for name, checksum in files.items():
            path = root / name
            require(
                not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and path.resolve().is_relative_to(root.resolve())
                and not path.is_symlink()
                and path.is_file()
                and file_sha256(path) == checksum,
                "Downloaded source bytes differ from the pinned preparation receipt",
            )
    return True
