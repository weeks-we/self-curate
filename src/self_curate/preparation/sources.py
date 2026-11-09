"""Explicit server downloads, immutable source receipts, and local parquet loading."""

from pathlib import Path, PurePosixPath
import shutil, urllib.request, zipfile, stat
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import (
    atomic_json,
    read_json,
    commit_directory,
    verify_directory,
    lease,
)


def download_sources(registry, output, *, execution, allow_download=True):
    require(execution == "server", "Downloads require explicit server execution")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    with lease(root / ".download.lease"):
        for role in registry["required_sources"]:
            spec = registry["datasets"].get(role, registry.get(role))
            target = root / role
            if (target / "COMMITTED.json").exists():
                verify_directory(target)
                receipt = read_json(target / "source-receipt.json")
                require(
                    receipt["repo_id"] == spec["repo_id"]
                    and receipt["revision"] == spec["revision"],
                    "Existing download belongs to another source revision",
                )
                receipts[role] = receipt
                continue
            require(
                allow_download,
                f"Offline source is missing or incomplete: {role}; run the CPU assets stage first",
            )
            from huggingface_hub import snapshot_download, HfApi

            if target.exists() and any(target.iterdir()):
                import uuid

                preserved = root / "incomplete" / (role + "-" + uuid.uuid4().hex)
                preserved.parent.mkdir(exist_ok=True)
                require(
                    target.resolve().parent == root
                    and preserved.resolve().is_relative_to(root),
                    "Source preservation escaped storage root",
                )
                target.rename(preserved)
            target.mkdir(exist_ok=True)
            data = target / "files"
            require(
                not data.exists(),
                "Incomplete source publication; preserve and reconcile it instead of mixing versions",
            )
            if role == "terminal_bench":
                archive = root / ("terminal-" + spec["revision"] + ".zip")
                if not archive.exists():
                    request = urllib.request.Request(
                        "https://codeload.github.com/"
                        + spec["repo_id"]
                        + "/zip/"
                        + spec["revision"]
                    )
                    partial = archive.with_suffix(".zip.partial")
                    with (
                        urllib.request.urlopen(request, timeout=300) as response,
                        partial.open("wb") as stream,
                    ):
                        shutil.copyfileobj(response, stream)
                    with zipfile.ZipFile(partial) as downloaded:
                        require(
                            downloaded.testzip() is None,
                            "Downloaded archive CRC failed",
                        )
                    partial.replace(archive)
                data.mkdir()
                with zipfile.ZipFile(archive) as z:
                    require(z.testzip() is None, "Downloaded source archive CRC failed")
                    for info in z.infolist():
                        parts = PurePosixPath(info.filename).parts[1:]
                        require(
                            not PurePosixPath(info.filename).is_absolute()
                            and ".." not in parts
                            and not stat.S_ISLNK(info.external_attr >> 16),
                            "Unsafe source archive path",
                        )
                        if not parts or info.is_dir():
                            continue
                        path = data.joinpath(*parts)
                        require(
                            path.resolve().is_relative_to(data.resolve()),
                            "Archive escaped destination",
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as source, path.open("xb") as stream:
                            shutil.copyfileobj(source, stream)
            else:
                kind = "model" if role == "model" else "dataset"
                require(
                    HfApi()
                    .repo_info(
                        spec["repo_id"], repo_type=kind, revision=spec["revision"]
                    )
                    .sha
                    == spec["revision"],
                    "Hub revision did not resolve to the pinned commit",
                )
                snapshot = Path(
                    snapshot_download(
                        repo_id=spec["repo_id"],
                        repo_type=kind,
                        revision=spec["revision"],
                        cache_dir=root / "hub-cache",
                    )
                )
                shutil.copytree(snapshot, data, symlinks=False)
            files = {
                p.relative_to(data).as_posix(): file_sha256(p)
                for p in sorted(data.rglob("*"))
                if p.is_file()
            }
            require(bool(files), "Downloaded source has no files")
            receipt = {
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "root": str(data),
                "files": files,
            }
            if "split" in spec:
                receipt.update(config=spec["config"], split=spec["split"])
            atomic_json(target / "source-receipt.json", receipt)
            commit_directory(target)
            receipts[role] = receipt
        atomic_json(
            root / "sources.lock.json",
            {"schema_version": "1", "source_receipts": receipts},
        )
    return receipts


def read_dataset(receipt):
    """Read only locally downloaded split files; never execute a dataset script."""
    import re
    from .historical import load_historical

    root = Path(receipt["root"])
    split = receipt["split"]
    files = [
        root / name
        for name in receipt["files"]
        if name.endswith(".parquet")
        and re.search(r"(^|[/_-])" + re.escape(split) + r"([/_.-]|$)", name)
    ]
    require(bool(files), "No unambiguous local parquet files for the locked split")
    if receipt["repo_id"] == "SWE-bench/SWE-smith-trajectories" and split == "train":
        return load_historical(
            sorted(files), checksums={str(p): file_sha256(p) for p in files}
        )
    import pyarrow.parquet as pq

    return tuple(
        row for path in sorted(files) for row in pq.read_table(path).to_pylist()
    )
