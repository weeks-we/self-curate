"""Checksummed immutable shared assets and separate consumer reuse receipts."""

import base64
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
import os
import shutil
from time import perf_counter

from self_curate.contracts import (
    ArtifactRef,
    ContractError,
    CostLedger,
    ErrorCode,
    canonical_json,
    parse_contract,
)
from self_curate.contracts.parsing import fields, load_json
from self_curate.contracts.types import integer, require, sha256
from self_curate.data.pool import _publish, _staging_directory, file_sha256, write_json
from .contracts import AssetSpec, Producer, parse_producer, parse_spec


def structured(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ContractError:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            IndexError,
            RecursionError,
            OverflowError,
        ) as exc:
            raise ContractError(
                ErrorCode.ARTIFACT_MISMATCH,
                "Invalid shared asset or I/O failure",
                reason=type(exc).__name__,
            ) from exc

    return call


def plain(path):
    require(
        not path.is_symlink()
        and not (getattr(path.lstat(), "st_file_attributes", 0) & 0x400),
        "Asset cannot contain links/reparse points",
    )


def relative_name(name):
    require(
        type(name) is str and bool(name) and "\\" not in name and ":" not in name,
        "Invalid asset payload path",
    )
    p = PurePosixPath(name)
    require(
        not p.is_absolute() and all(s not in (".", "..", "") for s in name.split("/")),
        "Unsafe asset payload path",
    )
    return name


def inventory(root):
    plain(root)
    require(root.is_dir(), "Asset payload must be a directory")
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            plain(Path(directory) / name)
        for name in files:
            p = Path(directory) / name
            require(
                p.is_file() and p.resolve().is_relative_to(root.resolve()),
                "Invalid payload file",
            )
            relative = relative_name(p.relative_to(root).as_posix())
            result[relative] = {"sha256": file_sha256(p), "size": p.stat().st_size}
    require(bool(result), "Empty assets cannot be published")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class StoredAsset:
    path: Path
    manifest_json: str
    manifest_hash: str

    @property
    def manifest(self):
        return load_json(self.manifest_json)

    @property
    def spec(self):
        return parse_spec(self.manifest["spec"])

    @property
    def payload(self):
        return self.path / "p"

    @property
    def reference(self):
        return ArtifactRef(
            "asset://" + self.spec.key_hash + "/manifest.json", self.manifest_hash
        )

    @property
    def origin_ledger(self):
        value = self.manifest["origin_ledger"]
        return None if value is None else parse_contract("CostLedger", value)

    def file(self, name):
        name = relative_name(name)
        entry = self.manifest["files"].get(name)
        require(entry is not None, "File is not in this asset")
        p = self.payload / name
        plain(p)
        require(
            p.resolve().is_relative_to(self.payload.resolve())
            and file_sha256(p) == entry["sha256"],
            "Asset file changed",
        )
        return p

    def resolve_evidence(self, reference):
        """Resolve a saved proof by hash inside this asset, even after moving storage."""
        require(isinstance(reference, ArtifactRef), "Expected evidence reference")
        names = sorted(
            n
            for n, entry in self.manifest["files"].items()
            if entry["sha256"] == reference.sha256
        )
        require(
            bool(names),
            "Evidence is external to this asset; supply its pinned dependency",
        )
        return self.file(names[0])


class AssetStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path_for(self, key):
        sha256(key, "asset key")
        # Full 256-bit identity in a shorter, portable directory name.
        return self.root / base64.urlsafe_b64encode(bytes.fromhex(key)).decode(
            "ascii"
        ).rstrip("=")

    @structured
    def get(self, expected: AssetSpec, *, reference=None):
        require(isinstance(expected, AssetSpec), "Expected exact AssetSpec")
        require(
            reference is None
            or (
                isinstance(reference, ArtifactRef)
                and reference.uri == "asset://" + expected.key_hash + "/manifest.json"
            ),
            "Pinned asset reference uses another input identity",
            ErrorCode.STALE_CACHE,
        )
        path = self.path_for(expected.key_hash)
        if not path.exists():
            require(
                reference is None,
                "Pinned shared asset is missing",
                ErrorCode.ARTIFACT_MISMATCH,
            )
            return None
        value = self.inspect(expected.key_hash, expected=expected)
        require(
            reference is None or value.reference == reference,
            "Pinned asset manifest changed",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        return value

    @structured
    def inspect(self, key, *, expected=None, _seen=()):
        require(key not in _seen, "Cyclic asset dependencies")
        path = self.path_for(key)
        plain(path)
        require(
            path.is_dir()
            and {p.name for p in path.iterdir()}
            == {"manifest.json", "COMPLETE.json", "p"},
            "Incomplete or unexpected asset contents",
        )
        for name in ("manifest.json", "COMPLETE.json", "p"):
            plain(path / name)
        complete = fields(
            load_json((path / "COMPLETE.json").read_text(encoding="utf-8")),
            ("schema_version", "key_hash", "manifest_hash"),
        )
        manifest_hash = file_sha256(path / "manifest.json")
        require(
            complete["schema_version"] == "1"
            and complete["key_hash"] == key
            and complete["manifest_hash"] == manifest_hash,
            "Asset publication marker mismatch",
        )
        row = fields(
            load_json((path / "manifest.json").read_text(encoding="utf-8")),
            (
                "schema_version",
                "key_hash",
                "spec",
                "producer",
                "files",
                "origin_ledger",
            ),
        )
        spec = parse_spec(row["spec"])
        require(
            row["schema_version"] == "1"
            and row["key_hash"] == key == spec.key_hash
            and (expected is None or spec == expected),
            "Asset input identity mismatch",
        )
        parse_producer(row["producer"])
        require(
            type(row["files"]) is dict and bool(row["files"]), "Asset files missing"
        )
        for name, entry in row["files"].items():
            relative_name(name)
            fields(entry, ("sha256", "size"))
            sha256(entry["sha256"], "payload checksum")
            integer(entry["size"], "payload size")
        require(
            inventory(path / "p") == row["files"],
            "Asset payload checksum or file set changed",
        )
        if row["origin_ledger"] is not None:
            require(
                parse_contract("CostLedger", row["origin_ledger"]).backend
                == spec.backend,
                "Mixed asset cost domain",
            )
        for ref in spec.dependencies:
            parent_key = ref.uri.split("/")[2]
            parent = self.inspect(parent_key, _seen=(*_seen, key))
            require(
                parent.manifest_hash == ref.sha256
                and parent.spec.backend == spec.backend,
                "Dependency changed or domain mismatch",
            )
        return StoredAsset(path, canonical_json(row), manifest_hash)

    @structured
    def publish(self, spec, source, *, producer, origin_ledger=None, validator=None):
        require(
            isinstance(spec, AssetSpec) and isinstance(producer, Producer),
            "Typed asset inputs/producer required",
        )
        require(
            origin_ledger is None
            or (
                isinstance(origin_ledger, CostLedger)
                and origin_ledger.backend == spec.backend
            ),
            "Invalid original cost ledger",
        )
        source = Path(source).absolute()
        plain(source)
        source = source.resolve()
        require(
            not self.root.is_relative_to(source)
            and not source.is_relative_to(self.root),
            "Source and shared store must be separate trees",
        )
        before = inventory(source)
        original_cost = None if origin_ledger is None else origin_ledger.to_dict()
        existing = self.get(spec)
        if existing is not None:
            require(
                existing.manifest["files"] == before
                and existing.manifest["producer"] == producer.to_dict()
                and existing.manifest["origin_ledger"] == original_cost,
                "Asset identity already has different output or producer; keep original evidence",
                ErrorCode.OUTPUT_EXISTS,
            )
            if validator is not None:
                validator(existing.payload)
            return existing
        for ref in spec.dependencies:
            dependency = self.inspect(ref.uri.split("/")[2])
            require(
                dependency.reference == ref and dependency.spec.backend == spec.backend,
                "Foreign or incompatible asset dependency",
            )
        self.root.mkdir(parents=True, exist_ok=True)
        stage = _staging_directory(self.path_for(spec.key_hash))
        shutil.copytree(source, stage / "p")
        require(
            inventory(stage / "p") == before == inventory(source),
            "Source changed during asset snapshot",
        )
        if validator is not None:
            validator(stage / "p")
        write_json(
            stage / "manifest.json",
            {
                "schema_version": "1",
                "key_hash": spec.key_hash,
                "spec": spec.to_dict(),
                "producer": producer.to_dict(),
                "files": before,
                "origin_ledger": original_cost,
            },
        )
        write_json(
            stage / "COMPLETE.json",
            {
                "schema_version": "1",
                "key_hash": spec.key_hash,
                "manifest_hash": file_sha256(stage / "manifest.json"),
            },
        )
        _publish(stage, self.path_for(spec.key_hash))
        return self.inspect(spec.key_hash, expected=spec)

    @structured
    def reuse(self, spec, *, consumer_run, receipt_path=None, reference=None):
        from self_curate.contracts.types import text

        text(consumer_run, "consumer run ID")
        started = perf_counter()
        asset = self.get(spec, reference=reference)
        if asset is None:
            return None
        return asset, self._receipt(
            asset, consumer_run=consumer_run, receipt_path=receipt_path, started=started
        )

    def _receipt(self, asset, *, consumer_run, receipt_path, started):
        from self_curate.contracts.types import text

        text(consumer_run, "consumer run ID")
        require(
            asset.path == self.path_for(asset.spec.key_hash)
            and file_sha256(asset.path / "manifest.json") == asset.manifest_hash,
            "Validated asset manifest changed before consumption",
        )
        receipt = {
            "schema_version": "1",
            "consumer_run": consumer_run,
            "asset": asset.reference.to_dict(),
            "producer_run": asset.manifest["producer"]["run_id"],
            "new_model_calls": 0,
            "new_training_supervised_tokens": 0,
            "io_wall_seconds": perf_counter() - started,
            "origin_cost_recharged": False,
        }
        if receipt_path is not None:
            path = Path(receipt_path).resolve()
            require(
                not path.is_relative_to(self.root),
                "Consumer receipt must not modify the shared store",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(canonical_json(receipt) + "\n")
        return receipt
