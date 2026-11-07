"""Immutable identities for shared experiment outputs; paths are not identities."""

from dataclasses import dataclass
import re

from self_curate.contracts import ArtifactRef, canonical_json, digest
from self_curate.contracts.parsing import artifact, fields, load_json
from self_curate.contracts.types import require, sha256, text


@dataclass(frozen=True)
class AssetSpec:
    kind: str
    backend: str
    identity_json: str
    dependencies: tuple[ArtifactRef, ...] = ()
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported asset identity")
        require(
            type(self.kind) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,47}", self.kind),
            "Invalid asset kind",
        )
        require(self.backend in ("mock", "real"), "Explicit asset domain required")
        require(
            type(self.identity_json) is str, "Asset identity must be immutable JSON"
        )
        value = load_json(self.identity_json)
        require(type(value) is dict and bool(value), "Nonempty asset inputs required")
        require(
            canonical_json(value) == self.identity_json, "Noncanonical asset identity"
        )
        require(type(self.dependencies) is tuple, "Immutable dependencies required")
        for ref in self.dependencies:
            require(
                isinstance(ref, ArtifactRef)
                and re.fullmatch(r"asset://[0-9a-f]{64}/manifest.json", ref.uri),
                "Dependencies must pin shared asset manifests",
            )
        require(
            len({r.uri for r in self.dependencies}) == len(self.dependencies),
            "Duplicate dependency",
        )
        require(
            tuple(sorted(self.dependencies, key=lambda r: r.uri)) == self.dependencies,
            "Dependencies must be ordered",
        )

    @classmethod
    def create(cls, kind, backend, identity, *, dependencies=()):
        return cls(
            kind,
            backend,
            canonical_json(identity),
            tuple(sorted(dependencies, key=lambda r: r.uri)),
        )

    @property
    def identity(self):
        return load_json(self.identity_json)

    @property
    def key_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "backend": self.backend,
            "identity": self.identity,
            "dependencies": [r.to_dict() for r in self.dependencies],
        }


@dataclass(frozen=True)
class Producer:
    run_id: str
    source_tree_hash: str
    environment_hash: str
    config_hash: str
    code_commit: str | None = None
    schema_version: str = "1"

    def __post_init__(self):
        require(self.schema_version == "1", "Unsupported producer stamp")
        text(self.run_id, "producer run ID")
        for name in ("source_tree_hash", "environment_hash", "config_hash"):
            sha256(getattr(self, name), name)
        require(
            self.code_commit is None
            or (
                type(self.code_commit) is str
                and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.code_commit)
            ),
            "Producer commit must be resolved; uncommitted runs use source_tree_hash and null commit",
        )

    def to_dict(self):
        return dict(vars(self))


def parse_spec(value):
    row = fields(
        value, ("schema_version", "kind", "backend", "identity", "dependencies")
    )
    require(type(row["dependencies"]) is list, "Dependencies must be an array")
    return AssetSpec(
        row["kind"],
        row["backend"],
        canonical_json(row["identity"]),
        tuple(artifact(r) for r in row["dependencies"]),
        row["schema_version"],
    )


def parse_producer(value):
    return Producer(
        **fields(
            value,
            (
                "schema_version",
                "run_id",
                "source_tree_hash",
                "environment_hash",
                "config_hash",
                "code_commit",
            ),
        )
    )
