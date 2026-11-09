"""Checked local task inputs and publication, shared by preparation and runners."""

from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from self_curate.contracts import ArtifactRef
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import read_json, atomic_json
from self_curate.training.artifacts import checkpoint_fingerprint


def local_path(ref, *, directory=False):
    parsed = urlsplit(ref.uri)
    require(
        parsed.scheme == "file" and not parsed.netloc,
        "Execution needs local pinned artifacts",
    )
    raw = Path(url2pathname(parsed.path))
    require(not raw.is_symlink(), "Artifact cannot be a symlink")
    path = raw.resolve()
    require(not path.is_symlink(), "Artifact cannot be a symlink")
    actual = checkpoint_fingerprint(path)[0] if directory else file_sha256(path)
    require(actual == ref.sha256, "Task/model artifact bytes changed")
    return path


def read_artifact(ref):
    return read_json(local_path(ref))


def publish_json(path, value):
    atomic_json(path, value)
    path = Path(path).resolve()
    return ArtifactRef(path.as_uri(), file_sha256(path))
