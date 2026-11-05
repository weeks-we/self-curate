"""Byte-level local checkpoint checks; remote object verification is adapter-owned."""

from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from self_curate.contracts import ArtifactRef, digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256


def checkpoint_fingerprint(root):
    root = Path(root)
    require(not root.is_symlink(), "Checkpoint root cannot be a symlink")
    root = root.resolve()
    require(root.is_dir(), "Checkpoint directory is missing")
    files = {}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(), "Checkpoint cannot contain symlinks")
        if p.is_file():
            files[p.relative_to(root).as_posix()] = file_sha256(p)
    require(bool(files), "Checkpoint contains no saved files")
    return digest(files), files


def verify_local_checkpoint(reference):
    require(isinstance(reference, ArtifactRef), "Expected checkpoint reference")
    uri = urlsplit(reference.uri)
    if uri.scheme.lower() != "file":
        return False  # Never claim verification of remote or synthetic objects.
    require(
        uri.netloc in ("", "localhost") and not uri.query and not uri.fragment,
        "Checkpoint file URI must refer to a local artifact",
    )
    path = Path(url2pathname(uri.path))
    require(
        path.is_dir() and checkpoint_fingerprint(path)[0] == reference.sha256,
        "Local candidate checkpoint changed or is incomplete",
    )
    return True
