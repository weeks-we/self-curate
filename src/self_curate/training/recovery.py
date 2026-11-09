"""Local atomic evidence, exclusive leases, and strict frozen-phase recovery."""

from contextlib import contextmanager
from pathlib import Path
import json
import os
import uuid
import time

from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".j" + uuid.uuid4().hex[:8])
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Antivirus/indexer handles can briefly prevent replacing a Windows file.
    # Retry only publication; never repeat model work or change the old bytes.
    for attempt in range(3):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checksums(root):
    root = Path(root)
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "COMMITTED.json"
    }


def commit_directory(root):
    atomic_json(Path(root) / "COMMITTED.json", {"checksums": checksums(root)})


def verify_directory(root):
    root = Path(root).resolve()
    saved = read_json(root / "COMMITTED.json")["checksums"]
    require(saved == checksums(root), "Recovery evidence file set or checksum changed")
    require(
        all(
            not p.is_symlink() and p.resolve().is_relative_to(root)
            for p in root.rglob("*")
        ),
        "Unsafe recovery evidence path",
    )


def publish_directory(source, target):
    """Bounded retries for Windows scanner handles, without repeating model work."""
    source, target = Path(source), Path(target)
    for attempt in range(3):
        require(
            source.is_dir() and not target.exists(),
            "Refuse to replace checkpoint evidence",
        )
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))


@contextmanager
def lease(path):
    """Kernel lock releases on crash; PID alone is never used to break a lease."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Another process owns this experiment/branch lease"
            ) from exc
        yield
    finally:
        stream.close()
