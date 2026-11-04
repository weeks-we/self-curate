"""Config validation, immutable pool publication and checked downstream reads."""

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import platform
import time
import uuid

from self_curate.contracts import (
    ContractError,
    ErrorCode,
    TokenizerRef,
    TrajectoryRecord,
    canonical_json,
    digest,
)
from self_curate.contracts.parsing import (
    fields,
    load_json,
    sequence,
    tokenizer_ref,
    tokenization,
)
from self_curate.contracts.types import Record, integer, require, text
from .normalize import normalize_trajectory
from .tokenization import FixtureTokenizer, PrecomputedTokenizer


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


@dataclass(frozen=True)
class ImportConfig(Record):
    input_path: Path
    backend: str
    skills: tuple[str, ...]
    max_context_tokens: int
    oversize_policy: str
    rejected_history_policy: str
    accepted_history_policy: str = "exclude"
    tokenizer: TokenizerRef | None = None
    schema_version: str = "1"

    def __post_init__(self):
        require(
            isinstance(self.input_path, Path),
            "input_path must be a Path",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            isinstance(self.backend, str)
            and self.backend in {"fixture", "precomputed"},
            "Unsupported backend",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            type(self.skills) is tuple and bool(self.skills),
            "Declare a nonempty skills tuple",
            ErrorCode.INVALID_CONFIG,
        )
        for skill in self.skills:
            text(skill, "skill")
        require(
            len(set(self.skills)) == len(self.skills),
            "Duplicate configured skill",
            ErrorCode.INVALID_CONFIG,
        )
        integer(self.max_context_tokens, "max_context_tokens", minimum=1)
        require(
            isinstance(self.oversize_policy, str)
            and self.oversize_policy in {"reject", "error"},
            "Invalid oversize_policy",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            isinstance(self.accepted_history_policy, str)
            and self.accepted_history_policy in {"exclude", "allow"},
            "Invalid accepted_history_policy",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            isinstance(self.rejected_history_policy, str)
            and self.rejected_history_policy in {"exclude", "allow"},
            "Invalid rejected_history_policy",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            (self.tokenizer is not None) == (self.backend == "precomputed"),
            "Only precomputed backend requires tokenizer identity",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            self.tokenizer is None or isinstance(self.tokenizer, TokenizerRef),
            "Invalid tokenizer identity",
            ErrorCode.INVALID_CONFIG,
        )
        require(
            self.schema_version == "1",
            "Unsupported config schema",
            ErrorCode.INVALID_CONFIG,
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "input_path": str(self.input_path),
            "backend": self.backend,
            "skills": list(self.skills),
            "max_context_tokens": self.max_context_tokens,
            "oversize_policy": self.oversize_policy,
            "rejected_history_policy": self.rejected_history_policy,
            "accepted_history_policy": self.accepted_history_policy,
            "tokenizer": None if self.tokenizer is None else self.tokenizer.to_dict(),
        }

    def tokenization_backend(self):
        return (
            FixtureTokenizer()
            if self.backend == "fixture"
            else PrecomputedTokenizer(self.tokenizer)
        )


def load_config(path: str | Path) -> ImportConfig:
    path = Path(path).resolve()
    try:
        row = fields(
            load_json(path.read_text(encoding="utf-8")),
            (
                "schema_version",
                "input_path",
                "backend",
                "skills",
                "max_context_tokens",
                "oversize_policy",
                "rejected_history_policy",
            ),
            ("accepted_history_policy", "tokenizer"),
        )
        text(row["input_path"], "input_path")
        input_path = (path.parent / row["input_path"]).resolve()
        return ImportConfig(
            **{
                **row,
                "input_path": input_path,
                "skills": tuple(sequence(row["skills"], "skills")),
                "tokenizer": (
                    tokenizer_ref(row["tokenizer"])
                    if row.get("tokenizer") is not None
                    else None
                ),
            }
        )
    except ContractError as exc:
        raise ContractError(ErrorCode.INVALID_CONFIG, str(exc), **exc.details) from exc


def _exclusion(record, cfg):
    if record.stats.context_tokens > cfg.max_context_tokens:
        if cfg.oversize_policy == "error":
            raise ContractError(
                ErrorCode.BUDGET_EXHAUSTED,
                "Trajectory exceeds context limit; not truncated",
                data_id=record.data_id,
            )
        return "CONTEXT_LIMIT"
    if cfg.accepted_history_policy == "exclude" and record.usage.accepted_sft:
        return "ACCEPTED_TRAINING_HISTORY"
    if any(use.decision == "pending" for use in record.usage.candidate_sft):
        return "PENDING_TRAINING_ATTEMPT"
    if cfg.rejected_history_policy == "exclude" and any(
        use.decision == "rejected" for use in record.usage.candidate_sft
    ):
        return "REJECTED_TRAINING_HISTORY"
    return None


def _implementation_hash():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            str(p.relative_to(root)).replace("\\", "/"): file_sha256(p)
            for p in sorted(root.rglob("*.py"))
        }
    )


def _staging_directory(output):
    """Create a short, exclusive sibling directory for atomic publication.

    Repeating output names and UUIDs at each nested handoff can exceed Windows
    path limits even when all final artifact paths fit. mkdir keeps inherited
    workspace ACLs on managed Windows hosts; exclusive creation retries name
    collisions and stays on the output's filesystem.
    """
    for _ in range(10):
        staging = output.parent / f".sc-{uuid.uuid4().hex[:16]}.staging"
        try:
            staging.mkdir()
        except FileExistsError:
            continue
        return staging
    raise ContractError(
        ErrorCode.ENVIRONMENT_FAILURE, "Cannot allocate staging directory"
    )


def _publish(staging, output):
    lock = output.parent / f".{output.name}.publish.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ContractError(
            ErrorCode.OUTPUT_EXISTS,
            "Another publication owns this output; inspect lock before recovery",
            lock=str(lock),
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical_json({"pid": os.getpid(), "staging": str(staging)}))
    try:
        for attempt in range(3):
            require(
                not output.exists(),
                "Output appeared during preparation",
                ErrorCode.OUTPUT_EXISTS,
            )
            try:
                staging.rename(output)
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        # A leftover lock is preferable to changing any completed evidence.
        for attempt in range(3):
            try:
                lock.unlink()
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))


def build_pool(config: ImportConfig, output: str | Path) -> dict:
    """Build a new pool. Existing outputs are never overwritten or resumed implicitly."""
    output = Path(output).resolve()
    require(
        not output.exists(),
        "Output already exists; choose a new directory",
        ErrorCode.OUTPUT_EXISTS,
        path=str(output),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_directory(output)
    try:
        backend = config.tokenization_backend()
        source_hash = hashlib.sha256()
        records, excluded, seen, total = [], [], set(), 0
        with config.input_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                source_hash.update(raw_line)
                require(
                    bool(raw_line.strip()),
                    "Blank JSONL row is invalid",
                    line=line_number,
                )
                try:
                    raw = load_json(raw_line.decode("utf-8"))
                    record = normalize_trajectory(raw, backend, config.skills)
                    require(
                        record.data_id not in seen,
                        "Duplicate data_id",
                        ErrorCode.DUPLICATE_ID,
                        data_id=record.data_id,
                    )
                    seen.add(record.data_id)
                    total += 1
                    reason = _exclusion(record, config)
                    if reason:
                        excluded.append(
                            {
                                "data_id": record.data_id,
                                "content_hash": record.content_hash,
                                "record_hash": record.record_hash,
                                "reason": reason,
                                "context_tokens": record.stats.context_tokens,
                            }
                        )
                    else:
                        records.append(record)
                except ContractError as exc:
                    raise ContractError(
                        exc.code, str(exc), **{**exc.details, "line": line_number}
                    ) from exc
        require(total > 0, "Input pool is empty", ErrorCode.INSUFFICIENT_POOL)
        require(
            bool(records),
            "No trajectories remain under the configured policies",
            ErrorCode.INSUFFICIENT_POOL,
        )
        records.sort(key=lambda r: r.data_id)
        excluded.sort(key=lambda r: r["data_id"])
        (staging / "trajectories.jsonl").write_text(
            "".join(canonical_json(r.to_dict()) + "\n" for r in records),
            encoding="utf-8",
            newline="\n",
        )
        (staging / "excluded.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in excluded),
            encoding="utf-8",
            newline="\n",
        )
        audit = {
            "schema_version": "1",
            "status": "passed",
            "scope": "m1_structural_validation",
            "source_authenticity_verified": False,
            "backend": config.backend,
            "synthetic": True if backend.identity.backend == "fixture" else None,
            "input_records": total,
            "included_records": len(records),
            "excluded_records": len(excluded),
            "duplicate_content_records": len(records)
            - len({r.content_hash for r in records}),
            "quality_counts": {
                s: sum(r.quality.status == s for r in records)
                for s in ("pass", "fail", "unknown")
            },
            "supervised_tokens": sum(r.stats.supervised_tokens for r in records),
            "context_tokens": sum(r.stats.context_tokens for r in records),
            "signal_records": sum(len(r.signals) for r in records),
            "records_with_signals": sum(bool(r.signals) for r in records),
            "gradient_features": sum(len(r.gradients) for r in records),
            "records_with_metadata": sum(r.metadata is not None for r in records),
            "missing_skill_annotations": {
                s: sum(
                    any(
                        m.skill == s and m.status == "missing"
                        for m in r.evidence_metrics
                    )
                    for r in records
                )
                for s in config.skills
            },
        }
        write_json(staging / "audit.json", audit)
        manifest = {
            "schema_version": "1",
            "backend": config.backend,
            "synthetic": audit["synthetic"],
            "tokenizer": backend.identity.to_dict(),
            "config": config.to_dict(),
            "input_sha256": source_hash.hexdigest(),
            "implementation_sha256": _implementation_hash(),
            "python_version": platform.python_version(),
            "input_records": total,
            "included_records": len(records),
            "excluded_records": len(excluded),
            "records": [
                {
                    "data_id": r.data_id,
                    "content_hash": r.content_hash,
                    "record_hash": r.record_hash,
                }
                for r in records
            ],
            "artifacts": {
                name: {"uri": name, "sha256": file_sha256(staging / name)}
                for name in ("trajectories.jsonl", "excluded.jsonl", "audit.json")
            },
        }
        manifest["pool_id"] = digest(manifest)
        write_json(staging / "manifest.json", manifest)
        # Rename publishes only a fully prepared tree; an interrupted stage is not a pool.
        _publish(staging, output)
        return manifest
    except Exception as exc:
        error = (
            exc.to_dict()
            if isinstance(exc, ContractError)
            else {"code": ErrorCode.ENVIRONMENT_FAILURE.value, "message": str(exc)}
        )
        write_json(staging / "failure.json", error)
        raise


@dataclass(frozen=True)
class PoolSnapshot:
    manifest: dict
    records: tuple[TrajectoryRecord, ...]
    excluded: tuple[dict, ...]

    def metric_view(self):
        from .view import MetricView

        return MetricView(self.records)


def _read_pool(path: str | Path) -> PoolSnapshot:
    """M2 entry point: verify every artifact and recompute all derived M1 fields."""
    root = Path(path).resolve()
    manifest = load_json((root / "manifest.json").read_text(encoding="utf-8"))
    require(
        type(manifest) is dict and manifest.get("schema_version") == "1",
        "Invalid pool schema",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        manifest.get("pool_id")
        == digest({k: v for k, v in manifest.items() if k != "pool_id"}),
        "Manifest checksum mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    expected_files = {"trajectories.jsonl", "excluded.jsonl", "audit.json"}
    require(
        set(manifest.get("artifacts", {})) == expected_files,
        "Pool artifact set mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    for name, ref in manifest["artifacts"].items():
        require(
            ref.get("uri") == name and file_sha256(root / name) == ref.get("sha256"),
            "Pool artifact checksum mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
            artifact=name,
        )

    class StoredBackend:
        identity = tokenizer_ref(manifest["tokenizer"])

        def encode(self, messages, provided=None):
            return tokenization(provided)

    try:
        stored_config = fields(
            manifest["config"],
            (
                "schema_version",
                "input_path",
                "backend",
                "skills",
                "max_context_tokens",
                "oversize_policy",
                "rejected_history_policy",
            ),
            ("accepted_history_policy", "tokenizer"),
        )
        config = ImportConfig(
            **{
                **stored_config,
                "input_path": Path(stored_config["input_path"]),
                "skills": tuple(sequence(stored_config["skills"], "skills")),
                "tokenizer": (
                    None
                    if stored_config.get("tokenizer") is None
                    else tokenizer_ref(stored_config["tokenizer"])
                ),
            }
        )
        require(
            manifest["backend"] == config.backend
            and config.tokenization_backend().identity == StoredBackend.identity,
            "Pool configuration and tokenizer identity disagree",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    except ContractError as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Malformed stored pool configuration",
            reason=str(exc),
        ) from exc

    records = []
    for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        row = load_json(line)
        raw = {
            k: row[k]
            for k in (
                "schema_version",
                "data_id",
                "source",
                "messages",
                "quality",
                "annotations",
                "usage",
                "tokenization",
            )
        }
        if "signals" in row:
            raw["signals"] = row["signals"]
        for field in ("metadata", "gradients", "annotation_record", "metadata_record"):
            if field in row:
                raw[field] = row[field]
        record = normalize_trajectory(
            raw, StoredBackend(), tuple(manifest["config"]["skills"])
        )
        # Keep old evidence readable and byte-identical; enrichment explicitly
        # upgrades a returned record, never rewrites the original pool.
        if row.get("normalization_version") in {"m1-v1", "m1-v2"}:
            record = replace(record, normalization_version=row["normalization_version"])
        require(
            canonical_json(record.to_dict()) == canonical_json(row),
            "Derived record fields are inconsistent",
            ErrorCode.ARTIFACT_MISMATCH,
        )
        records.append(record)
    excluded = tuple(
        load_json(line)
        for line in (root / "excluded.jsonl").read_text(encoding="utf-8").splitlines()
    )
    refs = [
        {
            "data_id": r.data_id,
            "content_hash": r.content_hash,
            "record_hash": r.record_hash,
        }
        for r in records
    ]
    ids = [r.data_id for r in records]
    excluded_ids = [r["data_id"] for r in excluded]
    require(
        ids == sorted(set(ids))
        and not set(ids).intersection(excluded_ids)
        and len(set(excluded_ids)) == len(excluded_ids),
        "Pool IDs are duplicated or out of order",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        refs == manifest["records"]
        and len(records) == manifest["included_records"]
        and len(excluded) == manifest["excluded_records"]
        and len(records) + len(excluded) == manifest["input_records"],
        "Pool counts or record identities mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    audit = load_json((root / "audit.json").read_text(encoding="utf-8"))
    expected_audit = {
        "schema_version": "1",
        "status": "passed",
        "scope": "m1_structural_validation",
        "source_authenticity_verified": False,
        "backend": config.backend,
        "synthetic": True if StoredBackend.identity.backend == "fixture" else None,
        "input_records": len(records) + len(excluded),
        "included_records": len(records),
        "excluded_records": len(excluded),
        "duplicate_content_records": len(records)
        - len({r.content_hash for r in records}),
        "quality_counts": {
            s: sum(r.quality.status == s for r in records)
            for s in ("pass", "fail", "unknown")
        },
        "supervised_tokens": sum(r.stats.supervised_tokens for r in records),
        "context_tokens": sum(r.stats.context_tokens for r in records),
        "signal_records": sum(len(r.signals) for r in records),
        "records_with_signals": sum(bool(r.signals) for r in records),
        "gradient_features": sum(len(r.gradients) for r in records),
        "records_with_metadata": sum(r.metadata is not None for r in records),
        "missing_skill_annotations": {
            s: sum(
                any(m.skill == s and m.status == "missing" for m in r.evidence_metrics)
                for r in records
            )
            for s in config.skills
        },
    }
    require(
        type(audit) is dict
        and set(audit).issubset(expected_audit)
        and all(
            canonical_json(value) == canonical_json(expected_audit[key])
            for key, value in audit.items()
        ),
        "Pool audit labels are inconsistent with actual records/configuration",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    if "gradient_features" in audit:
        require(
            audit["gradient_features"] == sum(len(r.gradients) for r in records)
            and audit["records_with_metadata"]
            == sum(r.metadata is not None for r in records),
            "Gradient/metadata audit totals are inconsistent",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    if "signal_records" in audit:
        require(
            audit["signal_records"] == sum(len(r.signals) for r in records)
            and audit["records_with_signals"] == sum(bool(r.signals) for r in records),
            "Signal audit totals are inconsistent",
            ErrorCode.ARTIFACT_MISMATCH,
        )
    require(
        audit["status"] == "passed"
        and audit["included_records"] == len(records)
        and audit["excluded_records"] == len(excluded)
        and audit["input_records"] == len(records) + len(excluded)
        and audit["supervised_tokens"]
        == sum(r.stats.supervised_tokens for r in records)
        and audit["context_tokens"] == sum(r.stats.context_tokens for r in records),
        "Audit totals are inconsistent",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    require(
        manifest["backend"] == manifest["config"]["backend"]
        and manifest["synthetic"]
        == (True if StoredBackend.identity.backend == "fixture" else None),
        "Backend or synthetic marker mismatch",
        ErrorCode.ARTIFACT_MISMATCH,
    )
    return PoolSnapshot(manifest, tuple(records), excluded)


def read_pool(path: str | Path) -> PoolSnapshot:
    """Load only a completed, consistent pool; malformed artifacts fail closed."""
    try:
        return _read_pool(path)
    except ContractError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        FileNotFoundError,
    ) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Missing or malformed pool artifact",
            reason=type(exc).__name__,
        ) from exc
