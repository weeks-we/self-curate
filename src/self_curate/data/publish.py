"""Publish signal-only enrichment as a new M1 pool; original snapshots stay intact."""

from pathlib import Path

from self_curate.contracts import ArtifactRef, canonical_json, digest
from self_curate.contracts.parsing import load_json
from self_curate.contracts.types import require
from .enrichment import attach_signals
from .pool import read_pool, file_sha256, write_json, _staging_directory, _publish
from .view import MetricView


def publish_signal_enrichment(source, records, output, *, receipt: ArtifactRef):
    require(
        isinstance(receipt, ArtifactRef),
        "Signal enrichment needs a computation receipt",
    )
    source, output = Path(source).resolve(), Path(output).resolve()
    snapshot = read_pool(source)
    original = {r.data_id: r for r in snapshot.records}
    supplied = tuple(records)
    MetricView(supplied)
    require(
        len({r.data_id for r in supplied}) == len(supplied), "Duplicate enrichment IDs"
    )
    for record in supplied:
        require(record.data_id in original, "Enrichment contains an unknown data ID")
        require(
            record == original[record.data_id]
            or attach_signals(original[record.data_id], record.signals) == record,
            "Signal enrichment changed data/masks/labels/usage outside forward signals",
        )
        original[record.data_id] = record
    merged = tuple(original[r.data_id] for r in snapshot.records)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "Signal pool output already exists")
    stage = _staging_directory(output)
    (stage / "trajectories.jsonl").write_text(
        "".join(canonical_json(r.to_dict()) + "\n" for r in merged),
        encoding="utf-8",
        newline="\n",
    )
    (stage / "excluded.jsonl").write_text(
        "".join(canonical_json(r) + "\n" for r in snapshot.excluded),
        encoding="utf-8",
        newline="\n",
    )
    audit = load_json((source / "audit.json").read_text(encoding="utf-8"))
    audit.update(
        signal_records=sum(len(r.signals) for r in merged),
        records_with_signals=sum(bool(r.signals) for r in merged),
    )
    write_json(stage / "audit.json", audit)
    manifest = {
        **snapshot.manifest,
        "records": [
            {
                "data_id": r.data_id,
                "content_hash": r.content_hash,
                "record_hash": r.record_hash,
            }
            for r in merged
        ],
        "parent_pool_id": snapshot.manifest["pool_id"],
        "signal_enrichment": receipt.to_dict(),
        "artifacts": {
            n: {"uri": n, "sha256": file_sha256(stage / n)}
            for n in ("trajectories.jsonl", "excluded.jsonl", "audit.json")
        },
    }
    manifest.pop("pool_id")
    manifest["pool_id"] = digest(manifest)
    write_json(stage / "manifest.json", manifest)
    read_pool(stage)
    _publish(stage, output)
    return manifest
