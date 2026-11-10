"""Build a checked, shareable result bundle without data, base weights or secrets."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import zipfile
from self_curate.contracts import digest

from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import read_json, atomic_json

SENSITIVE = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)$"
)


def redact(value):
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if SENSITIVE.search(k) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~-]+", r"\1[REDACTED]", value)
        value = re.sub(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        value = re.sub(
            r"(?i)([?&](?:api_key|token|access_token)=)[^&\s]+", r"\1[REDACTED]", value
        )
        return value
    return value


def _contains_training_payload(value):
    if isinstance(value, dict):
        examples = value.get("examples")
        if isinstance(examples, list) and any(
            isinstance(item, dict) and "input_ids" in item and "labels" in item for item in examples
        ):
            return True
        return any(_contains_training_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_training_payload(item) for item in value)
    return False


def verify_run(root):
    root = Path(root).resolve()
    marker = read_json(root / "completion.json")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "completion.json"
    }
    require(
        actual == set(marker["checksums"]), "Run output set changed before packaging"
    )
    for name, checksum in marker["checksums"].items():
        path = root / name
        require(
            not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and file_sha256(path) == checksum,
            "Run evidence changed before packaging",
        )
    return marker


def export_bundle(run, output=None):
    root = Path(run).resolve()
    verify_run(root)
    summary = read_json(root / "summary.json")
    output = (
        Path(output).resolve() if output else root.parent / "deliverables" / root.name
    )
    require(
        not output.is_relative_to(root) and not root.is_relative_to(output),
        "Delivery must be outside immutable run output",
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for metric in summary["initial_development"]["metrics"]:
        rows.append(
            {
                "split": "development",
                "branch": "initial",
                "stage": 0,
                "skill": metric["skill"],
                "mean_reward": metric["mean_reward"],
                "status": summary["initial_development"]["status"],
            }
        )
    for role, groups in (("development", summary["development"]),):
        for branch, reports in groups.items():
            for stage, report in enumerate(reports, 1):
                for metric in report["metrics"]:
                    rows.append(
                        {
                            "split": role,
                            "branch": branch,
                            "stage": stage,
                            "skill": metric["skill"],
                            "mean_reward": metric["mean_reward"],
                            "status": report["status"],
                        }
                    )
    for name, report in summary["benchmark"].items():
        suite_id = report.get("suite_id", "")
        point = name.removeprefix(summary["run_id"] + "-")
        if suite_id:
            point = point.removesuffix("-" + digest(suite_id)[:12])
        match = re.fullmatch(r"(.+)-stage([123])", point)
        scored = [
            v["result"]["reward"]
            for v in report.get("runs", [])
            if v.get("result") is not None and v["result"].get("reward") is not None
        ]
        rows.append(
            {
                "split": "benchmark",
                "suite": suite_id,
                "branch": match.group(1) if match else point,
                "stage": (
                    int(match.group(2)) if match else (0 if point == "initial" else "")
                ),
                "skill": "",
                "mean_reward": report["mean_reward"],
                "status": report["status"],
                "task_runs": len(report.get("runs", [])),
                "scored_runs": len(scored),
                "resolved_runs": (
                    sum(int(v) for v in scored)
                    if scored and all(v in (0, 1) for v in scored)
                    else ""
                ),
            }
        )
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "suite",
                "branch",
                "stage",
                "skill",
                "mean_reward",
                "status",
                "task_runs",
                "scored_runs",
                "resolved_runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "summary.json", redact(summary))
    atomic_json(
        output / "RESULTS_INDEX.json",
        {
            "run_id": summary["run_id"],
            "original_run_path": str(root),
            "backend": summary["backend"],
            "all_reports_valid": summary["all_reports_valid"],
            "training_weights_included": False,
            "raw_training_data_included": False,
            "retained_on_server": [
                "shared-assets",
                "data",
                "base-model",
                "branch adapter weights",
                "optimizer/RNG recovery states",
            ],
            "read_first": [
                "summary.json",
                "results.csv",
                "evidence/checkpoints.csv",
                "BUNDLE_MANIFEST.json",
            ],
        },
    )
    (output / "READ_ME.txt").write_text(
        "Read summary.json and results.csv first. Evidence and native logs are in results.zip. Base weights, raw training data and optimizer states stay on the server. Check SHA256SUMS.txt before transfer.\n",
        encoding="utf-8",
    )
    entries = {}
    archive = output / "results.zip"
    paths = [(p, p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file()]
    # A reused M0 measurement lives in shared storage, not in the new run.
    # Include its original task evidence too, after validating the asset manifest.
    if (root / "resume-plan.json").is_file():
        from self_curate.assets import AssetStore
        from self_curate.contracts.parsing import artifact

        phase = read_json(root / "resume-plan.json")
        reference = artifact(summary["shared_assets"]["m0_capability"])
        key = reference.uri.removeprefix("asset://").split("/")[0]
        stored = AssetStore(phase["asset_root"]).inspect(key)
        require(
            stored.reference == reference, "Baseline asset differs from frozen result"
        )
        paths.extend(
            (p, Path("shared-m0-capability") / p.relative_to(stored.payload))
            for p in sorted(stored.payload.rglob("*"))
            if p.is_file()
        )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, rel in paths:
            if not path.is_file():
                continue
            if (
                any(
                    part
                    in (
                        "weights",
                        "recovery",
                        "exposure-history",
                        "scoring-origin",
                        "capability-origin",
                        "strategy-origin",
                        "prepared-input-origin",
                    )
                    for part in rel.parts
                )
                or path.suffix.lower()
                not in (
                    ".json",
                    ".jsonl",
                    ".csv",
                    ".log",
                    ".txt",
                    ".md",
                    ".png",
                    ".svg",
                    ".diff",
                    ".patch",
                    ".toml",
                    ".yaml",
                )
                or path.name in ("records.jsonl", "historical-5016.jsonl", "job.json")
            ):
                continue
            require(not path.is_symlink(), "Do not package symlink targets")
            raw = path.read_bytes()
            if path.suffix == ".json" and _contains_training_payload(json.loads(raw.decode("utf-8"))):
                # Requests/plans can embed raw tokenized TrainingData as well as job.json.
                continue
            changed = False
            if path.suffix == ".json":
                transformed = (
                    json.dumps(
                        redact(json.loads(raw.decode("utf-8"))),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                changed = transformed != raw
                raw = transformed
            elif path.suffix == ".jsonl":
                raw = (
                    "\n".join(
                        json.dumps(
                            redact(json.loads(line)), ensure_ascii=False, sort_keys=True
                        )
                        for line in raw.decode("utf-8").splitlines()
                    )
                    + "\n"
                ).encode("utf-8")
                changed = raw != path.read_bytes()
            elif path.suffix in (
                ".log",
                ".txt",
                ".md",
                ".diff",
                ".patch",
                ".toml",
                ".yaml",
            ):
                raw = redact(raw.decode("utf-8", errors="replace")).encode("utf-8")
                changed = raw != path.read_bytes()
            name = "evidence/" + rel.as_posix()
            z.writestr(name, raw)
            entries[name] = {
                "source_sha256": file_sha256(path),
                "packed_sha256": hashlib.sha256(raw).hexdigest(),
                "sanitized_or_reformatted": changed,
            }
        for name in (
            "summary.json",
            "results.csv",
            "RESULTS_INDEX.json",
            "READ_ME.txt",
        ):
            z.write(output / name, name)
        manifest = {
            "schema_version": "1",
            "run_id": summary["run_id"],
            "entries": entries,
            "weights_and_data_excluded": True,
            "redaction_applied": True,
        }
        z.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        )
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "Delivery archive CRC check failed")
        for name, row in entries.items():
            require(
                hashlib.sha256(z.read(name)).hexdigest() == row["packed_sha256"],
                "Packed evidence changed",
            )
    (output / "SHA256SUMS.txt").write_text(
        file_sha256(archive) + "  results.zip\n", encoding="utf-8"
    )
    atomic_json(
        output / "bundle-verification.json",
        {
            "passed": True,
            "zip": str(archive),
            "sha256": file_sha256(archive),
            "bytes": archive.stat().st_size,
            "evidence_files": len(entries),
        },
    )
    return output


def verify_bundle(output):
    output = Path(output)
    receipt = read_json(output / "bundle-verification.json")
    archive = output / "results.zip"
    require(
        receipt.get("passed") is True and file_sha256(archive) == receipt["sha256"],
        "Delivery archive changed after packaging",
    )
    require(
        (output / "SHA256SUMS.txt").read_text(encoding="utf-8").split()[0]
        == receipt["sha256"],
        "Delivery checksum file changed",
    )
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "Delivery archive CRC check failed")
        manifest = json.loads(z.read("BUNDLE_MANIFEST.json"))
        for name, row in manifest["entries"].items():
            require(
                hashlib.sha256(z.read(name)).hexdigest() == row["packed_sha256"],
                "Delivery evidence changed",
            )
        for name in (
            "summary.json",
            "results.csv",
            "RESULTS_INDEX.json",
            "READ_ME.txt",
        ):
            require(
                z.read(name) == (output / name).read_bytes(),
                "Delivery companion file changed",
            )
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(export_bundle(args.run, args.output))


if __name__ == "__main__":
    main()
