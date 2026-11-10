"""Read the exact original 5016-row release without joining the later tool pool."""

from pathlib import Path
import argparse
import json
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, commit_directory

DATASET = "SWE-bench/SWE-smith-trajectories" #TODO 可更换
REVISION = "470d6f28b7219c95f43c47843cc7c69f63e077f0"


def load_historical(paths, *, checksums):
    require(
        set(checksums) == {str(Path(p)) for p in paths},
        "Need all downloaded shard checksums",
    )
    rows = []
    for filename in paths:
        path = Path(filename)
        require(
            file_sha256(path) == checksums[str(path)], "Historical source bytes changed"
        )
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            values = pq.read_table(path).to_pylist()
        elif path.suffix == ".json":
            values = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        else:
            raise ValueError("Historical shards must be parquet/JSON/JSONL")
        rows.extend(values)
    require(
        len(rows) == 5016,
        "Expected the complete historical 5016-row release, not a substitute tool subset",
    )
    result = []
    for index, row in enumerate(rows):
        require(
            type(row) is dict
            and type(row.get("instance_id")) is str
            and type(row.get("messages")) is list
            and bool(row["messages"]),
            "Malformed original trajectory",
        )
        for message in row["messages"]:
            require(
                type(message) is dict
                and message.get("role") in ("system", "user", "assistant", "tool")
                and (
                    type(message.get("content")) is str
                    or (
                        message.get("content") is None
                        and bool(message.get("tool_calls"))
                    )
                ),
                "Preserve original textual messages; unsupported source schema",
            )
        result.append(
            {
                "data_id": "historical-5016-" + str(index).zfill(5),
                "row_index": index,
                "instance_id": row["instance_id"],
                "raw_record": row,
                "raw_record_hash": digest(row),
                "source": {
                    "dataset": DATASET,
                    "revision": REVISION,
                    "config": "default",
                    "split": "train",
                },
                "quality_status": "unknown",
                "annotations_status": "pending",
            }
        )
    return tuple(result)


def export_historical(paths, checksums, output):
    rows = load_historical(paths, checksums=checksums)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    with (root / "historical-5016.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    atomic_json(
        root / "source.json",
        {
            "dataset": DATASET,
            "revision": REVISION,
            "split": "train",
            "rows": 5016,
            "source_files": checksums,
            "training_pool": "historical_exact",
            "later_tool_pool_used": False,
            "ready_for_training": False,
            "remaining": [
                "quality admission",
                "tokenizer/mask",
                "skill labels",
                "task isolation",
            ],
        },
    )
    commit_directory(root)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_historical(
        args.shards, json.loads(Path(args.checksums).read_text()), args.output
    )


if __name__ == "__main__":
    main()
