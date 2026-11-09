"""Read-only live progress view: python -m ...exp001.status --output RUN --watch."""

import argparse
import time
from pathlib import Path
from .recovery import read_json


def main():
    parser = argparse.ArgumentParser(
        description="Read EXP001 progress without loading models"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    while True:
        rows = []
        for path in [
            root / "progress/progress.json",
            *sorted(root.glob("branches/*/progress/progress.json")),
        ]:
            if path.is_file():
                row = read_json(path)
                total = "?" if row["total"] is None else row["total"]
                eta = (
                    "estimating"
                    if row["eta_seconds"] is None
                    else f"~{row['eta_seconds']:.0f}s"
                )
                age = max(0, time.time() - row["timestamp"])
                rows.append(
                    f"{row['label']}: {row['phase']} {row['completed']}/{total} ETA {eta}; updated {age:.0f}s ago"
                )
        print("\n".join(rows) if rows else "No progress published yet.", flush=True)
        if not args.watch or (root / "completion.json").is_file():
            return 0
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
