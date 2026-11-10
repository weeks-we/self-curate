"""Timestamped run/attempt identities and durable structured event logs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import uuid


def new_run_id():
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-") + hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:10]


class RunLog:
    def __init__(self, root, *, rank=0, identity=None):
        self.identity = identity or new_run_id()
        self.folder = Path(root) / self.identity / f"rank-{rank:02d}"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.path = self.folder / "events.jsonl"
        self.started = perf_counter()
        self.rank = rank

    def event(self, event, **details):
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "attempt_id": self.identity,
                  "rank": self.rank, "pid": os.getpid(), "elapsed_seconds": perf_counter() - self.started,
                  "event": event, **details}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record
