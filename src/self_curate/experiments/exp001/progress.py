"""Append-only progress events, atomic status, conservative stage-local ETA."""

from pathlib import Path
from time import time, perf_counter
import sys

from .recovery import atomic_json
from self_curate.contracts import canonical_json


class Progress:
    def __init__(self, output, *, label="experiment", console=True):
        self.output, self.label, self.console = Path(output), label, console
        self.output.mkdir(parents=True, exist_ok=True)
        self.started, self.origin = perf_counter(), None
        self.phase = None

    def update(self, phase, completed=0, total=None, **details):
        phase_key = (phase, details.get("evaluation_id"))
        if self.phase != phase_key or self.origin is None or completed < self.origin[0]:
            self.phase, self.origin = phase_key, (completed, perf_counter())
        advanced = completed - self.origin[0]
        elapsed = perf_counter() - self.origin[1]
        eta = (
            (total - completed) * elapsed / advanced
            if total is not None and advanced > 0 and completed < total
            else None
        )
        if total is not None and completed >= total:
            eta = 0.0
        value = dict(
            timestamp=time(),
            label=self.label,
            phase=phase,
            completed=completed,
            total=total,
            percent=None if not total else round(100 * completed / total, 2),
            eta_seconds=eta,
            eta_scope="current_phase_estimate",
            **details,
        )
        atomic_json(self.output / "progress.json", value)
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
        if self.console:
            count = str(completed) if total is None else f"{completed}/{total}"
            remaining = "estimating" if eta is None else f"~{eta:.0f}s"
            filled = 0 if not total else min(20, int(20 * completed / total))
            bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
            print(
                f"[{self.label}] {phase} {bar} {count} ETA {remaining}",
                file=sys.stderr,
                flush=True,
            )
