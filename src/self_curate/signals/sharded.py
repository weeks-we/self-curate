"""One frozen scoring process per authorized GPU; deterministic token-balanced shards."""

from pathlib import Path
from dataclasses import replace
import os, shutil, subprocess, sys, uuid
from self_curate.contracts import CostLedger, canonical_json, digest
from self_curate.contracts.parsing import model_ref
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, read_json
from .resident import ResidentSignalBackend
from .service import SignalService
from .contracts import ScoringBatch
from .io import load_scoring_batch, publish_batch, read_records
from .backends import ScoringFailure


def balanced_shards(records, count):
    require(type(count) is int and count > 0, "Positive worker count required")
    groups = [[] for _ in range(min(count, len(records)))]
    loads = [0] * len(groups)
    for index, record in sorted(
        enumerate(records), key=lambda x: (-len(x[1].tokenization.input_ids), x[0])
    ):
        rank = min(range(len(groups)), key=lambda i: (loads[i], i))
        groups[rank].append((index, record))
        loads[rank] += len(record.tokenization.input_ids)
    return tuple(tuple(r for _, r in sorted(group)) for group in groups)


class ClusterBackend(ResidentSignalBackend):
    @property
    def implementation_hash(self):
        # All workers use local cuda:0 after UUID isolation. This identity binds
        # the whole approved cluster/numerical profile, not one rank's UUID.
        return self.cluster_hash


class ShardedSignalService:
    def __init__(
        self,
        *,
        model,
        store_settings,
        gpu_uuids,
        output,
        max_input_tokens,
        execution,
        worker=None,
        backend="real",
    ):
        require(
            backend in ("real", "mock")
            and (
                execution == "server"
                if backend == "real"
                else execution == "local_mock" and worker is not None
            )
            and len(gpu_uuids) == len(set(gpu_uuids))
            and bool(gpu_uuids),
            "Explicit server GPUs or injected local mock workers required",
        )
        self.domain = backend
        self.model_ref, self.settings, self.gpu_uuids, self.output = (
            model,
            store_settings,
            tuple(gpu_uuids),
            Path(output),
        )
        import importlib.metadata
        from self_curate.training import peft_store

        self.cluster_hash = digest(
            {
                "code": file_sha256(Path(__file__)),
                "resident": file_sha256(Path(__file__).with_name("resident.py")),
                "store_code": file_sha256(Path(peft_store.__file__)),
                "store_settings": store_settings,
                "torch": importlib.metadata.version("torch"),
                "gpu_uuids": list(gpu_uuids),
                "policy": "BF16-frozen/float64-full-vocab-statistics/cuda0-uuid-isolated-v1",
            }
        )
        self.backend = type(
            "ClusterIdentity",
            (),
            {
                "backend": backend,
                "implementation_hash": self.cluster_hash,
                "close": lambda self: None,
            },
        )()
        self.reference = SignalService(
            model=model,
            backend=self.backend,
            output=output,
            max_input_tokens=max_input_tokens,
        )
        self.computation_hash = self.reference.computation_hash
        self.max_input_tokens = max_input_tokens
        self.worker = worker

    def spec(self, *args, **kwargs):
        return self.reference.spec(*args, **kwargs)

    def score(self, records, *, specs=None):
        records = tuple(records)
        specs = tuple(specs or (self.spec(),))
        label = "cluster-" + uuid.uuid4().hex[:12]
        folder = self.output / ("score-" + label)
        folder.mkdir(parents=True, exist_ok=False)
        children = []
        paths = []
        try:
            for rank, shard in enumerate(balanced_shards(records, len(self.gpu_uuids))):
                work = folder / "shards" / str(rank)
                work.mkdir(parents=True)
                (work / "inputs.jsonl").write_text(
                    "".join(canonical_json(r.to_dict()) + "\n" for r in shard),
                    encoding="utf-8",
                )
                job = {
                    "model": self.model_ref.to_dict(),
                    "store": self.settings,
                    "cluster_hash": self.cluster_hash,
                    "max_input_tokens": self.max_input_tokens,
                    "expected_computation_hash": self.computation_hash,
                    "specs": [s.to_dict() for s in specs],
                    "gpu_uuid": self.gpu_uuids[rank],
                }
                atomic_json(work / "job.json", job)
                if self.worker is not None:
                    self.worker(work)
                    paths.append(work)
                    continue
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=self.gpu_uuids[rank],
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                )
                env["PYTHONPATH"] = (
                    str(Path(__file__).parents[2])
                    + os.pathsep
                    + env.get("PYTHONPATH", "")
                )
                stream = (work / "worker.log").open("w", encoding="utf-8")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "self_curate.signals.sharded",
                        "--worker",
                        str(work),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                children.append((process, stream, work))
                paths.append(work)
            for process, stream, work in children:
                process.wait()
                stream.close()
            batches = []
            failures = []
            for work in paths:
                if (work / "finished.json").exists():
                    path = Path(read_json(work / "finished.json")["folder"])
                    batches.append(
                        (
                            path,
                            load_scoring_batch(
                                path,
                                expected_model=self.model_ref,
                                expected_computation_hash=self.computation_hash,
                            ),
                        )
                    )
                else:
                    failures.append(str(work))
            if failures:
                events = [e for _, batch in batches for e in batch.ledger.events]
                for work in paths:
                    for fail in work.glob("scores/score-*/failure.json"):
                        from self_curate.contracts import parse_contract

                        events.extend(
                            parse_contract(
                                "CostLedger", read_json(fail)["ledger"]
                            ).events
                        )
                ledger = CostLedger(label, tuple(events), self.domain)
                atomic_json(
                    folder / "cluster-failure.json",
                    {
                        "failed_workers": failures,
                        "ledger": ledger.to_dict(),
                        "unknown_worker_costs": True,
                    },
                )
                raise ScoringFailure(
                    "Scoring shard failed; preserve completed and unknown work",
                    ledger=ledger,
                )
            merged = {r.data_id: r for _, batch in batches for r in batch.records}
            outcomes = {
                (o.data_id, o.spec.spec_hash): o
                for _, batch in batches
                for o in batch.outcomes
            }
            require(
                set(merged) == {r.data_id for r in records},
                "Shards changed scoring coverage",
            )
            token_rows = []
            forward = 0
            for path, batch in batches:
                token_rows.extend(read_json(path / "token-statistics.json")["records"])
                for original in sorted(path.glob("fwd-*")):
                    shutil.copytree(original, folder / ("fwd-" + str(forward)))
                    forward += 1
            atomic_json(
                folder / "request.json",
                {
                    "schema_version": "2",
                    "input_records_file": "inputs.jsonl",
                    "token_statistics_file": "token-statistics.json",
                    "model_ref": self.model_ref.to_dict(),
                    "computation_hash": self.computation_hash,
                    "specs": [s.to_dict() for s in specs],
                    "records": [
                        {
                            "data_id": r.data_id,
                            "record_hash": r.record_hash,
                            "content_hash": r.content_hash,
                            "tokenization_hash": digest(r.tokenization.to_dict()),
                        }
                        for r in records
                    ],
                    "backend": self.domain,
                },
            )
            (folder / "inputs.jsonl").write_text(
                "".join(canonical_json(r.to_dict()) + "\n" for r in records),
                encoding="utf-8",
            )
            tokens = {r["data_id"]: r for r in token_rows}
            atomic_json(
                folder / "token-statistics.json",
                {
                    "schema_version": "1",
                    "model_hash": self.model_ref.model_hash,
                    "computation_hash": self.computation_hash,
                    "records": [
                        tokens[r.data_id] for r in records if r.data_id in tokens
                    ],
                },
            )
            batch = ScoringBatch(
                label,
                self.model_ref,
                self.computation_hash,
                tuple(merged[r.data_id] for r in records),
                tuple(outcomes[r.data_id, s.spec_hash] for r in records for s in specs),
                CostLedger(
                    label,
                    tuple(
                        replace(e, attempt_id=label)
                        for _, b in batches
                        for e in b.ledger.events
                    ),
                    self.domain,
                ),
                self.domain,
            )
            publish_batch(batch, folder)
            return batch
        finally:
            for process, stream, _ in children:
                # Do not orphan running scorer processes on launch/controller errors.
                if process.poll() is None:
                    process.terminate()
                process.wait()
                stream.close()


def worker(folder):
    from self_curate.training.peft_store import PeftModelStore, LoRASettings
    from self_curate.contracts.parsing import signal_spec

    work = Path(folder)
    job = read_json(work / "job.json")
    model = model_ref(job["model"])
    cfg = job["store"]
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == job["gpu_uuid"],
        "Scoring worker escaped its authorized UUID",
    )
    store = PeftModelStore(
        base_path=cfg["base_path"],
        initial_model=model,
        initial_adapter=cfg["initial_adapter"],
        settings=LoRASettings.from_dict(cfg["lora"]),
        base_fingerprint=cfg["base_fingerprint"],
        loader=cfg["loader"],
    )
    backend = ClusterBackend(store, backend="real", device="cuda:0")
    backend.cluster_hash = job["cluster_hash"]
    service = SignalService(
        model=model,
        backend=backend,
        output=work / "scores",
        max_input_tokens=job["max_input_tokens"],
    )
    require(
        service.computation_hash == job["expected_computation_hash"],
        "Worker scoring implementation changed",
    )
    try:
        batch = service.score(
            read_records(work / "inputs.jsonl"),
            specs=tuple(signal_spec(s) for s in job["specs"]),
        )
        atomic_json(
            work / "finished.json",
            {"folder": str(service.output / ("score-" + batch.batch_id))},
        )
    finally:
        backend.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    worker(parser.parse_args().worker)
