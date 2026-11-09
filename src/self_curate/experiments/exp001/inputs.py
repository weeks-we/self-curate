"""Register already prepared Coding inputs once; reject incomplete/leaking data."""

from dataclasses import dataclass
from pathlib import Path
import re

from self_curate.contracts import (
    ArtifactRef,
    CostLedger,
    ModelRef,
    TaskRecord,
    canonical_json,
    parse_contract,
    parse_task_record,
)
from self_curate.contracts.types import integer, require
from self_curate.contracts.workflow import WorkflowRecord
from self_curate.contracts.parsing import load_json, fields, model_ref, artifact
from self_curate.data import MetricView
from self_curate.data.pool import write_json, file_sha256
from self_curate.assets import AssetSpec
from self_curate.validation import TaskSuite


@dataclass(frozen=True)
class BenchmarkSuite(WorkflowRecord):
    suite_id: str
    tasks: tuple[TaskRecord, ...]
    seeds: tuple[int, ...]
    sampling: ArtifactRef
    protocol: ArtifactRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(
            self.schema_version == "1"
            and type(self.suite_id) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.suite_id)
            and self.backend in ("mock", "real"),
            "Invalid benchmark identity",
        )
        require(
            type(self.tasks) is tuple
            and bool(self.tasks)
            and all(
                isinstance(t, TaskRecord)
                and t.split == "test"
                and t.backend == self.backend
                for t in self.tasks
            ),
            "Benchmark tasks must retain test split",
        )
        require(
            len({t.task_id for t in self.tasks}) == len(self.tasks),
            "Duplicate benchmark task",
        )
        require(
            type(self.seeds) is tuple
            and bool(self.seeds)
            and len(set(self.seeds)) == len(self.seeds),
            "Explicit benchmark seeds required",
        )
        for seed in self.seeds:
            integer(seed, "benchmark seed")
        require(
            isinstance(self.sampling, ArtifactRef)
            and isinstance(self.protocol, ArtifactRef),
            "Fixed benchmark sampling/protocol required",
        )
        if self.backend == "real":
            require(
                not any(
                    r.uri.strip().lower().startswith(("mock://", "fixture://"))
                    for r in (
                        self.sampling,
                        self.protocol,
                        *(
                            r
                            for t in self.tasks
                            for r in (t.payload, t.scorer, t.environment)
                        ),
                    )
                ),
                "Mock benchmark cannot be real",
            )


@dataclass(frozen=True)
class PreparedInputs:
    records: tuple
    model: ModelRef
    development: TaskSuite
    benchmarks: tuple[BenchmarkSuite, ...]
    skills: tuple[str, ...]
    source_lock_json: str
    isolation_evidence: ArtifactRef
    origin_ledger: CostLedger

    def validate(self, config):
        require(
            type(self.records) is tuple
            and bool(self.records)
            and isinstance(self.model, ModelRef),
            "Prepared M1 pool and pinned initial model required",
        )
        view = MetricView(self.records)
        require(
            isinstance(self.development, TaskSuite)
            and self.development.backend == config.backend
            and self.origin_ledger.backend == config.backend
            and isinstance(self.isolation_evidence, ArtifactRef),
            "Preparation domain/evidence mismatch",
        )
        require(
            type(self.skills) is tuple
            and bool(self.skills)
            and len(set(self.skills)) == len(self.skills),
            "Frozen skill ontology required",
        )
        require(
            type(self.benchmarks) is tuple
            and bool(self.benchmarks)
            and all(
                isinstance(b, BenchmarkSuite) and b.backend == config.backend
                for b in self.benchmarks
            ),
            "Registered final benchmark suites required",
        )
        require(
            len({b.suite_id for b in self.benchmarks}) == len(self.benchmarks),
            "Duplicate benchmark suite",
        )
        lock = load_json(self.source_lock_json)
        require(type(lock) is dict and bool(lock), "Concrete source lock required")
        require(
            not any(
                v in ("main", "latest", "resolve_once", "")
                for k, v in lock.items()
                if "revision" in k.lower()
            ),
            "Source revisions must already be resolved",
        )
        require(
            len({r.content_hash for r in self.records}) == len(self.records),
            "Preparation must deduplicate candidate content before freezing the 10% denominator",
        )
        require(
            all(
                r.quality.status == "pass"
                and r.tokenization.identity == self.model.tokenizer
                and r.stats.context_tokens <= config.max_context_tokens
                and r.annotations is not None
                and set(s.skill for s in r.annotations.skills).issubset(self.skills)
                and not r.usage.candidate_sft
                for r in self.records
            ),
            "Pool needs quality pass, skill labels, complete fitting trajectories and no prior task training",
        )
        dev_ids = {t.task_id for t in self.development.tasks}
        train_ids = {r.data_id for r in self.records} | {
            r.source.record_id for r in self.records
        }
        train_families = {f for r in self.records for f in r.source.family_ids}
        dev_families = {f for t in self.development.tasks for f in t.family_ids}
        require(
            not train_ids.intersection(dev_ids)
            and not train_families.intersection(dev_families),
            "Train/development task or source-family overlap",
        )
        for suite in self.benchmarks:
            ids = {t.task_id for t in suite.tasks}
            families = {f for t in suite.tasks for f in t.family_ids}
            require(
                not ids.intersection(train_ids | dev_ids)
                and not families.intersection(train_families | dev_families),
                "Benchmark contamination with optimization data",
            )
        # Different final benchmark variants may overlap; report them separately.
        if config.backend == "real":
            settings = load_json(config.adapter_settings_json)
            from .sources import load_source_registry, validate_source_receipts

            binding = settings.get("source_registry")
            require(
                type(binding) is dict and set(binding) == {"path", "sha256"},
                "Bind the reviewed source registry before real preparation",
            )
            registry = load_source_registry(
                binding["path"], expected_sha256=binding["sha256"]
            )
            require(
                lock.get("source_registry_sha256") == binding["sha256"],
                "Prepared inputs use another source registry",
            )
            validate_source_receipts(registry, lock)
            if registry.get("training_selection", {}).get("mode") == "historical_exact":
                source = registry["datasets"]["author_history"]
                require(
                    lock.get("historical_raw_rows") == 5016
                    and len(self.records) <= 5016,
                    "Use the complete original 5016-row release and report admission exclusions",
                )
                require(
                    all(
                        r.source.source_id == source["repo_id"]
                        and r.source.revision == source["revision"]
                        for r in self.records
                    ),
                    "Prepared records came from the later tool pool or another source",
                )
            require(
                self.model.base_model_id == registry["model"]["repo_id"]
                and self.model.revision == registry["model"]["revision"],
                "Prepared model differs from reviewed official source pins",
            )
            require(
                settings.get("model_id") == self.model.base_model_id
                and settings.get("model_revision") == self.model.revision
                and re.fullmatch(r"[a-f0-9]{40,64}", self.model.revision),
                "Real initial model differs from the pinned configured revision",
            )
            require(
                settings.get("tuning") == "lora"
                and self.model.adapter_hash is not None,
                "EXP001 server profile requires a common frozen-base initial LoRA adapter",
            )
            require(
                set(self.skills)
                == {
                    "code_localization",
                    "failure_reproduction",
                    "error_analysis",
                    "hypothesis_revision",
                    "implementation_edit",
                    "test_construction",
                    "regression_check",
                },
                "Use the frozen seven-skill Coding ontology",
            )
            require(
                self.model.tokenizer.backend != "fixture"
                and not any(r.known_mock_artifacts for r in self.records),
                "Mock preparation cannot enter real execution",
            )
            require(
                not self.isolation_evidence.uri.strip()
                .lower()
                .startswith(("fixture://", "mock://")),
                "Real preparation requires external near-duplicate/source isolation audit",
            )
            require(
                all(t.family_ids for t in self.development.tasks)
                and all(r.source.family_ids for r in self.records)
                and all(t.family_ids for b in self.benchmarks for t in b.tasks),
                "Real data must declare source families",
            )
        return view

    def spec(self, backend):
        return AssetSpec.create(
            "coding_preparation",
            backend,
            {
                "records": [r.record_hash for r in self.records],
                "model_tokenizer": self.model.tokenizer.to_dict(),
                "development": self.development.to_dict(),
                "benchmarks": [b.to_dict() for b in self.benchmarks],
                "skills": list(self.skills),
                "source_lock": load_json(self.source_lock_json),
                "isolation_evidence": self.isolation_evidence.to_dict(),
            },
        )

    def register(self, store, output, *, producer):
        spec = self.spec(self.origin_ledger.backend)
        found = store.get(spec)
        if found is not None:
            return found
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "records.jsonl").write_text(
            "".join(canonical_json(r.to_dict()) + "\n" for r in self.records),
            encoding="utf-8",
        )
        write_json(output / "inputs.json", spec.identity)
        return store.publish(
            spec,
            output,
            producer=producer,
            origin_ledger=self.origin_ledger,
            validator=lambda p: require(
                load_json((p / "inputs.json").read_text(encoding="utf-8"))
                == spec.identity,
                "Prepared input identity changed",
            ),
        )


def load_prepared_manifest(path):
    """Portable output contract for real data preparation/label providers.

    This reads existing M1 records, labels and task manifests. It never fetches
    datasets, calls a Teacher, or declares unexecuted quality audits successful.
    """
    from self_curate.signals import read_records
    from self_curate.orchestration.io import parse_outer_contract

    path = Path(path).resolve()
    row = fields(
        load_json(path.read_text(encoding="utf-8")),
        (
            "schema_version",
            "files",
            "checksums",
            "skills",
            "source_lock",
            "isolation_evidence",
            "origin_ledger",
        ),
    )
    require(row["schema_version"] == "1", "Unknown prepared-manifest version")
    files = fields(row["files"], ("records", "model", "development", "benchmarks"))
    require(
        type(row["checksums"]) is dict and set(row["checksums"]) == set(files.values()),
        "Prepared inputs need exact file checksums",
    )
    resolved = {}
    for role, name in files.items():
        require(
            type(name) is str
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts,
            "Unsafe prepared-input path",
        )
        file = path.parent / name
        require(
            file.resolve().is_relative_to(path.parent)
            and not file.is_symlink()
            and file_sha256(file) == row["checksums"][name],
            "Prepared file changed or escaped manifest",
        )
        resolved[role] = file
    benchmarks = load_json(resolved["benchmarks"].read_text(encoding="utf-8"))
    require(
        type(benchmarks) is list and type(row["skills"]) is list,
        "Prepared suite/skill arrays required",
    )
    suites = []
    for value in benchmarks:
        value = fields(
            value,
            (
                "schema_version",
                "suite_id",
                "tasks",
                "seeds",
                "sampling",
                "protocol",
                "backend",
            ),
        )
        require(value["schema_version"] == "1", "Unknown benchmark schema")
        suites.append(
            BenchmarkSuite(
                value["suite_id"],
                tuple(parse_task_record(t) for t in value["tasks"]),
                tuple(value["seeds"]),
                artifact(value["sampling"]),
                artifact(value["protocol"]),
                value["backend"],
            )
        )
    return PreparedInputs(
        read_records(resolved["records"]),
        model_ref(load_json(resolved["model"].read_text(encoding="utf-8"))),
        parse_outer_contract(
            "TaskSuite", load_json(resolved["development"].read_text(encoding="utf-8"))
        ),
        tuple(suites),
        tuple(row["skills"]),
        canonical_json(row["source_lock"]),
        artifact(row["isolation_evidence"]),
        parse_contract("CostLedger", row["origin_ledger"]),
    )
