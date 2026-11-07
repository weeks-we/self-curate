"""Branch-local evidence of completed OUTER task SFT, distinct from inner candidates."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from self_curate.contracts import ArtifactRef, ContractError, ErrorCode, digest
from self_curate.contracts.parsing import artifact, fields, load_json
from self_curate.contracts.types import integer, require, sha256, text
from .pool import _publish, _staging_directory, file_sha256, write_json


@dataclass(frozen=True)
class OuterTrainingExposure:
    training_id: str
    attempt_id: str
    handoff_hash: str
    batch_id: str
    members: tuple[tuple[str, str], ...]
    starting_model_hash: str
    candidate_model_hash: str
    processed_supervised_tokens: int
    optimizer_steps: int
    receipt: ArtifactRef
    backend: str
    schema_version: str = "1"

    def __post_init__(self):
        require(
            self.schema_version == "1" and self.backend in ("mock", "real"),
            "Invalid outer exposure version/domain",
        )
        text(self.training_id, "outer training ID")
        text(self.attempt_id, "outer attempt ID")
        for name in (
            "handoff_hash",
            "batch_id",
            "starting_model_hash",
            "candidate_model_hash",
        ):
            sha256(getattr(self, name), name)
        require(
            type(self.members) is tuple and bool(self.members),
            "Outer exposure needs actual task data",
        )
        for member in self.members:
            require(
                type(member) is tuple and len(member) == 2, "Expected ID/content pair"
            )
            text(member[0], "outer data ID")
            sha256(member[1], "outer content hash")
        require(
            len({p[0] for p in self.members}) == len(self.members),
            "Duplicate outer training data ID",
        )
        integer(self.processed_supervised_tokens, "processed supervised tokens", 1)
        integer(self.optimizer_steps, "optimizer steps", 1)
        require(
            isinstance(self.receipt, ArtifactRef), "Physical training receipt required"
        )
        require(
            self.backend != "real"
            or not self.receipt.uri.strip()
            .lower()
            .startswith(("mock://", "fixture://")),
            "Mock exposure cannot be real",
        )

    def to_dict(self):
        return {
            **vars(self),
            "members": [list(p) for p in self.members],
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True)
class OuterExposureHistory:
    branch_id: str
    backend: str
    events: tuple[OuterTrainingExposure, ...] = ()
    unresolved: tuple[tuple[str, ArtifactRef], ...] = ()
    parent_hash: str | None = None
    schema_version: str = "1"
    _by_id: object = field(init=False, repr=False, compare=False)
    _contents: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        text(self.branch_id, "experiment branch ID")
        require(
            self.backend in ("mock", "real") and self.schema_version == "1",
            "Invalid exposure history",
        )
        if self.parent_hash is not None:
            sha256(self.parent_hash, "parent exposure history")
        require(
            type(self.events) is tuple
            and all(
                isinstance(e, OuterTrainingExposure) and e.backend == self.backend
                for e in self.events
            ),
            "Typed same-domain outer exposures required",
        )
        require(
            len({e.training_id for e in self.events}) == len(self.events),
            "Duplicate outer training event",
        )
        require(
            type(self.unresolved) is tuple, "Immutable unresolved outer work required"
        )
        for entry in self.unresolved:
            require(
                type(entry) is tuple
                and len(entry) == 2
                and isinstance(entry[1], ArtifactRef),
                "Invalid unresolved outer event",
            )
            text(entry[0], "unresolved training ID")
        require(
            len({e[0] for e in self.unresolved}) == len(self.unresolved),
            "Duplicate unresolved outer training",
        )
        require(
            not {e.training_id for e in self.events}.intersection(
                e[0] for e in self.unresolved
            ),
            "Completed exposure cannot remain unresolved",
        )
        lookup = {}
        for event in self.events:
            for data_id, content in event.members:
                require(
                    data_id not in lookup or lookup[data_id] == content,
                    "Outer history reused an ID for changed content",
                    ErrorCode.USAGE_CONFLICT,
                )
                lookup[data_id] = content
        object.__setattr__(self, "_by_id", MappingProxyType(lookup))
        object.__setattr__(self, "_contents", frozenset(lookup.values()))

    @property
    def history_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "branch_id": self.branch_id,
            "backend": self.backend,
            "parent_hash": self.parent_hash,
            "events": [e.to_dict() for e in self.events],
            "unresolved": [
                {"training_id": name, "receipt": ref.to_dict()}
                for name, ref in self.unresolved
            ],
        }

    def was_used(self, record):
        if record.data_id in self._by_id:
            require(
                self._by_id[record.data_id] == record.content_hash,
                "Current sample changed under an outer-used ID",
                ErrorCode.STALE_CACHE,
            )
        return record.data_id in self._by_id or record.content_hash in self._contents

    def record(self, handoff, outcome):
        from self_curate.orchestration.contracts import OuterHandoff
        from self_curate.training import TrainingOutcome

        require(
            isinstance(handoff, OuterHandoff) and isinstance(outcome, TrainingOutcome),
            "Outer handoff and completed physical training required",
        )
        request = outcome.request
        require(
            request.backend == self.backend == handoff.backend
            and request.attempt_id == handoff.attempt_id
            and request.data.kind == "subset"
            and request.starting_model == handoff.starting_model
            and request.batch_id == handoff.batch.batch_id
            and request.training_data_hash == handoff.batch.training_data_hash,
            "Training is not this outer task-subset exposure",
        )
        by_id = {m.data_id: m.content_hash for m in handoff.batch.members}
        require(
            tuple(x.data_id for x in request.data.examples)
            == handoff.batch.training_order
            and all(
                by_id.get(x.data_id) == x.content_hash for x in request.data.examples
            ),
            "Outer training did not consume the fixed selected manifest",
        )
        event = OuterTrainingExposure(
            request.training_id,
            request.attempt_id,
            handoff.record_hash,
            handoff.batch.batch_id,
            tuple((x.data_id, x.content_hash) for x in request.data.examples),
            request.starting_model.model_hash,
            outcome.result.candidate_model.model_hash,
            outcome.result.processed_supervised_tokens,
            outcome.result.optimizer_steps,
            outcome.record.receipt,
            self.backend,
        )
        old = next((e for e in self.events if e.training_id == event.training_id), None)
        require(
            old is None or old == event,
            "Conflicting physical outer exposure",
            ErrorCode.USAGE_CONFLICT,
        )
        events = self.events if old is not None else (*self.events, event)
        return replace(
            self,
            events=events,
            unresolved=tuple(e for e in self.unresolved if e[0] != event.training_id),
        )

    def mark_unresolved(self, training_id, receipt):
        text(training_id, "uncertain outer training")
        require(isinstance(receipt, ArtifactRef), "Failure evidence required")
        if any(e.training_id == training_id for e in self.events):
            return self
        existing = dict(self.unresolved)
        require(
            training_id not in existing or existing[training_id] == receipt,
            "Conflicting unresolved training evidence",
        )
        existing[training_id] = receipt
        return replace(self, unresolved=tuple(sorted(existing.items())))

    def fork(self, branch_id):
        return replace(self, branch_id=branch_id, parent_hash=self.history_hash)

    def save(self, output):
        output = Path(output).resolve()
        require(
            not output.exists(),
            "Exposure history output already exists",
            ErrorCode.OUTPUT_EXISTS,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = _staging_directory(output)
        write_json(stage / "history.json", self.to_dict())
        write_json(
            stage / "complete.json",
            {
                "history_hash": self.history_hash,
                "file_sha256": file_sha256(stage / "history.json"),
            },
        )
        _publish(stage, output)
        return output

    def persist(self, root):
        output = Path(root) / self.history_hash
        if output.exists():
            require(
                load_outer_exposures(
                    output,
                    expected_hash=self.history_hash,
                    expected_branch=self.branch_id,
                )
                == self,
                "Exposure snapshot changed",
            )
            return output
        return self.save(output)


def parse_outer_exposures(value):
    row = fields(
        value,
        (
            "schema_version",
            "branch_id",
            "backend",
            "parent_hash",
            "events",
            "unresolved",
        ),
    )
    require(
        type(row["events"]) is list and type(row["unresolved"]) is list,
        "Exposure arrays required",
    )
    events = []
    for event in row["events"]:
        event = fields(
            event,
            (
                "schema_version",
                "training_id",
                "attempt_id",
                "handoff_hash",
                "batch_id",
                "members",
                "starting_model_hash",
                "candidate_model_hash",
                "processed_supervised_tokens",
                "optimizer_steps",
                "receipt",
                "backend",
            ),
        )
        require(
            type(event["members"]) is list
            and all(type(v) is list and len(v) == 2 for v in event["members"]),
            "Invalid outer member array",
        )
        events.append(
            OuterTrainingExposure(
                **{
                    **event,
                    "members": tuple(tuple(v) for v in event["members"]),
                    "receipt": artifact(event["receipt"]),
                }
            )
        )
    unresolved = []
    for entry in row["unresolved"]:
        entry = fields(entry, ("training_id", "receipt"))
        unresolved.append((entry["training_id"], artifact(entry["receipt"])))
    return OuterExposureHistory(
        row["branch_id"],
        row["backend"],
        tuple(events),
        tuple(unresolved),
        row["parent_hash"],
        row["schema_version"],
    )


def load_outer_exposures(path, *, expected_hash, expected_branch):
    try:
        root = Path(path).resolve()
        require(
            {p.name for p in root.iterdir()} == {"history.json", "complete.json"},
            "Incomplete exposure snapshot",
        )
        for p in root.iterdir():
            require(
                not p.is_symlink()
                and not (getattr(p.lstat(), "st_file_attributes", 0) & 0x400),
                "Exposure snapshot cannot use links",
            )
        complete = fields(
            load_json((root / "complete.json").read_text(encoding="utf-8")),
            ("history_hash", "file_sha256"),
        )
        require(
            complete["file_sha256"] == file_sha256(root / "history.json"),
            "Exposure file changed",
        )
        value = parse_outer_exposures(
            load_json((root / "history.json").read_text(encoding="utf-8"))
        )
        require(
            value.history_hash == complete["history_hash"] == expected_hash
            and value.branch_id == expected_branch,
            "Wrong exposure branch/version",
        )
        return value
    except ContractError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        IndexError,
    ) as exc:
        raise ContractError(
            ErrorCode.ARTIFACT_MISMATCH,
            "Malformed outer exposure history",
            reason=type(exc).__name__,
        ) from exc
