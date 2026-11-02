"""Selection receipts passed unchanged to valuation and candidate training."""

from dataclasses import asdict, dataclass
import math

from .errors import ErrorCode
from .types import Record, digest, integer, require, sha256, text


@dataclass(frozen=True)
class BatchMember(Record):
    data_id: str
    content_hash: str
    record_hash: str
    tokenization_hash: str
    supervised_tokens: int
    branch: str

    def __post_init__(self):
        text(self.data_id, "data_id")
        text(self.branch, "branch")
        for field in ("content_hash", "record_hash", "tokenization_hash"):
            sha256(getattr(self, field), field)
        integer(self.supervised_tokens, "supervised_tokens", 1)


@dataclass(frozen=True)
class BatchManifest(Record):
    program_hash: str
    execution_hash: str
    snapshot_hash: str
    policy_hash: str
    members: tuple[BatchMember, ...]
    selection_order: tuple[str, ...]
    training_order: tuple[str, ...]
    training_seed: int
    branch_shares: tuple[tuple[str, float], ...] = ()
    model_hash: str | None = None
    synthetic: bool | None = None
    schema_version: str = "1"

    def __post_init__(self):
        for field in ("program_hash", "execution_hash", "snapshot_hash", "policy_hash"):
            sha256(getattr(self, field), field)
        require(
            type(self.members) is tuple
            and bool(self.members)
            and all(isinstance(m, BatchMember) for m in self.members),
            "Batch needs members",
        )
        ids = tuple(m.data_id for m in self.members)
        require(
            ids == tuple(sorted(set(ids))), "Batch members must be unique and sorted"
        )
        for order in (self.selection_order, self.training_order):
            require(
                type(order) is tuple
                and all(isinstance(i, str) for i in order)
                and len(order) == len(ids)
                and set(order) == set(ids),
                "Batch order must contain each exact member once",
            )
        integer(self.training_seed, "training_seed")
        require(type(self.branch_shares) is tuple, "Invalid branch shares")
        if self.branch_shares:
            require(
                all(
                    type(item) is tuple and len(item) == 2
                    for item in self.branch_shares
                ),
                "Invalid share entry",
            )
            names = []
            for name, share in self.branch_shares:
                text(name, "branch")
                require(
                    type(share) in (int, float)
                    and 0 < share <= 1
                    and math.isfinite(share),
                    "Invalid branch share",
                )
                names.append(name)
            require(
                names == sorted(set(names))
                and abs(sum(s for _, s in self.branch_shares) - 1) <= 1e-9,
                "Branch shares must be unique, sorted and sum to 1",
            )
            require(
                set(names) == {m.branch for m in self.members},
                "Every allocated branch must have members",
            )
        else:
            require(
                len({m.branch for m in self.members}) == 1,
                "Unmixed batch must have one branch",
            )
        if self.model_hash is not None:
            sha256(self.model_hash, "model_hash")
        require(
            self.synthetic is None or type(self.synthetic) is bool,
            "Invalid synthetic marker",
        )
        require(self.schema_version == "1", "Unsupported BatchManifest schema")

    @property
    def data_ids(self):
        return tuple(m.data_id for m in self.members)

    @property
    def supervised_tokens(self):
        return sum(m.supervised_tokens for m in self.members)

    @property
    def actual_branch_tokens(self):
        result = {name: 0 for name, _ in self.branch_shares}
        for member in self.members:
            result[member.branch] = (
                result.get(member.branch, 0) + member.supervised_tokens
            )
        return result

    @property
    def actual_branch_shares(self):
        return {
            name: tokens / self.supervised_tokens
            for name, tokens in self.actual_branch_tokens.items()
        }

    @property
    def batch_id(self):
        return digest(asdict(self))

    @property
    def training_data_hash(self):
        return digest(
            {
                "rows": [
                    (m.data_id, m.content_hash, m.tokenization_hash)
                    for m in self.members
                ],
                "order": self.training_order,
            }
        )

    def to_dict(self):
        return {
            **asdict(self),
            "batch_id": self.batch_id,
            "supervised_tokens": self.supervised_tokens,
            "actual_branch_tokens": self.actual_branch_tokens,
            "actual_branch_shares": self.actual_branch_shares,
            "training_data_hash": self.training_data_hash,
        }


@dataclass(frozen=True)
class NodeTrace(Record):
    node_id: str
    operator: str
    inputs: tuple[tuple[str, tuple[str, ...]], ...]
    output_ids: tuple[str, ...]
    exclusions: tuple[tuple[str, str, str | None], ...]
    details_json: str

    @property
    def details(self):
        from .parsing import load_json

        return load_json(self.details_json)

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "operator": self.operator,
            "inputs": self.inputs,
            "output_ids": self.output_ids,
            "exclusions": self.exclusions,
            "details": self.details,
        }


@dataclass(frozen=True)
class SelectionResult(Record):
    status: str
    program_hash: str
    execution_hash: str
    batch: BatchManifest | None
    traces: tuple[NodeTrace, ...]
    reason: str | None = None
    diagnostics_json: str = "{}"
    schema_version: str = "1"

    def __post_init__(self):
        require(self.status in ("feasible", "infeasible"), "Invalid selection status")
        require(
            (self.batch is not None) == (self.status == "feasible"),
            "Only feasible selection has a batch",
        )
        sha256(self.program_hash, "program_hash")
        sha256(self.execution_hash, "execution_hash")
        require(
            type(self.traces) is tuple
            and all(isinstance(t, NodeTrace) for t in self.traces),
            "Invalid execution trace",
        )
        if self.batch is not None:
            require(
                self.batch.program_hash == self.program_hash
                and self.batch.execution_hash == self.execution_hash,
                "Batch receipt belongs to another execution",
            )
        else:
            text(self.reason, "infeasibility reason")
        require(self.schema_version == "1", "Unsupported selection schema")

    def to_dict(self):
        from .parsing import load_json

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "program_hash": self.program_hash,
            "execution_hash": self.execution_hash,
            "batch": None if self.batch is None else self.batch.to_dict(),
            "traces": [
                {**t.to_dict(), "details": load_json(t.details_json)}
                for t in self.traces
            ],
            "reason": self.reason,
            "diagnostics": load_json(self.diagnostics_json),
        }


def verify_batch(manifest: BatchManifest, view, *, model_ref=None):
    """Verify exact training inputs; later feature enrichment is allowed."""
    require(isinstance(manifest, BatchManifest), "Expected BatchManifest")
    if model_ref is not None:
        from .signals import ModelRef

        require(isinstance(model_ref, ModelRef), "Expected ModelRef")
        require(
            manifest.model_hash == model_ref.model_hash,
            "Batch was selected for another model checkpoint",
            ErrorCode.STALE_CACHE,
        )
    for member in manifest.members:
        record = view.record(member.data_id)
        require(
            record.content_hash == member.content_hash
            and digest(record.tokenization.to_dict()) == member.tokenization_hash
            and record.stats.supervised_tokens == member.supervised_tokens,
            "Selected training data changed",
            ErrorCode.ARTIFACT_MISMATCH,
            data_id=member.data_id,
        )
    return manifest.training_order


def parse_batch_manifest(value):
    from .parsing import fields, sequence

    row = fields(
        value,
        (
            "schema_version",
            "program_hash",
            "execution_hash",
            "snapshot_hash",
            "policy_hash",
            "members",
            "selection_order",
            "training_order",
            "training_seed",
            "branch_shares",
            "model_hash",
            "synthetic",
            "batch_id",
            "supervised_tokens",
            "actual_branch_tokens",
            "actual_branch_shares",
            "training_data_hash",
        ),
    )
    members = tuple(
        BatchMember(
            **fields(
                m,
                (
                    "data_id",
                    "content_hash",
                    "record_hash",
                    "tokenization_hash",
                    "supervised_tokens",
                    "branch",
                ),
            )
        )
        for m in sequence(row["members"], "members")
    )
    shares = []
    for pair in sequence(row["branch_shares"], "branch_shares"):
        require(type(pair) is list and len(pair) == 2, "Invalid branch share pair")
        shares.append(tuple(pair))
    original = {
        k: row[k]
        for k in (
            "program_hash",
            "execution_hash",
            "snapshot_hash",
            "policy_hash",
            "training_seed",
            "model_hash",
            "synthetic",
            "schema_version",
        )
    }
    manifest = BatchManifest(
        **original,
        members=members,
        selection_order=tuple(sequence(row["selection_order"], "selection_order")),
        training_order=tuple(sequence(row["training_order"], "training_order")),
        branch_shares=tuple(shares),
    )
    computed = manifest.to_dict()
    for field in (
        "batch_id",
        "supervised_tokens",
        "actual_branch_tokens",
        "actual_branch_shares",
        "training_data_hash",
    ):
        require(
            row[field] == computed[field],
            "Batch checksum or derived totals mismatch",
            ErrorCode.ARTIFACT_MISMATCH,
            field=field,
        )
    return manifest
