"""Frozen system controls and externally supplied model/semantic contexts."""

from dataclasses import dataclass, field
from fractions import Fraction
import math
from pathlib import Path
from types import MappingProxyType

from self_curate.contracts import (
    ArtifactRef,
    ContractError,
    ErrorCode,
    ModelRef,
    PercentileReference,
    SignalSpec,
    digest,
)
from self_curate.contracts.parsing import (
    artifact,
    fields,
    load_json,
    model_ref,
    percentile_reference,
    sequence,
    signal_spec,
)
from self_curate.contracts.program import OPERATOR_VERSION
from self_curate.contracts.types import Record, integer, require, text, sha256
from self_curate.data import MetricView


def _unique(values, name):
    require(type(values) is tuple, f"{name} must be tuple")
    for value in values:
        text(value, name)
    require(len(set(values)) == len(values), f"Duplicate {name}")


@dataclass(frozen=True)
class QualityPolicy(Record):
    allow_unknown: bool = False
    exclude_accepted: bool = True
    rejected_policy: str = "exclude"
    excluded_ids: tuple[str, ...] = ()
    excluded_families: tuple[str, ...] = ()

    def __post_init__(self):
        require(
            type(self.allow_unknown) is bool and type(self.exclude_accepted) is bool,
            "Quality flags must be boolean",
        )
        require(self.rejected_policy in ("allow", "exclude"), "Invalid rejected_policy")
        for name in ("excluded_ids", "excluded_families"):
            _unique(getattr(self, name), name)
            object.__setattr__(self, name, tuple(sorted(getattr(self, name))))


@dataclass(frozen=True)
class BudgetPolicy(Record):
    token_target: int
    token_tolerance: float
    effective_batch: int
    share_tolerance: float = 0.1
    max_samples: int | None = None
    deduplicate_content: bool = True
    algorithm: str = "ordered-greedy-v1"

    def __post_init__(self):
        integer(self.token_target, "token_target", 1)
        require(
            self.token_target <= 2**53 - 1,
            "token_target exceeds safe JSON integer range",
        )
        integer(self.effective_batch, "effective_batch", 1)
        for name in ("token_tolerance", "share_tolerance"):
            value = getattr(self, name)
            require(
                type(value) in (int, float)
                and 0 <= value <= 1
                and math.isfinite(value),
                f"Invalid {name}",
            )
        require(self.token_tolerance < 1, "token_tolerance must be below 1")
        if self.max_samples is not None:
            integer(self.max_samples, "max_samples", self.effective_batch)
        require(
            type(self.deduplicate_content) is bool,
            "deduplicate_content must be boolean",
        )
        require(self.algorithm == "ordered-greedy-v1", "Unsupported packer algorithm")

    @property
    def minimum(self):
        return max(
            1, math.ceil(self.token_target * (1 - Fraction(str(self.token_tolerance))))
        )

    @property
    def maximum(self):
        return math.floor(self.token_target * (1 + Fraction(str(self.token_tolerance))))

    def branch_window(self, share):
        target = self.token_target * Fraction(str(share))
        margin = self.token_target * Fraction(str(self.share_tolerance))
        return target, max(0, math.ceil(target - margin)), math.floor(target + margin)


@dataclass(frozen=True)
class SignalContext(Record):
    spec: SignalSpec
    references: tuple[PercentileReference, ...] = ()

    def __post_init__(self):
        require(isinstance(self.spec, SignalSpec), "Invalid signal specification")
        require(
            type(self.references) is tuple
            and all(isinstance(r, PercentileReference) for r in self.references),
            "Invalid CDF references",
        )
        require(
            len({r.reference_group for r in self.references}) == len(self.references),
            "Ambiguous CDF group",
        )
        require(
            all(r.spec_hash == self.spec.spec_hash for r in self.references),
            "CDF and signal spec mismatch",
            ErrorCode.STALE_CACHE,
        )


@dataclass(frozen=True)
class SimilarityPair(Record):
    left_id: str
    right_id: str
    left_content_hash: str
    right_content_hash: str
    value: float

    def __post_init__(self):
        text(self.left_id, "left_id")
        text(self.right_id, "right_id")
        require(self.left_id != self.right_id, "Self similarity is not needed")
        sha256(self.left_content_hash, "left_content_hash")
        sha256(self.right_content_hash, "right_content_hash")
        require(
            type(self.value) in (int, float)
            and -1 <= self.value <= 1
            and math.isfinite(self.value),
            "Similarity must be finite in [-1,1]",
        )


@dataclass(frozen=True)
class SimilarityTable(Record):
    identity: ArtifactRef
    pairs: tuple[SimilarityPair, ...]
    _lookup: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        require(
            isinstance(self.identity, ArtifactRef),
            "Semantic table needs provenance identity",
        )
        require(
            type(self.pairs) is tuple
            and all(isinstance(p, SimilarityPair) for p in self.pairs),
            "Invalid pairwise similarities",
        )
        lookup = {}
        for pair in self.pairs:
            key = tuple(sorted((pair.left_id, pair.right_id)))
            require(key not in lookup, "Duplicate semantic pair")
            lookup[key] = pair
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    def lookup(self, left, right):
        return self.lookup_bound(
            left.data_id, left.content_hash, right.data_id, right.content_hash
        )

    def lookup_bound(self, left_id, left_content_hash, right_id, right_content_hash):
        pair = self._lookup.get(tuple(sorted((left_id, right_id))))
        if pair is None:
            return None
        hashes = {
            pair.left_id: pair.left_content_hash,
            pair.right_id: pair.right_content_hash,
        }
        require(
            hashes[left_id] == left_content_hash
            and hashes[right_id] == right_content_hash,
            "Semantic scores belong to another data version",
            ErrorCode.STALE_CACHE,
        )
        return pair.value

    def to_dict(self):
        return {
            "identity": self.identity.to_dict(),
            "pairs": [
                p.to_dict()
                for p in sorted(
                    self.pairs, key=lambda p: tuple(sorted((p.left_id, p.right_id)))
                )
            ],
        }


@dataclass(frozen=True)
class ExecutionContext:
    view: MetricView
    quality: QualityPolicy
    budget: BudgetPolicy
    model_ref: ModelRef | None = None
    signals: tuple[SignalContext, ...] = ()
    similarities: SimilarityTable | None = None
    training_seed: int = 0
    training_order: str = "seeded"
    max_nodes: int = 128
    _proofs: object = field(init=False, repr=False, compare=False)
    snapshot_hash: str = field(init=False)
    policy_hash: str = field(init=False)
    context_hash: str = field(init=False)

    def __post_init__(self):
        require(
            isinstance(self.view, MetricView)
            and isinstance(self.quality, QualityPolicy)
            and isinstance(self.budget, BudgetPolicy),
            "Invalid execution context",
        )
        require(
            self.model_ref is None or isinstance(self.model_ref, ModelRef),
            "Invalid active model",
        )
        require(
            type(self.signals) is tuple
            and all(isinstance(s, SignalContext) for s in self.signals),
            "Invalid signal contexts",
        )
        require(
            len({(s.spec.scope, s.spec.skill) for s in self.signals})
            == len(self.signals),
            "Ambiguous signal definition for scope",
        )
        if self.signals:
            require(self.model_ref is not None, "Signal contexts require active model")
        for signal in self.signals:
            require(
                all(
                    r.model_hash == self.model_ref.model_hash for r in signal.references
                ),
                "CDF belongs to another model",
                ErrorCode.STALE_CACHE,
            )
        require(
            self.similarities is None or isinstance(self.similarities, SimilarityTable),
            "Invalid semantic context",
        )
        integer(self.training_seed, "training_seed")
        integer(self.max_nodes, "max_nodes", 1)
        require(
            self.max_nodes <= 256 and self.training_order in ("seeded", "selection"),
            "Invalid node/order policy",
        )
        proofs = {}
        pool_tokenizer = None
        for data_id in self.view.data_ids:
            row = self.view.record(data_id)
            if pool_tokenizer is None:
                pool_tokenizer = row.tokenization.identity
            require(
                row.tokenization.identity == pool_tokenizer,
                "A selection context cannot mix tokenizer/template identities",
                ErrorCode.TOKENIZATION_MISMATCH,
            )
            if self.model_ref is not None:
                require(
                    row.tokenization.identity == self.model_ref.tokenizer,
                    "Active model tokenizer differs from pool",
                    ErrorCode.STALE_CACHE,
                )
            proofs[data_id] = (
                row.content_hash,
                row.record_hash,
                digest(row.tokenization.to_dict()),
                row.stats.supervised_tokens,
            )
        object.__setattr__(self, "_proofs", MappingProxyType(proofs))
        object.__setattr__(self, "snapshot_hash", digest(sorted(proofs.items())))
        object.__setattr__(
            self,
            "policy_hash",
            digest(
                {
                    "quality": self.quality.to_dict(),
                    "budget": self.budget.to_dict(),
                    "training_seed": self.training_seed,
                    "training_order": self.training_order,
                    "operator_version": OPERATOR_VERSION,
                }
            ),
        )
        object.__setattr__(self, "context_hash", digest(self.to_dict()))

    def to_dict(self):
        return {
            "schema_version": "1",
            "snapshot_hash": self.snapshot_hash,
            "quality_gate": self.quality.to_dict(),
            "budget_pack": self.budget.to_dict(),
            "model_ref": None if self.model_ref is None else self.model_ref.to_dict(),
            "signal_contexts": [s.to_dict() for s in self.signals],
            "similarities": (
                None if self.similarities is None else self.similarities.to_dict()
            ),
            "training_seed": self.training_seed,
            "training_order": self.training_order,
            "max_nodes": self.max_nodes,
            "operator_version": OPERATOR_VERSION,
        }

    def signal_context(self, scope, skill, reference_group):
        match = next(
            (s for s in self.signals if (s.spec.scope, s.spec.skill) == (scope, skill)),
            None,
        )
        if match is None or self.model_ref is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL,
                "No fixed signal backend context for requested scope",
                scope=scope,
                skill=skill,
            )
        reference = None
        if reference_group is not None:
            reference = next(
                (r for r in match.references if r.reference_group == reference_group),
                None,
            )
            require(
                reference is not None,
                "Reference group is not registered in frozen context",
                ErrorCode.INVALID_PROGRAM,
                reference_group=reference_group,
            )
        return match.spec, reference


def parse_context(value, view):
    try:
        fields(
            value,
            ("schema_version", "quality_gate", "budget_pack"),
            (
                "model_ref",
                "signal_contexts",
                "similarities",
                "training_seed",
                "training_order",
                "max_nodes",
                "snapshot_hash",
                "operator_version",
            ),
        )
        require(value["schema_version"] == "1", "Unsupported context schema")
        require(
            value.get("operator_version", OPERATOR_VERSION) == OPERATOR_VERSION,
            "Context uses another operator version",
            ErrorCode.STALE_CACHE,
        )
        q = fields(
            value["quality_gate"],
            (),
            (
                "allow_unknown",
                "exclude_accepted",
                "rejected_policy",
                "excluded_ids",
                "excluded_families",
            ),
        )
        q = {
            **q,
            "excluded_ids": tuple(sequence(q.get("excluded_ids", []), "excluded_ids")),
            "excluded_families": tuple(
                sequence(q.get("excluded_families", []), "excluded_families")
            ),
        }
        b = fields(
            value["budget_pack"],
            ("token_target", "token_tolerance", "effective_batch"),
            ("share_tolerance", "max_samples", "deduplicate_content", "algorithm"),
        )
        contexts = []
        for item in sequence(value.get("signal_contexts", []), "signal_contexts"):
            fields(item, ("spec",), ("references",))
            contexts.append(
                SignalContext(
                    signal_spec(item["spec"]),
                    tuple(
                        percentile_reference(r)
                        for r in sequence(item.get("references", []), "references")
                    ),
                )
            )
        semantic = value.get("similarities")
        if semantic is not None:
            fields(semantic, ("identity", "pairs"))
            pairs = tuple(
                SimilarityPair(
                    **fields(
                        p,
                        (
                            "left_id",
                            "right_id",
                            "left_content_hash",
                            "right_content_hash",
                            "value",
                        ),
                    )
                )
                for p in sequence(semantic["pairs"], "pairs")
            )
            semantic = SimilarityTable(artifact(semantic["identity"]), pairs)
        context = ExecutionContext(
            view,
            QualityPolicy(**q),
            BudgetPolicy(**b),
            None if value.get("model_ref") is None else model_ref(value["model_ref"]),
            tuple(contexts),
            semantic,
            value.get("training_seed", 0),
            value.get("training_order", "seeded"),
            value.get("max_nodes", 128),
        )
        if "snapshot_hash" in value:
            require(
                value["snapshot_hash"] == context.snapshot_hash,
                "Frozen context belongs to another pool snapshot",
                ErrorCode.STALE_CACHE,
            )
        return context
    except ContractError as exc:
        if exc.code == ErrorCode.STALE_CACHE:
            raise
        raise ContractError(ErrorCode.INVALID_CONFIG, str(exc), **exc.details) from exc


def load_context(path, view):
    return parse_context(load_json(Path(path).read_text(encoding="utf-8")), view)
