"""Compute versioned NLL/entropy and enrich immutable M1 records by ID."""

import base64
from dataclasses import replace, fields
from pathlib import Path
from threading import RLock
import uuid

from self_curate.contracts import (
    ArtifactRef,
    ComputeCost,
    ContractError,
    CostEvent,
    CostLedger,
    ErrorCode,
    ModelRef,
    ResourceUsage,
    SignalSpec,
    canonical_json,
    digest,
)
from self_curate.contracts.types import require, integer, sha256
from self_curate.data import MetricView, attach_signals, make_signal, signal_binding
from self_curate.data.pool import file_sha256, write_json
from .backends import ScoringFailure
from .contracts import ForwardRequest, TokenMetricResult, SignalOutcome, ScoringBatch
from .math import aggregate

COMPUTATION_ID = "causal-full-vocab-forward-v1"


def _positions(record, spec):
    binding = signal_binding(record, spec)
    mask = record.tokenization.assistant_mask
    if spec.scope == "skill_evidence":
        annotation = next(a for a in record.annotations.skills if a.skill == spec.skill)
        selected = set(
            p
            for span in annotation.spans
            for p in range(span.start, span.end)
            if mask[p]
        )
        mask = tuple(p in selected for p in range(len(mask)))
    positions = tuple(p for p, value in enumerate(mask) if value)
    require(
        len(positions) == binding.token_count
        and digest(list(mask)) == binding.selection_hash,
        "Scoring position selection disagrees with M1 binding",
    )
    require(
        positions[0] > 0,
        "Selected token at position zero has no causal prediction; provide its preceding context",
        ErrorCode.TOKENIZATION_MISMATCH,
    )
    return binding, positions


class SignalService:
    def __init__(self, *, model, backend, output, max_input_tokens=None):
        require(isinstance(model, ModelRef), "SignalService needs pinned ModelRef")
        require(backend.backend in ("mock", "real"), "Explicit signal backend required")
        if max_input_tokens is not None:
            integer(max_input_tokens, "maximum complete input tokens", 2)
        require(
            backend.backend != "real" or model.tokenizer.backend != "fixture",
            "Fixture model cannot be scored as real",
        )
        self._model, self.backend, self.output = model, backend, Path(output).resolve()
        self.max_input_tokens = max_input_tokens
        self._lock, self._tokens = RLock(), {}
        self._computation_hash = self._implementation()

    @property
    def model_ref(self):
        return self._model

    @property
    def computation_hash(self):
        return self._computation_hash

    def _implementation(self):
        from . import contracts, math

        backend_hash = self.backend.implementation_hash
        sha256(backend_hash, "scoring backend implementation_hash")

        return digest(
            {
                "computation_id": COMPUTATION_ID,
                "sources": {
                    p.name: file_sha256(p)
                    for p in (
                        Path(__file__),
                        Path(contracts.__file__),
                        Path(math.__file__),
                    )
                },
                "backend": backend_hash,
                "domain": self.backend.backend,
                "max_input_tokens": self.max_input_tokens,
            }
        )

    def _fixed(self):
        require(
            self._implementation() == self._computation_hash,
            "Scoring implementation/precision/configuration changed; create a new service",
            ErrorCode.STALE_CACHE,
        )

    def spec(
        self, *, scope="assistant", skill=None, reduction="token_mean", log_base="e"
    ):
        return SignalSpec(
            scope, reduction, log_base, COMPUTATION_ID, self.computation_hash, skill
        )

    def compute(self, record, model_ref, spec):
        """Implements the existing data.SignalProvider protocol."""
        require(
            model_ref == self.model_ref,
            "SignalProvider requested another model checkpoint",
            ErrorCode.STALE_CACHE,
        )
        result = self.score((record,), specs=(spec,)).outcomes[0]
        if result.signal is None:
            raise ContractError(
                ErrorCode.MISSING_SIGNAL, result.reason, data_id=record.data_id
            )
        return result.signal

    def score_provider(self, provider, *, specs=None):
        return self.score(provider.records(), specs=specs)

    def score(self, records, *, specs=None):
        with self._lock:
            return self._score(tuple(records), specs=specs)

    def _score(self, records, *, specs):
        from .io import publish_batch

        self._fixed()
        view = MetricView(
            records
        )  # Validate tokens, masks, labels and all ID joins once.
        specs = (self.spec(),) if specs is None else tuple(specs)
        require(
            bool(specs)
            and all(
                isinstance(s, SignalSpec)
                and s.computation_id == COMPUTATION_ID
                and s.computation_hash == self.computation_hash
                for s in specs
            ),
            "Use this service's versioned SignalSpec",
            ErrorCode.STALE_CACHE,
        )
        require(
            len({s.spec_hash for s in specs}) == len(specs),
            "Duplicate signal specifications",
        )
        plans = {}
        for record in records:
            require(
                record.tokenization.identity == self.model_ref.tokenizer,
                "Data tokenizer/template differs from scoring model",
                ErrorCode.STALE_CACHE,
            )
            require(
                self.backend.backend != "real" or not record.known_mock_artifacts,
                "Known mock data cannot be scored as real",
            )
            require(
                self.max_input_tokens is None
                or len(record.tokenization.input_ids) <= self.max_input_tokens,
                "Complete input exceeds scoring budget; truncation is forbidden",
            )
            row = []
            for spec in specs:
                try:
                    binding, positions = _positions(record, spec)
                    row.append((spec, binding, positions, None))
                except ContractError as exc:
                    if exc.code != ErrorCode.MISSING_SIGNAL:
                        raise
                    row.append((spec, None, (), str(exc)))
            plans[record.data_id] = row
        self.output.mkdir(parents=True, exist_ok=True)
        label = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")
        folder = self.output / ("score-" + label)
        folder.mkdir(exist_ok=False)
        (folder / "inputs.jsonl").write_text(
            "".join(canonical_json(r.to_dict()) + "\n" for r in records),
            encoding="utf-8",
            newline="\n",
        )
        write_json(
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
                "backend": self.backend.backend,
            },
        )
        updated, outcomes, events = dict((r.data_id, r) for r in records), [], []
        used_tokens = []
        try:
            for record in records:
                plan = plans[record.data_id]
                direct = {}
                for spec, binding, _, reason in plan:
                    if reason is None:
                        prior = next(
                            (
                                s
                                for s in record.signals
                                if s.binding == binding
                                and s.model_ref == self.model_ref
                                and s.spec == spec
                                and s.nll is not None
                                and s.entropy is not None
                            ),
                            None,
                        )
                        if prior is not None:
                            direct[spec.spec_hash] = replace(
                                prior,
                                nll_percentile=None,
                                entropy_percentile=None,
                                percentile_reference=None,
                            )
                key = digest(
                    (
                        record.data_id,
                        record.content_hash,
                        digest(record.tokenization.to_dict()),
                        self.model_ref.to_dict(),
                        self.computation_hash,
                    )
                )
                cached = self._tokens.get(key, {})
                wanted = {
                    p
                    for spec, _, positions, reason in plan
                    if reason is None and spec.spec_hash not in direct
                    for p in positions
                }
                missing = tuple(sorted(wanted - set(cached)))
                if missing:
                    request = ForwardRequest(
                        "forward-" + label + "-" + str(len(events)),
                        record.data_id,
                        record.content_hash,
                        digest(record.tokenization.to_dict()),
                        self.model_ref,
                        record.tokenization.input_ids,
                        missing,
                        self.computation_hash,
                        self.backend.backend,
                    )
                    work = folder / ("fwd-" + str(len(events)))
                    work.mkdir()
                    write_json(work / "request.json", request.to_dict())
                    usage, response = ResourceUsage(), None
                    try:
                        response = self.backend.score(request, work / "computation")
                        if isinstance(getattr(response, "usage", None), ResourceUsage):
                            usage = response.usage
                        require(
                            isinstance(response, TokenMetricResult),
                            "Backend must return TokenMetricResult",
                        )
                        require(
                            response.request_hash == request.record_hash
                            and response.positions == request.positions
                            and response.backend == request.backend,
                            "Scoring result changed request/model/data/position order",
                            ErrorCode.ARTIFACT_MISMATCH,
                        )
                        require(
                            all(
                                getattr(usage, name) in (None, 0)
                                for name in (
                                    "new_gradient_samples",
                                    "new_gradient_tokens",
                                    "training_supervised_tokens",
                                )
                            ),
                            "Forward scoring backend reported gradient or training work",
                        )
                        self._fixed()
                        write_json(work / "result.json", response.to_dict())
                    except Exception as exc:
                        if isinstance(getattr(exc, "usage", None), ResourceUsage):
                            usage = exc.usage
                        error = {"type": type(exc).__name__, "message": str(exc)}
                        events.append(
                            CostEvent(
                                "score-" + request.record_hash,
                                label,
                                "scoring",
                                record.data_id,
                                self.model_ref.model_hash,
                                "failed",
                                usage,
                                ArtifactRef(
                                    "record://scoring-failure/" + digest(error),
                                    digest(error),
                                ),
                                self.backend.backend,
                            )
                        )
                        raise
                    events.append(
                        CostEvent(
                            "score-" + request.record_hash,
                            label,
                            "scoring",
                            record.data_id,
                            self.model_ref.model_hash,
                            "completed",
                            usage,
                            ArtifactRef(
                                "record://token-statistics/" + response.record_hash,
                                response.record_hash,
                            ),
                            self.backend.backend,
                        )
                    )
                    cached = {
                        **cached,
                        **{
                            p: (n, h, response.evidence)
                            for p, n, h in zip(
                                response.positions,
                                response.token_nll,
                                response.token_entropy,
                            )
                        },
                    }
                    self._tokens[key] = cached
                positions_used = tuple(sorted(wanted))
                if positions_used:
                    used_tokens.append(
                        {
                            "data_id": record.data_id,
                            "positions": positions_used,
                            "token_nll": [cached[p][0] for p in positions_used],
                            "token_entropy": [cached[p][1] for p in positions_used],
                            "evidence": [
                                cached[p][2].to_dict() for p in positions_used
                            ],
                        }
                    )
                additions = []
                row_outcomes = []
                for spec, _, positions, reason in plan:
                    if reason is not None:
                        row_outcomes.append(
                            SignalOutcome(record.data_id, spec, "missing", None, reason)
                        )
                    elif spec.spec_hash in direct:
                        signal = direct[spec.spec_hash]
                        additions.append(signal)
                        row_outcomes.append(
                            SignalOutcome(record.data_id, spec, "cached", signal)
                        )
                    else:
                        refs = tuple(
                            sorted(
                                {cached[p][2] for p in positions},
                                key=lambda a: (a.uri, a.sha256),
                            )
                        )
                        signal = make_signal(
                            record,
                            self.model_ref,
                            spec,
                            provenance=refs,
                            nll=aggregate(tuple(cached[p][0] for p in positions), spec),
                            entropy=aggregate(
                                tuple(cached[p][1] for p in positions), spec
                            ),
                        )
                        additions.append(signal)
                        row_outcomes.append(
                            SignalOutcome(
                                record.data_id,
                                spec,
                                (
                                    "computed"
                                    if set(positions).intersection(missing)
                                    else "cached"
                                ),
                                signal,
                            )
                        )
                updated[record.data_id] = attach_signals(record, additions)
                # attach_signals may merge provenance with an existing complementary record.
                for item in row_outcomes:
                    if item.signal is not None:
                        attached = next(
                            s
                            for s in updated[record.data_id].signals
                            if s.cache_key == item.signal.cache_key
                        )
                        item = replace(item, signal=attached)
                    outcomes.append(item)
            batch = ScoringBatch(
                label,
                self.model_ref,
                self.computation_hash,
                tuple(updated[i] for i in view.data_ids),
                tuple(outcomes),
                CostLedger("signals-" + label, tuple(events), self.backend.backend),
                self.backend.backend,
            )
            write_json(
                folder / "token-statistics.json",
                {
                    "schema_version": "1",
                    "model_hash": self.model_ref.model_hash,
                    "computation_hash": self.computation_hash,
                    "records": used_tokens,
                },
            )
            publish_batch(batch, folder)
            return batch
        except Exception as exc:
            ledger = CostLedger("signals-" + label, tuple(events), self.backend.backend)
            try:
                partial = folder / "partial-records.jsonl"
                partial.write_text(
                    "".join(
                        canonical_json(updated[i].to_dict()) + "\n"
                        for i in view.data_ids
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                write_json(
                    folder / "failure.json",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "outcomes": [o.to_dict() for o in outcomes],
                        "ledger": ledger.to_dict(),
                        "partial_records": {
                            "uri": partial.name,
                            "sha256": file_sha256(partial),
                        },
                        "updated_records": [
                            {"data_id": r.data_id, "record_hash": r.record_hash}
                            for r in updated.values()
                        ],
                    },
                )
            except OSError:
                pass
            # Unknown totals remain unknown; the detailed ledger retains known subtotals.
            summary = ledger.summary()
            usage = ResourceUsage(
                ComputeCost(
                    summary["wall_seconds"]["total"],
                    summary["gpu_seconds"]["total"],
                    summary["processed_tokens"]["total"],
                ),
                **{
                    f.name: summary[f.name]["total"]
                    for f in fields(ResourceUsage)
                    if f.name not in ("compute", "schema_version")
                },
            )
            raise ScoringFailure(
                str(exc),
                usage=usage,
                ledger=ledger,
                records=tuple(updated[i] for i in view.data_ids),
                outcomes=outcomes,
            ) from exc
