"""Strict, checksummed signal handoff and normalized M1 record input/output."""

from dataclasses import fields as dataclass_fields, replace
from pathlib import Path, PurePosixPath
from functools import wraps

from self_curate.contracts import (
    TrajectoryRecord,
    canonical_json,
    parse_contract,
    digest,
    ContractError,
    ErrorCode,
)
from self_curate.contracts.parsing import (
    fields,
    load_json,
    model_ref,
    signal_record,
    signal_spec,
    tokenization,
    tokenizer_ref,
    artifact,
)
from self_curate.contracts.types import require, integer
from self_curate.contracts.workflow import finite
from self_curate.contracts.workflow_parsing import CONTRACT_TYPES, _decode
from self_curate.data import normalize_trajectory, attach_signals
from self_curate.data.pool import file_sha256, write_json
from .contracts import ForwardRequest, TokenMetricResult, SignalOutcome, ScoringBatch
from .math import aggregate


def _structured_input(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ContractError:
            raise
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            IndexError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise ContractError(
                ErrorCode.INVALID_RECORD,
                "Malformed signal input or handoff",
                reason=type(exc).__name__,
            ) from exc

    return wrapped


@_structured_input
def parse_forward_contract(kind, value):
    registry = {
        **CONTRACT_TYPES,
        "ForwardRequest": ForwardRequest,
        "TokenMetricResult": TokenMetricResult,
    }
    require(kind in ("ForwardRequest", "TokenMetricResult"), "Unknown scoring contract")
    return _decode(registry[kind], value, kind, registry=registry)


@_structured_input
def parse_scoring_record(row):
    allowed = tuple(f.name for f in dataclass_fields(TrajectoryRecord) if f.init)
    fields(
        row,
        (
            "schema_version",
            "data_id",
            "source",
            "messages",
            "quality",
            "tokenization",
            "usage",
            "evidence_metrics",
        ),
        tuple(
            n
            for n in allowed
            if n
            not in (
                "schema_version",
                "data_id",
                "source",
                "messages",
                "quality",
                "tokenization",
                "usage",
                "evidence_metrics",
            )
        ),
    )

    class StoredTokens:
        identity = tokenizer_ref(row["tokenization"]["identity"])

        def encode(self, messages, provided=None):
            return tokenization(provided)

    raw = {
        k: row[k]
        for k in (
            "schema_version",
            "data_id",
            "source",
            "messages",
            "quality",
            "tokenization",
            "usage",
        )
    }
    for name in (
        "annotations",
        "signals",
        "metadata",
        "gradients",
        "annotation_record",
        "metadata_record",
    ):
        if name in row:
            raw[name] = row[name]
    record = normalize_trajectory(
        raw, StoredTokens(), tuple(v["skill"] for v in row["evidence_metrics"])
    )
    if row.get("normalization_version") in ("m1-v1", "m1-v2"):
        record = replace(record, normalization_version=row["normalization_version"])
    require(
        canonical_json(record.to_dict()) == canonical_json(row),
        "Stored M1 record labels/counts do not match its inputs",
    )
    return record


@_structured_input
def read_records(path):
    """Input interface for caller-provided, already normalized M1 JSONL records."""
    return tuple(
        parse_scoring_record(load_json(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )


def publish_batch(batch, folder):
    folder = Path(folder)
    require(
        not (folder / "checksums.json").exists(),
        "Scoring output is already complete",
        ErrorCode.OUTPUT_EXISTS,
    )
    (folder / "records.jsonl").write_text(
        "".join(canonical_json(r.to_dict()) + "\n" for r in batch.records),
        encoding="utf-8",
        newline="\n",
    )
    write_json(folder / "result.json", batch.to_dict())
    write_json(
        folder / "checksums.json",
        {
            p.relative_to(folder).as_posix(): file_sha256(p)
            for p in folder.rglob("*")
            if p.is_file()
        },
    )


@_structured_input
def load_scoring_batch(folder, *, expected_model, expected_computation_hash):
    folder = Path(folder).resolve()
    checks = load_json((folder / "checksums.json").read_text(encoding="utf-8"))
    require(
        type(checks) is dict
        and {"request.json", "records.jsonl", "result.json"}.issubset(checks),
        "Incomplete scoring handoff",
    )
    actual = {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and p != folder / "checksums.json"
    }
    require(set(checks) == actual, "Scoring handoff file set changed")
    for name, sha in checks.items():
        relative = PurePosixPath(name)
        file = folder / name
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and file.resolve().is_relative_to(folder)
            and not file.is_symlink()
            and file_sha256(file) == sha,
            "Scoring artifact checksum/path mismatch",
        )
    row = fields(
        load_json((folder / "result.json").read_text(encoding="utf-8")),
        (
            "schema_version",
            "batch_id",
            "model_ref",
            "computation_hash",
            "backend",
            "records",
            "outcomes",
            "ledger",
        ),
    )
    require(
        row["schema_version"] == "1"
        and model_ref(row["model_ref"]) == expected_model
        and row["computation_hash"] == expected_computation_hash,
        "Scoring handoff model/computation version mismatch",
    )
    require(type(row["outcomes"]) is list, "Outcomes must be an array")
    outcomes = []
    for value in row["outcomes"]:
        value = fields(value, ("data_id", "spec", "status", "signal", "reason"))
        outcomes.append(
            SignalOutcome(
                value["data_id"],
                signal_spec(value["spec"]),
                value["status"],
                None if value["signal"] is None else signal_record(value["signal"]),
                value["reason"],
            )
        )
    batch = ScoringBatch(
        row["batch_id"],
        expected_model,
        expected_computation_hash,
        read_records(folder / "records.jsonl"),
        tuple(outcomes),
        parse_contract("CostLedger", row["ledger"]),
        row["backend"],
    )
    require(
        canonical_json(batch.to_dict()) == canonical_json(row),
        "Scoring metadata/record identities changed",
    )
    requested = fields(
        load_json((folder / "request.json").read_text(encoding="utf-8")),
        (
            "schema_version",
            "model_ref",
            "computation_hash",
            "specs",
            "records",
            "backend",
        ),
        ("input_records_file", "token_statistics_file"),
    )
    require(
        requested["schema_version"] in ("1", "2")
        and model_ref(requested["model_ref"]) == expected_model
        and requested["computation_hash"] == expected_computation_hash
        and requested["backend"] == batch.backend,
        "Scoring request/result conditions changed",
    )
    require(
        type(requested["specs"]) is list
        and bool(requested["specs"])
        and type(requested["records"]) is list,
        "Scoring input IDs/specs must be arrays",
    )
    specs = tuple(signal_spec(s) for s in requested["specs"])
    from .service import COMPUTATION_ID, _positions

    require(
        len({s.spec_hash for s in specs}) == len(specs)
        and all(
            s.computation_id == COMPUTATION_ID
            and s.computation_hash == expected_computation_hash
            for s in specs
        ),
        "Duplicate or foreign requested signal scopes",
    )
    require(
        [(o.data_id, o.spec) for o in batch.outcomes]
        == [(r.data_id, s) for r in batch.records for s in specs],
        "Scoring outcome coverage/order differs from requested inputs/scopes",
    )
    require(
        len(requested["records"]) == len(batch.records), "Scoring input count changed"
    )
    for before, after in zip(requested["records"], batch.records):
        before = fields(
            before, ("data_id", "record_hash", "content_hash", "tokenization_hash")
        )
        require(
            before["data_id"] == after.data_id
            and before["content_hash"] == after.content_hash
            and before["tokenization_hash"] == digest(after.tokenization.to_dict()),
            "Scoring inputs changed during label update",
        )
    inputs = None
    token_inputs = None
    if requested["schema_version"] == "2":
        require(
            requested.get("input_records_file") == "inputs.jsonl"
            and requested.get("token_statistics_file") == "token-statistics.json",
            "Missing exact scoring input/statistic snapshots",
        )
        originals = read_records(folder / "inputs.jsonl")
        require(len(originals) == len(batch.records), "Original input count changed")
        for old, identity, enriched in zip(
            originals, requested["records"], batch.records
        ):
            require(
                old.record_hash == identity["record_hash"]
                and old.data_id == enriched.data_id
                and (
                    old == enriched or attach_signals(old, enriched.signals) == enriched
                ),
                "Scoring changed input labels or history outside forward signals",
            )
        inputs = {r.data_id: r for r in originals}
        token_inputs = _read_token_inputs(
            folder / "token-statistics.json", batch.model_ref, batch.computation_hash
        )
    else:
        require(
            "input_records_file" not in requested
            and "token_statistics_file" not in requested,
            "Legacy scoring request cannot declare unchecked snapshots",
        )
    by_id = {r.data_id: r for r in batch.records}
    forward_tokens = {}
    consumed = set()
    for path in folder.glob("fwd-*/request.json"):
        forward = parse_forward_contract(
            "ForwardRequest", load_json(path.read_text(encoding="utf-8"))
        )
        result = parse_forward_contract(
            "TokenMetricResult",
            load_json((path.parent / "result.json").read_text(encoding="utf-8")),
        )
        data = by_id.get(forward.data_id)
        require(
            data is not None
            and forward.model_ref == expected_model
            and forward.computation_hash == expected_computation_hash
            and forward.input_ids == data.tokenization.input_ids
            and forward.content_hash == data.content_hash
            and forward.tokenization_hash == digest(data.tokenization.to_dict())
            and forward.backend == batch.backend,
            "Forward evidence uses another model/data/input version",
        )
        require(
            result.request_hash == forward.record_hash
            and result.positions == forward.positions
            and result.backend == batch.backend,
            "Forward output/request binding changed",
        )
        token_map = forward_tokens.setdefault(data.data_id, {})
        require(
            not set(token_map).intersection(forward.positions),
            "Duplicate forward statistics for the same input positions",
        )
        token_map.update(
            (p, (n, h, result.evidence))
            for p, n, h in zip(result.positions, result.token_nll, result.token_entropy)
        )
        event = next(
            (
                e
                for e in batch.ledger.events
                if e.event_id == "score-" + forward.record_hash
            ),
            None,
        )
        require(
            event is not None
            and event.event_id not in consumed
            and event.usage == result.usage
            and event.receipt.sha256 == result.record_hash
            and event.subject_id == data.data_id,
            "Scoring forward cost/evidence mismatch",
        )
        consumed.add(event.event_id)
    require(
        consumed == {e.event_id for e in batch.ledger.events},
        "Missing or extraneous scoring forward events",
    )
    _validate_aggregations(batch, forward_tokens, token_inputs, inputs, _positions)
    batch.metric_view()
    return batch


def _read_token_inputs(path, model, computation_hash):
    value = fields(
        load_json(path.read_text(encoding="utf-8")),
        ("schema_version", "model_hash", "computation_hash", "records"),
    )
    require(
        value["schema_version"] == "1"
        and value["model_hash"] == model.model_hash
        and value["computation_hash"] == computation_hash
        and type(value["records"]) is list,
        "Token statistic snapshot identity changed",
    )
    result = {}
    for row in value["records"]:
        row = fields(
            row, ("data_id", "positions", "token_nll", "token_entropy", "evidence")
        )
        require(
            all(
                type(row[k]) is list and len(row[k]) == len(row["positions"])
                for k in ("positions", "token_nll", "token_entropy", "evidence")
            )
            and bool(row["positions"])
            and type(row["data_id"]) is str
            and row["data_id"] not in result,
            "Duplicate or misaligned token statistic inputs",
        )
        positions = row["positions"]
        for p in positions:
            integer(p, "statistic target position", 1)
        require(positions == sorted(set(positions)), "Unordered statistic positions")
        tokens = {}
        for p, n, h, ref in zip(
            positions, row["token_nll"], row["token_entropy"], row["evidence"]
        ):
            finite(n, "token NLL", 0)
            finite(h, "token entropy", 0)
            tokens[p] = (n, h, artifact(ref))
        result[row["data_id"]] = tokens
    return result


def _validate_aggregations(batch, forward_tokens, token_inputs, inputs, select):
    by_id = {r.data_id: r for r in batch.records}
    allowed, used = {}, {}
    for item in batch.outcomes:
        if item.signal is None:
            continue
        _, positions = select(by_id[item.data_id], item.spec)
        selected = set(positions)
        allowed.setdefault(item.data_id, set()).update(selected)
        current = forward_tokens.get(item.data_id, {})
        direct = inputs is not None and any(
            s.binding == item.signal.binding
            and s.model_ref == item.signal.model_ref
            and s.spec == item.spec
            and s.nll is not None
            and s.entropy is not None
            for s in inputs[item.data_id].signals
        )
        if direct:
            require(item.status == "cached", "Existing complete signal was recomputed")
            continue
        require(
            (
                item.status
                == ("computed" if selected.intersection(current) else "cached")
                if inputs is not None
                else item.status != "computed" or bool(selected.intersection(current))
            ),
            "Signal computed/cache status disagrees with physical forwards",
        )
        known = current if token_inputs is None else token_inputs.get(item.data_id, {})
        if token_inputs is not None:
            require(
                selected.issubset(known), "Missing token inputs for signal aggregate"
            )
            used.setdefault(item.data_id, set()).update(selected)
        if selected.issubset(known):
            require(
                item.signal.nll
                == aggregate(tuple(known[p][0] for p in positions), item.spec)
                and item.signal.entropy
                == aggregate(tuple(known[p][1] for p in positions), item.spec),
                "Signal aggregate differs from its per-token statistics",
            )
        require(
            all(
                known[p][2] in item.signal.provenance
                for p in selected.intersection(known)
            ),
            "Signal provenance omits a consumed token statistic",
        )
    for data_id, values in forward_tokens.items():
        require(
            set(values).issubset(allowed.get(data_id, set())),
            "Forward scored positions outside requested supervised scopes",
        )
        if token_inputs is not None:
            require(
                all(
                    token_inputs.get(data_id, {}).get(p) == v for p, v in values.items()
                ),
                "Token inputs differ from the physical forward result",
            )
    if token_inputs is not None:
        require(
            {k: set(v) for k, v in token_inputs.items()} == used,
            "Missing or extraneous aggregate token inputs",
        )
