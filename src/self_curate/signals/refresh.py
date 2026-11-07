"""Round 1 full initialization; later exact refresh for focused, outer-unused data."""

import base64
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from time import perf_counter
import uuid

from self_curate.contracts import (
    CostLedger,
    ContractError,
    ErrorCode,
    SignalSpec,
    canonical_json,
    digest,
)
from self_curate.contracts.parsing import fields, load_json, signal_spec
from self_curate.contracts.types import integer, require, text
from self_curate.data import (
    MetricView,
    OuterExposureHistory,
    attach_signals,
    signal_binding,
)
from self_curate.data.outer_usage import parse_outer_exposures
from self_curate.data.pool import _publish, _staging_directory, file_sha256, write_json
from .backends import ScoringFailure
from .service import SignalService
from .io import _structured_input, load_scoring_batch, read_records
from .refresh_contracts import SkillFocus, parse_skill_focus


@dataclass(frozen=True)
class RefreshPlan:
    request_json: str
    eligible_ids: tuple[str, ...]
    refresh_ids: tuple[str, ...]
    cached_ids: tuple[str, ...]
    used_ids: tuple[str, ...]
    outside_focus_ids: tuple[str, ...]
    missing_skill_ids: tuple[str, ...]
    status: str

    @property
    def request(self):
        return load_json(self.request_json)

    @property
    def plan_hash(self):
        return digest(self.to_dict())

    def to_dict(self):
        return {
            "schema_version": "1",
            "request": self.request,
            "status": self.status,
            **{
                n: list(getattr(self, n))
                for n in (
                    "eligible_ids",
                    "refresh_ids",
                    "cached_ids",
                    "used_ids",
                    "outside_focus_ids",
                    "missing_skill_ids",
                )
            },
        }


@dataclass(frozen=True)
class RefreshResult:
    plan: RefreshPlan
    records: tuple
    ledger: CostLedger
    output: Path
    scoring: object = None

    @property
    def metric_view(self):
        return MetricView(self.records)

    @property
    def status(self):
        return "completed" if self.plan.status == "ready" else self.plan.status

    def focused_context(self, template):
        """Explicitly create a NEW inner-search snapshot; never mutate an active search."""
        from self_curate.operators import ExecutionContext, SignalContext
        from self_curate.contracts.parsing import model_ref

        require(
            self.status == "completed" and bool(self.plan.eligible_ids),
            "No complete focused pool for inner selection",
        )
        require(
            isinstance(template, ExecutionContext)
            and template.model_ref == model_ref(self.plan.request["model_ref"]),
            "New inner context needs current accepted ModelRef",
        )
        ids = set(self.plan.eligible_ids)
        specs = tuple(signal_spec(s) for s in self.plan.request["specs"])
        return replace(
            template,
            view=MetricView(tuple(r for r in self.records if r.data_id in ids)),
            signals=tuple(SignalContext(s) for s in specs),
        )


class RefreshFailure(ScoringFailure):
    def __init__(self, message, *, records, ledger, plan, output, cause):
        super().__init__(
            message, records=records, ledger=ledger, usage=getattr(cause, "usage", None)
        )
        self.plan, self.output = plan, output


def _complete(record, model, specs):
    for spec in specs:
        try:
            binding = signal_binding(record, spec)
        except ContractError as exc:
            if exc.code != ErrorCode.MISSING_SIGNAL:
                raise
            return False
        if not any(
            s.model_ref == model
            and s.spec == spec
            and s.binding == binding
            and s.nll is not None
            and s.entropy is not None
            for s in record.signals
        ):
            return False
    return True


def _plan(
    records,
    *,
    model,
    specs,
    computation_hash,
    backend,
    known_skills,
    branch_id,
    outer_round,
    history,
    focus,
    max_refresh_records,
    max_context_tokens,
):
    integer(outer_round, "outer round", 1)
    view = MetricView(records)
    require(
        isinstance(history, OuterExposureHistory)
        and history.branch_id == branch_id
        and history.backend == backend,
        "Wrong experiment branch or exposure domain",
    )
    require(
        all(r.tokenization.identity == model.tokenizer for r in records),
        "Refresh pool uses another tokenizer/template",
        ErrorCode.STALE_CACHE,
    )
    require(
        backend != "real"
        or (
            model.tokenizer.backend != "fixture"
            and not any(r.known_mock_artifacts for r in records)
        ),
        "Mock data/model cannot enter real refresh",
    )
    if outer_round == 1:
        require(
            not history.events and not history.unresolved and focus is None,
            "First-round initialization requires an empty outer history and no Lopt restriction",
        )
    elif focus is not None:
        require(
            isinstance(focus, SkillFocus)
            and focus.outer_round == outer_round
            and focus.model_ref == model,
            "Lopt direction targets another round/model",
            ErrorCode.STALE_CACHE,
        )
        require(
            set(focus.skills).issubset(known_skills), "Unknown Lopt skill direction"
        )
        require(
            backend != "real"
            or not focus.feedback.uri.strip()
            .lower()
            .startswith(("mock://", "fixture://")),
            "Mock outer feedback cannot drive real refresh",
        )
        require(
            all(s.scope == "assistant" or s.skill in focus.skills for s in specs),
            "Skill-scoped metric is outside Lopt directions",
        )
    status = (
        "blocked_outer_exposure"
        if history.unresolved
        else ("awaiting_focus" if outer_round > 1 and focus is None else "ready")
    )
    eligible, needed, cached, used, outside, missing = [], [], [], [], [], []
    labels = view.static_batch(view.data_ids, ("skills",)).column("skills")
    for record, skills in zip(records, labels):
        if outer_round > 1 and history.was_used(record):
            used.append(record.data_id)
            continue
        if outer_round > 1 and (
            focus is None
            or not focus.skills
            or skills is None
            or not set(skills).intersection(focus.skills)
        ):
            outside.append(record.data_id)
            if skills is None:
                missing.append(record.data_id)
            continue
        eligible.append(record.data_id)
        (cached if _complete(record, model, specs) else needed).append(record.data_id)
    if status == "ready":
        require(
            max_refresh_records is None or len(needed) <= max_refresh_records,
            "Refresh row budget exceeded before inference",
            ErrorCode.BUDGET_EXHAUSTED,
        )
        wanted = set(needed)
        tokens = sum(r.stats.context_tokens for r in records if r.data_id in wanted)
        require(
            max_context_tokens is None or tokens <= max_context_tokens,
            "Refresh complete-context budget exceeded before inference",
            ErrorCode.BUDGET_EXHAUSTED,
        )
    request = {
        "schema_version": "1",
        "branch_id": branch_id,
        "outer_round": outer_round,
        "model_ref": model.to_dict(),
        "backend": backend,
        "computation_hash": computation_hash,
        "specs": [s.to_dict() for s in specs],
        "known_skills": list(known_skills),
        "focus": None if focus is None else focus.to_dict(),
        "history_hash": history.history_hash,
        "pool_hash": digest([r.record_hash for r in records]),
        "max_refresh_records": max_refresh_records,
        "max_context_tokens": max_context_tokens,
    }
    return RefreshPlan(
        canonical_json(request),
        tuple(eligible),
        tuple(needed),
        tuple(cached),
        tuple(used),
        tuple(outside),
        tuple(missing),
        status,
    )


class FocusedRefreshService:
    def __init__(
        self,
        scorer,
        *,
        branch_id,
        known_skills,
        output,
        specs=None,
        max_refresh_records=None,
        max_context_tokens=None,
    ):
        require(
            isinstance(scorer, SignalService), "Use the existing exact SignalService"
        )
        text(branch_id, "refresh branch")
        require(
            type(known_skills) is tuple and len(set(known_skills)) == len(known_skills),
            "Unique immutable skill ontology required",
        )
        for skill in known_skills:
            text(skill, "known skill")
        for name, value in (
            ("max_refresh_records", max_refresh_records),
            ("max_context_tokens", max_context_tokens),
        ):
            if value is not None:
                integer(value, name, 0)
        specs = (scorer.spec(),) if specs is None else tuple(specs)
        require(
            bool(specs) and all(isinstance(s, SignalSpec) for s in specs),
            "Typed signal specs required",
        )
        require(
            len({s.spec_hash for s in specs}) == len(specs)
            and all(
                s.computation_hash == scorer.computation_hash
                and s.computation_id == scorer.spec().computation_id
                for s in specs
            ),
            "Use current scorer specs",
        )
        require(
            all(s.scope == "assistant" or s.skill in known_skills for s in specs),
            "Metric scope outside skill ontology",
        )
        self.scorer, self.branch_id, self.known_skills = scorer, branch_id, known_skills
        self.output, self.specs = Path(output).resolve(), specs
        self.max_refresh_records, self.max_context_tokens = (
            max_refresh_records,
            max_context_tokens,
        )

    def plan(self, records, *, outer_round, history, focus=None):
        self.scorer._fixed()
        return _plan(
            tuple(records),
            model=self.scorer.model_ref,
            specs=self.specs,
            computation_hash=self.scorer.computation_hash,
            backend=self.scorer.backend.backend,
            known_skills=self.known_skills,
            branch_id=self.branch_id,
            outer_round=outer_round,
            history=history,
            focus=focus,
            max_refresh_records=self.max_refresh_records,
            max_context_tokens=self.max_context_tokens,
        )

    def initialize(self, records, *, history):
        return self.run_round(records, outer_round=1, history=history)

    def run_round(self, records, *, outer_round, history, focus=None):
        records = tuple(records)
        plan = self.plan(records, outer_round=outer_round, history=history, focus=focus)
        return self.execute(plan, records, history=history, focus=focus)

    def execute(self, plan, records, *, history, focus=None):
        records = tuple(records)
        require(
            isinstance(plan, RefreshPlan)
            and self.plan(
                records,
                outer_round=plan.request["outer_round"],
                history=history,
                focus=focus,
            )
            == plan,
            "Refresh inputs/history/focus changed after planning",
            ErrorCode.STALE_CACHE,
        )
        started = perf_counter()
        self.output.mkdir(parents=True, exist_ok=True)
        label = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")
        output = self.output / ("refresh-" + label)
        stage = _staging_directory(output)
        write_json(stage / "plan.json", plan.to_dict())
        write_json(stage / "history.json", history.to_dict())
        _write_records(stage / "inputs.jsonl", records)
        batch = None
        ledger = CostLedger("refresh-" + label, (), self.scorer.backend.backend)
        updated = {r.data_id: r for r in records}
        try:
            if plan.status == "ready" and plan.refresh_ids:
                wanted = set(plan.refresh_ids)
                selected = tuple(r for r in records if r.data_id in wanted)
                batch = self.scorer.score(selected, specs=self.specs)
                ledger = batch.ledger
                require(
                    batch.model_ref == self.scorer.model_ref
                    and batch.computation_hash == self.scorer.computation_hash
                    and tuple(r.data_id for r in batch.records) == plan.refresh_ids,
                    "Scoring returned another refresh batch",
                )
                for record in batch.records:
                    updated[record.data_id] = record
                original = self.scorer.output / ("score-" + batch.batch_id)
                shutil.copytree(original, stage / "scoring")
                require(
                    load_scoring_batch(
                        stage / "scoring",
                        expected_model=self.scorer.model_ref,
                        expected_computation_hash=self.scorer.computation_hash,
                    )
                    == batch,
                    "Scoring copy changed",
                )
            values = tuple(updated[r.data_id] for r in records)
            _write_records(stage / "records.jsonl", values)
            write_json(
                stage / "result.json",
                _result_dict(plan, values, ledger, batch, perf_counter() - started),
            )
            write_json(
                stage / "checksums.json",
                {
                    p.relative_to(stage).as_posix(): file_sha256(p)
                    for p in stage.rglob("*")
                    if p.is_file()
                },
            )
            _publish(stage, output)
            return RefreshResult(plan, values, ledger, output, batch)
        except Exception as exc:
            if isinstance(getattr(exc, "ledger", None), CostLedger):
                ledger = exc.ledger
            for record in getattr(exc, "records", ()):
                if record.data_id in updated:
                    updated[record.data_id] = record
            values = tuple(updated[r.data_id] for r in records)
            try:
                _write_records(stage / "partial-records.jsonl", values)
                write_json(
                    stage / "failure.json",
                    {
                        "error": str(exc),
                        "plan_hash": plan.plan_hash,
                        "ledger": ledger.to_dict(),
                        "partial_records_sha256": file_sha256(
                            stage / "partial-records.jsonl"
                        ),
                    },
                )
            except OSError:
                pass
            raise RefreshFailure(
                str(exc),
                records=values,
                ledger=ledger,
                plan=plan,
                output=stage,
                cause=exc,
            ) from exc


def _write_records(path, records):
    path.write_text(
        "".join(canonical_json(r.to_dict()) + "\n" for r in records),
        encoding="utf-8",
        newline="\n",
    )


def _result_dict(plan, records, ledger, batch, wall):
    return {
        "schema_version": "1",
        "plan_hash": plan.plan_hash,
        "status": "completed" if plan.status == "ready" else plan.status,
        "records": [
            {"data_id": r.data_id, "record_hash": r.record_hash} for r in records
        ],
        "ledger": ledger.to_dict(),
        "scoring_batch_hash": None if batch is None else digest(batch.to_dict()),
        "local_wall_seconds": wall,
        "new_forward_calls": len(ledger.events),
        "missing_metric_cells": (
            0 if batch is None else sum(o.signal is None for o in batch.outcomes)
        ),
    }


@_structured_input
def load_refresh_result(
    path,
    *,
    expected_model,
    expected_computation_hash,
    expected_plan_hash,
    expected_history_hash,
):
    from self_curate.contracts import parse_contract
    from self_curate.contracts.workflow import finite

    root = Path(path).resolve()
    checks = load_json((root / "checksums.json").read_text(encoding="utf-8"))
    require(type(checks) is dict, "Refresh checksum table required")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "checksums.json"
    }
    require(
        set(checks) == actual
        and {
            "plan.json",
            "history.json",
            "inputs.jsonl",
            "records.jsonl",
            "result.json",
        }.issubset(actual),
        "Incomplete refresh output",
    )
    for name, checksum in checks.items():
        file = root / name
        require(
            file.resolve().is_relative_to(root)
            and not file.is_symlink()
            and file_sha256(file) == checksum,
            "Refresh output path/checksum mismatch",
        )
    raw_plan = load_json((root / "plan.json").read_text(encoding="utf-8"))
    row = fields(
        raw_plan,
        (
            "schema_version",
            "request",
            "status",
            "eligible_ids",
            "refresh_ids",
            "cached_ids",
            "used_ids",
            "outside_focus_ids",
            "missing_skill_ids",
        ),
    )
    request = fields(
        row["request"],
        (
            "schema_version",
            "branch_id",
            "outer_round",
            "model_ref",
            "backend",
            "computation_hash",
            "specs",
            "known_skills",
            "focus",
            "history_hash",
            "pool_hash",
            "max_refresh_records",
            "max_context_tokens",
        ),
    )
    require(
        row["schema_version"] == request["schema_version"] == "1"
        and digest(raw_plan) == expected_plan_hash
        and request["model_ref"] == expected_model.to_dict()
        and request["computation_hash"] == expected_computation_hash
        and request["history_hash"] == expected_history_hash,
        "Refresh output uses another model/plan/history",
    )
    history = parse_outer_exposures(
        load_json((root / "history.json").read_text(encoding="utf-8"))
    )
    require(
        history.history_hash == expected_history_hash,
        "Refresh outer exposure history changed",
    )
    before, after = read_records(root / "inputs.jsonl"), read_records(
        root / "records.jsonl"
    )
    specs = tuple(signal_spec(s) for s in request["specs"])
    focus = None if request["focus"] is None else parse_skill_focus(request["focus"])
    plan = _plan(
        before,
        model=expected_model,
        specs=specs,
        computation_hash=expected_computation_hash,
        backend=request["backend"],
        known_skills=tuple(request["known_skills"]),
        branch_id=request["branch_id"],
        outer_round=request["outer_round"],
        history=history,
        focus=focus,
        max_refresh_records=request["max_refresh_records"],
        max_context_tokens=request["max_context_tokens"],
    )
    require(plan.to_dict() == raw_plan, "Refresh eligibility/routing changed")
    batch = None
    if plan.status == "ready" and plan.refresh_ids:
        batch = load_scoring_batch(
            root / "scoring",
            expected_model=expected_model,
            expected_computation_hash=expected_computation_hash,
        )
        require(
            tuple(r.data_id for r in batch.records) == plan.refresh_ids
            and tuple(o.spec for o in batch.outcomes)
            == tuple(s for _ in batch.records for s in specs),
            "Wrong scoring rows/scopes in refresh",
        )
        require(
            read_records(root / "scoring" / "inputs.jsonl")
            == tuple(r for r in before if r.data_id in set(plan.refresh_ids)),
            "Scorer consumed different input labels/history",
        )
    else:
        require(
            not (root / "scoring").exists(),
            "Unexpected scoring evidence for a no-work refresh",
        )
    expected = {r.data_id: r for r in before}
    if batch is not None:
        for record in batch.records:
            old = expected[record.data_id]
            require(
                attach_signals(old, record.signals) == record,
                "Refresh changed labels outside signals",
            )
            expected[record.data_id] = record
    require(
        after == tuple(expected[r.data_id] for r in before),
        "Unrequested or outer-used data changed during refresh",
    )
    result = fields(
        load_json((root / "result.json").read_text(encoding="utf-8")),
        (
            "schema_version",
            "plan_hash",
            "status",
            "records",
            "ledger",
            "scoring_batch_hash",
            "local_wall_seconds",
            "new_forward_calls",
            "missing_metric_cells",
        ),
    )
    ledger = parse_contract("CostLedger", result["ledger"])
    require(
        batch is None
        and not ledger.events
        and ledger.backend == request["backend"]
        or batch is not None
        and ledger == batch.ledger,
        "Refresh cost differs from actual scoring",
    )
    finite(result["local_wall_seconds"], "refresh wall time", 0)
    require(
        result
        == _result_dict(plan, after, ledger, batch, result["local_wall_seconds"]),
        "Refresh counts/result metadata changed",
    )
    return RefreshResult(plan, after, ledger, root, batch)
