"""Explicit mock routing; numerical oracles remain in test_signal_computation."""

from dataclasses import replace
import json

import pytest

from helpers import golden_record
from test_signal_computation import service, MatrixProvider
from test_outer_cycle import make_cycle, Decisions
from self_curate import contracts as c
from self_curate.data import (
    PrecomputedTokenizer,
    normalize_trajectory,
    record_candidate_training,
    OuterExposureHistory,
    OuterTrainingExposure,
    load_outer_exposures,
)
from self_curate.data.pool import file_sha256
from self_curate.signals import (
    FocusedRefreshService,
    SkillFocus,
    RefreshFailure,
    ScoringFailure,
    load_refresh_result,
    focus_from_outer_decision,
)


SKILLS = ("recovery", "testing", "empty", "missing")


def ref(name):
    return c.ArtifactRef("fixture://" + name, c.digest(name))


def row(name, skills=("recovery",), *, content=None):
    raw, tokenizer = golden_record()
    raw["data_id"] = name
    raw["source"]["record_id"] = name
    raw["messages"][0]["content"] = content or name
    if skills is None:
        raw.pop("annotations")
    else:
        raw["annotations"]["skills"] = [
            s for s in raw["annotations"]["skills"] if s["skill"] in skills
        ]
    return normalize_trajectory(raw, PrecomputedTokenizer(tokenizer), SKILLS)


def setup(tmp_path, *, model=None, provider=None, **kwargs):
    scorer, provider, _ = service(tmp_path, model, provider=provider)
    refresh = FocusedRefreshService(
        scorer,
        branch_id="branch",
        known_skills=SKILLS,
        output=tmp_path / "refresh",
        **kwargs,
    )
    return refresh, provider, OuterExposureHistory("branch", "mock")


def focus(refresh, skills=("recovery",), *, outer_round=2):
    return SkillFocus(
        outer_round,
        refresh.scorer.model_ref,
        skills,
        ref("outer-feedback"),
        "external Lopt direction fixture",
    )


def exposure(history, record):
    """Declared synthetic completed OUTER update, separate from inner uses."""
    return replace(
        history,
        events=(
            *history.events,
            OuterTrainingExposure(
                "outer-update-" + record.data_id,
                "outer-attempt",
                c.digest("handoff"),
                c.digest("batch"),
                ((record.data_id, record.content_hash),),
                c.digest("parent"),
                c.digest("candidate"),
                7,
                1,
                ref("physical-update"),
                "mock",
            ),
        ),
    )


def reload(result, refresh, history):
    return load_refresh_result(
        result.output,
        expected_model=refresh.scorer.model_ref,
        expected_computation_hash=refresh.scorer.computation_hash,
        expected_plan_hash=result.plan.plan_hash,
        expected_history_hash=history.history_hash,
    )


def test_first_round_scores_whole_pool_including_unlabelled_and_inner_sft(tmp_path):
    refresh, provider, history = setup(tmp_path)
    rows = (row("a"), row("b", ("testing",)), row("c", None))
    rows = (
        record_candidate_training(
            rows[0],
            c.TrainingUse("inner-only", "m0", ref("inner-training"), "accepted"),
        ),
        *rows[1:],
    )
    result = refresh.initialize(rows, history=history)
    assert result.status == "completed"
    assert result.plan.refresh_ids == ("a", "b", "c")
    assert len(provider.requests) == 3
    assert all(
        len(r.signals) == 1
        and r.signals[0].nll is not None
        and r.signals[0].entropy is not None
        for r in result.records
    )
    assert rows[0].usage == result.records[0].usage.__class__(
        candidate_sft=rows[0].usage.candidate_sft,
        accepted_sft=rows[0].usage.accepted_sft,
    )
    assert reload(result, refresh, history) == result


def test_exact_initial_assets_are_reused_without_forward(tmp_path):
    refresh, provider, history = setup(tmp_path)
    first = refresh.initialize((row("a"), row("b")), history=history)
    second = refresh.initialize(first.records, history=history)
    assert len(provider.requests) == 2
    assert second.plan.cached_ids == ("a", "b") and not second.plan.refresh_ids
    assert second.records == first.records and not second.ledger.events
    assert reload(second, refresh, history) == second


def test_next_model_only_focus_union_outer_unused_and_preserves_old_versions(tmp_path):
    first, _, history = setup(tmp_path / "initial")
    rows = (
        row("outer-used"),
        row("inner-only"),
        row("other", ("testing",)),
        row("both", ("recovery", "testing")),
        row("missing", None),
    )
    initial = first.initialize(rows, history=history)
    history = exposure(history, rows[0])
    model = replace(first.scorer.model_ref, weights_hash=c.digest("accepted-m1"))
    refresh, provider, _ = setup(tmp_path / "next", model=model)
    selected = focus(refresh, ("recovery", "testing"))
    initial_rows = list(initial.records)
    initial_rows[1] = record_candidate_training(
        initial_rows[1], c.TrainingUse("inner", "m0", ref("inner"), "rejected")
    )
    result = refresh.run_round(
        initial_rows, outer_round=2, history=history, focus=selected
    )
    assert result.plan.used_ids == ("outer-used",)
    assert result.plan.refresh_ids == ("inner-only", "other", "both")
    assert result.plan.missing_skill_ids == ("missing",)
    assert [q.data_id for q in provider.requests] == ["inner-only", "other", "both"]
    for before, after in zip(initial_rows, result.records):
        if before.data_id in result.plan.refresh_ids:
            assert {s.model_ref for s in after.signals} == {
                first.scorer.model_ref,
                model,
            }
        else:
            assert before == after
    assert reload(result, refresh, history) == result


def test_outside_focus_does_not_get_current_model_metrics_or_stale_fallback(tmp_path):
    refresh, provider, history = setup(tmp_path)
    rows = (row("a"), row("b", ("testing",)))
    result = refresh.run_round(
        rows, outer_round=2, history=history, focus=focus(refresh)
    )
    assert result.plan.eligible_ids == ("a",) and result.plan.outside_focus_ids == (
        "b",
    )
    assert result.records[1] == rows[1] and not result.records[1].signals
    assert len(provider.requests) == 1


def test_same_model_revisit_caches_and_new_direction_only_adds_missing_rows(tmp_path):
    refresh, provider, history = setup(tmp_path)
    rows = (row("a"), row("b", ("testing",)))
    first = refresh.run_round(
        rows, outer_round=2, history=history, focus=focus(refresh)
    )
    again = refresh.run_round(
        first.records,
        outer_round=3,
        history=history,
        focus=focus(refresh, outer_round=3),
    )
    assert again.plan.cached_ids == ("a",) and not again.ledger.events
    changed = refresh.run_round(
        again.records,
        outer_round=4,
        history=history,
        focus=focus(refresh, ("recovery", "testing"), outer_round=4),
    )
    assert changed.plan.cached_ids == ("a",) and changed.plan.refresh_ids == ("b",)
    assert [r.data_id for r in provider.requests] == ["a", "b"]
    assert reload(changed, refresh, history) == changed


@pytest.mark.parametrize(
    "direction,status", [(None, "awaiting_focus"), ((), "completed")]
)
def test_missing_or_explicit_empty_direction_does_no_inference(
    tmp_path, direction, status
):
    refresh, provider, history = setup(tmp_path)
    result = refresh.run_round(
        (row("a"),),
        outer_round=2,
        history=history,
        focus=None if direction is None else focus(refresh, direction),
    )
    assert result.status == status and not provider.requests
    assert not result.plan.eligible_ids and not result.ledger.events
    assert reload(result, refresh, history) == result


@pytest.mark.parametrize("kind", ["round", "model", "skill", "branch", "domain"])
def test_wrong_identity_fails_before_inference(tmp_path, kind):
    refresh, provider, history = setup(tmp_path)
    direction = focus(refresh)
    if kind == "round":
        direction = replace(direction, outer_round=3)
    elif kind == "model":
        direction = replace(
            direction,
            model_ref=replace(direction.model_ref, weights_hash=c.digest("wrong")),
        )
    elif kind == "skill":
        direction = replace(direction, skills=("invented",))
    elif kind == "branch":
        history = history.fork("other")
    else:
        history = OuterExposureHistory("branch", "real")
    with pytest.raises(c.ContractError):
        refresh.run_round((row("a"),), outer_round=2, history=history, focus=direction)
    assert not provider.requests


@pytest.mark.parametrize(
    "kwargs", [{"max_refresh_records": 1}, {"max_context_tokens": 17}]
)
def test_budget_preflight_never_truncates_pool(tmp_path, kwargs):
    refresh, provider, history = setup(tmp_path, **kwargs)
    with pytest.raises(c.ContractError) as error:
        refresh.initialize((row("a"), row("b")), history=history)
    assert error.value.code == c.ErrorCode.BUDGET_EXHAUSTED and not provider.requests


def test_planned_work_cannot_be_executed_after_outer_history_changes(tmp_path):
    refresh, provider, history = setup(tmp_path)
    rows, direction = (row("a"), row("b")), focus(refresh)
    plan = refresh.plan(rows, outer_round=2, history=history, focus=direction)
    with pytest.raises(c.ContractError):
        refresh.execute(plan, rows, history=exposure(history, rows[0]), focus=direction)
    assert not provider.requests


def test_content_alias_is_outer_used_but_changed_same_id_is_error(tmp_path):
    refresh, provider, history = setup(tmp_path)
    used, alias = row("a", content="same"), row("alias", content="same")
    assert used.content_hash == alias.content_hash
    history = exposure(history, used)
    result = refresh.run_round(
        (alias,), outer_round=2, history=history, focus=focus(refresh)
    )
    assert result.plan.used_ids == ("alias",) and not provider.requests
    with pytest.raises(c.ContractError):
        history.was_used(row("a", content="changed"))


def test_branch_fork_and_exposure_snapshot_roundtrip(tmp_path):
    history = exposure(OuterExposureHistory("branch", "mock"), row("a"))
    fork = history.fork("new-branch")
    assert fork.events == history.events and fork.parent_hash == history.history_hash
    output = fork.persist(tmp_path)
    assert fork.persist(tmp_path) == output
    assert (
        load_outer_exposures(
            output, expected_hash=fork.history_hash, expected_branch="new-branch"
        )
        == fork
    )
    with pytest.raises(c.ContractError):
        load_outer_exposures(
            output, expected_hash=fork.history_hash, expected_branch="wrong"
        )


def test_unknown_partial_update_blocks_refresh_and_preserves_records(tmp_path):
    refresh, provider, history = setup(tmp_path)
    history = history.mark_unresolved("partial", ref("partial-update"))
    rows = (row("a"),)
    result = refresh.run_round(
        rows, outer_round=2, history=history, focus=focus(refresh)
    )
    assert result.status == "blocked_outer_exposure" and result.records == rows
    assert not provider.requests and reload(result, refresh, history) == result


@pytest.mark.parametrize("field", ["records", "plan", "result", "history"])
def test_saved_refresh_rejects_tampering_even_if_checksums_updated(tmp_path, field):
    refresh, _, history = setup(tmp_path)
    result = refresh.initialize((row("a"), row("b")), history=history)
    filename = {
        "records": "records.jsonl",
        "plan": "plan.json",
        "result": "result.json",
        "history": "history.json",
    }[field]
    path = result.output / filename
    if field == "records":
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if field == "plan":
            value["used_ids"] = ["a"]
        elif field == "history":
            value["branch_id"] = "other"
        else:
            value["new_forward_calls"] = 0
        path.write_text(c.canonical_json(value), encoding="utf-8")
    checks_path = result.output / "checksums.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    checks[filename] = file_sha256(path)
    checks_path.write_text(c.canonical_json(checks), encoding="utf-8")
    with pytest.raises(c.ContractError):
        reload(result, refresh, history)


class FailSecond(MatrixProvider):
    def forward(self, request):
        if self.requests:
            raise ScoringFailure(
                "explicit second forward failure",
                usage=c.ResourceUsage(scoring_tokens=3),
            )
        return super().forward(request)


def test_failed_refresh_keeps_completed_rows_actual_costs_and_full_pool(tmp_path):
    default, _, history = setup(tmp_path / "base")
    provider = FailSecond(default.scorer.model_ref)
    refresh, _, _ = setup(tmp_path / "failure", provider=provider)
    rows = (row("a"), row("b"), row("outside", ("testing",)))
    with pytest.raises(RefreshFailure) as error:
        refresh.run_round(rows, outer_round=2, history=history, focus=focus(refresh))
    failure = error.value
    assert failure.records[0].signals and not failure.records[1].signals
    assert failure.records[2] == rows[2]
    assert len(failure.ledger.events) == 2
    assert (failure.output / "failure.json").is_file()
    assert not (failure.output / "checksums.json").exists()


def test_current_metrics_are_read_by_a_new_operator_context(tmp_path):
    from helpers_m2 import context

    refresh, _, history = setup(tmp_path)
    result = refresh.run_round(
        (row("a"), row("b", ("testing",))),
        outer_round=2,
        history=history,
        focus=focus(refresh),
    )
    old = context(result.records, model=refresh.scorer.model_ref, signals=())
    old_hash = old.context_hash
    new = result.focused_context(old)
    assert new.view.data_ids == ("a",)
    assert (
        new.model_ref == refresh.scorer.model_ref
        and new.signals[0].spec == refresh.scorer.spec()
    )
    assert old.context_hash == old_hash and new.context_hash != old_hash


def tracked_cycle(tmp_path):
    cycle, trainer, runner, producer, search = make_cycle(tmp_path)
    # Test-only attach to the existing accepted fixture; constructor path tested below.
    cycle._outer_exposures = OuterExposureHistory("branch", "mock")
    return cycle, trainer, runner, producer, search


@pytest.mark.parametrize("outcome", ["accept", "reject"])
def test_actual_outer_subset_is_used_before_decision_and_after_rejection(
    tmp_path, outcome
):
    cycle, trainer, _, _, _ = tracked_cycle(tmp_path)
    cycle.step()
    assert (
        cycle.phase == "awaiting_validation" and len(cycle.outer_exposures.events) == 1
    )
    event = cycle.outer_exposures.events[0]
    assert event.training_id == trainer.requests[0].training_id
    assert (
        cycle.outer_exposures.record(cycle.handoff, cycle.trainings[0])
        == cycle.outer_exposures
    )
    assert all(cycle.outer_exposures.was_used(r) for r in cycle.updated_records)
    cycle.step()
    context = cycle.pending_decision
    decision = Decisions([outcome]).decide(context)
    direction = focus_from_outer_decision(
        context,
        decision,
        outer_round=2,
        skills=("recovery",),
        rationale="fixture direction",
    )
    expected = (
        context.training.result.candidate_model
        if outcome == "accept"
        else context.accepted_model
    )
    assert direction.model_ref == expected
    cycle.submit_decision(decision)
    assert len(cycle.outer_exposures.events) == 1


def test_curation_training_does_not_mark_raw_task_pool_as_used_again(tmp_path):
    cycle, _, _, _, _ = tracked_cycle(tmp_path)
    cycle.run(decider=Decisions(["accept", "accept"]))
    assert len(cycle.trainings) == 2 and len(cycle.outer_exposures.events) == 1
    with pytest.raises(c.ContractError):
        cycle.outer_exposures.record(cycle.handoff, cycle.trainings[1])


def test_pending_outer_decision_cannot_drive_refresh_focus(tmp_path):
    cycle, _, _, _, _ = tracked_cycle(tmp_path)
    cycle.step()
    cycle.step()
    decision = Decisions(["needs_more_evidence"]).decide(cycle.pending_decision)
    with pytest.raises(c.ContractError):
        focus_from_outer_decision(
            cycle.pending_decision,
            decision,
            outer_round=2,
            skills=(),
            rationale="pending",
        )


def test_first_round_rejects_prior_exposures_or_lopt_restriction(tmp_path):
    refresh, provider, history = setup(tmp_path)
    rows = (row("a"),)
    with pytest.raises(c.ContractError):
        refresh.initialize(rows, history=exposure(history, rows[0]))
    with pytest.raises(c.ContractError):
        refresh.run_round(rows, outer_round=1, history=history, focus=focus(refresh))
    assert not provider.requests


def test_two_metric_scopes_share_one_forward_and_empty_evidence_stays_missing(tmp_path):
    refresh, provider, history = setup(tmp_path)
    scorer = refresh.scorer
    refresh = FocusedRefreshService(
        scorer,
        branch_id="branch",
        known_skills=SKILLS,
        output=tmp_path / "scoped",
        specs=(scorer.spec(), scorer.spec(scope="skill_evidence", skill="recovery")),
    )
    result = refresh.run_round(
        (row("a"),), outer_round=2, history=history, focus=focus(refresh)
    )
    assert len(provider.requests) == 1 and len(result.records[0].signals) == 2
    assert reload(result, refresh, history) == result


def test_outer_history_constructor_integration_and_wrong_domain(tmp_path):
    from self_curate.orchestration import OuterCycle
    from self_curate.training import TrainingService
    from self_curate.validation import TaskValidator
    from test_outer_cycle import recipe

    original, trainer, runner, producer, search = make_cycle(tmp_path / "source")
    kwargs = dict(
        expected_model=search.best.request.model_ref,
        expected_batch_id=search.best.request.batch.batch_id,
        trainer=TrainingService(trainer),
        validator=TaskValidator(runner),
        suite=original._suite,
        subset_recipe=recipe(),
        curation_recipe=recipe(epochs=2),
        curation_provider=producer,
    )
    view = search.best.prepared.context.view
    cycle = OuterCycle(
        search,
        view,
        output=tmp_path / "tracked",
        outer_exposures=OuterExposureHistory("branch", "mock"),
        **kwargs,
    )
    cycle.step()
    assert len(cycle.outer_exposures.events) == 1
    assert (
        cycle.state()["outer_exposure_history_hash"]
        == cycle.outer_exposures.history_hash
    )
    with pytest.raises(c.ContractError):
        OuterCycle(
            search,
            view,
            output=tmp_path / "wrong",
            outer_exposures=OuterExposureHistory("branch", "real"),
            **kwargs,
        )


def test_partial_outer_training_is_explicitly_unknown_not_silently_unused(tmp_path):
    from self_curate.training import TrainingService, TrainingFailure

    class Partial:
        backend = "mock"

        def train(self, request, output):
            raise TrainingFailure(
                "partial update fixture",
                usage=c.ResourceUsage(training_supervised_tokens=1),
            )

    cycle, _, _, _, _ = tracked_cycle(tmp_path)
    cycle._trainer = TrainingService(Partial())
    cycle.step()
    assert cycle.phase == "blocked" and cycle.outer_exposures.unresolved
    assert not cycle.outer_exposures.events
