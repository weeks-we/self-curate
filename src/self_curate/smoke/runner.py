"""CPU-only nested integration acceptance; deliberately not a real experiment runner."""

from dataclasses import replace
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import socket
import sys
from time import perf_counter
import zipfile

from self_curate import contracts as c
from self_curate.agents import RoleTeam, default_prompts
from self_curate.data import MetricView, OuterExposureHistory
from self_curate.data.pool import file_sha256, write_json
from self_curate.operators import BudgetPolicy, ExecutionContext, QualityPolicy, SignalContext, prepare_selection
from self_curate.orchestration import OuterCycle
from self_curate.search import InnerSearch, StopDecision
from self_curate.signals.refresh import FocusedRefreshService
from self_curate.signals.refresh_contracts import focus_from_outer_decision
from self_curate.signals.service import SignalService
from self_curate.training import SFTRecipe, TrainingService
from self_curate.validation import TaskValidator
from . import mock


def check(condition, message):
    if not condition:
        raise RuntimeError("Acceptance failed: " + message)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema_version", "backend", "execution", "seed", "pool_size", "outer_rounds",
                "inner_evaluations", "values", "proposal_shares", "decision_scenarios", "focus_skills"}
    check(set(config) == expected, "configuration keys differ from the smoke protocol")
    check(config["schema_version"] == "1" and config["backend"] == "mock" and
          config["execution"] == "cpu_mock", "only explicit CPU mock is supported")
    check(type(config["seed"]) is int and config["seed"] >= 0, "integer seed required")
    check(type(config["pool_size"]) is int and 48 <= config["pool_size"] <= 96,
          "small pool must contain 48..96 synthetic records")
    check(config["outer_rounds"] == 3 and config["inner_evaluations"] == 5, "this acceptance is exactly 3 x 5")
    check(config["values"] == [0.1, 0.4, 0.2, 0.3, 0.15] and
          config["proposal_shares"] == [0.625, 0.375, 0.75, 0.25], "fixed mock best-not-last scenario required")
    check(config["decision_scenarios"] == [["accept", "accept"], ["needs_more_evidence", "reject"],
                                            ["accept", "reject"]], "fixed decision coverage required")
    check(config["focus_skills"] == list(mock.SKILLS), "known synthetic skill directions required")
    return config


def run(config, output):
    check(not output.exists(), "output exists; use --verify-only or choose a fresh directory")
    output.mkdir(parents=True)
    started = perf_counter()
    write_json(output / "config.json", config)
    source_root = Path(__file__).resolve().parents[1]
    source_checks = {p.relative_to(source_root).as_posix(): file_sha256(p)
                     for p in sorted(source_root.rglob("*.py"))}
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "backend": "mock", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "network": "blocked", "source_sha256": source_checks})
    pool, model = mock.records(config["pool_size"]), mock.initial_model()
    exposures = OuterExposureHistory("nested-smoke", "mock")
    task_suite, tasks, trainer = mock.suite(), mock.MockTasks(), mock.MockTrainer()
    validator = TaskValidator(tasks)
    baseline = validator.evaluate(model, task_suite, attempt_id="initial", evaluation_id="baseline",
                                  output=output / "initial-validation")
    objective = c.ObjectiveSpec("nested-smoke", (c.SkillWeight("recovery", 0.7), c.SkillWeight("regression", 0.3)),
                                (c.SkillConstraint("retain-regression", "regression", -0.01, None),),
                                mock.ref("estimator"), mock.ref("acceptance-policy"))
    policy = c.EditPolicy(0.125, 0.01, 0.99, 4, 4, 128, 2, 30)
    recipe = SFTRecipe(learning_rate=0.01, epochs=1, accumulation_steps=8, max_sequence_tokens=128, seed=config["seed"])
    rows, previous, parent_memories, role_calls = [], None, {}, 0
    for number in range(1, 4):
        folder = output / f"round-{number:02d}"
        folder.mkdir()
        print(f"[CPU mock] outer {number}/3: refresh, inner 0/5", flush=True)
        starting_model = model
        scorer = SignalService(model=model, backend=mock.MockSignals(), output=folder / "signals")
        refresh = FocusedRefreshService(scorer, branch_id=exposures.branch_id, known_skills=mock.SKILLS,
                                       output=folder / "refresh")
        focus = None if previous is None else focus_from_outer_decision(
            previous[0], previous[1], outer_round=number, skills=tuple(config["focus_skills"]),
            rationale="Configured synthetic Lopt direction input; no real skill inference")
        before_signals = {r.data_id: tuple(r.signals) for r in pool}
        prior_used = {r.data_id for r in pool if exposures.was_used(r)}
        refreshed = refresh.run_round(pool, outer_round=number, history=exposures, focus=focus)
        check(refreshed.status == "completed", "refresh did not complete")
        check(not prior_used.intersection(refreshed.plan.eligible_ids), "outer-exposed sample reentered selection")
        check(number != 1 or len(refreshed.plan.refresh_ids) == len(pool), "first initialization was not full pool")
        check(all(all(s in r.signals for s in before_signals[r.data_id]) for r in refreshed.records),
              "refresh dropped old metric versions")
        pool = refreshed.records
        eligible = tuple(r for r in pool if r.data_id in set(refreshed.plan.eligible_ids))
        context = ExecutionContext(MetricView(eligible), QualityPolicy(), BudgetPolicy(64, 0, 8, 0),
                                   model, (SignalContext(scorer.spec()),), training_seed=config["seed"])
        prepared = prepare_selection(mock.program(config["seed"]), context, objective, backend="mock",
                                     attempt_id=f"outer-{number}", invocation_id="initial-selection")
        check(prepared.status == "feasible", "initial selection infeasible")
        request = c.ValidationRequest(f"round-{number}", model, prepared.program, prepared.selection.batch,
                                       objective, context.context_hash, "predicted_gain", mock.ref("valuation-protocol"), "mock")
        client = mock.MockRoles(model, config["proposal_shares"])
        prompts = {r: replace(p, instructions=f"CPU acceptance mock only: private {r} instructions")
                   for r, p in default_prompts().items()}
        team = RoleTeam(client, model=model, session_id=f"round-{number}", attempt_id=f"outer-{number}",
                        backend="mock", prompts=prompts)
        search = InnerSearch(team, prepared, request, policy, search_id=f"search-{number}", use_model=True)
        for role in ("lexec", "leval", "lopt"):
            # Old history stays immutable in its old model namespace. Only an explicit
            # role-local summary/reference and shared outer evidence enter the new session.
            team.runtime.generate(role, {"round_bootstrap": number, "own_previous_memory": parent_memories.get(role),
                                        "outer_feedback": None if previous is None else previous[1].to_dict()})
        values = mock.MockValues(config["values"])
        stop = lambda state: StopDecision(True, "Configured CPU acceptance limit") if state.evaluated_count >= 5 else StopDecision(False)
        # A genuine inner save/restore in the middle of every search, with no reissue.
        for _ in range(3):
            interim = search.step(values)
            check(interim.status != "blocked", "inner midpoint blocked")
        check(search.state.evaluated_count == 2, "inner midpoint evaluation count")
        search.save(folder / "inner-midpoint")
        before_restore = len(client.calls)
        search = InnerSearch.restore(folder / "inner-midpoint", client, context.view,
                    expected_request=request, expected_policy=policy, expected_search_id=f"search-{number}",
                    expected_session_id=f"round-{number}", expected_attempt_id=f"outer-{number}",
                    expected_prompts=prompts)
        check(len(client.calls) == before_restore, "inner restore reissued model calls")
        result = search.run(values, stop_condition=stop)
        check(result.status == "stopped" and len(values.calls) == 5, "inner did not execute five evaluations")
        check(result.best.request == values.calls[1] and result.best.request.batch != values.calls[-1].batch,
              "historical best was replaced by the last selection")
        search.save(folder / "inner-finished")
        search.publish_best(folder / "best")
        write_json(folder / "role-calls.json", client.calls)
        namespaces = []
        for role in ("lexec", "leval", "lopt"):
            history = search.team.runtime.history(role)
            check(history and all(e.role == role and e.model_hash == model.model_hash for e in history),
                  "role/model history crossed a namespace")
            namespaces.append(search.team.runtime.namespace(role))
            parent_memories[role] = {"role": role, "model_hash": model.model_hash,
                                     "history_hash": history[-1].record_hash,
                                     "history_file": f"round-{number:02d}/inner-finished",
                                     "best_batch_id": result.best.request.batch.batch_id}
        check(len(set(namespaces)) == 3, "role namespaces collided")
        own_calls = {role: {e.call_id for e in search.team.runtime.history(role)}
                     for role in ("lexec", "leval", "lopt")}
        for call in client.calls:
            role, payload = call["role"], call["payload"]
            check(payload["model_ref"] == model.to_dict() and payload["prompt_hash"] == prompts[role].prompt_hash,
                  "role call used another model or prompt")
            check(all(item["call_id"] in own_calls[role] for item in payload["history"]),
                  "another role's history entered the model input")
        check(len(client.calls) == 15, "expected three bootstrap plus twelve iterative role calls")
        role_calls += len(client.calls)
        print(f"[CPU mock] outer {number}/3: inner 5/5; train / validate / decide", flush=True)
        cycle = OuterCycle(result, context.view, expected_model=model,
                           expected_batch_id=result.best.request.batch.batch_id,
                           trainer=TrainingService(trainer), validator=validator, suite=task_suite,
                           subset_recipe=recipe, curation_recipe=recipe, output=folder / "outer",
                           curation_provider=mock.MockCuration(), acceptance_policy=mock.ref("acceptance-policy"),
                           baseline=baseline, outer_exposures=exposures)
        decisions = mock.MockDecisions(config["decision_scenarios"][number - 1])
        training_start = len(trainer.requests)
        cycle.run(decider=decisions)
        if number == 2:
            check(cycle.phase == "awaiting_decision" and len(cycle.trainings) == 1, "insufficient evidence failed to wait")
            cycle.request_revalidation(reason="Configured mock additional evidence")
            cycle.run(decider=decisions)
            check(len(cycle.trainings) == 1 and len(cycle.evaluations) == 2, "revalidation retrained the subset")
        expected_trainings = 1 if number == 2 else 2
        check(len(trainer.requests) - training_start == expected_trainings, "unexpected stage training count")
        first = trainer.requests[training_start]
        check(tuple(e.data_id for e in first.data.examples) == result.best.request.batch.training_order,
              "outer did not consume exact best training order")
        check(first.starting_model == starting_model, "outer subset started from wrong model")
        if expected_trainings == 2:
            check(trainer.requests[training_start + 1].starting_model == cycle.trainings[0].result.candidate_model,
                  "curation did not continue subset weights")
        expected_model = (cycle.trainings[-1].result.candidate_model if number == 1 else
                          starting_model if number == 2 else cycle.trainings[0].result.candidate_model)
        check(cycle.accepted_model == expected_model, "accept/reject retained the wrong model")
        model, exposures = cycle.accepted_model, cycle.outer_exposures
        changed = {r.data_id: r for r in cycle.updated_records}
        pool = tuple(changed.get(r.data_id, r) for r in pool)
        check(all(exposures.was_used(context.view.record(i)) for i in result.best.request.batch.training_order),
              "physical training exposure lost, including rejected subset")
        previous = decisions.contexts[-1], cycle.decisions[-1]
        for evaluation in cycle.evaluations:
            if evaluation.report.model_ref == model:
                baseline = evaluation
        check(baseline.report.model_ref == model, "next round baseline model is stale")
        row = {"round": number, "inner_evaluations": len(values.calls), "starting_model": starting_model.model_hash,
               "accepted_model": model.model_hash, "phase": cycle.phase, "trainings": len(cycle.trainings),
               "task_evaluations": len(cycle.evaluations), "best_batch_id": result.best.request.batch.batch_id,
               "best_ids": list(result.best.request.batch.training_order), "best_value": result.best.validation.total_value,
               "last_value": result.state.current.validation.total_value, "refreshed": len(refreshed.plan.refresh_ids),
               "cached": len(refreshed.plan.cached_ids), "used_excluded": len(refreshed.plan.used_ids),
               "outer_exposure_events": len(exposures.events), "role_calls": len(client.calls),
               "decisions": [d.outcome for d in cycle.decisions]}
        rows.append(row)
        write_json(folder / "summary.json", row)
    check(len(tasks.requests) == 42 and len(trainer.requests) == 5 and role_calls == 45, "aggregate counts mismatch")
    check(not set(rows[0]["best_ids"]).intersection(rows[1]["best_ids"]) and
          not set(rows[2]["best_ids"]).intersection(rows[0]["best_ids"] + rows[1]["best_ids"]),
          "outer-used samples were selected again")
    check(rows[2]["refreshed"] == 0 and rows[2]["cached"] > 0, "unchanged rejected model did not reuse exact metric cache")
    check(source_checks == {p.relative_to(source_root).as_posix(): file_sha256(p)
                            for p in sorted(source_root.rglob("*.py"))}, "source changed while running")
    write_json(output / "exposures.json", exposures.to_dict())
    summary = {"status": "passed", "backend": "mock", "execution": "cpu_mock", "outer_rounds": 3,
               "inner_evaluations": 15, "role_calls": role_calls, "training_calls": len(trainer.requests),
               "task_rollouts": len(tasks.requests), "rounds": rows, "seconds": perf_counter() - started,
               "inner_midpoint_restore_verified": True, "outer_crash_resume_supported": False,
               "research_effectiveness_verified": False}
    write_json(output / "summary.json", summary)
    with (output / "rounds.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    checks = {p.relative_to(output).as_posix(): file_sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "checksums.json", checks)
    verify(output)
    return summary


def verify(output):
    checks = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checks.items():
        path = (output / name).resolve()
        check(path.is_relative_to(output.resolve()) and path.is_file() and file_sha256(path) == expected,
              "evidence changed: " + name)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    check(summary["status"] == "passed" and summary["inner_evaluations"] == 15 and
          summary["training_calls"] == 5 and summary["task_rollouts"] == 42, "summary acceptance counts")
    check([r["phase"] for r in summary["rounds"]] == ["completed", "rejected", "rejected"], "outer decision phases")
    return summary


def package(output):
    """Share only this mock run; verify every archived byte before delivery."""
    destination = output.with_name(output.name + ".results.zip")
    expected = {p.relative_to(output).as_posix(): file_sha256(p)
                for p in sorted(output.rglob("*")) if p.is_file()}
    if not destination.exists():
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as bundle:
            for name in expected:
                bundle.write(output / name, name)
    import hashlib
    with zipfile.ZipFile(destination) as bundle:
        check(bundle.testzip() is None and set(bundle.namelist()) == set(expected), "zip members or CRC mismatch")
        check(all(hashlib.sha256(bundle.read(name)).hexdigest() == value for name, value in expected.items()),
              "zip evidence differs from run")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def blocked(*a, **kw):
        raise RuntimeError("Networking is disabled in nested CPU mock acceptance")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    output = Path(args.output).resolve()
    try:
        if args.verify_only:
            result = verify(output)
        else:
            check(args.config is not None, "--config required for a new run")
            result = run(load_config(args.config), output)
        bundle = package(output)
        print(c.canonical_json({"status": result["status"], "backend": "mock", "output": str(output),
                                "outer_rounds": 3, "inner_evaluations": 15, "results_zip": str(bundle)}), flush=True)
        return 0
    except Exception as exc:
        print(c.canonical_json({"status": "failed", "error": str(exc), "output": str(output)}), flush=True)
        return 2
