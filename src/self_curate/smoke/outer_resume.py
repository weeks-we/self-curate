"""Cross-process OuterCycle interruption acceptance for the CPU-only smoke run."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from self_curate import contracts as c
from self_curate.contracts.parsing import model_ref, artifact
from self_curate.agents.prompts import PromptTemplate
from self_curate.data import MetricView
from self_curate.data.outer_usage import parse_outer_exposures
from self_curate.data.pool import write_json
from self_curate.orchestration import OuterCycle
from self_curate.orchestration.io import parse_outer_contract
from self_curate.search import InnerSearch
from self_curate.signals.io import parse_scoring_record
from self_curate.training import TrainingService
from self_curate.validation import TaskValidator
from . import mock


class SimulatedProcessExit(BaseException):
    pass


def restore_kwargs(job, trainer, tasks):
    return dict(expected_model=model_ref(job["model"]), expected_batch_id=job["batch_id"],
                trainer=TrainingService(trainer), validator=TaskValidator(tasks),
                suite=parse_outer_contract("TaskSuite", job["suite"]),
                subset_recipe=parse_outer_contract("SFTRecipe", job["recipes"]["subset"]),
                curation_recipe=parse_outer_contract("SFTRecipe", job["recipes"]["curation"]),
                output=job["output"], curation_provider=mock.MockCuration(),
                acceptance_policy=artifact(job["policy"]),
                baseline=parse_outer_contract("TaskEvaluation", job["baseline"]),
                outer_exposures=parse_outer_exposures(job["exposures"]))


def recover_in_process(cycle, *, number, result, view, inner_request, edit_policy, prompts,
                       folder, trainer, tasks, decisions, outcomes):
    job = {"backend": "mock", "round": number, "view": [view.record(i).to_dict() for i in view.data_ids],
           "inner_request": inner_request.to_dict(), "edit_policy": edit_policy.to_dict(),
           "prompts": {r: p.to_dict() for r, p in prompts.items()},
           "inner_checkpoint": str((folder / "inner-finished").resolve()),
           "model": cycle.handoff.starting_model.to_dict(), "batch_id": cycle.handoff.batch.batch_id,
           "suite": cycle._suite.to_dict(), "recipes": {k: v.to_dict() for k, v in cycle._recipes.items()},
           "policy": cycle._policy.to_dict(), "baseline": cycle._baseline.to_dict(),
           "exposures": cycle.outer_exposures.to_dict(), "output": str((folder / "outer").resolve()),
           "outcomes": outcomes}
    # Interrupt AFTER durable service output / decision intent, BEFORE controller commit.
    try:
        if number == 1:
            original = cycle._trainer.train
            def train_then_exit(*args, **kwargs):
                original(*args, **kwargs)
                raise SimulatedProcessExit()
            cycle._trainer.train = train_then_exit
            try:
                cycle.step()
            finally:
                cycle._trainer.train = original
        elif number == 2:
            cycle.step()
            original = cycle._validator.evaluate
            def validate_then_exit(*args, **kwargs):
                original(*args, **kwargs)
                raise SimulatedProcessExit()
            cycle._validator.evaluate = validate_then_exit
            try:
                cycle.step()
            finally:
                cycle._validator.evaluate = original
        else:
            cycle.step()
            cycle.step()
            original = cycle._save
            def decision_then_exit(event):
                if event == "decision_recorded":
                    raise SimulatedProcessExit()
                return original(event)
            cycle._save = decision_then_exit
            try:
                cycle.step(decider=decisions)
            finally:
                cycle._save = original
    except SimulatedProcessExit:
        pass
    else:
        raise RuntimeError("Mock interruption point was not reached")
    path = folder / "outer-resume-input.json"
    write_json(path, job)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "self_curate.smoke.outer_resume", "--job", str(path.resolve()),
               "--job-hash", c.digest(job)]
    child = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (folder / "outer-resume-worker.log").write_text(child.stdout + child.stderr, encoding="utf-8")
    if child.returncode != 0:
        raise RuntimeError("Outer resume subprocess failed: " + child.stderr[-2000:])
    report = json.loads((folder / "outer-resume-worker.json").read_text(encoding="utf-8"))
    trainer.requests.extend(parse_outer_contract("TrainingRequest", v) for v in report["new_trainings"])
    tasks.requests.extend(parse_outer_contract("TaskRunRequest", v) for v in report["new_rollouts"])
    decisions.contexts.extend(parse_outer_contract("DecisionContext", v) for v in report["new_contexts"])
    restored = OuterCycle.restore(result, view, **restore_kwargs(job, trainer, tasks))
    before = len(trainer.requests), len(tasks.requests), len(restored.decisions)
    restored.run(decider=decisions)
    if before != (len(trainer.requests), len(tasks.requests), len(restored.decisions)):
        raise RuntimeError("Terminal outer recovery repeated work")
    write_json(folder / "outer-resume-verification.json", {
        "passed": True, "subprocess_pid": report["pid"], "parent_pid": os.getpid(),
        "scenario": ("training_output_before_controller_commit", "validation_output_before_controller_commit",
                     "accepted_decision_before_controller_commit")[number - 1],
        "terminal_resume_did_not_repeat_work": True})
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--job-hash", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    def blocked(*a, **kw):
        raise RuntimeError("No network in outer CPU resume acceptance")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    path = Path(args.job)
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("backend") != "mock" or c.digest(job) != args.job_hash:
        raise RuntimeError("Wrong CPU recovery input or checksum")
    view = MetricView(tuple(parse_scoring_record(v) for v in job["view"]))
    request = c.parse_feedback_contract("ValidationRequest", job["inner_request"])
    policy = c.parse_feedback_contract("EditPolicy", job["edit_policy"])
    number = job["round"]
    prompts = {r: PromptTemplate(**v) for r, v in job["prompts"].items()}
    search = InnerSearch.restore(job["inner_checkpoint"], mock.MockRoles(request.model_ref, []), view,
                 expected_request=request, expected_policy=policy, expected_search_id=f"search-{number}",
                 expected_session_id=f"round-{number}", expected_attempt_id=f"outer-{number}", expected_prompts=prompts)
    trainer, tasks = mock.MockTrainer(), mock.MockTasks()
    cycle = OuterCycle.restore(search.result(), view, **restore_kwargs(job, trainer, tasks))
    decisions = mock.MockDecisions(job["outcomes"][len(cycle.decisions):])
    cycle.run(decider=decisions)
    if number == 2:
        if cycle.phase != "awaiting_decision":
            raise RuntimeError("Insufficient-evidence branch did not wait after resume")
        cycle.request_revalidation(reason="Configured CPU recovery additional evidence")
        cycle.run(decider=decisions)
    if cycle.phase not in ("completed", "rejected"):
        raise RuntimeError("Recovered outer did not reach expected terminal phase: " + str(cycle.state()))
    write_json(path.parent / "outer-resume-worker.json", {
        "pid": os.getpid(), "state": cycle.state(),
        "new_trainings": [r.to_dict() for r in trainer.requests],
        "new_rollouts": [r.to_dict() for r in tasks.requests],
        "new_contexts": [v.to_dict() for v in decisions.contexts]})


if __name__ == "__main__":
    main()
