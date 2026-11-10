"""Atomic, hash-chained OuterCycle journals and process-independent recovery."""

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
from pathlib import Path
import platform

from self_curate.contracts import digest, parse_contract
from self_curate.contracts.parsing import artifact, model_ref, usage_history
from self_curate.contracts.types import require
from self_curate.data import attach_usage
from self_curate.data.outer_usage import parse_outer_exposures
from self_curate.data.pool import file_sha256
from self_curate.training.artifacts import verify_local_checkpoint
from self_curate.training.execution import code_identity
from self_curate.training.recovery import atomic_json, read_json, lease
from self_curate.training.runlog import new_run_id


def source_identity():
    root = Path(__file__).resolve().parents[1]
    return digest({"python": platform.python_version(), "sources": {
        p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*.py"))}})


def initial_anchor(cycle):
    return digest({"handoff": cycle._handoff.to_dict(), "subset": cycle._subset.to_dict(),
                   "suite": cycle._suite.to_dict(), "recipes": {k: v.to_dict() for k, v in cycle._recipes.items()},
                   "policy": None if cycle._policy is None else cycle._policy.to_dict(),
                   "baseline": None if cycle._baseline is None else cycle._baseline.to_dict(),
                   "exposures": None if cycle._outer_exposures is None else cycle._outer_exposures.to_dict(),
                   "records": {k: v.record_hash for k, v in cycle._records.items()},
                   "trainer": code_identity(cycle._trainer.backend),
                   "validator": code_identity(cycle._validator.runner),
                   "provider": None if cycle._provider is None else code_identity(cycle._provider)})


def journals(root):
    folder = Path(root) / "journal"
    values = sorted(folder.glob("[0-9]*.json"))
    require(all(p.stem.isdigit() and not p.is_symlink() for p in values), "Unsafe outer journal")
    require([int(p.stem) for p in values] == list(range(1, len(values) + 1)), "Outer journal sequence is incomplete")
    return values


@contextmanager
def transaction(cycle):
    if cycle._lease_depth:
        yield
        return
    with lease(cycle._output / ".outer.lease"):
        require(len(journals(cycle._output)) == cycle._journal_count,
                "Stale OuterCycle instance; restore the current journal before continuing")
        cycle._lease_depth += 1
        try:
            yield
        finally:
            cycle._lease_depth -= 1


def serialized(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with transaction(self):
            return function(self, *args, **kwargs)
    return wrapped


def save(cycle, event):
    require(source_identity() == cycle._source_identity, "Runtime source changed during outer cycle")
    previous = cycle._journal_hash
    recovery = {
        "version": "outer-recovery-v1", "source": cycle._source_identity, "anchor": cycle._initial_anchor,
        "previous_hash": previous, "pending_operation": cycle._pending_operation,
        "resume_io": cycle._resume_io, "retries": cycle._recovery_retries,
        "training": None if cycle._training is None else cycle._training.to_dict(),
        "evaluation": None if cycle._evaluation is None else cycle._evaluation.to_dict(),
        "trainings": [v.to_dict() for v in cycle._trainings],
        "evaluations": [v.to_dict() for v in cycle._evaluations],
        "baseline": None if cycle._baseline is None else cycle._baseline.to_dict(),
        "curation": None if cycle._curation is None else cycle._curation.to_dict(),
        "outer_exposures": None if cycle._outer_exposures is None else cycle._outer_exposures.to_dict(),
        "usage": {k: v.usage.to_dict() for k, v in cycle._records.items()},
    }
    envelope = {"event": event, "state": cycle.state(), "ledger": cycle.ledger.to_dict(),
                "usage": [r.usage.to_dict() for r in cycle.updated_records], "recovery": recovery}
    checksum = digest(envelope)
    envelope["journal_hash"] = checksum
    index = cycle._journal_count + 1
    path = cycle._output / "journal" / (str(index).zfill(6) + ".json")
    require(not path.exists(), "Outer journal cannot be overwritten")
    atomic_json(path, envelope)
    cycle._journal_count, cycle._journal_hash = index, checksum


def restore(cycle):
    from .io import parse_outer_contract, load_task_evaluation
    with lease(cycle._output / ".outer.lease"):
        cycle._lease_depth = 1
        try:
            for filename, expected in (
                ("handoff.json", cycle._handoff.to_dict()), ("subset_data.json", cycle._subset.to_dict()),
                ("suite.json", cycle._suite.to_dict()),
                ("recipes.json", {k: v.to_dict() for k, v in cycle._recipes.items()}),
            ):
                require(digest(read_json(cycle._output / filename)) == digest(expected),
                        "Frozen outer input changed: " + filename)
            paths = journals(cycle._output)
            require(bool(paths), "No committed outer journal; old nonrecoverable runs are not migrated")
            previous, last = None, None
            for path in paths:
                value = read_json(path)
                checksum = value.pop("journal_hash", None)
                require(checksum == digest(value), "Outer journal checksum changed")
                saved = value.get("recovery", {})
                require(saved.get("version") == "outer-recovery-v1" and saved["previous_hash"] == previous,
                        "Missing recovery payload or broken outer journal chain")
                require(saved["source"] == cycle._source_identity and saved["anchor"] == cycle._initial_anchor,
                        "Outer recovery changed source, model, best batch, data, suite, recipe, policy or backend")
                previous, last = checksum, value
            saved, state = last["recovery"], last["state"]
            require(state["handoff_hash"] == cycle._handoff.record_hash, "Wrong outer attempt")
            cycle._journal_count, cycle._journal_hash = len(paths), previous
            cycle._phase, cycle._stage, cycle._reason = state["phase"], state["stage"], state["reason"]
            cycle._accepted = model_ref(state["accepted_model"])
            cycle._policy = None if state["policy"] is None else artifact(state["policy"])
            cycle._decisions = [parse_outer_contract("AcceptanceDecision", v) for v in state["decisions"]]
            cycle._events = list(parse_contract("CostLedger", last["ledger"]).events)
            cycle._trainings = [parse_outer_contract("TrainingOutcome", v) for v in saved["trainings"]]
            cycle._evaluations = [parse_outer_contract("TaskEvaluation", v) for v in saved["evaluations"]]
            cycle._training = None if saved["training"] is None else parse_outer_contract("TrainingOutcome", saved["training"])
            cycle._evaluation = None if saved["evaluation"] is None else parse_outer_contract("TaskEvaluation", saved["evaluation"])
            cycle._baseline = None if saved["baseline"] is None else parse_outer_contract("TaskEvaluation", saved["baseline"])
            cycle._curation = None if saved["curation"] is None else parse_outer_contract("TrainingData", saved["curation"])
            cycle._outer_exposures = None if saved["outer_exposures"] is None else parse_outer_exposures(saved["outer_exposures"])
            require(set(saved["usage"]) == set(cycle._records), "Restored outer usage has different members")
            cycle._records = {k: attach_usage(v, usage_history(saved["usage"][k])) for k, v in cycle._records.items()}
            cycle._pending_operation, cycle._resume_io = saved["pending_operation"], saved["resume_io"]
            cycle._recovery_retries = saved["retries"]
            for outcome in cycle._trainings:
                verify_local_checkpoint(outcome.result.checkpoint)
                stage = outcome.request.data.kind
                require(digest(read_json(cycle._output / ("train-" + stage) / "outcome.json")) == digest(outcome.to_dict()),
                        "Saved training evidence changed")
            for i, evaluation in enumerate(cycle._evaluations):
                verified = load_task_evaluation(cycle._output / ("validate-" + str(i)),
                           expected_model=evaluation.report.model_ref, expected_suite=cycle._suite,
                           expected_checkpoint=evaluation.checkpoint)
                require(verified == evaluation, "Saved evaluation evidence changed")
            if cycle._phase in ("training", "validating"):
                cycle._phase = "awaiting_training" if cycle._phase == "training" else "awaiting_validation"
                cycle._resume_io = True
            # An atomic decision intent can precede a committed controller journal.
            # Complete that same decision; never call the decider a second time.
            intent = cycle._output / ("decision-" + str(len(cycle._decisions)) + ".json")
            pointer = read_json(cycle._output / "accepted.json")
            require(pointer["handoff_hash"] == cycle._handoff.record_hash, "Accepted pointer belongs to another attempt")
            if intent.exists():
                require(cycle._phase == "awaiting_decision", "Decision intent outside decision phase")
                value = read_json(intent)
                context = parse_outer_contract("DecisionContext", value["context"])
                decision = parse_outer_contract("AcceptanceDecision", value["decision"])
                require(context == cycle.pending_decision, "Uncommitted decision has another context")
                require(model_ref(pointer["model"]) in (
                        cycle._accepted, context.training.result.candidate_model if decision.outcome == "accept" else cycle._accepted),
                        "Accepted pointer is unrelated to the durable decision")
                cycle.submit_decision(decision)
            curation = cycle._output / "curation_data.json"
            if cycle._phase == "awaiting_curation_data" and curation.exists():
                cycle.supply_curation_data(parse_outer_contract("TrainingData", read_json(curation)))
            # accepted.json is an attempt-local pointer, never authority over journal decisions.
            pointer = read_json(cycle._output / "accepted.json")
            require(pointer["handoff_hash"] == cycle._handoff.record_hash and
                    model_ref(pointer["model"]) == cycle._accepted, "Accepted pointer disagrees with committed decisions")
            cycle._save("controller_restored")
            atomic_json(cycle._output / "recovery-attempts" / (new_run_id() + ".json"),
                        {"phase": cycle._phase, "stage": cycle._stage,
                         "pending_operation": cycle._pending_operation, "journal_hash": cycle._journal_hash})
        finally:
            cycle._lease_depth = 0
