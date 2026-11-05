"""Exact subset SFT -> validate -> decide -> curation SFT -> validate -> decide.

Rejected candidates never replace accepted state. No automatic retry, hidden
acceptance threshold, final-test access, or claim of crash recovery is included.
"""

from pathlib import Path
import os

from self_curate.contracts import (
    ArtifactRef,
    CostLedger,
    TrainingUse,
    digest,
)
from self_curate.contracts.types import require
from self_curate.data import record_candidate_training, resolve_training
from self_curate.data.pool import write_json
from self_curate.search import SearchResult
from self_curate.training import (
    SFTRecipe,
    TrainingData,
    TrainingRequest,
    TrainingOutcome,
    TrainingService,
    prepare_subset,
)
from self_curate.validation import TaskEvaluation, TaskSuite, TaskValidator
from self_curate.training.artifacts import verify_local_checkpoint
from .contracts import AcceptanceDecision, DecisionContext, OuterHandoff


class OuterCycle:
    def __init__(
        self,
        search_result: SearchResult,
        view,
        *,
        expected_model,
        expected_batch_id,
        trainer: TrainingService,
        validator: TaskValidator,
        suite: TaskSuite,
        subset_recipe: SFTRecipe,
        curation_recipe: SFTRecipe,
        output,
        curation_provider=None,
        acceptance_policy=None,
        baseline=None,
        history_artifacts=(),
    ):
        require(
            isinstance(search_result, SearchResult)
            and search_result.status == "stopped"
            and search_result.state.phase == "stopped"
            and search_result.state.unresolved_operation_json is None,
            "Outer training requires a cleanly stopped inner search",
        )
        best = search_result.require_best()
        require(
            isinstance(suite, TaskSuite) and suite.backend == best.request.backend,
            "Validation suite differs from search execution domain",
        )
        require(
            isinstance(subset_recipe, SFTRecipe)
            and isinstance(curation_recipe, SFTRecipe),
            "Both training recipes must be explicitly provided",
        )
        subset = prepare_subset(
            best,
            view,
            expected_model=expected_model,
            expected_batch_id=expected_batch_id,
        )
        self._handoff = OuterHandoff(
            best.prepared.attempt_id,
            digest(search_result.to_dict()),
            best.request.request_hash,
            best.request.batch,
            expected_model,
            best.request.backend,
            history_artifacts,
        )
        require(
            acceptance_policy is None or isinstance(acceptance_policy, ArtifactRef),
            "Invalid acceptance policy",
        )
        if acceptance_policy is not None and best.request.backend == "real":
            require(
                not acceptance_policy.uri.strip()
                .lower()
                .startswith(("fixture://", "mock://")),
                "Known mock acceptance policy cannot enter real training",
            )
        if baseline is not None:
            require(
                isinstance(baseline, TaskEvaluation)
                and baseline.protocol_verified
                and baseline.report.model_ref == expected_model
                and baseline.suite == suite,
                "Baseline must use the same accepted model and exact task protocol",
            )
        self._trainer, self._validator = trainer, validator
        self._suite, self._provider = suite, curation_provider
        self._policy = acceptance_policy
        self._recipes = {"subset": subset_recipe, "curation": curation_recipe}
        self._subset, self._curation = subset, None
        self._baseline = baseline
        self._accepted = expected_model
        self._phase, self._stage = "awaiting_training", "subset"
        self._training, self._evaluation = None, None
        self._trainings, self._evaluations, self._decisions = [], [], []
        self._events = list(search_result.state.ledger.events)
        self._records = {i: view.record(i) for i in best.request.batch.training_order}
        self._reason = None
        self._journal_count = 0
        self._output = Path(output).resolve()
        self._output.mkdir(parents=True, exist_ok=False)
        write_json(self._output / "handoff.json", self._handoff.to_dict())
        write_json(self._output / "subset_data.json", subset.to_dict())
        write_json(self._output / "suite.json", suite.to_dict())
        write_json(
            self._output / "recipes.json",
            {k: v.to_dict() for k, v in self._recipes.items()},
        )
        self._write_accepted(self._accepted, decision=None)
        self._save("created")

    @property
    def handoff(self):
        return self._handoff

    @property
    def accepted_model(self):
        return self._accepted

    @property
    def phase(self):
        return self._phase

    @property
    def stage(self):
        return self._stage

    @property
    def trainings(self):
        return tuple(self._trainings)

    @property
    def evaluations(self):
        return tuple(self._evaluations)

    @property
    def decisions(self):
        return tuple(self._decisions)

    @property
    def updated_records(self):
        return tuple(self._records[i] for i in self._handoff.batch.training_order)

    @property
    def ledger(self):
        return CostLedger(
            "outer-" + self._handoff.record_hash,
            tuple(self._events),
            self._handoff.backend,
        )

    @property
    def pending_decision(self):
        if self._phase != "awaiting_decision":
            return None
        return DecisionContext(
            self._handoff.attempt_id,
            self._stage,
            self._accepted,
            self._training,
            self._evaluation,
            self._baseline,
            self._policy,
        )

    def configure_acceptance(self, policy):
        require(
            isinstance(policy, ArtifactRef)
            and not self._decisions
            and self._policy is None,
            "Acceptance policy is already pinned",
        )
        if self._handoff.backend == "real":
            require(
                not policy.uri.strip().lower().startswith(("fixture://", "mock://")),
                "Known mock acceptance policy cannot enter a real attempt",
            )
        self._policy = policy
        self._save("acceptance_policy_configured")

    def supply_curation_data(self, data):
        require(
            self._stage == "curation"
            and self._phase == "awaiting_curation_data"
            and self._curation is None,
            "Curation inputs cannot be changed during training",
        )
        require(
            isinstance(data, TrainingData)
            and data.kind == "curation"
            and data.source_hash == self._handoff.record_hash
            and data.tokenizer == self._accepted.tokenizer
            and data.backend == self._handoff.backend,
            "Curation data must bind this stopped search, model and tokenizer",
        )
        # Validate complete sequences and explicit recipe before making the input runnable.
        self._request(data)
        write_json(self._output / "curation_data.json", data.to_dict())
        self._curation, self._phase = data, "awaiting_training"
        self._save("curation_data_ready")

    def _request(self, data):
        return TrainingRequest(
            "sft-" + self._stage + "-" + self._handoff.record_hash,
            self._handoff.attempt_id,
            self._accepted,
            data,
            (
                self._handoff.batch.batch_id
                if self._stage == "subset"
                else data.record_hash
            ),
            (
                self._handoff.batch.training_data_hash
                if self._stage == "subset"
                else data.record_hash
            ),
            self._recipes[self._stage],
            self._handoff.backend,
        )

    def _merge(self, ledger):
        by_id = {e.event_id: e for e in self._events}
        for event in ledger.events:
            require(event.backend == self._handoff.backend, "Mixed outer cost domains")
            require(
                event.event_id not in by_id or by_id[event.event_id] == event,
                "Conflicting physical cost event",
            )
            if event.event_id not in by_id:
                self._events.append(event)
                by_id[event.event_id] = event

    def step(self, *, decider=None):
        if self._phase in ("completed", "rejected", "blocked"):
            return self.state()
        if self._phase == "awaiting_decision":
            if decider is None or self._policy is None:
                return self.state()
            context = self.pending_decision
            if (
                self._decisions
                and self._decisions[-1].context_hash == context.record_hash
            ):
                return (
                    self.state()
                )  # An insufficient-evidence decision waits for new evidence.
            # Policy exceptions do not dispatch training or change accepted weights.
            self.submit_decision(decider.decide(context))
            return self.state()
        if self._phase == "awaiting_curation_data":
            if self._provider is not None:
                self.supply_curation_data(self._provider.prepare(self._handoff))
            return self.state()
        try:
            if self._phase == "awaiting_training":
                data = self._subset if self._stage == "subset" else self._curation
                request = self._request(data)
                self._phase = "training"
                self._save("training_started")
                result = self._trainer.train(
                    request, self._output / ("train-" + self._stage)
                )
                require(
                    isinstance(result, TrainingOutcome),
                    "Invalid training service outcome",
                )
                self._merge(result.ledger)
                require(
                    result.request == request,
                    "Trainer returned another fixed training request",
                )
                self._training = result
                self._trainings.append(result)
                if self._stage == "subset":
                    use = TrainingUse(
                        request.training_id,
                        request.starting_model.model_hash,
                        result.record.receipt,
                        "pending",
                    )
                    self._records = {
                        i: record_candidate_training(r, use)
                        for i, r in self._records.items()
                    }
                self._phase = "awaiting_validation"
                self._save("training_completed")
            elif self._phase == "awaiting_validation":
                self._phase = "validating"
                self._save("validation_started")
                result = self._validator.evaluate(
                    self._training.result.candidate_model,
                    self._suite,
                    attempt_id=self._handoff.attempt_id,
                    evaluation_id="outer-"
                    + self._handoff.record_hash
                    + "-"
                    + str(len(self._evaluations)),
                    output=self._output / ("validate-" + str(len(self._evaluations))),
                    checkpoint=self._training.result.checkpoint,
                )
                require(
                    isinstance(result, TaskEvaluation),
                    "Invalid task evaluation outcome",
                )
                self._merge(result.ledger)
                require(
                    result.suite == self._suite
                    and result.report.model_ref == self._training.result.candidate_model
                    and result.checkpoint == self._training.result.checkpoint
                    and result.protocol_verified
                    and all(
                        r.attempt_id == self._handoff.attempt_id
                        for r in result.execution_requests
                    ),
                    "Validator returned another candidate or task protocol",
                )
                self._evaluation = result
                self._evaluations.append(result)
                self._phase = "awaiting_decision"
                self._save("validation_completed")
        except Exception as exc:
            if isinstance(getattr(exc, "ledger", None), CostLedger):
                self._merge(exc.ledger)
            self._phase, self._reason = "blocked", str(exc)
            self._save("operation_failed")
        return self.state()

    def run(self, *, decider=None):
        while True:
            before = (self._phase, self._stage, len(self._decisions))
            self.step(decider=decider)
            if self._phase in ("completed", "rejected", "blocked"):
                break
            if self._phase == "awaiting_decision":
                if decider is None or self._policy is None:
                    break
                if (
                    self._decisions
                    and self._decisions[-1].outcome == "needs_more_evidence"
                    and self._decisions[-1].context_hash
                    == self.pending_decision.record_hash
                ):
                    break
            if before == (self._phase, self._stage, len(self._decisions)):
                break
        return self.state()

    def submit_decision(self, decision):
        require(
            isinstance(decision, AcceptanceDecision),
            "Expected typed acceptance decision",
        )
        if decision in self._decisions:
            return self.state()  # Exactly repeated delivery is not another promotion.
        context = self.pending_decision
        require(
            context is not None
            and self._policy is not None
            and decision.context_hash == context.record_hash,
            "Decision is not bound to the current candidate/evaluation/policy",
        )
        if self._handoff.backend == "real":
            require(
                not any(
                    a.uri.strip().lower().startswith(("fixture://", "mock://"))
                    for a in (self._policy, decision.evidence)
                ),
                "Mock acceptance evidence cannot promote real weights",
            )
        if decision.outcome == "accept":
            require(
                context.evaluation.report.status == "valid",
                "An unscored/invalid capability report cannot be accepted",
            )
            verify_local_checkpoint(self._training.result.checkpoint)
        write_json(
            self._output / ("decision-" + str(len(self._decisions)) + ".json"),
            {"context": context.to_dict(), "decision": decision.to_dict()},
        )
        if decision.outcome == "accept":
            self._write_accepted(
                self._training.result.candidate_model, decision=decision
            )
        self._decisions.append(decision)
        if self._stage == "subset" and decision.outcome in ("accept", "reject"):
            label = "accepted" if decision.outcome == "accept" else "rejected"
            self._records = {
                i: resolve_training(r, self._training.request.training_id, label)
                for i, r in self._records.items()
            }
        if decision.outcome == "accept":
            self._accepted = self._training.result.candidate_model
            self._baseline = self._evaluation
            self._reason = None
            if self._stage == "subset":
                self._stage, self._phase = "curation", "awaiting_curation_data"
            else:
                self._phase = "completed"
        elif decision.outcome == "reject":
            self._phase, self._reason = "rejected", decision.reason
        else:
            self._reason = decision.reason
        self._save("decision_recorded")
        return self.state()

    def request_revalidation(self, *, reason):
        from self_curate.contracts.types import text

        text(reason, "revalidation reason")
        require(
            self._phase == "awaiting_decision"
            and self._decisions
            and self._decisions[-1].outcome == "needs_more_evidence"
            and self._decisions[-1].context_hash == self.pending_decision.record_hash,
            "Revalidation requires an explicit insufficient-evidence decision",
        )
        self._phase, self._reason = "awaiting_validation", reason
        self._save("revalidation_requested")

    def _write_accepted(self, model, *, decision):
        # This pointer belongs only to this outer attempt, never a global model service.
        pending = self._output / (".accepted-" + os.urandom(8).hex() + ".tmp")
        write_json(
            pending,
            {
                "model": model.to_dict(),
                "decision": (None if decision is None else decision.to_dict()),
                "handoff_hash": self._handoff.record_hash,
            },
        )
        with pending.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, self._output / "accepted.json")

    def _save(self, event):
        self._journal_count += 1
        folder = self._output / "journal"
        folder.mkdir(exist_ok=True)
        path = folder / (str(self._journal_count).zfill(6) + ".json")
        require(not path.exists(), "Outer journal cannot be overwritten")
        write_json(
            path,
            {
                "event": event,
                "state": self.state(),
                "ledger": self.ledger.to_dict(),
                "usage": [r.usage.to_dict() for r in self.updated_records],
            },
        )

    def state(self):
        return {
            "schema_version": "1",
            "phase": self._phase,
            "stage": self._stage,
            "handoff_hash": self._handoff.record_hash,
            "backend": self._handoff.backend,
            "accepted_model": self._accepted.to_dict(),
            "training_ids": [x.record.training_id for x in self._trainings],
            "capability_ids": [x.report.report_id for x in self._evaluations],
            "decisions": [x.to_dict() for x in self._decisions],
            "policy": None if self._policy is None else self._policy.to_dict(),
            "reason": self._reason,
            "global_model_committed": False,
            "resume_implemented": False,
        }
