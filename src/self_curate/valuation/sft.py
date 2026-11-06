"""Candidate SFT -> development task reward delta; accepted weights stay frozen."""

from threading import RLock
from pathlib import Path
import uuid

from self_curate.contracts import ArtifactRef, CostLedger, digest
from self_curate.contracts.types import require
from self_curate.data.pool import write_json, file_sha256
from self_curate.training import SFTRecipe, TrainingRequest, TrainingOutcome
from self_curate.validation import TaskSuite, TaskEvaluation
from .common import (
    ValuationFailure,
    artifact,
    candidate_data,
    combine_events,
    cost_ledger,
    make_result,
    new_run,
    require_isolation,
    training_usage_records,
    usage_from_ledger,
    implementation_identity,
    invoke_service,
    restore_usage,
)


class SFTValidator:
    """Every candidate starts at the same model; no candidate is auto-accepted.

    An optional imported baseline is verified but not recharged. Otherwise the
    first call evaluates the frozen baseline once and records its physical costs.
    The baseline protocol and all SFT settings are part of this evaluator's ID.
    """

    def __init__(
        self, *, model, view, trainer, validator, suite, recipe, output, baseline=None
    ):
        require(
            isinstance(suite, TaskSuite) and isinstance(recipe, SFTRecipe),
            "Typed SFT validation settings required",
        )
        require(
            trainer.backend.backend == validator.runner.backend == suite.backend,
            "Mixed candidate evaluation domains",
        )
        if suite.backend == "real":
            require(
                all(
                    getattr(a, "implementation_hash", None) is not None
                    for a in (trainer.backend, validator.runner)
                ),
                "Real candidate validation adapters must declare implementation_hash including configuration",
            )
        if baseline is not None:
            self._check_evaluation(baseline, model, suite, baseline.checkpoint)
            require(baseline.report.status == "valid", "Imported baseline is invalid")
        self.model, self.view, self.trainer, self.validator = (
            model,
            view,
            trainer,
            validator,
        )
        self.suite, self.recipe, self.output = suite, recipe, output
        self.backend = suite.backend
        self._baseline, self._baseline_identity = baseline, (
            None if baseline is None else baseline.record_hash
        )
        # A fresh implicit baseline is a new observation, even with equal seeds.
        # Stable-boundary continuation must restore its exact saved reference.
        self._reference_namespace = (
            "auto-" + uuid.uuid4().hex
            if baseline is None
            else "imported-" + baseline.record_hash
        )
        self.protocol = ArtifactRef(
            "record://leval/sft/" + self.identity, self.identity
        )
        self.trainings, self.evaluations, self.results = [], [], []
        self._records = {}
        self._lock = RLock()

    @property
    def identity(self):
        return digest(
            {
                "method": "candidate-sft-reward-delta-v1",
                "model": self.model.to_dict(),
                "suite": self.suite.record_hash,
                "recipe": self.recipe.record_hash,
                "baseline": self._baseline_identity,
                "baseline_policy": "single-immutable-reference-v2",
                "reference_namespace": self._reference_namespace,
                "implementation": implementation_identity(
                    self.trainer,
                    self.trainer.backend,
                    self.validator,
                    self.validator.runner,
                ),
            }
        )

    @property
    def baseline(self):
        return self._baseline

    @property
    def updated_records(self):
        return tuple(self._records[i] for i in sorted(self._records))

    def save_reference(self, output):
        """Persist the exact baseline for a stable-boundary search continuation."""
        with self._lock:
            require(self._baseline is not None, "No valid baseline has executed")
            path = Path(output).resolve()
            path.mkdir(parents=True, exist_ok=False)
            write_json(
                path / "reference.json",
                {
                    "schema_version": "2",
                    "model_hash": self.model.model_hash,
                    "suite_hash": self.suite.record_hash,
                    "recipe_hash": self.recipe.record_hash,
                    "protocol": self.protocol.to_dict(),
                    "baseline_identity": self._baseline_identity,
                    "reference_namespace": self._reference_namespace,
                    "baseline": self._baseline.to_dict(),
                    "usage_records": [
                        {
                            "data_id": r.data_id,
                            "content_hash": r.content_hash,
                            "tokenization_hash": digest(r.tokenization.to_dict()),
                            "usage": r.usage.to_dict(),
                        }
                        for r in self.updated_records
                    ],
                },
            )
            return artifact(path / "reference.json")

    @classmethod
    def restore_reference(cls, path, *, expected_hash, expected_protocol, **settings):
        """Read verified old evidence; do not re-run or recharge its baseline."""
        from self_curate.contracts.parsing import (
            fields,
            load_json,
            artifact as parse_artifact,
        )
        from self_curate.orchestration.io import parse_outer_contract

        source = Path(path)
        require(file_sha256(source) == expected_hash, "SFT reference checksum changed")
        raw = fields(
            load_json(source.read_text(encoding="utf-8")),
            (
                "schema_version",
                "model_hash",
                "suite_hash",
                "recipe_hash",
                "protocol",
                "baseline_identity",
                "baseline",
            ),
            ("usage_records", "reference_namespace"),
        )
        require(raw["schema_version"] in ("1", "2"), "Unsupported SFT reference")
        require(
            raw["schema_version"] != "2"
            or {"usage_records", "reference_namespace"}.issubset(raw),
            "Missing candidate exposure history/reference namespace",
        )
        baseline = parse_outer_contract("TaskEvaluation", raw["baseline"])
        require(
            raw["baseline_identity"] in (None, baseline.record_hash),
            "Baseline origin changed",
        )
        provider = cls(**settings, baseline=baseline)
        require(
            raw["model_hash"] == provider.model.model_hash
            and raw["suite_hash"] == provider.suite.record_hash
            and raw["recipe_hash"] == provider.recipe.record_hash,
            "Restored evaluator settings changed",
        )
        provider._baseline_identity = raw["baseline_identity"]
        if "reference_namespace" in raw:
            from self_curate.contracts.types import text

            text(raw["reference_namespace"], "reference namespace")
            provider._reference_namespace = raw["reference_namespace"]
        require(
            parse_artifact(raw["protocol"]) == expected_protocol
            and provider.identity == expected_protocol.sha256,
            "Restored evaluation protocol changed",
        )
        provider.protocol = expected_protocol
        provider._records = restore_usage(provider.view, raw.get("usage_records", []))
        return provider

    @staticmethod
    def _check_evaluation(value, model, suite, checkpoint, attempt_id=None):
        require(
            isinstance(value, TaskEvaluation)
            and value.protocol_verified
            and value.report.model_ref == model
            and value.suite == suite
            and value.checkpoint == checkpoint,
            "Task evaluation changed model/suite/checkpoint/protocol",
        )
        if attempt_id is not None:
            require(
                all(r.attempt_id == attempt_id for r in value.execution_requests),
                "Candidate evaluation attempt mismatch",
            )

    def evaluate(self, request):
        with self._lock:
            return self._evaluate(request)

    def _evaluate(self, request):
        require(
            request.model_ref == self.model
            and request.backend == self.backend
            and request.measure == "reward_delta"
            and request.protocol == self.protocol
            and self.protocol.sha256 == self.identity,
            "SFT request changed frozen model/method/protocol",
        )
        data = candidate_data(request, self.view)
        require(
            all(
                len(x.input_ids) <= self.recipe.max_sequence_tokens
                for x in data.examples
            ),
            "Complete SFT trajectories exceed the recipe; truncation is forbidden",
        )
        require_isolation(self.view, data, self.suite)
        require(
            {w.skill for w in request.objective.weights}.issubset(
                {s for t in self.suite.tasks for s in t.skills}
            ),
            "Development suite misses objective skills",
        )
        run = new_run(self.output, request)
        events, training, evaluated, baseline = (), None, None, None
        attempt = "candidate-" + run.name

        def details():
            return {
                "method": "candidate_sft",
                "starting_model": self.model.to_dict(),
                "baseline": (
                    None if self._baseline is None else self._baseline.to_dict()
                ),
                "attempted_baseline": (
                    baseline.to_dict() if isinstance(baseline, TaskEvaluation) else None
                ),
                "training": (
                    training.to_dict()
                    if isinstance(training, TrainingOutcome)
                    else None
                ),
                "evaluation": (
                    evaluated.to_dict()
                    if isinstance(evaluated, TaskEvaluation)
                    else None
                ),
                "ledger": cost_ledger(run, events, self.backend).to_dict(),
                "usage_summary": cost_ledger(run, events, self.backend).summary(),
                "accepted_model_changed": False,
                "candidate_automatically_accepted": False,
                "usage_records": [r.to_dict() for r in self.updated_records],
            }

        try:
            if self._baseline is None:
                baseline, costs = invoke_service(
                    lambda: self.validator.evaluate(
                        self.model,
                        self.suite,
                        attempt_id=attempt,
                        evaluation_id="baseline-" + run.name,
                        output=run / "baseline",
                    ),
                    run=run,
                    stage="baseline",
                    operation="evaluation",
                    subject_id="baseline-" + run.name,
                    model=self.model,
                    attempt_id=attempt,
                    backend=self.backend,
                )
                events = combine_events(events, costs)
                self._check_evaluation(baseline, self.model, self.suite, None, attempt)
                if baseline.report.status != "valid":
                    result = make_result(
                        request,
                        run,
                        values={},
                        usage=usage_from_ledger(cost_ledger(run, events, self.backend)),
                        details=details(),
                        reason="BASELINE_TASK_EVIDENCE_UNAVAILABLE",
                    )
                    self.results.append(result)
                    return result
                self._baseline = baseline
                self.evaluations.append(baseline)
            train_request = TrainingRequest(
                "inner-subset-" + run.name,
                attempt,
                self.model,
                data,
                request.batch.batch_id,
                request.batch.training_data_hash,
                self.recipe,
                self.backend,
            )
            training, costs = invoke_service(
                lambda: self.trainer.train(train_request, run / "training"),
                run=run,
                stage="training",
                operation="training",
                subject_id=train_request.training_id,
                model=self.model,
                attempt_id=attempt,
                backend=self.backend,
            )
            events = combine_events(events, costs)
            require(
                isinstance(training, TrainingOutcome)
                and training.request == train_request,
                "Candidate training returned another request",
            )
            self.trainings.append(training)
            training_usage_records(request, training, self.view, self._records)
            evaluated, costs = invoke_service(
                lambda: self.validator.evaluate(
                    training.result.candidate_model,
                    self.suite,
                    attempt_id=attempt,
                    evaluation_id="candidate-" + run.name,
                    output=run / "candidate",
                    checkpoint=training.result.checkpoint,
                ),
                run=run,
                stage="candidate",
                operation="evaluation",
                subject_id="candidate-" + run.name,
                model=training.result.candidate_model,
                attempt_id=attempt,
                backend=self.backend,
            )
            events = combine_events(events, costs)
            self._check_evaluation(
                evaluated,
                training.result.candidate_model,
                self.suite,
                training.result.checkpoint,
                attempt,
            )
            self.evaluations.append(evaluated)
            values = {}
            if evaluated.report.status == "valid":
                baseline_values = {
                    m.skill: m.mean_reward for m in self._baseline.report.metrics
                }
                current_values = {
                    m.skill: m.mean_reward for m in evaluated.report.metrics
                }
                values = {
                    w.skill: current_values[w.skill] - baseline_values[w.skill]
                    for w in request.objective.weights
                }
            result = make_result(
                request,
                run,
                values=values,
                usage=usage_from_ledger(cost_ledger(run, events, self.backend)),
                details=details(),
                reason=(
                    None
                    if evaluated.report.status == "valid"
                    else "CANDIDATE_TASK_EVIDENCE_UNAVAILABLE"
                ),
            )
            self.results.append(result)
            return result
        except Exception as exc:
            if isinstance(getattr(exc, "ledger", None), CostLedger):
                events = combine_events(events, exc.ledger.events)
            ledger = cost_ledger(run, events, self.backend)
            usage = usage_from_ledger(ledger)
            try:
                write_json(
                    run / "failure.json",
                    {
                        **details(),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                evidence = artifact(run / "failure.json")
            except Exception:
                evidence = None
            raise ValuationFailure(str(exc), usage=usage, evidence=evidence) from exc
