"""Small, explicitly synthetic adapters for the nested CPU acceptance run."""

from dataclasses import replace
import json
from pathlib import Path

from self_curate import contracts as c
from self_curate.contracts.parsing import message
from self_curate.data import FixtureTokenizer, normalize_trajectory
from self_curate.data.pool import file_sha256, write_json
from self_curate.signals.contracts import TokenMetricResult
from self_curate.training import BackendTrainingResult, TrainingData, TrainingExample
from self_curate.validation import TaskRunResult, TaskSuite

SKILLS = ("recovery", "regression")


def ref(name):
    return c.ArtifactRef("mock://nested-smoke/" + name, c.digest(name))


def initial_model():
    return c.ModelRef("nested-smoke-only", "1", c.digest("synthetic-initial-weights"),
                      FixtureTokenizer.identity, "fixture")


def records(count):
    result = []
    for i in range(count):
        name = f"data-{i:04d}"
        raw = {
            "schema_version": "1", "data_id": name,
            "source": {"source_id": "synthetic-coding", "revision": "1",
                       "record_id": name, "artifact": ref(name).to_dict(),
                       "family_ids": [name]},
            "messages": [{"role": "user", "content": "Synthetic task " + name},
                         {"role": "assistant", "content": f"fix-{i:04d}"}],
            "quality": {"status": "pass", "method": "executable",
                        "verifier_id": "mock-only", "evidence": ref("verdict-" + name).to_dict()},
            "metadata": {"task_type": "bugfix", "repository": f"repo-{i % 2}",
                         "duplicate_groups": []},
        }
        encoded = FixtureTokenizer().encode(tuple(message(m) for m in raw["messages"]))
        start = encoded.message_spans[1].start
        raw["annotations"] = {
            "tokenizer_hash": encoded.identity.tokenizer_hash,
            "template_hash": encoded.identity.template_hash,
            "skills": [{"skill": SKILLS[i % 2], "annotator_id": "mock-only",
                        "spans": [{"message_index": 1, "start": start, "end": start + 4}],
                        "complete_events": 1, "behaviors": ["test_after_final_edit"]}],
        }
        result.append(normalize_trajectory(raw, FixtureTokenizer(), SKILLS))
    return tuple(result)


def program(seed):
    # Eight equal-length examples allow exact integer quota changes by 1/8.
    return c.parse_program({
        "schema_version": "1", "program_id": "nested-seed", "seed": seed,
        "output": "mix", "nodes": [
            {"id": "a-skill", "op": "SkillSelect", "inputs": ["eligible_pool"],
             "params": {"skills": ["recovery"]}},
            {"id": "a", "op": "NLLSelect", "inputs": ["a-skill"],
             "params": {"interval": [0, 100], "rank_direction": "ascending"}},
            {"id": "b-skill", "op": "SkillSelect", "inputs": ["eligible_pool"],
             "params": {"skills": ["regression"]}},
            {"id": "b", "op": "EntropySelect", "inputs": ["b-skill"],
             "params": {"scale": "raw", "interval": [0, 100], "rank_direction": "ascending"}},
            {"id": "mix", "op": "MixAllocate", "inputs": ["a", "b"],
             "params": {"branch_token_shares": {"a": 0.5, "b": 0.5}}},
        ]})


def suite():
    tasks = tuple(c.TaskRecord(name, "1", (skill,), ref(name), ref("environment"),
                              ref("scorer"), "development", c.TaskBudget(8, 32, 512, 30), "mock")
                  for name, skill in (("dev-a", "recovery"), ("dev-b", "recovery"),
                                      ("dev-c", "regression")))
    return TaskSuite("nested-mock-dev", tasks, (11, 12), ref("sampling"),
                     ref("task-protocol"), "task_mean", "mock")


class MockSignals:
    backend = "mock"

    @property
    def implementation_hash(self):
        return file_sha256(Path(__file__))

    def score(self, request, output):
        Path(output).mkdir(parents=True, exist_ok=False)
        value = 1 + int(c.digest((request.data_id, request.model_ref.model_hash))[:6], 16) / 2**24
        evidence = Path(output) / "declared-token-values.json"
        write_json(evidence, {"mock": True, "request_hash": request.record_hash,
                              "positions": request.positions, "nll": value, "entropy": value / 2})
        return TokenMetricResult(request.record_hash, request.positions,
                                 (value,) * len(request.positions), (value / 2,) * len(request.positions),
                                 c.ArtifactRef(evidence.resolve().as_uri(), file_sha256(evidence)),
                                 c.ResourceUsage(), "mock")


class MockRoles:
    def __init__(self, model, shares):
        self.model_ref, self.shares, self.calls = model, iter(shares), []

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        self.calls.append({"role": role, "system_prompt": system_prompt, "payload": payload})
        work = payload["work"]
        if "round_bootstrap" in work:
            content = {"acknowledged": c.digest(work)}
        elif role == "lexec":
            content = {"action": "execute", "program_hash": work["program_hash"],
                       "context_hash": work["context_hash"]}
        elif role == "leval":
            content = {"source_feedback_hash": work["feedback_hash"],
                       "commentary": "Synthetic interpretation; numeric facts unchanged.",
                       "hypotheses": ["A configured mock proposal needs evaluation."]}
        else:
            source = work["input"]
            raw = source["parent_program"]
            raw = json.loads(c.canonical_json(raw))
            raw["parent_id"] = raw["program_id"]
            raw["program_id"] = "mock-child-" + c.digest(source)[:20]
            share = next(self.shares)
            next(n for n in raw["nodes"] if n["id"] == "mix")["params"]["branch_token_shares"] = {
                "a": share, "b": 1 - share}
            content = {"program": raw, "rationale": "Configured mock quota change",
                       "hypothesis": "Synthetic only", "cited_feedback_id": source["feedback"]["feedback_id"]}
        return {"content": content}


class MockValues:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def evaluate(self, request):
        self.calls.append(request)
        value = next(self.values)
        return c.ValidationResult("mock-value-" + str(len(self.calls)), request.request_hash,
                                  "valid", tuple(c.SkillValue(w.skill, value) for w in request.objective.weights),
                                  sum(w.weight * value for w in request.objective.weights),
                                  tuple(c.ConstraintResult(b.name, value,
                                        (b.minimum is None or value >= b.minimum) and
                                        (b.maximum is None or value <= b.maximum))
                                        for b in request.objective.constraints),
                                  None, ref("declared-value-" + request.request_hash), c.ResourceUsage(), "mock")


class MockTrainer:
    backend = "mock"

    def __init__(self):
        self.requests = []

    def train(self, request, output):
        self.requests.append(request)
        tokens = request.data.supervised_tokens * request.recipe.epochs
        return BackendTrainingResult(replace(request.starting_model, weights_hash=c.digest(request.record_hash)),
                                     ref("declared-weights-" + request.record_hash), tokens,
                                     request.recipe.epochs * ((len(request.data.examples) +
                                         request.recipe.accumulation_steps - 1) // request.recipe.accumulation_steps),
                                     None, c.ResourceUsage(training_supervised_tokens=tokens))


class MockTasks:
    backend = "mock"

    def __init__(self):
        self.requests = []

    def execute(self, request, output):
        self.requests.append(request)
        return TaskRunResult(request.record_hash, ref("actions"), ref("observations"),
                             c.digest("tokens"), c.digest("mask"), 2, 0.5, "completed",
                             "Synthetic outcome", c.ResourceUsage(c.ComputeCost(0, 0, 2), tool_calls=1))


class MockCuration:
    def prepare(self, handoff):
        return TrainingData("curation", handoff.record_hash, handoff.starting_model.tokenizer,
                            (TrainingExample("curation-" + handoff.attempt_id, c.digest(handoff.record_hash),
                                             (1, 2, 3), (-100, 2, 3)),), (ref("synthetic-curation"),), "mock")


class MockDecisions:
    def __init__(self, outcomes):
        self.outcomes, self.contexts = iter(outcomes), []

    def decide(self, context):
        from self_curate.orchestration import AcceptanceDecision
        self.contexts.append(context)
        return AcceptanceDecision(context.record_hash, next(self.outcomes),
                                  "Configured mock branch, not a research acceptance rule", ref("decision"))
