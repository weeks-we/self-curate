"""Explicit CPU fixture: tiny random model and declared task rewards, no downloads."""

from pathlib import Path
from types import SimpleNamespace
import math

from self_curate import contracts as c
from self_curate.data import PrecomputedTokenizer, normalize_trajectory
from self_curate.data.pool import file_sha256
from self_curate.signals import SignalService
from self_curate.signals.torch_backend import TorchSignalBackend
from self_curate.training.torch_backend import StateDictModelStore
from self_curate.training.continuous import TorchContinuousBackend
from self_curate.validation import TaskSuite, TaskValidator, TaskRunResult
from self_curate.agents.prompts import default_prompts, PromptTemplate
from .inputs import PreparedInputs, BenchmarkSuite
from .runner import random_program
from .selection import count_context


def ref(name):
    return c.ArtifactRef("fixture://exp001/" + name, c.digest(name))


def tiny_factory():
    import torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.embedding.weight.requires_grad_(False)
            self.head = torch.nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    with torch.random.fork_rng():
        torch.manual_seed(42)
        return Tiny()


class FixtureRunner:
    backend = "mock"

    def __init__(self):
        self.requests = []

    def execute(self, request, output):
        self.requests.append(request)
        return TaskRunResult(
            request.record_hash,
            ref("actions"),
            ref("observations"),
            c.digest("tokenization"),
            c.digest("action-mask"),
            2,
            0.5,
            "completed",
            "declared synthetic task reward",
            c.ResourceUsage(c.ComputeCost(0, 0, 2), tool_calls=1),
        )


class FixtureClient:
    def __init__(self, model):
        self.model_ref, self.calls = model, []

    def generate_json(self, *, role, system_prompt, payload, timeout_seconds):
        self.calls.append((role, payload))
        if role == "lopt":
            k = payload["work"]["count"]
            raw = random_program(k, 42).to_dict()
            raw["program_id"] = "fixture-static-m0"
            raw["nodes"] = [
                {
                    "id": "rank",
                    "op": "NLLSelect",
                    "inputs": ["eligible_pool"],
                    "params": {
                        "scope": "assistant",
                        "skill": None,
                        "scale": "raw",
                        "interval": None,
                        "keep_fraction": 1.0,
                        "rank_direction": "ascending",
                        "reference_group": None,
                        "missing_policy": "error",
                    },
                },
                {
                    "id": "sample",
                    "op": "SampleSelect",
                    "inputs": ["rank"],
                    "params": {
                        "method": "ranked",
                        "size": k,
                        "token_target": None,
                        "seed": 42,
                        "strata": [],
                        "missing_policy": "error",
                    },
                },
            ]
            content = {
                "program": raw,
                "rationale": "Explicit fixture frozen M0 ranking",
                "hypothesis": "Unverified fixture only",
                "cited_feedback_id": payload["work"]["initial_capability"]["report_id"],
            }
        elif role == "lexec":
            content = {
                "action": "execute",
                "program_hash": payload["work"]["program_hash"],
                "context_hash": payload["work"]["context_hash"],
            }
        else:
            raise AssertionError("EXP001 must never invoke Leval")
        return {
            "content": content,
            "usage": c.ResourceUsage(
                agent_input_tokens=5, agent_output_tokens=2
            ).to_dict(),
        }


class MockPorts:
    backend = "mock"
    task_execution_hash = c.digest("exp001-declared-task-fixture-v1")
    agent_implementation_hash = c.digest("exp001-fixture-client-v1")
    agent_sampling = ref("agent-sampling-seed42")
    edit_policy = c.EditPolicy(0.1, 0.01, 0.99, 4, 4, 128, 2, 30)

    def __init__(self, config, output, asset_root):
        self.output, self.asset_root = Path(output), Path(asset_root)
        self.validator = TaskValidator(FixtureRunner())
        self.benchmark_runner = FixtureRunner()
        self.prompts = default_prompts()
        self.prompts["lopt"] = PromptTemplate(
            "lopt",
            "exp001-initial-policy",
            "1",
            "Propose one fixed complete program using M0 task feedback and M0 metrics. Return exactly the requested count. No Leval or test access.",
        )
        self.objective = c.ObjectiveSpec(
            "exp001-fixed-selection",
            (c.SkillWeight("recovery", 1.0),),
            (),
            ref("no-proxy-used"),
            ref("no-checkpoint-acceptance"),
        )

    def preflight(self, config):
        return {
            "passed": True,
            "backend": "mock",
            "device": "cpu",
            "target_model_loaded": False,
        }

    def prepare(self, config, output):
        import torch

        identity = c.TokenizerRef(
            "explicit-tiny-fixture",
            "1",
            c.digest("tiny-tokenizer"),
            "tiny-template",
            c.digest("tiny-template"),
            "external",
        )
        root = self.asset_root.parent / "mock-initial-model"
        root.mkdir(parents=True, exist_ok=True)
        initial = root / "state.pt"
        if not initial.exists():
            torch.save(tiny_factory().state_dict(), initial)
        self.model = c.ModelRef(
            "explicit-tiny-random-fixture",
            "1",
            file_sha256(initial),
            identity,
            "fixture",
        )
        self.store = StateDictModelStore(
            tiny_factory, initial_model=self.model, initial_state=initial
        )
        self.client = FixtureClient(self.model)
        self.scorer = SignalService(
            model=self.model,
            backend=TorchSignalBackend(self.store, backend="mock"),
            output=self.output / "scoring-origin",
        )
        rows = []
        from self_curate.contracts.parsing import load_json
        from self_curate.contracts.types import integer

        count = load_json(config.adapter_settings_json).get("pool_size", 30)
        integer(count, "mock pool size", 30)
        for i in range(count):
            raw = {
                "schema_version": "1",
                "data_id": "fixture-" + str(i),
                "source": {
                    "source_id": "explicit-synthetic-coding",
                    "revision": "fixture-v1",
                    "record_id": "fixture-" + str(i),
                    "artifact": ref("raw-" + str(i)).to_dict(),
                    "family_ids": ["train-family-" + str(i)],
                },
                "messages": [
                    {"role": "user", "content": "fixture task " + str(i)},
                    {"role": "assistant", "content": "fixture answer " + str(i)},
                ],
                "quality": {
                    "status": "pass",
                    "method": "executable",
                    "verifier_id": "fixture-only",
                    "evidence": ref("fixture-verdict").to_dict(),
                },
                "tokenization": {
                    "identity": identity.to_dict(),
                    "input_ids": [1, 2, 3, 4 + i % 3, 5, 6],
                    "assistant_mask": [False, False, False, True, True, False],
                    "message_spans": [{"start": 1, "end": 2}, {"start": 3, "end": 5}],
                },
                "annotations": {
                    "tokenizer_hash": identity.tokenizer_hash,
                    "template_hash": identity.template_hash,
                    "skills": [
                        {
                            "skill": "recovery",
                            "annotator_id": "explicit-fixture",
                            "spans": [{"message_index": 1, "start": 3, "end": 5}],
                        }
                    ],
                },
            }
            rows.append(
                normalize_trajectory(raw, PrecomputedTokenizer(identity), ("recovery",))
            )

        def task(name, split):
            return c.TaskRecord(
                name,
                "fixture-v1",
                ("recovery",),
                ref(name + "-payload"),
                ref("environment"),
                ref("scorer"),
                split,
                c.TaskBudget(2, 8, 32, 5.0),
                "mock",
                (name + "-family",),
            )

        dev = TaskSuite(
            "fixture-development",
            (task("dev-1", "development"),),
            (42,),
            ref("sampling"),
            ref("protocol"),
            "task_mean",
            "mock",
        )
        bench = BenchmarkSuite(
            "fixture-benchmark",
            (task("test-1", "test"),),
            (42,),
            ref("sampling"),
            ref("protocol"),
            "mock",
        )
        return PreparedInputs(
            tuple(rows),
            self.model,
            dev,
            (bench,),
            ("recovery",),
            c.canonical_json(
                {
                    "dataset_revision": "explicit-fixture-v1",
                    "author_selection": "synthetic-not-SWE-smith",
                }
            ),
            ref("declared-source-and-near-duplicate-isolation-fixture"),
            c.CostLedger("fixture-preparation", (), "mock"),
        )

    def context(self, records, config):
        k = math.floor(len(records) * config.selection_fraction)
        return count_context(records, model=self.model, count=k, seed=config.seed)

    def training_backend(self, branch):
        return TorchContinuousBackend(self.store, backend="mock", device="cpu")

    def attach_training(self, config, model):
        inputs = self.prepare(config, self.output / "unused-worker-preparation")
        from self_curate.contracts.types import require

        require(inputs.model == model, "Worker initial model differs from frozen plan")

    def release_initial(self):
        pass  # CPU fixture model instances are isolated and weakly held.


def build(config, output, asset_root):
    return MockPorts(config, output, asset_root)
