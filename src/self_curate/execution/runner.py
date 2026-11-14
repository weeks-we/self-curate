"""Concrete TaskRunner for development SWE-smith, SWE benchmarks and Harbor."""

from pathlib import Path
from time import perf_counter
from self_curate.contracts import ComputeCost, ResourceUsage, digest
from self_curate.validation import TaskRunResult
from self_curate.contracts.parsing import artifact
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json
from .artifacts import read_artifact, publish_json, local_path
from .agent import run_mini
from .model import CompletionClient, AdapterRouter
from .graders import SWEJudge, HarborJudge
from .runtime import verify_runtime


class ExecutionFailure(RuntimeError):
    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


class CodingTaskRunner:
    def __init__(
        self,
        router,
        tokenizer_factory,
        *,
        execution,
        backend="real",
        swe_judge=None,
        harbor=None,
        pro=None,
        mini=run_mini,
        client_factory=CompletionClient,
        verify=verify_runtime,
    ):
        require(
            backend in ("real", "mock")
            and (backend != "real" or execution == "server"),
            "Real harness requires explicit server execution",
        )
        self.router, self.tokenizer_factory = router, tokenizer_factory
        self.backend, self.swe_judge, self.harbor = (
            backend,
            swe_judge or SWEJudge(),
            harbor or HarborJudge(),
        )
        from .pro_v2 import ProV2Judge
        self.pro = pro or ProV2Judge()
        self.mini, self.client_factory, self.verify = mini, client_factory, verify

    def execute(self, request, output):
        require(request.backend == self.backend, "Task execution domain mismatch")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        calls = []
        generated = None
        inputs = None
        tool_calls = None

        def usage():
            return ResourceUsage(
                ComputeCost(perf_counter() - started, None, None),
                agent_input_tokens=inputs,
                agent_output_tokens=generated,
                tool_calls=tool_calls,
            )

        try:
            environment, scorer, payload = map(
                read_artifact,
                (request.task.environment, request.task.scorer, request.task.payload),
            )
            policy = read_artifact(request.protocol)
            sampling = read_artifact(request.sampling)
            require(
                sampling
                == {"schema_version": "1", "temperature": 0, "seed_policy": "request"},
                "EXP001 task runner currently supports fixed greedy sampling",
            )
            kind = environment["harness"]
            require(
                kind == scorer["harness"]
                and kind in ("swe_smith", "swe_bench", "swe_rebench", "harbor", "swe_pro_v2"),
                "Task environment/scorer mismatch",
            )
            lock = read_artifact(artifact(scorer["runtime_lock"]))
            self.verify(lock, kinds=("harbor",) if kind in ("harbor", "swe_pro_v2") else (("mini",) if kind == "swe_rebench" else ("mini", kind)))
            atomic_json(output / "request.json", request.to_dict())
            with self.router.bind(request, output) as binding:
                if kind in ("harbor", "swe_pro_v2"):
                    local_path(artifact(environment["task_directory"]), directory=True)
                    require(
                        environment["preserve_native_resources"] is True,
                        "Do not replace Harbor native task resources",
                    )
                    row = (self.pro if kind == "swe_pro_v2" else self.harbor).run(
                        request,
                        payload["task_path"],
                        binding,
                        output / "harbor",
                        timeout=scorer["timeout_seconds"],
                    )
                    generated, inputs = row["generated_tokens"], row["input_tokens"]
                    require(
                        type(generated) is int and generated >= 0,
                        "Harbor omitted native output token usage",
                    )
                    reward, status, reason = (
                        row["reward"],
                        row["status"],
                        "harbor_native_verifier",
                    )
                    trace = row["raw"]
                    tokenization = digest(
                        {
                            "tokenizer": request.model_ref.tokenizer.to_dict(),
                            "harbor": trace,
                        }
                    )
                    mask = digest(
                        {
                            "schema": "harbor-native-trace-v1",
                            "trace": trace,
                            "gradient_mask_available": False,
                        }
                    )
                else:
                    instance = payload["instance"]
                    require(
                        instance["instance_id"] == request.task.task_id,
                        "Task text/ID mismatch",
                    )
                    require(
                        environment["checkout"]
                        == (
                            instance["instance_id"]
                            if kind == "swe_smith"
                            else instance["base_commit"]
                        ),
                        "Wrong repository start state",
                    )
                    for key in (
                        "recent_tool_observations",
                        "per_call_output_tokens",
                        "tool_timeout_seconds",
                        "model_timeout_seconds",
                    ):
                        require(
                            type(policy[key]) is int and policy[key] > 0,
                            "Invalid task execution budget",
                        )
                    client = self.client_factory(binding, output / "model-calls")
                    row = self.mini(
                        client,
                        self.tokenizer_factory(request.model_ref),
                        request,
                        {
                            **environment,
                            "problem_statement": instance["problem_statement"],
                        },
                        policy,
                        output / "agent",
                    )
                    calls = client.calls
                    tool_calls = sum(
                        len(m.get("extra", {}).get("actions", []))
                        for m in row.get("actions", [])
                    )
                    generated = (
                        sum(c["output_tokens"] for c in calls)
                        if all(c["output_tokens"] is not None for c in calls)
                        else None
                    )
                    inputs = (
                        sum(c["input_tokens"] for c in calls)
                        if all(c["input_tokens"] is not None for c in calls)
                        else None
                    )
                    require(
                        generated is not None, "Task has unaccounted inference work"
                    )
                    reason = row["exit_status"]
                    tokenization, mask = (
                        row["tokenization_hash"],
                        row["action_mask_hash"],
                    )
                    if row["error"] is not None:
                        reward, status = None, "environment_failed"
                    elif not row["patch"].strip():
                        reward, status = 0.0, "task_failed"
                    else:
                        judged = self.swe_judge.grade(
                            request,
                            instance,
                            row["patch"],
                            kind,
                            output / "grading",
                            timeout=scorer["timeout_seconds"],
                        )
                        reward = judged["reward"]
                        status = (
                            "completed" if reward is not None else "environment_failed"
                        )
                        atomic_json(output / "grading-summary.json", judged)
                        if judged.get("status") == "timeout" and reward is not None:
                            status = "task_failed"
                            reason += ";official_test_timeout"
                    trace = row
            actions = publish_json(
                output / "actions.json",
                {"request_hash": request.record_hash, "trace": trace},
            )
            observations = publish_json(
                output / "observations.json",
                {
                    "execution_directory": str(output),
                    "calls": calls,
                    "status": status,
                    "stop_reason": reason,
                    "full_native_logs_retained": True,
                },
            )
            result = TaskRunResult(
                request.record_hash,
                actions,
                observations,
                tokenization,
                mask,
                generated,
                reward,
                status,
                reason,
                usage(),
            )
            atomic_json(output / "task-result.json", result.to_dict())
            atomic_json(
                output / "timing-tokens.json",
                {
                    "wall_seconds": perf_counter() - started,
                    "input_tokens": inputs,
                    "output_tokens": generated,
                    "gpu_seconds": None,
                    "gpu_seconds_estimated": False,
                    "native_gradient_masks_available": False,
                },
            )
            return result
        except Exception as exc:
            atomic_json(
                output / "failure.json",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "usage": usage().to_dict(),
                },
            )
            raise ExecutionFailure(str(exc), usage()) from exc


def build_task_runner(settings, model, *, execution):
    from .model import http_json
    from self_curate.training.recovery import read_json
    from .artifacts import local_path

    pool = read_json(local_path(artifact(settings["serving_pool"])))
    router = AdapterRouter(
        pool["entries"],
        model,
        settings["initial_adapter"],
        transport=http_json,
        slot_root=settings.get("slot_root"),
    )

    def tokenizer_factory(reference):
        require(
            reference.tokenizer == model.tokenizer,
            "Tokenizer changed between checkpoints",
        )
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            settings["tokenizer_path"], local_files_only=True, trust_remote_code=False
        )

    from .pro_v2 import ProV2Judge
    return CodingTaskRunner(
        router,
        tokenizer_factory,
        execution=execution,
        swe_judge=SWEJudge(
            python=settings.get("swe_python") or __import__("sys").executable,
            pythons={"swe_rebench": settings.get("rebench_python") or settings.get("swe_python") or __import__("sys").executable}
        ),
        pro=ProV2Judge(executable=settings.get("harbor_executable", "harbor"), tooling=settings.get("pro_tooling")),
        harbor=HarborJudge(
            executable=settings.get("harbor_executable", "harbor"),
            gpu_pool=settings.get("harbor_gpu_pool", ()),
            gpu_slot_root=settings.get("harbor_gpu_slot_root"),
            installed_root=settings.get("harbor_installed_root"),
        ),
    )
