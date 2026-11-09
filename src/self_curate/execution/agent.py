"""mini-SWE-agent integration with explicit token/time budgets and full evidence."""

from copy import deepcopy
from pathlib import Path
import json
import shlex
import time
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute bash in the task container.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}
SYSTEM = "You are a software engineering agent. Use the bash tool to inspect, edit and test the repository. When finished run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not alter evaluation infrastructure."


class BudgetReached(RuntimeError):
    pass


class ModelOutputError(RuntimeError):
    pass


class BudgetEnvironment:
    def __init__(self, environment, model):
        self.environment, self.model = environment, model

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def execute(self, action, **kwargs):
        remaining = self.model.request.task.budget.max_wall_seconds - (
            time.perf_counter() - self.model.started
        )
        if remaining <= 0:
            raise BudgetReached("wall_time_limit")
        return self.environment.execute(
            action,
            timeout=min(self.model.policy["tool_timeout_seconds"], max(0.1, remaining)),
        )


class MiniModel:
    def __init__(self, client, tokenizer, request, policy, output):
        self.client, self.tokenizer, self.request, self.policy, self.output = (
            client,
            tokenizer,
            request,
            policy,
            Path(output),
        )
        self.started = time.perf_counter()
        self.tokens = 0
        self.inputs = []

    def get_template_vars(self):
        return {}

    def format_message(self, **kwargs):
        return kwargs

    def query(self, messages):
        elapsed = time.perf_counter() - self.started
        if elapsed >= self.request.task.budget.max_wall_seconds:
            raise BudgetReached("wall_time_limit")
        if len(self.client.calls) >= self.request.task.budget.max_steps:
            raise BudgetReached("step_limit")
        if self.tokens >= self.request.task.budget.max_output_tokens:
            raise BudgetReached("output_token_limit")
        sent = deepcopy(
            [
                {k: v for k, v in m.items() if k != "extra"}
                for m in messages
                if m["role"] != "exit"
            ]
        )
        tools = [i for i, m in enumerate(sent) if m["role"] == "tool"]
        for i in tools[: -self.policy["recent_tool_observations"]]:
            sent[i][
                "content"
            ] = "[Older observation omitted by fixed input policy; full output retained on disk.]"
        ids = self.tokenizer.apply_chat_template(
            sent, tools=[BASH_TOOL], tokenize=True, add_generation_prompt=True
        )
        require(
            type(ids) is list and all(type(i) is int for i in ids),
            "Tokenizer must return token IDs",
        )
        remaining = self.request.task.budget.max_output_tokens - self.tokens
        available = self.request.task.budget.max_context_tokens - len(ids)
        if available <= 0:
            raise BudgetReached("context_limit")
        cap = min(remaining, self.policy["per_call_output_tokens"], available)
        self.inputs.append(ids)
        response = self.client.complete(
            sent,
            max_tokens=cap,
            seed=self.request.seed,
            tools=[BASH_TOOL],
            timeout=max(
                0.1,
                min(
                    self.policy["model_timeout_seconds"],
                    self.request.task.budget.max_wall_seconds - elapsed,
                ),
            ),
        )
        usage = response["usage"]
        self.tokens += usage["completion_tokens"]
        require(
            usage["completion_tokens"] <= cap,
            "Inference service exceeded output budget",
        )
        message = deepcopy(response["choices"][0]["message"])
        actions = []
        try:
            for call in message.get("tool_calls") or []:
                if call["function"]["name"] != "bash":
                    raise ModelOutputError("unsupported_tool")
                args = json.loads(call["function"]["arguments"])
                if (
                    type(args) is not dict
                    or set(args) != {"command"}
                    or type(args["command"]) is not str
                ):
                    raise ModelOutputError("malformed_bash_action")
                actions.append({"command": args["command"], "tool_call_id": call["id"]})
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ModelOutputError("malformed_tool_call") from exc
        if not actions:
            raise ModelOutputError("no_tool_call")
        message["extra"] = {"actions": actions, "cost": 0.0, "cost_usd_known": False}
        return message

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [
            {
                "role": "tool",
                "tool_call_id": a["tool_call_id"],
                "content": json.dumps(o, ensure_ascii=False),
            }
            for a, o in zip(message["extra"]["actions"], outputs)
        ]

    def serialize(self):
        return {"model_binding": self.client.binding}


def run_mini(
    client,
    tokenizer,
    request,
    environment,
    policy,
    output,
    *,
    agent_factory=None,
    environment_factory=None,
):
    """No gold patch/tests enter model messages. All commands run in Docker."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if agent_factory is None:
        from minisweagent.agents.default import DefaultAgent

        agent_factory = DefaultAgent
    if environment_factory is None:
        from minisweagent.environments.docker import DockerEnvironment

        environment_factory = DockerEnvironment
    require(
        environment["image"].find("@sha256:") > 0,
        "Pin the task image digest before inference",
    )
    env = None
    agent = None
    model = MiniModel(client, tokenizer, request, policy, output)
    exit_status = "environment_failed"
    patch = ""
    error = None
    try:
        env = environment_factory(
            image=environment["image"],
            cwd=environment["cwd"],
            timeout=policy["tool_timeout_seconds"],
            container_timeout=str(int(request.task.budget.max_wall_seconds) + 120)
            + "s",
            run_args=["--rm", "--label", "self-curate-request=" + request.record_hash],
        )
        setup = env.execute(
            {"command": "git reset --hard " + shlex.quote(environment["checkout"])}
        )
        require(setup["returncode"] == 0, "Task repository checkout failed")
        # Keep untracked submission files in the diff. No gold/test patch is applied.
        agent = agent_factory(
            model,
            BudgetEnvironment(env, model),
            system_template=policy.get("system_prompt", SYSTEM),
            instance_template="{{ task }}",
            step_limit=request.task.budget.max_steps,
            cost_limit=0.0,
            wall_time_limit_seconds=int(request.task.budget.max_wall_seconds),
            output_path=output / "mini-trajectory.json",
        )
        try:
            result = agent.run(task=environment["problem_statement"])
            exit_status = result.get("exit_status", "unknown_agent_exit")
        except (BudgetReached, ModelOutputError) as exc:
            exit_status = str(exc)
        diff = env.execute({"command": "git add -N . && git diff --binary"}, timeout=10)
        require(diff["returncode"] == 0, "Could not extract model patch")
        patch = diff["output"]
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if agent is not None:
            # This is our stable evidence format; native mini logs are also kept.
            atomic_json(output / "messages.json", agent.messages)
        if env is not None:
            env.cleanup()
    atomic_json(
        output / "patch.json",
        {"patch": patch, "exit_status": exit_status, "error": error},
    )
    atomic_json(
        output / "tokenization.json",
        {
            "tokenizer": request.model_ref.tokenizer.to_dict(),
            "input_ids_by_call": model.inputs,
            "native_generated_token_mask_available": False,
        },
    )
    return {
        "patch": patch,
        "exit_status": exit_status,
        "error": error,
        "calls": client.calls,
        "actions": (
            []
            if agent is None
            else [m for m in agent.messages if m["role"] == "assistant"]
        ),
        "observations": (
            [] if agent is None else [m for m in agent.messages if m["role"] == "tool"]
        ),
        "tokenization_hash": digest(
            {"tokenizer": request.model_ref.tokenizer.to_dict(), "inputs": model.inputs}
        ),
        "action_mask_hash": digest(
            {
                "format": "task-tool-actions-semantic-v1",
                "messages": [] if agent is None else agent.messages,
                "native_generated_token_mask_available": False,
            }
        ),
    }
