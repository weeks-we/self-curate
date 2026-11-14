"""Official SWE-smith/SWE-bench subprocess adapters and native Harbor trials."""

from pathlib import Path
import sys
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json, read_json
from .process import execute
from .artifacts import read_artifact


class SWEJudge:
    def __init__(self, *, python=sys.executable, process=execute, pythons=None):
        self.python, self.process = python, process
        self.pythons = dict(pythons or {})

    def grade(self, request, instance, patch, harness, output, *, timeout):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        prediction = {
            "instance_id": instance["instance_id"],
            "model_patch": patch,
            "model_name_or_path": "sc-" + request.model_ref.model_hash,
        }
        value = {
            "harness": harness,
            "instance": instance,
            "prediction": prediction,
            "run_id": "sc-" + request.record_hash[:32],
            "timeout": timeout,
            "expected_image": read_artifact(request.task.environment)["image"],
            "runtime_lock": read_artifact(
                __import__(
                    "self_curate.contracts.parsing", fromlist=["artifact"]
                ).artifact(read_artifact(request.task.scorer)["runtime_lock"])
            ),
        }
        atomic_json(output / "input.json", value)
        receipt = self.process(
            [
                self.pythons.get(harness, self.python),
                "-m",
                "self_curate.execution.judge_worker",
                "--input",
                str((output / "input.json").resolve()),
                "--output",
                str((output / "official-result.json").resolve()),
                "--execution",
                "server",
            ],
            cwd=output,
            output=output / "process",
            timeout=timeout + 300,
        )
        if not (output / "official-result.json").exists():
            return {
                "status": "error",
                "reward": None,
                "reason": (
                    "grader_timeout" if receipt["timed_out"] else "missing_report"
                ),
            }
        return read_json(output / "official-result.json")


class HarborJudge:
    def __init__(
        self,
        *,
        executable="harbor",
        process=execute,
        gpu_pool=(),
        gpu_slot_root=None,
        installed_root=None,
    ):
        self.executable, self.process = executable, process
        self.gpu_pool, self.gpu_slot_root = tuple(gpu_pool), gpu_slot_root
        self.installed_root = Path(installed_root) if installed_root else None

    def run(self, request, task_path, binding, output, *, timeout):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        require(binding.get("relay_grant_root"), "Harbor SDK requires a scoped inference relay")
        require(self.installed_root is not None, "Prepare SDK Harbor images on workstation before running")
        config = {
            "task": {"path": str(Path(task_path).resolve())},
            "trial_name": "sc-" + request.record_hash[:24],
            "trials_dir": str((output / "trials").resolve()),
            "timeout_multiplier": 1.0,
            "agent": {
                "import_path": "self_curate.execution.harbor_claude:ClaudeSDKAgent",
                "model_name": binding["model_name"],
                "base_url": binding.get(
                    "container_base_url", binding["base_url"]
                ).rstrip("/"),
                "model_api_key_name": binding.get("api_key_env"),
                "kwargs": {
                    "harness": binding.get("harness", {}),
                    "gate": __import__("self_curate.execution.claude_sdk", fromlist=["policy_config"]).policy_config(
                        request.task.budget, read_artifact(request.protocol), seed=request.seed, temperature=read_artifact(request.sampling)["temperature"]),
                    "max_turns": request.task.budget.max_steps,
                    "tool_seconds": read_artifact(request.protocol)["tool_timeout_seconds"],
                },
            },
            "environment": {"type": "docker"},
            "verifier": {"disable": False},
        }
        from .harbor_resources import gpu_layout, write_override
        from .resources import exclusive_gpus
        from contextlib import nullcontext
        from .relay import trial_grant

        layout = gpu_layout(task_path)
        if self.installed_root is not None:
            from self_curate.training.artifacts import checkpoint_fingerprint
            from self_curate.server.harbor_prepare import runtime_overlay

            installed_file = (
                self.installed_root / Path(task_path).name / "installed.json"
            )
            require(
                installed_file.exists(),
                "Harbor agent environment was not prepared; run assets on the network node first",
            )
            installed = read_json(installed_file)
            require(
                installed["task_checksum"]
                == checkpoint_fingerprint(Path(task_path))[0],
                "Preinstalled task source changed",
            )
            overlay = runtime_overlay(installed, output / "installed-environment.yaml")
            config["environment"]["extra_docker_compose"] = [str(overlay.resolve())]
            require(installed.get("harness_backend") == "claude_agent_sdk",
                    "Old mini-SWE installation cannot authorize Claude execution")
            config["agent"]["env"] = {"UV_OFFLINE": "1"}
            config["environment"][
                "delete"
            ] = False  # Native down, preserving prepared images.
        count = sum(layout.values())
        require(
            not count or self.gpu_slot_root is not None,
            "Native Harbor GPU task has no exclusive reservation pool",
        )
        with (
            exclusive_gpus(
                self.gpu_slot_root or output / "no-gpu-slots",
                self.gpu_pool,
                count,
                timeout=timeout,
            ) as assigned,
            (
                trial_grant(
                    binding["relay_grant_root"], binding["model_name"], timeout=timeout
                )
                if binding.get("relay_grant_root")
                else nullcontext(None)
            ) as scoped_key,
        ):
            if assigned:
                override = write_override(
                    layout, assigned, output / "gpu-reservation.yaml"
                )
                config["environment"].setdefault("extra_docker_compose", []).append(
                    str(override.resolve())
                )
                atomic_json(
                    output / "gpu-allocation.json",
                    {
                        "layout": layout,
                        "gpu_uuids": list(assigned),
                        "native_counts_preserved": True,
                    },
                )
            atomic_json(output / "harbor-config.json", config)
            process_kwargs = {}
            if scoped_key is not None:
                import os

                allowed = {
                    "PATH",
                    "HOME",
                    "USER",
                    "LANG",
                    "LC_ALL",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                    "REQUESTS_CA_BUNDLE",
                    "CURL_CA_BUNDLE",
                    "PYTHONPATH",
                    "VIRTUAL_ENV",
                    "CONDA_PREFIX",
                    "TMPDIR",
                    "DOCKER_HOST",
                    "DOCKER_CONTEXT",
                    "DOCKER_CONFIG",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "NO_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "no_proxy",
                }
                process_kwargs["env"] = {
                    k: v for k, v in os.environ.items() if k in allowed
                }
                process_kwargs["env"]["SELF_CURATE_TRIAL_KEY"] = scoped_key
                config["agent"]["model_api_key_name"] = "SELF_CURATE_TRIAL_KEY"
                atomic_json(output / "harbor-config.json", config)
            result = self.process(
                [
                    self.executable,
                    "trials",
                    "start",
                    "--config",
                    str((output / "harbor-config.json").resolve()),
                ],
                cwd=output,
                output=output / "process",
                timeout=timeout,
                **process_kwargs,
            )
        path = output / "trials" / config["trial_name"] / "result.json"
        if not path.exists():
            return {
                "status": "environment_failed",
                "reward": None,
                "generated_tokens": None,
                "input_tokens": None,
                "raw": {"process": result},
            }
        row = read_json(path)
        require(row["task_name"] == Path(task_path).name, "Harbor substituted a task")
        valid = row.get("reward_valid", row.get("exception_info") is None) and bool(
            (row.get("verifier_result") or {}).get("rewards")
        )
        rewards = (row.get("verifier_result") or {}).get("rewards", {})
        require(
            not valid or set(rewards) == {"reward"},
            "Configure an explicit reward adapter for multi-metric Harbor tasks",
        )
        agent = row.get("agent_result") or {}
        return {
            "status": "completed" if valid else "environment_failed",
            "reward": float(rewards["reward"]) if valid else None,
            "generated_tokens": agent.get("n_output_tokens"),
            "input_tokens": agent.get("n_input_tokens"),
            "raw": row,
        }
