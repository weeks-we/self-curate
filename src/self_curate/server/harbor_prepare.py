"""Network-side native agent installation snapshots; never solve/score tasks."""

from pathlib import Path
import importlib.metadata
import json
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json, read_json
from self_curate.training.artifacts import checkpoint_fingerprint


async def install_snapshot(task_path, root):
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    import docker

    task_path, root = Path(task_path).resolve(), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("mini-swe-agent")
    identity = digest(
        {
            "task_checksum": checkpoint_fingerprint(task_path)[0],
            "mini_version": version,
            "harbor_version": importlib.metadata.version("harbor"),
            "harbor_source": importlib.metadata.distribution("harbor").read_text(
                "direct_url.json"
            ),
        }
    )
    receipt = root / "installed.json"
    if receipt.exists():
        saved = read_json(receipt)
        require(
            saved["identity"] == identity, "Installed task environment identity changed"
        )
        return saved
    # This stage has no model and no evaluation. Explicitly disable GPU devices
    # only for the installation container; execution uses the original task counts.
    from self_curate.execution.harbor_resources import gpu_layout

    layout = gpu_layout(task_path)
    lines = ["services:"]
    for name in sorted({"main", *layout}):
        lines += [
            "  " + json.dumps(name) + ":",
            "    gpus: !reset null",
            "    deploy:",
            "      resources:",
            "        reservations:",
            "          devices: !reset []",
        ]
    no_gpu = root / "installation-no-gpu.yaml"
    no_gpu.write_text("\n".join(lines) + "\n")
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task_path)},
            "trial_name": "prepare-" + identity[:20],
            "trials_dir": str(root / "trials"),
            "install_only": True,
            "agent": {
                "name": "mini-swe-agent",
                "kwargs": {"version": version},
                "model_name": "openai/preparation-only-never-infer",
                "base_url": "http://127.0.0.1:9/v1",
                "model_api_key_name": "SELF_CURATE_PREPARATION_ONLY_KEY",
                "env": {"SELF_CURATE_PREPARATION_ONLY_KEY": "not-a-real-key"},
            },
            "environment": {
                "type": "docker",
                "delete": False,
                "override_gpus": 0,
                "kwargs": {"keep_containers": True},
                "extra_docker_compose": [str(no_gpu)],
            },
            "verifier": {"disable": True},
        }
    )
    require(
        config.install_only and config.verifier.disable,
        "Preparation must not execute an agent or verifier",
    )
    trial = Trial(config)
    # Native install may run package managers in the task container here, on the
    # CPU/network node. No model inference is configured or allowed.
    result = await trial.run()
    require(
        result.exception_info is None,
        "Harbor installation failed; inspect preparation trial logs",
    )
    env = trial.agent_environment
    row = await env._run_docker_compose_command(["ps", "--all", "--quiet", "main"])
    ids = row.stdout.split()
    require(len(ids) == 1, "Installation main container was not retained")
    client = docker.from_env()
    try:
        container = client.containers.get(ids[0])
        require(
            not any(container.attrs.get("HostConfig", {}).get("DeviceRequests") or []),
            "Installation container unexpectedly requested GPUs",
        )
        name = "self-curate/harbor-installed:" + identity
        image = container.commit(
            repository="self-curate/harbor-installed", tag=identity
        )
        saved = {
            "identity": identity,
            "task_path": str(task_path),
            "task_checksum": checkpoint_fingerprint(task_path)[0],
            "image": name,
            "image_id": image.id,
            "mini_version": version,
            "install_only": True,
            "research_gpu_resources_changed": False,
        }
        atomic_json(receipt, saved)
    finally:
        client.close()
        await env._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
    return saved


def runtime_overlay(receipt, output):
    """Use the snapshot without rebuilding/pulling; all native resource limits stay."""
    path = Path(output)
    path.write_text(
        "services:\n  main:\n    image: "
        + json.dumps(receipt["image_id"])
        + "\n    pull_policy: never\n    build: !reset null\n",
        encoding="utf-8",
    )
    return path
