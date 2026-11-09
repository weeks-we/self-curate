"""A private engineering-only Harbor task; never a benchmark model retry."""

from pathlib import Path
from self_curate.contracts import TaskBudget
from self_curate.execution.registration import register_harbor_task


def create_smoke_files(root, *, gpu_count):
    path = Path(root) / "task"
    path.mkdir(parents=True, exist_ok=True)
    (path / "environment").mkdir(exist_ok=True)
    (path / "tests").mkdir(exist_ok=True)
    (path / "task.toml").write_text(
        'version = "1.0"\n[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n[environment]\nbuild_timeout_sec=600\ngpus='
        + str(gpu_count)
        + "\n",
        encoding="utf-8",
    )
    # Resolve/pin this engineering image, without changing benchmark task images.
    image = "ubuntu:24.04"
    if gpu_count:
        image = "nvidia/cuda:12.8.1-base-ubuntu24.04"
    import docker

    client = docker.from_env()
    try:
        actual = client.images.get(image)
        image = actual.attrs["RepoDigests"][0]
    finally:
        client.close()
    (path / "environment/Dockerfile").write_text(
        "FROM "
        + image
        + "\nRUN apt-get update && apt-get install -y python3 curl ca-certificates && rm -rf /var/lib/apt/lists/*\nWORKDIR /app\n",
        encoding="utf-8",
    )
    instruction = "Create /app/readiness-marker.txt containing READY. Then finish. This is a deployment smoke task, not a benchmark."
    if gpu_count:
        instruction += " Run nvidia-smi -L and save its complete output to /app/readiness-gpus.txt."
    (path / "instruction.md").write_text(instruction, encoding="utf-8")
    test = "#!/bin/bash\nmkdir -p /logs/verifier\nif grep -qx READY /app/readiness-marker.txt"
    if gpu_count:
        test += (
            ' && [ "$(grep -c GPU /app/readiness-gpus.txt)" -eq '
            + str(gpu_count)
            + " ]"
        )
    test += "; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n"
    (path / "tests/test.sh").write_text(test, encoding="utf-8")
    return path


def harbor_smoke_task(root, runtime_lock, *, gpu_count):
    path = create_smoke_files(root, gpu_count=gpu_count)
    return register_harbor_task(
        path,
        task_id="engineering-harbor-readiness",
        skills=("implementation_edit",),
        budget=TaskBudget(75, 262144, 131072, 1200),
        runtime_lock=runtime_lock,
        output=Path(root) / "registered",
    )
