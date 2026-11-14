"""Read-only server prerequisite inventory; never install host drivers or kill jobs."""

import sys, subprocess, importlib.metadata, re
from self_curate.contracts.types import require


def inspect_server(gpu_uuids, *, run=subprocess.run):
    require(
        sys.version_info >= (3, 12),
        "Current Harbor server profile requires Python >=3.12",
    )
    report = {"python": sys.version, "packages": {}}
    for name in (
        "torch",
        "transformers",
        "peft",
        "vllm",
        "claude-agent-sdk",
        "swesmith",
        "swebench",
        "harbor",
        "pyarrow",
        "huggingface-hub",
        "docker",
        "PyYAML",
    ):
        report["packages"][name] = importlib.metadata.version(name)
    rows = run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout
    parsed = [
        tuple(part.strip() for part in row.split(",")) for row in rows.splitlines()
    ]
    selected = [row for row in parsed if row[0] in gpu_uuids]
    require(
        len(selected) == 8 and all("H200" in row[1] for row in selected),
        "Authorized eight-H200 hardware profile does not match this server",
    )
    report["gpus"] = selected
    report["docker"] = run(
        ["docker", "version", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout
    compose = run(
        ["docker", "compose", "version", "--short"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    version = re.search(r"(\d+)\.(\d+)\.(\d+)", compose)
    require(
        version is not None and tuple(map(int, version.groups())) >= (2, 24, 4),
        "Docker Compose >=2.24.4 is required for explicit GPU reservation overlays",
    )
    report["docker_compose"] = compose
    processes = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout
    active = [
        line
        for line in processes.splitlines()
        if line.split(",")[0].strip() in gpu_uuids
    ]
    require(
        not active,
        "An authorized GPU already has a compute process; reconcile existing jobs rather than starting duplicate services",
    )
    return report
