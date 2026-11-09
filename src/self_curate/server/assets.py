"""CPU/network-only provisioning; GPU execution consumes verified local assets."""

from pathlib import Path
import asyncio
import argparse
import json
import subprocess
import tomllib
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, read_json, lease
from self_curate.preparation.sources import download_sources, read_dataset


async def harbor_images(task_path, output):
    """Use the locked Harbor compose generator, without starting task containers.

    This is an explicit compatibility boundary: changed native APIs fail setup,
    never substitute a home-made task environment or consume a GPU.
    """
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.environments.definition import should_use_prebuilt_docker_image
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    task = Task(task_dir=Path(task_path))
    paths = TrialPaths(trial_dir=Path(output))
    paths.mkdir()
    env = DockerEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="self-curate-prepare-" + digest(str(task_path))[:16],
        trial_paths=paths,
        task_env_config=task.config.environment,
    )
    # Same native overlays as start(), but no up/run and no GPU reservation.
    env._mounts_compose_path = env._write_mounts_compose_file()
    env._env_compose_path = env._write_env_compose_file()
    env._resources_compose_path = await env._write_resources_compose_file()
    env._use_prebuilt = should_use_prebuilt_docker_image(
        env.environment_dir,
        docker_image=env.task_env_config.docker_image,
        force_build=False,
    )
    await env._run_docker_compose_command(["pull", "--ignore-buildable"])
    if not env._use_prebuilt:
        await env._run_docker_compose_command(["build"])
    result = await env._run_docker_compose_command(["config", "--images"])
    require(type(result.stdout) is str, "Unsupported Harbor compose result API")
    images = tuple(dict.fromkeys(result.stdout.split()))
    require(images, "Harbor produced no image inventory")
    native = tomllib.loads((Path(task_path) / "task.toml").read_text())
    # Separately defined verifier environments need their own preparation adapter.
    # Do not falsely publish an offline-ready receipt for an unprepared verifier.
    require(
        not native.get("verifier", {}).get("environment"),
        "Separate Harbor verifier environment requires explicit image preparation",
    )
    return images


def image_inventory(names, *, run=subprocess.run):
    result = {}
    for name in sorted(set(names)):
        row = json.loads(
            run(
                ["docker", "image", "inspect", name],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            ).stdout
        )[0]
        result[name] = {
            "id": row["Id"],
            "repo_digests": sorted(row.get("RepoDigests") or []),
        }
    return result


def verify_assets(path, registry, *, inventory=image_inventory):
    value = read_json(path)
    require(
        value["source_registry_hash"] == digest(registry),
        "Offline source registry changed",
    )
    require(
        value["images"] == inventory(value["images"]),
        "Task image cache is missing/changed; import the prepared images on this Docker daemon",
    )
    pins = Path(value["image_pins_file"])
    require(
        file_sha256(pins) == value["image_pins_sha256"], "Prepared image pins changed"
    )
    for name, checksum in value.get("installed_files", {}).items():
        require(
            file_sha256(Path(name)) == checksum,
            "Prepared Harbor snapshot receipt changed",
        )
    return value


def prepare_assets(plan_path, *, stage, execution):
    require(
        execution == "server", "Network provisioning requires explicit server execution"
    )
    require(stage in ("download", "assets"), "Unknown provisioning stage")
    from self_curate.experiments.exp001.sources import load_source_registry
    from self_curate.execution.native import ImagePins

    project = Path(__file__).parents[3]
    registry = load_source_registry(project / "configs/experiments/exp001/sources.json")
    plan = read_json(plan_path)
    require(Path(plan["storage_root"]).is_absolute(), "Absolute storage root required")
    root = Path(plan["storage_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lease(root / ".assets.lease"):
        receipts = download_sources(registry, root / "sources", execution=execution)
        if stage == "download":
            return {
                "status": "downloaded",
                "sources": str(root / "sources/sources.lock.json"),
                "gpu_required": False,
            }
        receipt_path = root / "offline-assets.json"
        if receipt_path.exists():
            verify_assets(receipt_path, registry)
            return {
                "status": "assets_ready",
                "receipt": str(receipt_path),
                "gpu_required": False,
            }
        tasks = read_dataset(receipts["development_tasks"])
        benchmarks = [
            r
            for name in ("benchmark_full", "benchmark_lite", "benchmark_verified")
            for r in read_dataset(receipts[name])
        ]
        pin_path = root / "task-images/image-pins.json"
        pins = ImagePins(
            [*tasks, *benchmarks],
            pin_path,
            harnesses={
                **{r["instance_id"]: "swe_smith" for r in tasks},
                **{r["instance_id"]: "swe_bench" for r in benchmarks},
            },
        )
        names = set()
        for i, row in enumerate([*tasks, *benchmarks]):
            names.add(pins[row["instance_id"]])
            print(
                json.dumps(
                    {
                        "phase": "task_image_prepare",
                        "completed": i + 1,
                        "total": len(tasks) + len(benchmarks),
                    }
                ),
                flush=True,
            )
        terminal = sorted(
            p.parent
            for p in Path(receipts["terminal_bench"]["root"]).glob(
                plan["terminal_task_glob"]
            )
        )
        require(
            terminal,
            "No native Terminal tasks found; check the pinned source and task glob",
        )
        for i, path in enumerate(terminal):
            names.update(
                asyncio.run(
                    harbor_images(path, root / "task-images/harbor" / path.name)
                )
            )
            from .harbor_prepare import install_snapshot

            installed = asyncio.run(
                install_snapshot(
                    path, root / "task-images/harbor-installed" / path.name
                )
            )
            names.add(installed["image"])
            print(
                json.dumps(
                    {
                        "phase": "terminal_image_prepare",
                        "completed": i + 1,
                        "total": len(terminal),
                    }
                ),
                flush=True,
            )
        # The engineering smoke task has its own base, not a benchmark retry.
        for name in ("ubuntu:24.04", "nvidia/cuda:12.8.1-base-ubuntu24.04"):
            subprocess.run(["docker", "pull", name], check=True)
            names.add(name)
        from .smoke_task import create_smoke_files
        from self_curate.execution.harbor_resources import gpu_layout

        demand = max((sum(gpu_layout(p).values()) for p in terminal), default=0)
        require(demand < 8, "Native task GPU demand leaves no inference GPU")
        smoke = create_smoke_files(root / "task-images/engineering", gpu_count=demand)
        names.update(
            asyncio.run(harbor_images(smoke, root / "task-images/engineering-build"))
        )
        installed = asyncio.run(
            install_snapshot(smoke, root / "task-images/harbor-installed" / smoke.name)
        )
        names.add(installed["image"])
        atomic_json(
            receipt_path,
            {
                "schema_version": "1",
                "source_registry_hash": digest(registry),
                "images": image_inventory(names),
                "image_pins_file": str(pin_path),
                "image_pins_sha256": file_sha256(pin_path),
                "terminal_task_count": len(terminal),
                "gpu_required": False,
                "harbor_installed_root": str(root / "task-images/harbor-installed"),
                "installed_files": {
                    str(p): file_sha256(p)
                    for p in sorted(
                        (root / "task-images/harbor-installed").glob("*/installed.json")
                    )
                },
                "note": "Prebuilt images and source bytes; not a task score, not proof of network-free task scripts",
            },
        )
        return {
            "status": "assets_ready",
            "receipt": str(receipt_path),
            "gpu_required": False,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", choices=("download", "assets"), default="assets")
    parser.add_argument("--execution", choices=("server",), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_assets(args.plan, stage=args.stage, execution=args.execution)
        )
    )


if __name__ == "__main__":
    main()
