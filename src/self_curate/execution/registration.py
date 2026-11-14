"""Register task text, its Docker environment and official scorer together."""

from pathlib import Path
import re
from self_curate.contracts import TaskRecord, digest
from self_curate.contracts.types import require
from .artifacts import publish_json


def register_swe_task(
    instance,
    *,
    harness,
    split,
    skills,
    budget,
    image,
    runtime_lock,
    output,
    grader_timeout_seconds=3600,
):
    require(harness in ("swe_smith", "swe_bench", "swe_rebench"), "Unknown SWE task source")
    require(
        type(grader_timeout_seconds) is int and grader_timeout_seconds > 0,
        "Positive grader process timeout required",
    )
    require(
        re.fullmatch(r"[A-Za-z0-9_.-]+", instance["instance_id"]) is not None,
        "Unsafe instance ID",
    )
    require(
        re.search(r"@sha256:[0-9a-f]{64}$", image) is not None,
        "Task image must be digest-pinned",
    )
    require(
        type(instance.get("problem_statement")) is str
        and instance["problem_statement"],
        "Task needs its problem statement",
    )
    checkout = (
        instance["instance_id"] if harness == "swe_smith" else instance["base_commit"]
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    # Golden/test patches remain in the scorer payload and never enter LM input.
    payload = publish_json(
        root / "payload.json", {"schema_version": "1", "instance": instance}
    )
    env = publish_json(
        root / "environment.json",
        {
            "schema_version": "1",
            "harness": harness,
            "image": image,
            "cwd": "/testbed",
            "checkout": checkout,
        },
    )
    scorer = publish_json(
        root / "scorer.json",
        {
            "schema_version": "1",
            "harness": harness,
            "runtime_lock": runtime_lock.to_dict(),
            "timeout_seconds": grader_timeout_seconds,
        },
    )
    return TaskRecord(
        instance["instance_id"],
        digest(instance),
        tuple(skills),
        payload,
        env,
        scorer,
        split,
        budget,
        "real",
        ("repo:" + instance["repo"],),
    )


def register_harbor_task(task_path, *, task_id, skills, budget, runtime_lock, output):
    from self_curate.training.artifacts import checkpoint_fingerprint
    from self_curate.contracts import ArtifactRef

    path = Path(task_path).resolve()
    require(
        (path / "task.toml").is_file() and (path / "instruction.md").is_file(),
        "Register the complete downloaded Harbor task directory",
    )
    import tomllib

    native = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
    if not all(
        type(
            native.get(name, {}).get(
                "timeout_sec" if name != "environment" else "build_timeout_sec"
            )
        )
        in (int, float)
        for name in ("agent", "verifier", "environment")
    ):
        from harbor.models.task.config import TaskConfig

        native = TaskConfig.model_validate(native).model_dump(mode="json")
    require(
        all(
            type(
                native.get(name, {}).get(
                    "timeout_sec" if name != "environment" else "build_timeout_sec"
                )
            )
            in (int, float)
            for name in ("agent", "verifier", "environment")
        ),
        "Resolve native Harbor timeout defaults before registering the task",
    )
    total = (
        native["agent"]["timeout_sec"]
        + native["verifier"]["timeout_sec"]
        + native["environment"]["build_timeout_sec"]
    )
    require(
        budget.max_wall_seconds >= total,
        "Outer timeout would shorten native Harbor task budgets",
    )
    fingerprint = checkpoint_fingerprint(path)[0]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    payload = publish_json(
        root / "payload.json",
        {"schema_version": "1", "task_path": str(path), "task_checksum": fingerprint},
    )
    env = publish_json(
        root / "environment.json",
        {
            "schema_version": "1",
            "harness": "harbor",
            "task_directory": ArtifactRef(path.as_uri(), fingerprint).to_dict(),
            "preserve_native_resources": True,
        },
    )
    scorer = publish_json(
        root / "scorer.json",
        {
            "schema_version": "1",
            "harness": "harbor",
            "runtime_lock": runtime_lock.to_dict(),
            "timeout_seconds": budget.max_wall_seconds,
        },
    )
    return TaskRecord(
        task_id,
        fingerprint,
        tuple(skills),
        payload,
        env,
        scorer,
        "test",
        budget,
        "real",
        ("terminal:" + task_id,),
    )
