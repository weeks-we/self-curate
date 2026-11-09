"""Explicit official bug/reference sanity checks, never replay host trajectory commands."""

from pathlib import Path
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json, atomic_json
from self_curate.execution.process import execute


def check_tasks(
    tasks, images, runtime_lock, output, *, execution, timeout=3600, process=execute
):
    require(
        execution == "server",
        "Official environment sanity requires explicit server execution",
    )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for task in tasks:
        folder = root / task["instance_id"]
        folder.mkdir(exist_ok=True)
        result_path = folder / "sanity.json"
        if result_path.exists():
            row = read_json(result_path)
            require(row["instance"] == task, "Sanity task changed")
            reports.append(row)
            continue
        checks = {}
        for kind in ("bug", "reference"):
            work = folder / kind
            work.mkdir(exist_ok=True)
            require(
                not (work / "dispatched.json").exists(),
                "Unknown in-flight sanity call; reconcile before retry",
            )
            value = {
                "harness": "swe_smith",
                "instance": task,
                "prediction": {
                    "instance_id": task["instance_id"],
                    "model_patch": "" if kind == "bug" else task["patch"],
                    "model_name_or_path": "official-sanity-" + kind,
                },
                "run_id": "sanity-" + kind + "-" + task["instance_id"],
                "timeout": timeout,
                "is_gold": kind == "reference",
                "expected_image": images[task["instance_id"]],
                "runtime_lock": runtime_lock,
            }
            atomic_json(work / "input.json", value)
            atomic_json(work / "dispatched.json", {"kind": kind})
            import sys

            process(
                [
                    sys.executable,
                    "-m",
                    "self_curate.execution.judge_worker",
                    "--input",
                    str((work / "input.json").resolve()),
                    "--output",
                    str((work / "result.json").resolve()),
                    "--execution",
                    "server",
                ],
                cwd=work,
                output=work / "process",
                timeout=timeout + 300,
            )
            require((work / "result.json").exists(), "Sanity grader produced no result")
            checks[kind] = read_json(work / "result.json")
        row = {
            "instance": task,
            "checks": checks,
            "passed": checks["bug"]["reward"] == 0
            and checks["reference"]["reward"] == 1,
        }
        atomic_json(result_path, row)
        reports.append(row)
    atomic_json(
        root / "report.json",
        {"checks": reports, "all_passed": all(r["passed"] for r in reports)},
    )
    require(
        all(r["passed"] for r in reports),
        "Official bug/reference sanity failed; do not silently replace tasks",
    )
    return reports
