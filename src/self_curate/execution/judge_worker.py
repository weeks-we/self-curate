"""One isolated official grading process; logs stay beneath its private cwd."""

import argparse
import json
from pathlib import Path
from self_curate.training.recovery import atomic_json


def image_binding(native, expected):
    """Accept official results only if the native image equals the inference image."""
    import docker

    client = docker.from_env()
    try:
        pinned = client.images.get(expected)
        actual = client.images.get(native)
        if actual.id != pinned.id:
            raise RuntimeError(
                "Official grader image differs from the pinned inference image"
            )
        return {
            "native_image": native,
            "expected_image": expected,
            "image_id": actual.id,
        }
    finally:
        client.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execution", required=True, choices=["server"])
    args = parser.parse_args(argv)
    value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    instance, prediction = value["instance"], value["prediction"]
    try:
        if value.get("runtime_lock"):
            from .runtime import verify_runtime

            verify_runtime(
                value["runtime_lock"],
                kinds=(
                    ("swe_smith", "swe_bench")
                    if value["harness"] == "swe_smith"
                    else (value["harness"],)
                ),
            )
        if value["harness"] == "swe_smith":
            from swesmith.harness.eval import run_evaluation

            if value.get("expected_image"):
                from swesmith.profiles import registry

                profile = registry.get_from_inst(instance)
                if profile.timeout > value["timeout"]:
                    raise RuntimeError(
                        "Grader process timeout would shorten the native SWE-smith test timeout"
                    )
                before = image_binding(profile.image_name, value["expected_image"])
            if value.get("is_gold"):
                result = run_evaluation(
                    prediction, instance, value["run_id"], is_gold=True
                )
            else:
                result = run_evaluation(prediction, instance, value["run_id"])
            if (
                value.get("expected_image")
                and image_binding(profile.image_name, value["expected_image"]) != before
            ):
                raise RuntimeError("SWE-smith grader image changed during execution")
            status = result["status"]
            if status == "completed" and type(result.get("resolved")) is not bool:
                raise RuntimeError(
                    "Official SWE-smith grader omitted its resolved flag"
                )
            reward = (
                float(result["resolved"])
                if status in ("completed", "timeout")
                else None
            )
        elif value["harness"] in ("swe_bench", "swe_rebench"):
            import importlib

            run_evaluation = importlib.import_module("swebench.harness.run_evaluation")
            from .native import test_spec, spec_image, evaluate_bench

            if hasattr(run_evaluation, "_docker_client"):
                client = run_evaluation._docker_client()
            else:
                import docker

                client = docker.from_env()
            try:
                spec = test_spec(instance)
                if value.get("expected_image"):
                    before = image_binding(spec_image(spec), value["expected_image"])
                result = evaluate_bench(
                    instance,
                    prediction,
                    client,
                    value["run_id"],
                    value["timeout"],
                )
                if (
                    value.get("expected_image")
                    and image_binding(spec_image(spec), value["expected_image"])
                    != before
                ):
                    raise RuntimeError(
                        "SWE-bench grader image changed during execution"
                    )
            finally:
                client.close()
            if result is None:
                raise RuntimeError("Official harness produced no report")
            _, report = result
            row = report[instance["instance_id"]]
            if type(row.get("resolved")) is not bool:
                raise RuntimeError("Missing official resolved flag")
            status = "completed"
            reward = float(row["resolved"])
        else:
            raise ValueError("Unsupported grader")
        atomic_json(
            args.output,
            {
                "status": status,
                "reward": reward,
                "official_result": result,
                "image_binding": locals().get("before"),
            },
        )
        return 0
    except Exception as exc:
        atomic_json(
            args.output,
            {
                "status": "error",
                "reward": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
