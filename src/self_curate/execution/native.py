"""Small explicit compatibility boundary around installed official SWE APIs."""

import inspect
from self_curate.contracts.types import require


def test_spec(instance):
    import importlib

    utils = importlib.import_module("swebench.harness.utils")
    if all(
        k in instance for k in ("image", "eval_script", "log_parser", "eval_type")
    ) and callable(getattr(utils, "make_test_spec", None)):
        make_test_spec = utils.make_test_spec
    else:
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError:
            require(
                callable(getattr(utils, "make_test_spec", None)),
                "Installed SWE-bench cannot prepare this source schema",
            )
            make_test_spec = utils.make_test_spec
    kwargs = (
        {"namespace": "swebench"}
        if "namespace" in inspect.signature(make_test_spec).parameters
        else {}
    )
    return make_test_spec(instance, **kwargs)


def spec_image(spec):
    return spec.image if hasattr(spec, "image") else spec.instance_image_key


def evaluate_bench(instance, prediction, client, run_id, timeout):
    from swebench.harness.run_evaluation import run_instance

    spec = test_spec(instance)
    available = {
        "test_spec": spec,
        "spec": spec,
        "instance": instance,
        "pred": prediction,
        "prediction": prediction,
        "client": client,
        "run_id": run_id,
        "timeout": timeout,
        "rm_image": False,
        "force_rebuild": False,
        "rewrite_reports": False,
    }
    params = inspect.signature(run_instance).parameters
    require(
        all(
            name in available
            or p.default is not inspect.Parameter.empty
            or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for name, p in params.items()
        ),
        "Unsupported official SWE-bench API; lock a compatible environment",
    )
    result = run_instance(
        **{name: available[name] for name in params if name in available}
    )
    if isinstance(result, dict) and "completed" in result:
        require(
            result["completed"] is True and type(result.get("resolved")) is bool,
            "Official SWE-bench evaluation did not complete",
        )
        return (
            instance["instance_id"],
            {
                instance["instance_id"]: {
                    "resolved": result["resolved"],
                    "native_summary": result,
                }
            },
        )
    if isinstance(result, dict) and instance["instance_id"] in result:
        return instance["instance_id"], result
    return result


class ImagePins(dict):
    def __init__(self, tasks, output, *, harnesses, allow_pull=True):
        from pathlib import Path
        from self_curate.training.recovery import read_json

        self.tasks = {t["instance_id"]: t for t in tasks}
        self.output = Path(output)
        self.harnesses = harnesses
        self.allow_pull = allow_pull
        super().__init__()  # Resolved pins are external artifacts, not mutable config.
        self.pins = read_json(self.output) if self.output.exists() else {}

    def __missing__(self, task_id):
        from self_curate.training.recovery import atomic_json
        import docker

        task = self.tasks[task_id]
        if self.harnesses[task_id] == "swe_bench":
            native = spec_image(test_spec(task))
        else:
            from swesmith.profiles import registry

            native = registry.get_from_inst(task).image_name
        client = docker.from_env()
        try:
            if task_id in self.pins:
                pinned = self.pins[task_id]
                client.images.get(pinned)
                return pinned
            try:
                image = client.images.get(native)
            except docker.errors.ImageNotFound:
                require(
                    self.allow_pull,
                    f"Offline task image missing: {native}; prepare/import images on this Docker daemon first",
                )
                image = client.images.pull(native)
            digests = image.attrs.get("RepoDigests") or []
            require(
                bool(digests),
                "Official task image has no immutable RepoDigest; publish/pin it before execution",
            )
            pinned = next(
                (d for d in digests if d.split("@")[0] == native.split(":")[0]),
                digests[0],
            )
            self.pins[task_id] = pinned
            atomic_json(self.output, self.pins)
            return pinned
        finally:
            client.close()
