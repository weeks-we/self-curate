"""Lock and verify installed harness APIs; no containers start during imports."""

from pathlib import Path
import importlib
import importlib.metadata
import inspect
from self_curate.contracts.types import require
from self_curate.contracts import digest
from self_curate.data.pool import file_sha256
from .artifacts import publish_json

APIS = {
    "mini": ("mini-swe-agent", "minisweagent.agents.default", "DefaultAgent"),
    "swe_smith": ("swesmith", "swesmith.harness.eval", "run_evaluation"),
    "swe_bench": ("swebench", "swebench.harness.run_evaluation", "run_instance"),
    "swe_rebench": ("swebench", "swebench.harness.run_evaluation", "run_instance"),
    "harbor": ("harbor", "harbor.models.trial.config", "TrialConfig"),
}


def runtime_identity(kinds):
    result = {}
    for kind in kinds:
        distribution, module, name = APIS[kind]
        loaded = importlib.import_module(module)
        api = getattr(loaded, name)
        path = Path(inspect.getsourcefile(api))
        package = importlib.import_module(module.split(".")[0])
        package_root = Path(package.__file__).parent
        result[kind] = {
            "distribution": distribution,
            "version": importlib.metadata.version(distribution),
            "module": module,
            "api": name,
            "source_sha256": file_sha256(path),
            "signature": str(inspect.signature(api)),
            "package_python_sha256": digest(
                {
                    p.relative_to(package_root).as_posix(): file_sha256(p)
                    for p in sorted(package_root.rglob("*.py"))
                }
            ),
        }
    return result


def verify_runtime(lock, *, kinds):
    require(lock["schema_version"] == "1", "Unknown harness lock")
    actual = runtime_identity(kinds)
    require(
        all(lock["apis"].get(k) == v for k, v in actual.items()),
        "Installed harness version/API changed",
    )
    return actual


def lock_runtime(output, *, kinds):
    return publish_json(
        output,
        {
            "schema_version": "1",
            "apis": runtime_identity(kinds),
            "container_smoke_passed": False,
            "real_model_smoke_passed": False,
        },
    )


def main(argv=None):
    import argparse,json
    parser=argparse.ArgumentParser(description="CPU-only installed harness API/source lock")
    parser.add_argument("--kinds",nargs="+",choices=list(APIS),required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(lock_runtime(args.output,kinds=tuple(args.kinds)).to_dict()))
    return 0


if __name__=="__main__":raise SystemExit(main())
