"""Dedicated EXP001 entry point; no implicit real-model or API fallback."""

import argparse
from pathlib import Path

from self_curate.contracts import canonical_json
from .config import load_config
from .runner import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EXP001 Random + frozen M0 policy; 4 branches, 3 training waves"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--handoff")
    parser.add_argument(
        "--package-results",
        action="store_true",
        help="Write shareable results.zip outside the immutable run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume frozen training/task boundaries; verify completed runs without new calls",
    )
    parser.add_argument(
        "--parallel-training",
        action="store_true",
        help="Four isolated worker processes; always enabled for server execution",
    )
    parser.add_argument("--train-branch", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.train_branch:
            from .training import run_worker

            run_worker(
                load_config(args.config),
                args.output,
                args.assets,
                args.train_branch,
                resume=args.resume,
            )
            return 0
        result = run_experiment(
            load_config(args.config),
            output=Path(args.output),
            asset_root=Path(args.assets),
            resume=args.resume,
            handoff=args.handoff,
            parallel_training=args.parallel_training,
        )
        delivery = None
        if args.package_results:
            from .delivery import export_bundle, verify_bundle

            destination = (
                Path(args.output).resolve().parent
                / "deliverables"
                / Path(args.output).name
            )
            if args.resume and (destination / "bundle-verification.json").exists():
                verify_bundle(destination)
                delivery = str(destination)
            else:
                delivery = str(export_bundle(args.output, destination))
        print(
            canonical_json(
                {
                    "status": result["status"],
                    "backend": result["backend"],
                    "all_reports_valid": result["all_reports_valid"],
                    "selected_per_mode": result["selected_per_mode"],
                    "output": str(Path(args.output).resolve()),
                    "delivery_directory": delivery,
                }
            )
        )
        return 0
    except Exception as exc:
        print(canonical_json({"status": "blocked", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
