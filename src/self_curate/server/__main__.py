"""Run server deployment stages explicitly; never silently switch to a mock."""

import argparse, json


def main():
    parser = argparse.ArgumentParser(
        description="EXP001 pinned-source server preparation, capacity, smoke and run"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpu-uuids", nargs=8)
    parser.add_argument(
        "--stage",
        choices=["download", "assets", "prepare", "preflight", "run", "all"],
        default="all",
    )
    parser.add_argument("--execution", choices=["server"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.stage in ("download", "assets"):
            from .assets import prepare_assets

            result = prepare_assets(
                args.plan, stage=args.stage, execution=args.execution
            )
            print(json.dumps(result, ensure_ascii=False))
            return 0
        from .bootstrap import bootstrap

        result = bootstrap(
            args.plan,
            gpu_uuids=args.gpu_uuids or (),
            stage=args.stage,
            execution=args.execution,
            resume=args.resume,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result.get("all_reports_valid") is False else 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
