"""Local M1 CLI; no downloading, model initialization, training or benchmarks."""

import argparse
import sys

from self_curate.contracts import ContractError, ErrorCode, canonical_json
from self_curate.data.pool import build_pool, load_config, read_pool


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="self-curate",
        description="Self-Curate: data contracts and deterministic selection",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "validate-config", help="Validate import config and input path"
    )
    check.add_argument("--config", required=True)
    normalize = commands.add_parser("normalize", help="Build a new immutable data pool")
    normalize.add_argument("--config", required=True)
    normalize.add_argument("--output", required=True)
    inspect = commands.add_parser(
        "inspect-pool", help="Verify checksums and recompute record metrics"
    )
    inspect.add_argument("--pool", required=True)
    validate = commands.add_parser(
        "validate-program", help="Validate operator schemas and DAG dependencies"
    )
    validate.add_argument("--program", required=True)
    select = commands.add_parser("select", help="Execute a frozen selection program")
    select.add_argument("--pool", required=True)
    select.add_argument("--program", required=True)
    select.add_argument("--context", required=True)
    select.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            if not config.input_path.is_file():
                raise ContractError(
                    ErrorCode.INVALID_CONFIG,
                    "Input file not found",
                    path=str(config.input_path),
                )
            config.tokenization_backend()
            result = {"status": "valid", "config": config.to_dict()}
        elif args.command == "normalize":
            result = build_pool(load_config(args.config), args.output)
        elif args.command == "validate-program":
            from self_curate.operators import load_program

            program = load_program(args.program)
            result = {
                "status": "valid",
                "recipe_hash": program.recipe_hash,
                "nodes": len(program.nodes),
            }
        elif args.command == "select":
            from self_curate.operators import select_to_directory

            result = select_to_directory(
                args.pool, args.program, args.context, args.output
            )
            print(canonical_json(result.to_dict()))
            return 0 if result.status == "feasible" else 3
        else:
            snapshot = read_pool(args.pool)
            result = {
                "status": "verified",
                "pool_id": snapshot.manifest["pool_id"],
                "synthetic": snapshot.manifest["synthetic"],
                "records": len(snapshot.records),
                "excluded_records": len(snapshot.excluded),
                "signal_records": sum(len(r.signals) for r in snapshot.records),
                "gradient_features": sum(len(r.gradients) for r in snapshot.records),
            }
        print(canonical_json(result))
        return 0
    except ContractError as exc:
        print(
            canonical_json({"status": "error", "error": exc.to_dict()}), file=sys.stderr
        )
        return 2
    except (OSError, UnicodeError) as exc:
        print(
            canonical_json(
                {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.ENVIRONMENT_FAILURE.value,
                        "message": str(exc),
                    },
                }
            ),
            file=sys.stderr,
        )
        return 2
