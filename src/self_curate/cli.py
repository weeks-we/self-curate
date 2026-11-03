"""Local data/selection/contracts CLI; no model initialization or training."""

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
    workflow = commands.add_parser(
        "validate-workflow", help="Validate P0 record identities without running models"
    )
    workflow.add_argument("--workflow", required=True)
    workflow.add_argument("--pool", help="Optionally verify exact M1 token/mask inputs")
    workflow.add_argument("--expected-model", help="System-owned ModelRef JSON")
    workflow.add_argument(
        "--expected-batch", help="System-owned exact selected BatchManifest JSON"
    )
    workflow.add_argument(
        "--expected-objective", help="System-owned ObjectiveSpec JSON"
    )
    workflow.add_argument(
        "--expected-context",
        help="System-owned selection context JSON (requires --pool)",
    )
    prepare = commands.add_parser(
        "prepare-selection", help="Run P1 and publish an unvalued P0 handoff"
    )
    for flag in (
        "pool",
        "program",
        "context",
        "objective",
        "output",
        "attempt-id",
        "invocation-id",
    ):
        prepare.add_argument("--" + flag, required=True)
    prepare.add_argument("--backend", choices=("mock", "real"), required=True)
    prepared = commands.add_parser(
        "inspect-preparation",
        help="Verify and replay a P1 preparation on its original pool",
    )
    prepared.add_argument("--prepared", required=True)
    prepared.add_argument("--pool", required=True)
    prepared.add_argument("--expected-model")
    prepared.add_argument("--expected-batch")
    prepared.add_argument("--expected-objective")
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
        elif args.command == "validate-workflow":
            from self_curate.contracts import load_workflow, validate_workflow
            from self_curate.operators import load_batch
            from self_curate.contracts.parsing import load_json, model_ref
            from self_curate.contracts import parse_contract
            from pathlib import Path

            view = None if args.pool is None else read_pool(args.pool).metric_view()
            expected_model = (
                None
                if args.expected_model is None
                else model_ref(
                    load_json(Path(args.expected_model).read_text(encoding="utf-8"))
                )
            )
            expected_objective = (
                None
                if args.expected_objective is None
                else parse_contract(
                    "ObjectiveSpec",
                    load_json(
                        Path(args.expected_objective).read_text(encoding="utf-8")
                    ),
                )
            )
            expected_context_hash = None
            expected_batch_id = (
                None
                if args.expected_batch is None
                else load_batch(args.expected_batch).batch_id
            )
            if args.expected_context is not None:
                if view is None:
                    raise ContractError(
                        ErrorCode.INVALID_CONFIG, "--expected-context requires --pool"
                    )
                from self_curate.operators import load_context

                expected_context_hash = load_context(
                    args.expected_context, view
                ).context_hash
            bundle = load_workflow(
                args.workflow,
                view=view,
                expected_model=expected_model,
                expected_objective=expected_objective,
                expected_selection_context_hash=expected_context_hash,
                expected_batch_id=expected_batch_id,
            )
            result = validate_workflow(
                bundle,
                view,
                expected_model=expected_model,
                expected_objective=expected_objective,
                expected_selection_context_hash=expected_context_hash,
                expected_batch_id=expected_batch_id,
            )
        elif args.command == "prepare-selection":
            from self_curate.operators import prepare_to_directory

            prepared = prepare_to_directory(
                args.pool,
                args.program,
                args.context,
                args.objective,
                args.output,
                backend=args.backend,
                attempt_id=args.attempt_id,
                invocation_id=args.invocation_id,
            )
            print(
                canonical_json(
                    {
                        **prepared.to_dict(),
                        "error": prepared.error,
                        "output": args.output,
                    }
                )
            )
            return {"feasible": 0, "infeasible": 3, "failed": 2}[prepared.status]
        elif args.command == "inspect-preparation":
            from self_curate.operators import verify_preparation
            from self_curate.operators import load_batch
            from self_curate.contracts import parse_contract
            from self_curate.contracts.parsing import load_json, model_ref
            from pathlib import Path

            expected_model = (
                None
                if args.expected_model is None
                else model_ref(
                    load_json(Path(args.expected_model).read_text(encoding="utf-8"))
                )
            )
            expected_objective = (
                None
                if args.expected_objective is None
                else parse_contract(
                    "ObjectiveSpec",
                    load_json(
                        Path(args.expected_objective).read_text(encoding="utf-8")
                    ),
                )
            )
            result = verify_preparation(
                args.prepared,
                read_pool(args.pool).metric_view(),
                expected_model=expected_model,
                expected_objective=expected_objective,
                expected_batch_id=(
                    None
                    if args.expected_batch is None
                    else load_batch(args.expected_batch).batch_id
                ),
            )
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
