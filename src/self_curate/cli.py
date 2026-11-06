"""Local data/selection/contracts CLI and explicitly synthetic workflow demos."""

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
    feedback_check = commands.add_parser(
        "validate-feedback",
        help="Check external evidence binding and arithmetic; no verification run",
    )
    feedback_check.add_argument("--request", required=True)
    feedback_check.add_argument("--result", required=True)
    feedback_step = commands.add_parser(
        "feedback-step", help="Leval -> Lopt -> P1 -> next validation request"
    )
    for flag in (
        "prepared",
        "pool",
        "request",
        "result",
        "policy",
        "output",
        "attempt-id",
        "invocation-id",
    ):
        feedback_step.add_argument("--" + flag, required=True)
    request_validation = commands.add_parser(
        "request-validation",
        help="Create an external-verifier request from a verified P1 handoff",
    )
    for flag in ("prepared", "pool", "protocol", "request-id", "output"):
        request_validation.add_argument("--" + flag, required=True)
    request_validation.add_argument(
        "--measure", choices=("predicted_gain", "reward_delta"), required=True
    )
    role_demo = commands.add_parser(
        "roles-demo", help="Run three isolated roles on one explicit CPU fixture client"
    )
    role_demo.add_argument("--output", required=True)
    role_demo.add_argument("--project-root", default=".")
    role_demo.add_argument("--prompts")
    inner_demo = commands.add_parser(
        "inner-demo",
        help="Run a complete mock inner search with an explicit demo stopping hook",
    )
    inner_demo.add_argument("--output", required=True)
    inner_demo.add_argument("--project-root", default=".")
    inner_demo.add_argument("--prompts")
    outer_demo = commands.add_parser(
        "outer-demo",
        help="Run explicit mock subset/curation SFT, task evaluation and acceptance",
    )
    outer_demo.add_argument("--output", required=True)
    outer_demo.add_argument("--project-root", default=".")
    leval_demo = commands.add_parser(
        "leval-demo",
        help="Run a mock NICE or candidate-SFT Leval loop with history feedback",
    )
    leval_demo.add_argument("--method", choices=("nice", "sft"), required=True)
    leval_demo.add_argument("--output", required=True)
    leval_demo.add_argument("--project-root", default=".")
    signals_demo = commands.add_parser(
        "signals-demo",
        help="Compute synthetic NLL/entropy, enrich M1, consume raw operators",
    )
    signals_demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "signals-demo":
            from self_curate.signals.demo import run_signals_demo

            result = run_signals_demo(args.output)
        elif args.command == "leval-demo":
            from self_curate.valuation.demo import run_valuation_demo

            result = run_valuation_demo(
                args.output, method=args.method, project_root=args.project_root
            )
        elif args.command == "outer-demo":
            from self_curate.orchestration.demo import run_outer_demo

            result = run_outer_demo(args.output, project_root=args.project_root)
        elif args.command == "inner-demo":
            from self_curate.search.demo import run_inner_demo

            result = run_inner_demo(
                args.output,
                project_root=args.project_root,
                prompt_directory=args.prompts,
            )
        elif args.command == "roles-demo":
            from self_curate.agents.demo import run_role_demo

            result = run_role_demo(
                args.output,
                project_root=args.project_root,
                prompt_directory=args.prompts,
            )
        elif args.command == "request-validation":
            from pathlib import Path
            from self_curate.agents import request_from_preparation
            from self_curate.contracts.parsing import load_json, artifact

            request = request_from_preparation(
                args.prepared,
                args.pool,
                request_id=args.request_id,
                measure=args.measure,
                protocol=artifact(
                    load_json(Path(args.protocol).read_text(encoding="utf-8"))
                ),
            )
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(canonical_json(request.to_dict()) + "\n")
            result = {
                "status": "awaiting_external_validation",
                "request_hash": request.request_hash,
                "output": str(output),
            }
        elif args.command == "validate-feedback":
            from pathlib import Path
            from self_curate.contracts import parse_feedback_contract
            from self_curate.contracts.parsing import load_json
            from self_curate.agents import check_external_result

            request = parse_feedback_contract(
                "ValidationRequest",
                load_json(Path(args.request).read_text(encoding="utf-8")),
            )
            external = parse_feedback_contract(
                "ValidationResult",
                load_json(Path(args.result).read_text(encoding="utf-8")),
            )
            check_external_result(request, external)
            result = {
                "status": "consistent",
                "validation_status": external.status,
                "request_hash": request.request_hash,
                "result_hash": external.record_hash,
                "validation_implemented": False,
            }
        elif args.command == "feedback-step":
            from self_curate.agents import feedback_step_to_directory

            step = feedback_step_to_directory(
                args.prepared,
                args.pool,
                args.request,
                args.result,
                args.policy,
                args.output,
                attempt_id=args.attempt_id,
                invocation_id=args.invocation_id,
            )
            print(canonical_json({**step.to_dict(), "output": args.output}))
            return (
                0
                if step.status == "awaiting_validation"
                else (2 if step.status == "candidate_failed" else 3)
            )
        elif args.command == "validate-config":
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
