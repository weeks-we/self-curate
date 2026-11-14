"""Server setup/smoke commands; all expensive execution is explicit."""

import argparse
from dataclasses import replace
from pathlib import Path
from self_curate.contracts import TaskBudget
from self_curate.contracts.parsing import model_ref
from self_curate.orchestration.io import load_outer_contract
from self_curate.training.recovery import read_json, atomic_json
from self_curate.contracts.types import require
from .artifacts import publish_json


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock-runtime")
    lock.add_argument("--output", required=True)
    lock.add_argument(
        "--kinds", nargs="+", default=["claude_sdk", "swe_smith", "swe_bench", "harbor"]
    )
    register = sub.add_parser("register-swe")
    register.add_argument("--instances", required=True)
    register.add_argument("--images", required=True)
    register.add_argument("--runtime-lock", required=True)
    register.add_argument(
        "--harness", choices=["swe_smith", "swe_bench"], required=True
    )
    register.add_argument("--split", choices=["development", "test"], required=True)
    register.add_argument("--skills", nargs="+", required=True)
    register.add_argument("--budget", required=True)
    register.add_argument("--output", required=True)
    register.add_argument("--grader-timeout-seconds", type=int, default=3600)
    harbor = sub.add_parser("register-harbor")
    for name in ("task-path", "task-id", "budget", "runtime-lock", "output"):
        harbor.add_argument("--" + name, required=True)
    harbor.add_argument("--skills", nargs="+", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--request", required=True)
    smoke.add_argument("--settings", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--execution", choices=["server"], required=True)
    init = sub.add_parser("init-lora")
    init.add_argument("--base", required=True)
    init.add_argument("--model-template", required=True)
    init.add_argument("--lora", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--execution", choices=["server"], required=True)
    args = parser.parse_args()
    if args.command == "lock-runtime":
        from .runtime import lock_runtime

        print(lock_runtime(args.output, kinds=args.kinds).to_dict())
    elif args.command == "register-swe":
        from .registration import register_swe_task
        from self_curate.contracts import ArtifactRef
        from self_curate.data.pool import file_sha256

        runtime = Path(args.runtime_lock).resolve()
        reference = ArtifactRef(runtime.as_uri(), file_sha256(runtime))
        budget = TaskBudget(**read_json(args.budget))
        images = read_json(args.images)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        tasks = []
        for row in read_json(args.instances):
            tasks.append(
                register_swe_task(
                    row,
                    harness=args.harness,
                    split=args.split,
                    skills=args.skills,
                    budget=budget,
                    image=images[row["instance_id"]],
                    runtime_lock=reference,
                    output=output / row["instance_id"],
                    grader_timeout_seconds=args.grader_timeout_seconds,
                ).to_dict()
            )
        atomic_json(output / "tasks.json", tasks)
    elif args.command == "register-harbor":
        from .registration import register_harbor_task
        from self_curate.contracts import ArtifactRef
        from self_curate.data.pool import file_sha256

        runtime = Path(args.runtime_lock).resolve()
        task = register_harbor_task(
            args.task_path,
            task_id=args.task_id,
            skills=args.skills,
            budget=TaskBudget(**read_json(args.budget)),
            runtime_lock=ArtifactRef(runtime.as_uri(), file_sha256(runtime)),
            output=args.output,
        )
        atomic_json(Path(args.output) / "task.json", task.to_dict())
    elif args.command == "smoke":
        from .runner import build_task_runner

        request = load_outer_contract("TaskRunRequest", args.request)
        runner = build_task_runner(
            read_json(args.settings), request.model_ref, execution=args.execution
        )
        result = runner.execute(request, args.output)
        require(
            result.reward is not None,
            "Real harness smoke did not produce an official score",
        )
        print(result.to_dict())
    else:
        import torch, transformers
        from self_curate.training.peft_store import LoRASettings, initialize_adapter
        from self_curate.training.artifacts import checkpoint_fingerprint
        from self_curate.data.pool import file_sha256

        model = model_ref(read_json(args.model_template))
        settings = LoRASettings.from_dict(read_json(args.lora))
        base = Path(args.base).resolve()
        require(
            checkpoint_fingerprint(base)[0] == model.weights_hash,
            "Base source fingerprint changed",
        )
        require(
            file_sha256(base / "tokenizer.json") == model.tokenizer.tokenizer_hash
            and file_sha256(base / "chat_template.jinja")
            == model.tokenizer.template_hash,
            "Tokenizer/template mismatch",
        )
        actual = transformers.AutoModelForImageTextToText.from_pretrained(
            str(base),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )
        checksum = initialize_adapter(
            actual,
            args.output,
            settings,
            base_model_id=model.base_model_id,
            revision=model.revision,
        )
        reference = replace(
            model,
            parameterization="lora:" + settings.config_hash,
            adapter_hash=checksum,
        )
        publish_json(
            Path(args.output).parent / "initial-model.json", reference.to_dict()
        )


if __name__ == "__main__":
    main()
