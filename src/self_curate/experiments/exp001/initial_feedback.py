"""One-shot Leval interpretation, shared by later experiments via AssetStore."""
from pathlib import Path
from self_curate.assets import AssetSpec
from self_curate.agents.runtime import RoleRuntime
from self_curate.contracts import digest
from self_curate.contracts.parsing import fields, artifact
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json, atomic_json, lease
from self_curate.data.pool import file_sha256

TEMPLATE = "exp001_initial_capability_report"


def initial_feedback(inputs, ports, config, store, output, producer, capability_ref):
    if isinstance(capability_ref, dict):
        capability_ref = artifact(capability_ref)
    report = inputs.to_dict()
    feedback_hash = digest(report)
    work = {"kind": "initial-capability-report-v1", "source_feedback_hash": feedback_hash,
            "current_result": report, "related_history": [],
            "instruction": "Interpret only M0 development facts. No NICE, selection valuation, or benchmark."}
    spec = AssetSpec.create("initial_feedback", config.backend, {
        "model": ports.client.model_ref.to_dict(), "work": work,
        "prompt": ports.prompts["leval"].to_dict(),
        "prompt_hash": ports.prompts["leval"].prompt_hash,
        "feedback_implementation": file_sha256(Path(__file__)),
        "sampling": ports.agent_sampling.to_dict(),
        "implementation": ports.agent_implementation_hash,
        "timeout": config.lopt_timeout_seconds,
    }, dependencies=(capability_ref,))
    receipt = Path(output) / "feedback-reuse.json"
    with lease(store.root / (".feedback-" + spec.key_hash + ".lease")):
        reused = store.reuse(spec, consumer_run=producer.run_id, receipt_path=receipt)
        events = ()
        if reused is None:
            runtime = RoleRuntime(ports.client, model=ports.client.model_ref,
                                  session_id="initial-feedback-" + spec.key_hash[:20],
                                  backend=config.backend, prompts=ports.prompts)
            origin = Path(output) / "feedback-origin"
            require(not origin.exists(), "Uncommitted initial Leval call requires reconciliation")
            origin.mkdir()
            atomic_json(origin / "dispatched.json", {"spec": spec.to_dict(), "work": work})
            def validate(content):
                fields(content, ("source_feedback_hash", "commentary", "hypotheses"))
                require(content["source_feedback_hash"] == feedback_hash, "Leval cited another report")
                require(type(content["commentary"]) is str and bool(content["commentary"].strip()), "Empty commentary")
                require(type(content["hypotheses"]) is list and all(type(v) is str for v in content["hypotheses"]), "Invalid hypotheses")
            try:
                reply = runtime.generate("leval", work, timeout_seconds=config.lopt_timeout_seconds, validator=validate)
            finally:
                runtime.save(origin / "roles")
            atomic_json(origin / "feedback.json", reply.content)
            store.publish(spec, origin, producer=producer, origin_ledger=runtime.ledger())
            events = runtime.ledger().events
            reused = store.reuse(spec, consumer_run=producer.run_id, receipt_path=receipt)
        asset = reused[0]
        content = read_json(asset.file("feedback.json"))
        require(content["source_feedback_hash"] == feedback_hash, "Reused feedback mismatch")
        return content, events, asset.reference
