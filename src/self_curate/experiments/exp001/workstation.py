"""CPU workstation preparation primitives for the new Rebench release.
Separate from legacy bootstrap: no implicit 5016-row loader, GPU, or annotation.
"""
import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from self_curate.training.recovery import atomic_json, read_json, lease
from self_curate.contracts import digest
from self_curate.contracts.types import require


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"Non-object row {number}")
                yield value


def sample_tasks(source, count, seed, excluded_ids=()):
    require(type(count) is int and count > 0, "Positive task candidate count required")
    unique = {}
    excluded = set(excluded_ids)
    for row in source:
        identity = row.get("instance_id")
        require(isinstance(identity, str) and identity, "Task instance_id required")
        require(identity not in unique, "Duplicate task ID; reconcile source before sampling")
        unique[identity] = row
    available = sorted(set(unique) - excluded)
    require(len(available) >= count, "Insufficient isolated tasks")
    random.Random(seed).shuffle(available)
    selected = available[:count]
    return [unique[k] for k in selected]


def light_gate(source, excluded_ids, token_counter, max_context, *, fingerprint_func=None, state=None):
    """No execution or semantic judging. token_counter returns (input, supervised).
    It must use the frozen target tokenizer/template and assistant payload mask.
    """
    require(type(max_context) is int and max_context > 0, "Positive context limit required")
    accepted, audit = [], []
    state = {} if state is None else state
    seen = state.setdefault('seen', set())
    identities = state.setdefault('identities', set())
    excluded = set(excluded_ids)
    for row in source:
        identity = row.get("trajectory_id")
        task = row.get("instance_id")
        reason = None
        messages = row.get("trajectory")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except (ValueError, TypeError):
                messages = None
        if not isinstance(identity, str) or not identity or not isinstance(task, str) or not task:
            reason = "MISSING_ID"
        elif identity in identities:
            reason = "DUPLICATE_TRAJECTORY_ID"
        elif row.get("truncated") is True or row.get("is_truncated") is True or row.get("finish_reason") in ("length","context_length","max_tokens"):
            reason = "EXPLICIT_TRUNCATION"
        elif row.get("resolved") not in (1, True) or type(row.get("resolved")) not in (int, bool):
            reason = "NOT_CONFIRMED_RESOLVED"
        elif task in excluded:
            reason = "HELD_OUT_TASK"
        elif not isinstance(messages, list) or not messages:
            reason = "INVALID_MESSAGES"
        else:
            pending = set()
            try:
                for message in messages:
                    require(isinstance(message, dict), "Non-object message")
                    require(message.get("role") in ("system", "user", "assistant", "tool"), "Unsupported role")
                    for call in message.get("tool_calls") or []:
                        key = call.get("id")
                        require(isinstance(key, str) and key and key not in pending, "Invalid call ID")
                        pending.add(key)
                    if message["role"] == "tool":
                        key = message.get("tool_call_id")
                        require(key in pending, "Orphan tool result")
                        pending.remove(key)
                require(not pending, "Missing tool results")
                fingerprint = fingerprint_func(messages,row.get("tools",[])) if fingerprint_func else digest({"messages": messages, "tools": row.get("tools", [])})
                if fingerprint in seen:
                    reason = "EXACT_DUPLICATE"
            except (ValueError, TypeError, KeyError) as exc:
                reason = "STRUCTURAL_INVALID:" + type(exc).__name__
            if reason is None:
                # Tokenizer/service failures are infrastructure, never quality rejects.
                n_input, n_supervised = token_counter(messages, row.get("tools", []))
                require(type(n_input) is int and type(n_supervised) is int and 0 <= n_supervised <= n_input, "Invalid token counts")
                if n_supervised == 0:
                    reason = "NO_SUPERVISION"
                elif n_input > max_context:
                    reason = "OVER_CONTEXT"
                else:
                    seen.add(fingerprint)
        if isinstance(identity, str) and identity:
            identities.add(identity)
        audit.append({"trajectory_id": identity, "instance_id": task, "reason": reason or "PASS"})
        if reason is None:
            accepted.append(row)
    return accepted, audit


def teacher_budgets(task_candidates, eligible_trajectories):
    require(all(type(x) is int and x >= 0 for x in (task_candidates, eligible_trajectories)), "Actual counts required")
    total = task_candidates + eligible_trajectories
    require(total > 0, "Empty annotation workload")
    return {"primary_max_calls": math.ceil(1.25 * total),
            "review_max_calls": math.ceil(0.10 * total),
            "basis": {"task_candidates": task_candidates, "eligible_trajectories": eligible_trajectories},
            "note": "Call caps, not currency caps. Pilot/context validation required."}


def resolve_sources(plan):
    from urllib.request import Request, urlopen
    resolved = {}
    for role, spec in plan["sources"].items():
        prefix = "models" if spec["repo_type"] == "model" else "datasets"
        url = "https://huggingface.co/api/" + prefix + "/" + spec["repo_id"]
        if spec.get("revision"):
            url += "/revision/" + spec["revision"]
        with urlopen(Request(url, headers={"User-Agent": "self-curate-source-lock"}), timeout=60) as response:
            metadata = json.load(response)
        revision = metadata.get("sha")
        require(isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision), "Source commit unavailable")
        require(not spec.get("revision") or revision == spec["revision"], "Pinned source identity mismatch")
        resolved[role] = {**spec, "revision": revision, "metadata": metadata}
    return {"schema_version": "rebench-source-lock-v1", "plan_hash": digest(plan), "sources": resolved}



def download_locked(lock_path, storage):
    """Workstation-only snapshots. Never called by the cluster runtime."""
    from huggingface_hub import snapshot_download
    lock = read_json(lock_path)
    require(lock.get("schema_version") == "rebench-source-lock-v1", "Wrong source lock")
    root = Path(storage).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipts = {}
    with lease(root / ".download.lease"):
        for role, spec in lock["sources"].items():
            require(re.fullmatch("[a-z][a-z0-9_]*", role), "Unsafe source role")
            require(re.fullmatch("[0-9a-f]{40}", spec["revision"]), "Immutable source revision required")
            target = root / role
            receipt = target / "snapshot-receipt.json"
            identity = {k: spec[k] for k in ("repo_id", "repo_type", "revision")}
            if receipt.exists():
                value = read_json(receipt)
                require(value["identity"] == identity, "Existing source has another identity")
                current = source_inventory(target / "files")
                require(current == value["files"], "Source bytes changed")
                receipts[role] = value
                continue
            # Preserve incomplete snapshots; snapshot_download can resume only the
            # same locked revision, verified by the durable intent below.
            target.mkdir(parents=True, exist_ok=True)
            intent = target / "download-intent.json"
            if intent.exists():
                require(read_json(intent) == identity, "Incomplete source has another revision")
            else:
                require(not (target / "files").exists(), "Unowned snapshot requires reconciliation")
                atomic_json(intent, identity)
            snapshot_download(repo_id=spec["repo_id"], repo_type=spec["repo_type"],
                              revision=spec["revision"], local_dir=target / "files")
            value = {"identity": identity, "files": source_inventory(target / "files")}
            require(bool(value["files"]), "Empty source snapshot")
            atomic_json(receipt, value)
            receipts[role] = value
    return {"schema_version": "rebench-download-v1", "source_lock_hash": digest(lock),
            "sources": receipts, "task_environments_prepared": False}


def source_inventory(root):
    from self_curate.data.pool import file_sha256
    root = Path(root)
    require(root.is_dir(), "Missing source directory")
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == ".cache":
            continue
        require(not path.is_symlink(), "Snapshot must contain actual portable files")
        if path.is_file():
            result[relative.as_posix()] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return result



def export_table(lock_path, storage, role, output):
    """Export only the explicitly locked HF config/split from local parquet shards."""
    import fnmatch
    from self_curate.data.pool import file_sha256
    lock=read_json(lock_path);spec=lock["sources"][role]
    require(spec["repo_type"]=="dataset","Only dataset snapshots can be exported")
    root=Path(storage).resolve()/role
    saved=read_json(root/"snapshot-receipt.json")
    require(saved["identity"]=={k:spec[k] for k in ("repo_id","repo_type","revision")},"Export source identity changed")
    require(source_inventory(root/"files")==saved["files"],"Export snapshot changed")
    config=spec.get("config","default");split=spec["split"]
    configs=spec["metadata"].get("cardData",{}).get("configs",[])
    match=[c for c in configs if c.get("config_name")==config]
    require(len(match)==1,"No exact data_files config in locked metadata; supply reviewed source metadata, never guess split")
    patterns=[]
    for row in match[0].get("data_files",[]):
        if row["split"]==split:
            patterns.extend([row["path"]] if isinstance(row["path"],str) else row["path"])
    files=sorted(n for n in saved["files"] if n.endswith(".parquet") and any(fnmatch.fnmatch(n,p) for p in patterns))
    require(bool(files),"No pinned parquet shards for requested config/split")
    output=Path(output).resolve()
    require(not output.exists(),"Do not overwrite frozen row export")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.name+".partial-"+__import__("uuid").uuid4().hex[:8])
    import pyarrow.parquet as pq
    count=0
    with temporary.open("w",encoding="utf-8") as stream:
        for name in files:
            for batch in pq.ParquetFile(root/"files"/name).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    def serial(value):
                        import datetime
                        if isinstance(value,(datetime.datetime,datetime.date,datetime.time)):return value.isoformat()
                        raise TypeError("Unsupported source scalar: "+type(value).__name__)
                    stream.write(json.dumps(row,ensure_ascii=False,default=serial)+"\n");count+=1
    require(count>0,"Empty dataset export")
    __import__("os").replace(temporary,output)
    result={"repo_id":spec["repo_id"],"revision":spec["revision"],"config":config,
            "split":split,"rows":count,"files":files,"sha256":file_sha256(output)}
    atomic_json(output.with_suffix(output.suffix+".source.json"),result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", choices=["workstation"], required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("resolve")
    command.add_argument("--plan", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("download")
    command.add_argument("--lock", required=True)
    command.add_argument("--storage", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("export-table")
    command.add_argument("--lock", required=True)
    command.add_argument("--storage", required=True)
    command.add_argument("--role", required=True)
    command.add_argument("--output", required=True)
    command = sub.add_parser("sample-tasks")
    command.add_argument("--input", required=True)
    command.add_argument("--count", type=int, required=True)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--exclude-ids", required=True, help="JSON list of benchmark/family exclusion IDs")
    command.add_argument("--output", required=True)
    command = sub.add_parser("teacher-budget")
    command.add_argument("--task-candidates", type=int, required=True)
    command.add_argument("--eligible-trajectories", type=int, required=True)
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    with lease(target.parent / ("." + target.name + ".lease")):
        require(not target.exists(), "Never overwrite a frozen preparation artifact")
        if args.action == "resolve":
            plan = read_json(args.plan)
            require(plan.get("execution_location") == "workstation", "Not a workstation plan")
            result = resolve_sources(plan)
        elif args.action == "download":
            result = download_locked(args.lock, args.storage)
        elif args.action == "export-table":
            result = export_table(args.lock, args.storage, args.role, args.output)
        elif args.action == "sample-tasks":
            exclusions = read_json(args.exclude_ids)
            require(type(exclusions) is list and all(type(x) is str for x in exclusions), "Exclusion IDs must be a list")
            selected = sample_tasks(rows(args.input), args.count, args.seed, exclusions)
            result = {"schema_version": "task-candidates-v1", "seed": args.seed,
                      "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                      "exclusions_hash": digest(exclusions), "tasks": selected,
                      "status": "unlabelled_not_final_development"}
        else:
            result = teacher_budgets(args.task_candidates, args.eligible_trajectories)
        if args.action != 'export-table':
            atomic_json(target, result)
    print(json.dumps({"status": "prepared", "output": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
