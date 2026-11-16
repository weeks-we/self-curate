"""Offline bundle and selected-image transfer. No cluster builds or installs."""
from pathlib import Path
import argparse
import json
import subprocess
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import read_json,atomic_json,lease


def inventory(root):
    root=Path(root).resolve()
    require(root.is_dir(),"Missing asset root")
    values={}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(),"Offline assets cannot contain symlinks")
        if path.is_file():
            relative=path.relative_to(root).as_posix()
            require(not any(p in (".git","private") for p in Path(relative).parts)
                    and not path.name.endswith((".key",".pem")) and path.name!=".env",
                    "Credentials/private state cannot enter offline bundle")
            values[relative]={"sha256":file_sha256(path),"bytes":path.stat().st_size}
    require(bool(values),"Cannot seal an empty bundle")
    return values


def seal(root,receipt,*,images=None):
    require(not Path(receipt).resolve().is_relative_to(Path(root).resolve()),"Receipt must be outside sealed tree")
    value={"schema_version":"rebench-offline-v1","files":inventory(root),
           "images":images or {},"network_download_allowed":False}
    value["hash"]=digest(value)
    require(not Path(receipt).exists(),"Do not overwrite a sealed bundle")
    atomic_json(receipt,value)
    return value


def verify(root,receipt):
    value=read_json(receipt)
    expected={k:v for k,v in value.items() if k!="hash"}
    require(value.get("schema_version")=="rebench-offline-v1" and value["hash"]==digest(expected),"Bundle receipt changed")
    require(inventory(root)==value["files"],"Offline bundle bytes/file set changed")
    return value


def image_transfer(images,archive,*,mode,run=subprocess.run,expected=None):
    require(mode in ("export","import"),"Unknown image transfer")
    archive=Path(archive).resolve()
    require(type(images) is list and bool(images) and all(type(i) is str and i and not i.startswith("-") for i in images),"Explicit image inventory required")
    names=sorted(set(images))
    if mode=="export":
        require(not archive.exists(),"Do not overwrite image archive")
        archive.parent.mkdir(parents=True,exist_ok=True)
        records={}
        for name in names:
            values=json.loads(run(["docker","image","inspect",name],capture_output=True,text=True,check=True).stdout)
            require(len(values)==1,"Ambiguous image")
            records[name]={"id":values[0]["Id"],"repo_digests":values[0].get("RepoDigests") or []}
        run(["docker","image","save","-o",str(archive),*names],check=True)
        return {"images":records,"archive_sha256":file_sha256(archive)}
    require(expected is not None and set(expected["images"])==set(names),"Image import inventory mismatch")
    require(file_sha256(archive)==expected["archive_sha256"],"Image archive bytes changed")
    run(["docker","image","load","-i",str(archive)],check=True)
    for name,record in expected["images"].items():
        actual=json.loads(run(["docker","image","inspect",name],capture_output=True,text=True,check=True).stdout)
        require(len(actual)==1 and actual[0]["Id"]==record["id"],"Imported image differs")
    return {"status":"imported_verified","images":expected["images"]}


def run_workstation_checks(manifest,output,*,process=None):
    """Run pinned official oracle/nop/network probes; no target solving or GPU.
    Caller declares expected evidence files; the CLI never fabricates probe results.
    """
    from self_curate.execution.process import execute
    process = process or execute
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    require(manifest.get("location")=="workstation","Environment preparation belongs on workstation")
    results=[]
    for index,item in enumerate(manifest["checks"]):
        require(item["kind"] in ("oracle","nop","network_probe","git_probe","dependency_probe","cli_probe"),"Unsupported preflight check")
        require(type(item["argv"]) is list and all(type(x) is str for x in item["argv"]),"Shell-free official command required")
        folder=output/str(index)
        if (folder/"receipt.json").exists():
            saved=read_json(folder/"receipt.json")
            require(saved["request_hash"]==digest(item),"Workstation probe changed")
            for name,sha in saved["evidence"].items():require(file_sha256(Path(item["cwd"])/name)==sha,"Probe evidence changed")
            results.append(saved);continue
        require(not folder.exists(),"Unknown in-flight probe; reconcile before retry")
        env=dict(__import__("os").environ);env["CUDA_VISIBLE_DEVICES"]=""
        result=process(item["argv"],cwd=item["cwd"],output=folder,timeout=item["timeout_seconds"],env=env)
        require(result["returncode"]==0 and not result["timed_out"],"Workstation official probe failed")
        evidence={}
        for name,expect in item["evidence"].items():
            path=(Path(item["cwd"])/name).resolve()
            require(path.is_relative_to(Path(item["cwd"]).resolve()),"Probe evidence escaped cwd")
            value=read_json(path)
            def field(value,key):
                for part in key.split("."):
                    require(isinstance(value,dict) and part in value,"Probe field missing")
                    value=value[part]
                return value
            require(all(field(value,k)==v for k,v in expect.items()),"Official probe expectations not met")
            evidence[name]=file_sha256(path)
        require(bool(evidence),"A successful command alone is not environment evidence")
        receipt={"request_hash":digest(item),"evidence":evidence,"process":result}
        atomic_json(folder/"receipt.json",receipt);results.append(receipt)
    return {"checks":results,"gpu_capacity_verified":False}



def publish_pro_protocol(manifest, checks_root, tooling, tasks, prepared, output):
    """Build runtime capability evidence from checked workstation probe artifacts."""
    from self_curate.training.artifacts import checkpoint_fingerprint
    from self_curate.execution.artifacts import local_path
    from self_curate.contracts.parsing import artifact
    require(manifest.get("location")=="workstation" and manifest.get("backend") in ("real","mock"),"Declare probe provenance")
    completed=run_workstation_checks(manifest,checks_root)
    task_map={Path(p).name:checkpoint_fingerprint(Path(p))[0] for p in tasks}
    require(all(v.get("harness_backend")=="claude_agent_sdk" for v in prepared.values()),
            "Old mini agent preparation is incompatible with Claude protocol")
    require(set(prepared)==set(task_map),"Prepared Pro environments do not cover exact task set")
    kinds={name:set() for name in task_map}
    network=False
    sampling=False
    required_generation=manifest.get("generation_policy")
    generation_keys={"temperature","top_p","top_k","thinking","preserve_thinking","reasoning_effort","min_p","presence_penalty","repetition_penalty"}
    require(isinstance(required_generation,dict) and set(required_generation)==generation_keys,
            "Freeze the complete actual Qwen generation policy for the Pro probe")
    for item in manifest["checks"]:
        kind=item["kind"]
        if kind in ("oracle","nop","dependency_probe"):
            require(item.get("task_id") in kinds,"Probe refers to another Pro task")
            kinds[item["task_id"]].add(kind)
            if kind in ("oracle","nop"):
                expect=1 if kind=="oracle" else 0
                require(any(e.get("verifier_result.rewards.reward")==expect for e in item["evidence"].values()),
                        "Pro oracle/nop must check the official resolved reward")
        if kind=="network_probe":
            require(any(e.get("model_endpoint_allowed") is True and e.get("outside_network_blocked") is True
                        for e in item["evidence"].values()),"Network probe lacks allow/deny evidence")
            network=True
        if kind=="cli_probe" and any(e.get("harness_backend")=="claude_agent_sdk"
                and all(e.get("effective_generation."+k)==v for k,v in required_generation.items())
                and all(e.get("effective_sampling."+k)==required_generation[k] for k in ("temperature","top_p","top_k","min_p","presence_penalty","repetition_penalty"))
                for e in item["evidence"].values()):
            sampling=True
    require(sampling,"Probe actual engine generation, thinking and sampling plus Claude harness routing")
    require(network and all({"oracle","nop","dependency_probe"}<=v for v in kinds.values()),"Incomplete offline Pro probe coverage")
    require(type(manifest.get("model_endpoints")) is list and bool(manifest["model_endpoints"]),"Freeze allowed model endpoints")
    require(manifest.get("network_overlay") is not None,"Freeze the Docker model-only network overlay")
    local_path(artifact(manifest["network_overlay"]))
    value={"schema_version":"pro-v2-claude-sdk-offline-v2","backend":manifest["backend"],
           "offline_agent_verified":True,"offline_verifier_verified":True,"fresh_sandbox_verified":True,
           "sampling_verified":True,"generation_policy":required_generation,
           "greedy_sampling_verified":required_generation["temperature"]==0,
           "environment":"docker","tooling_checksum":checkpoint_fingerprint(Path(tooling))[0],
           "task_checksums":task_map,"prepared_environments":prepared,
           "model_endpoints":manifest["model_endpoints"],"network_overlay":manifest["network_overlay"],
           "probe_manifest_hash":digest(manifest),"probe_receipts_hash":digest(completed)}
    require(not Path(output).exists(),"Do not overwrite frozen Pro protocol")
    atomic_json(output,value)
    return value

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location",choices=["workstation","cluster"],required=True)
    sub=parser.add_subparsers(dest="stage",required=True)
    for name in ("seal","verify"):
        cmd=sub.add_parser(name);cmd.add_argument("--root",required=True);cmd.add_argument("--receipt",required=True)
    for name in ("export-images","import-images"):
        cmd=sub.add_parser(name);cmd.add_argument("--images",required=True);cmd.add_argument("--archive",required=True);cmd.add_argument("--receipt",required=True)
    cmd=sub.add_parser("checks");cmd.add_argument("--manifest",required=True);cmd.add_argument("--output",required=True)
    cmd=sub.add_parser("pro-receipt")
    cmd.add_argument("--manifest",required=True);cmd.add_argument("--checks-root",required=True)
    cmd.add_argument("--tooling",required=True);cmd.add_argument("--tasks",required=True)
    cmd.add_argument("--prepared",required=True);cmd.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    require(args.location=="workstation" or args.stage in ("verify","import-images"),"Cluster cannot download/build/install/prepare")
    if args.stage=="seal":result=seal(args.root,args.receipt)
    elif args.stage=="verify":result=verify(args.root,args.receipt)
    elif args.stage=="checks":result=run_workstation_checks(read_json(args.manifest),args.output)
    elif args.stage=="pro-receipt":result=publish_pro_protocol(read_json(args.manifest),args.checks_root,args.tooling,
          [p.parent for p in Path(args.tasks).glob("*/task.toml")],read_json(args.prepared),args.output)
    else:
        images=read_json(args.images)
        if args.stage=="export-images":
            require(not Path(args.receipt).exists(),"Image receipt already exists")
            result=image_transfer(images,args.archive,mode="export")
            atomic_json(args.receipt,result)
        else:result=image_transfer(images,args.archive,mode="import",expected=read_json(args.receipt))
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
