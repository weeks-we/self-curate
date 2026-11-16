"""Selected task environments: workstation preparation and native registration."""
from pathlib import Path
from dataclasses import replace
import argparse
import asyncio
import json
import subprocess
from self_curate.contracts import TaskBudget,parse_task_record
from self_curate.contracts.parsing import artifact
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json,atomic_json
from self_curate.execution.registration import register_swe_task,register_harbor_task
from self_curate.execution.pro_v2 import register_pro_task
from self_curate.preparation.isolation import family_ids
from self_curate.experiments.exp001.workstation import rows


def catalog(bindings):
    verified=list(rows(bindings["verified"]))
    pro=list(rows(bindings["pro"]))
    terminal=[]
    for task in sorted(Path(bindings["terminal_tasks"]).glob("*/task.toml")):
        folder=task.parent
        terminal.append({"instance_id":"terminal:"+folder.name,
                         "problem_statement":(folder/"instruction.md").read_text(encoding="utf-8"),
                         "family_ids":["terminal:"+folder.name]})
    require(len({v["instance_id"] for v in verified})==len(verified),"Duplicate Verified task")
    require(len({v["instance_id"] for v in pro})==len(pro),"Duplicate Pro task")
    if bindings.get("backend","real")=="real":
        require(len(verified)==500 and len(pro)==642,"Wrong frozen Verified / Pro V2 task sets")
    require(bool(terminal),"No frozen Terminal tasks")
    return verified,pro,terminal


def native_image(row,python):
    code="import json,sys; from self_curate.execution.native import test_spec,spec_image; print(spec_image(test_spec(json.load(sys.stdin))))"
    result=subprocess.run([python,"-c",code],input=json.dumps(row),capture_output=True,text=True,check=True,timeout=120)
    name=result.stdout.strip()
    require(bool(name) and "\n" not in name,"Official harness omitted image")
    return name


def prepare_images(bindings,*,run=subprocess.run,resolver=native_image,harbor=None):
    from self_curate.server.assets import harbor_images
    selected=read_json(bindings["selected"])
    verified,pro,terminal=catalog(bindings)
    root=Path(bindings["output"]).resolve();root.mkdir(parents=True,exist_ok=True)
    pins={};names=set()
    for kind,values,python in (("swe_rebench",selected["tasks"],bindings["rebench_python"]),
                               ("swe_bench",verified,bindings["swe_python"])):
        for row in values:
            name=resolver(row,python)
            require(type(name) is str and name and not name.startswith("-"),"Invalid native image name")
            run(["docker","image","pull",name],check=True)
            actual=json.loads(run(["docker","image","inspect",name],capture_output=True,text=True,check=True).stdout)[0]
            digests=actual.get("RepoDigests") or []
            require(bool(digests),"Publish native task image with immutable RepoDigest")
            pins[row["instance_id"]]=digests[0];names.add(digests[0])
    pro_lookup={r["instance_id"] for r in pro}
    for parent,ids in ((Path(bindings["pro_tasks"]),pro_lookup),(Path(bindings["terminal_tasks"]),{r["instance_id"].split(":",1)[1] for r in terminal})):
        for key in sorted(ids):
            folder=parent/key
            require((folder/"task.toml").is_file(),"Missing exact Harbor task directory")
            prepared=harbor(folder,root/"harbor-image-work"/key) if harbor else asyncio.run(harbor_images(folder,root/"harbor-image-work"/key))
            names.update(prepared)
    atomic_json(root/"swe-image-pins.json",pins)
    atomic_json(root/"image-names.json",sorted(names))
    return {"pins":pins,"images":sorted(names),"all_selected_development_tasks":len(selected["tasks"])}


async def verifier_snapshot(task_path,root):
    """Official nop on a fresh image, never a target or gold solution.
    Snapshot stays verifier-only; do not expose hidden verifier files to the agent.
    """
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    from self_curate.training.artifacts import checkpoint_fingerprint
    import docker
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    identity=checkpoint_fingerprint(Path(task_path))[0]
    if (root/"verifier.json").exists():
        saved=read_json(root/"verifier.json")
        require(saved["task_checksum"]==identity,"Verifier task changed")
        return saved
    cfg=TrialConfig.model_validate({"task":{"path":str(Path(task_path).resolve())},
        "trial_name":"nop-cache","trials_dir":str(root/"trials"),
        "agent":{"name":"nop"},"environment":{"type":"docker","delete":False,
            "override_gpus":0,"kwargs":{"keep_containers":True}},"verifier":{"disable":False}})
    trial=Trial(cfg)
    env=None;client=None
    try:
        result=await trial.run()
        env=trial.agent_environment
        value=result.model_dump(mode="json")
        require(value.get("exception_info") is None
                and (value.get("verifier_result") or {}).get("rewards",{}).get("reward")==0,
                "Pro nop environment must execute successfully and remain unresolved")
        container_ids=(await env._run_docker_compose_command(["ps","--all","--quiet","main"])).stdout.split()
        require(len(container_ids)==1,"No retained official nop environment")
        client=docker.from_env();container=client.containers.get(container_ids[0])
        require(not container.attrs.get("HostConfig",{}).get("DeviceRequests"),"Workstation verifier cache requested GPUs")
        image=container.commit(repository="self-curate/pro-verifier-cache",tag=identity)
        saved={"task_checksum":identity,"image_id":image.id,"provenance":"official_nop_not_agent_or_gold",
               "nop_result":value}
        atomic_json(root/"verifier.json",saved)
        return saved
    finally:
        if client is not None:client.close()
        if env is not None:await env._run_docker_compose_command(["down","--volumes","--remove-orphans"])


def install(bindings):
    from self_curate.server.harbor_prepare import install_snapshot
    root=Path(bindings["output"]).resolve()
    _,pro,terminal=catalog(bindings)
    prepared={}
    images=set(read_json(root/"image-names.json"))
    for row in pro:
        name=row["instance_id"];folder=Path(bindings["pro_tasks"])/name
        agent=asyncio.run(install_snapshot(folder,root/"installed"/name))
        verifier=asyncio.run(verifier_snapshot(folder,root/"verifiers"/name))
        prepared[name]={"agent_image":agent["image_id"],"verifier_image":verifier["image_id"],
                        "harness_backend":agent["harness_backend"],"sdk_version":agent["sdk_version"]}
        images.update((agent["image_id"],verifier["image_id"]))
    for row in terminal:
        name=row["instance_id"].split(":",1)[1]
        value=asyncio.run(install_snapshot(Path(bindings["terminal_tasks"])/name,root/"installed"/name))
        images.add(value["image_id"])
    atomic_json(root/"pro-prepared-environments.json",prepared)
    atomic_json(root/"image-names.json",sorted(images))
    return prepared


def register(bindings):
    root=Path(bindings["output"]).resolve()
    selected=read_json(bindings["selected"]);verified,pro,terminal=catalog(bindings)
    pins=read_json(root/"swe-image-pins.json")
    lock=artifact(bindings["runtime_lock"])
    rebench_lock=artifact(bindings.get("rebench_runtime_lock",bindings["runtime_lock"]))
    swe_budget=TaskBudget(**bindings["swe_budget"])
    pro_budget=TaskBudget(**bindings.get("pro_budget",bindings.get("harbor_budget",{})))
    terminal_budget=TaskBudget(**bindings.get("terminal_budget",bindings.get("harbor_budget",{})))
    development=[];benchmarks={"swe_verified":[],"swe_pro_v2":[],"terminal_bench":[]}
    def swe(row,kind,split,skills):
        destination=root/"registered"/kind/row["instance_id"]
        if (destination/"task.json").exists():
            task=parse_task_record(read_json(destination/"task.json"))
            from self_curate.execution.artifacts import read_artifact
            require(read_artifact(task.payload)["instance"]==row and task.budget==swe_budget,"Task registration changed")
            require(read_artifact(task.scorer)["runtime_lock"]==(rebench_lock if kind=="swe_rebench" else lock).to_dict(),"Registered runtime changed")
            return task
        task=register_swe_task(row,harness=kind,split=split,skills=skills,budget=swe_budget,
            image=pins[row["instance_id"]],runtime_lock=rebench_lock if kind=="swe_rebench" else lock,output=destination)
        task=replace(task,family_ids=family_ids(row))
        atomic_json(destination/"task.json",task.to_dict())
        return task
    for row in selected["tasks"]:
        development.append(swe(row,"swe_rebench","development",tuple(selected["assignments"][row["instance_id"]])).to_dict())
    for row in verified:benchmarks["swe_verified"].append(swe(row,"swe_bench","test",("benchmark",)).to_dict())
    for suite,values,parent in (("swe_pro_v2",pro,Path(bindings["pro_tasks"])),
                                ("terminal_bench",terminal,Path(bindings["terminal_tasks"]))):
        for row in values:
            name=row["instance_id"].split(":",1)[-1] if suite=="terminal_bench" else row["instance_id"]
            destination=root/"registered"/suite/name
            saved=destination/"task.json"
            if saved.exists():
                task=parse_task_record(read_json(saved))
                require(task.budget==(pro_budget if suite=="swe_pro_v2" else terminal_budget),"Registered Harbor budget changed; use new environment output")
                from self_curate.execution.artifacts import read_artifact
                require(read_artifact(task.scorer)["runtime_lock"]==lock.to_dict(),"Registered Harbor runtime changed")
                if suite=="swe_pro_v2":
                    require(read_artifact(task.environment)["pro_protocol_receipt"]==artifact(bindings["pro_protocol_receipt"]).to_dict(),"Registered Pro receipt changed")
            elif suite=="swe_pro_v2":
                task=register_pro_task(parent/name,task_id=row["instance_id"],skills=("benchmark",),budget=pro_budget,
                     runtime_lock=lock,protocol_receipt=artifact(bindings["pro_protocol_receipt"]),output=destination)
                task=replace(task,family_ids=family_ids(row))
                atomic_json(saved,task.to_dict())
            else:
                task=register_harbor_task(parent/name,task_id=row["instance_id"],skills=("benchmark",),budget=terminal_budget,
                     runtime_lock=lock,output=destination)
                task=replace(task,family_ids=family_ids(row))
                atomic_json(saved,task.to_dict())
            benchmarks[suite].append(task.to_dict())
    atomic_json(root/"development-records.json",development)
    atomic_json(root/"benchmark-records.json",benchmarks)
    return {"development":str(root/"development-records.json"),"benchmarks":str(root/"benchmark-records.json")}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location",choices=["workstation"],required=True)
    parser.add_argument("--bindings",required=True)
    parser.add_argument("--stage",choices=["catalog","images","install","register"],required=True)
    args=parser.parse_args(argv);bindings=read_json(args.bindings)
    require(bindings.get("execution_location")=="workstation","Environment preparation cannot run on cluster")
    root=Path(bindings["output"]);root.mkdir(parents=True,exist_ok=True)
    from self_curate.training.recovery import lease
    with lease(root/".environments.lease"):
        if args.stage=="catalog":
            values=catalog(bindings)
            target=root/"benchmark-instances.jsonl"
            target.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for group in values for r in group),encoding="utf-8")
            result={"benchmark_instances":str(target)}
        elif args.stage=="images":result=prepare_images(bindings)
        elif args.stage=="install":result=install(bindings)
        else:result=register(bindings)
        atomic_json(root/(args.stage+"-complete.json"),result)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
