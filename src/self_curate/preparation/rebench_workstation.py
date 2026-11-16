"""Workstation-only Rebench stage runner. No implicit GPU assignment or old bootstrap."""
import argparse
import json
from pathlib import Path
from dataclasses import replace
from self_curate.contracts import digest, canonical_json, CostLedger, parse_task_record
from self_curate.contracts.parsing import model_ref, artifact
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json,atomic_json,lease
from self_curate.execution.artifacts import publish_json,read_artifact
from self_curate.data.pool import file_sha256
from self_curate.preparation.rebench import (
    ontology,annotate_tasks,select_skill_groups,prepare_records,publish_prepared,
)
from self_curate.preparation.isolation import IsolationIndex
from self_curate.preparation.provider import AnnotationClient
from self_curate.preparation.teacher import load_settings
from self_curate.preparation.pipeline import annotation_ledger
from self_curate.experiments.exp001.workstation import rows,sample_tasks


def renderer_from(bindings,model):
    from transformers import AutoTokenizer
    from self_curate.preparation.rendering import ChatTemplateTokenizer
    tokenizer=AutoTokenizer.from_pretrained(bindings["tokenizer_path"],local_files_only=True,trust_remote_code=False)
    return ChatTemplateTokenizer(tokenizer,model.tokenizer)


def stage(bindings,phase, *, client=None,review=None,renderer=None):
    root=Path(bindings["output"]).resolve(); root.mkdir(parents=True,exist_ok=True)
    require(bindings.get("execution_location")=="workstation","Preparation cannot run on the cluster")
    settings=load_settings(bindings["ontology"])
    if "review_mode" in bindings:settings={**settings,"review_mode":bindings["review_mode"]}
    from .teacher import review_mode
    mode=review_mode(settings)
    skills=ontology(settings)
    require(bindings.get("development_per_skill",50)==50,"This experiment fixes 50 tasks per primary skill")
    require(bindings.get("execution_location")=="workstation","Preparation cannot run on the cluster")
    task_rows=list(rows(bindings["tasks"]))
    test_rows=list(rows(bindings["benchmark_instances"]))
    index=IsolationIndex(bindings.get("near_duplicate_threshold",0.85))
    for row in test_rows:index.add(row)
    if bindings.get("candidate_count") is None:
        covered={r["instance_id"] for r in rows(bindings["trajectories"]) if r.get("resolved")==1}
        population=bindings.get("task_population","success_covered")
        available=sum(not index.conflicts(t) and (population=="all_tasks" or (t["instance_id"] not in covered if population=="success_uncovered" else t["instance_id"] in covered)) for t in task_rows)
        bindings = {**bindings, "candidate_count": min(available,max(1000,200*len(skills)))}
    if bindings.get("backend","real") == "real":
        registry=read_json(bindings["source_registry"])
        for name,source in (("tasks","development_tasks"),("trajectories","rebench_trajectories")):
            if name=="trajectories" and phase=="tasks" and not bindings.get("helper_trajectories",True):
                continue
            source_receipt=read_json(str(bindings[name])+".source.json")
            spec=registry["datasets"][source]
            require(source_receipt["repo_id"]==spec["repo_id"]
                    and source_receipt["revision"]==spec["revision"]
                    and source_receipt["split"]==spec["split"]
                    and source_receipt["config"]==spec["config"]
                    and source_receipt["sha256"]==file_sha256(Path(bindings[name])),
                    "Local row export is not from the locked source")
    intent={"phase":phase,"ontology":digest(settings),
            "implementation_hash":digest({str(p.name):file_sha256(p) for p in (
                Path(__file__),Path(__file__).with_name("rebench.py"),
                Path(__file__).with_name("teacher.py"),Path(__file__).with_name("development_asset.py"),
                Path(__file__).with_name("provider.py"),Path(__file__).with_name("teacher_grouped.py"),
                Path(__file__).parents[1]/"data/task_links.py",
                Path(__file__).parents[1]/"experiments/exp001/workstation.py")}),
            "tasks":file_sha256(Path(bindings["tasks"])),
            "benchmarks":file_sha256(Path(bindings["benchmark_instances"])),
            "seed":bindings.get("seed",42),"trajectory_grouping":bindings.get("trajectory_grouping",{}),
            "teacher":bindings.get("teacher"),"review_teacher":bindings.get("review_teacher"),"review_mode":mode,
            "max_context_tokens":bindings.get("max_context_tokens",131072),
            "model_sha256":file_sha256(Path(bindings["model"])) if phase!="tasks" else None,
            "registered_inputs":{k:file_sha256(Path(bindings[k])) for k in ("registered_development","registered_benchmarks","source_registry","source_lock") if phase=="publish"}}
    if phase=="publish":
        intent["evaluation_protocols"]={k:file_sha256(Path(v)) for k,v in bindings.get("benchmark_protocol_files",{}).items()}
        intent["development_protocol"]=bindings.get("protocol") or file_sha256(Path(bindings.get("task_protocol_file","configs/experiments/exp001/swe-task-protocol.json")))
        intent["sampling"]=bindings.get("sampling")
        intent["task_generation"]=bindings.get("task_generation",{})
    if phase=="tasks":
        intent["candidate_count"]=bindings["candidate_count"]
        intent["task_population"]=bindings.get("task_population","success_covered")
        intent["helper_trajectories"]=bindings.get("helper_trajectories",True)
        intent["helper_source_sha256"]=file_sha256(Path(bindings["trajectories"])) if bindings.get("helper_trajectories",True) else None
        intent["development_asset"]=bindings.get("development_asset")
    else:
        intent["trajectories"]=file_sha256(Path(bindings["trajectories"]))
        intent["development"]=file_sha256(root/"selected-development.json")
    frozen=root/(phase+"-intent.json")
    if frozen.exists():require(read_json(frozen)==intent,"Stage inputs changed; use a new preparation root")
    else:atomic_json(frozen,intent)
    teacher = {**bindings["teacher"], "max_calls": bindings["teacher"].get("max_calls") or __import__("math").ceil(1.25*bindings["candidate_count"])}
    client=client or AnnotationClient({**teacher,"backend":bindings.get("backend","real"),"execution":"workstation"},Path(bindings.get("teacher_cache",root/"teacher")))
    review=None if mode=="off" else (review or (AnnotationClient({**bindings["review_teacher"],"max_calls":bindings["review_teacher"].get("max_calls") or __import__("math").ceil(.1*bindings["candidate_count"]),"backend":bindings.get("backend","real"),"execution":"workstation"},Path(bindings.get("review_cache",root/"review"))) if bindings.get("review_teacher") else None))
    client.settings["max_calls_is_explicit"]=bindings["teacher"].get("max_calls") is not None
    if review is not None:review.settings["max_calls_is_explicit"]=bindings.get("review_teacher",{}).get("max_calls") is not None
    client.usage_path = root/"primary-requests.json"
    if review is not None:review.usage_path = root/"review-requests.json"
    if phase=="tasks":
        if bindings.get("development_asset"):
            from .development_asset import load
            result=load(bindings["development_asset"],expected_ontology_hash=digest(settings),
                expected_source={"tasks_sha256":file_sha256(Path(bindings["tasks"])),
                                 "trajectory_revision":bindings.get("trajectory_revision")})
            require(all(not index.conflicts(t) for t in result["tasks"]),"Reused development tasks overlap new benchmarks")
            result={**result,"development_asset":bindings["development_asset"]}
            atomic_json(root/"selected-development.json",result)
            return result
        covered={r["instance_id"] for r in rows(bindings["trajectories"]) if r.get("resolved")==1}
        population=bindings.get("task_population","success_covered")
        require(population in ("success_covered","success_uncovered","all_tasks"),"Unknown development task population")
        eligible=[t for t in task_rows if not index.conflicts(t) and
                  (population=="all_tasks" or (t["instance_id"] not in covered if population=="success_uncovered" else t["instance_id"] in covered))]
        require(len(eligible)>=bindings["candidate_count"],"Not enough isolated tasks in the declared task population")
        if settings.get("task_labels_only"):
            require(population=="success_uncovered","Task-only profile requires development tasks outside successful trajectories")
            from .task_label_pool import label_tasks
            successful_tasks=[t for t in task_rows if t["instance_id"] in covered]
            require({t["instance_id"] for t in successful_tasks}==covered,"Successful trajectory/task join incomplete")
            needed=__import__('math').ceil(1.25*(len(covered)+bindings["candidate_count"]))
            if not client.settings.get("max_calls_is_explicit"):client.settings["max_calls"]=needed
            label_tasks(successful_tasks,settings,client,root/"training-task-labels",review)
            atomic_json(root/"task-populations.json",{"successful_task_count":len(covered),
                "development_population":"success_uncovered","eligible_development_count":len(eligible)})
        candidates=sample_tasks(eligible,bindings["candidate_count"],bindings.get("seed",42))
        atomic_json(root/"task-candidates.json",candidates)
        helpers={}
        if settings.get("annotation_contract")=="coding-teacher-v1" and bindings.get("helper_trajectories",True):
            candidate_ids={t["instance_id"] for t in candidates}
            for row in rows(bindings["trajectories"]):
                key=row["instance_id"]
                if key in candidate_ids and row.get("resolved")==1:
                    if key not in helpers or row["trajectory_id"]<helpers[key]["trajectory_id"]:helpers[key]=row
        labels=annotate_tasks(candidates,settings,client,root/"task-labels",review,helpers=helpers)
        selected,assignments,groups=select_skill_groups(candidates,labels,skills,seed=bindings.get("seed",42),per_skill=50)
        if settings.get("group_policy")=="unique-primary-v1":
            require(len(selected)==50*len(skills),"Primary skill groups must contain unique tasks")
        result={"candidate_count":bindings["candidate_count"],"tasks":list(selected),"assignments":assignments,"groups":groups,
                "per_skill":50,"unique_tasks":len(selected),"ontology_hash":digest(settings)}
        # Match the portable JSON representation on both first publication and reuse.
        result=json.loads(canonical_json(result))
        from self_curate.data.task_links import task_identity,validate_task_join
        task_spec=(read_json(bindings["source_registry"])["datasets"]["development_tasks"]
                   if bindings.get("backend","real")=="real" else
                   {"repo_id":bindings.get("task_source_id","fixture-task-source"),"revision":bindings.get("task_source_revision","fixture")})
        selected_by_id={t["instance_id"]:t for t in result["tasks"]}
        members={key:[] for key in selected_by_id}
        for raw in rows(bindings["trajectories"]):
            if raw.get("instance_id") in members:
                validate_task_join(raw,selected_by_id[raw["instance_id"]])
                require(type(raw.get("trajectory_id")) is str and raw["trajectory_id"],"Missing linked trajectory ID")
                members[raw["instance_id"]].append(raw["trajectory_id"])
        result["task_links"]={key:{**task_identity(t,source_id=task_spec["repo_id"],revision=task_spec["revision"]),
                                  "trajectory_ids":sorted(set(members[key]))} for key,t in selected_by_id.items()}
        atomic_json(root/"selected-task-relations.json",result["task_links"])
        if settings.get("annotation_contract")=="coding-teacher-v1":
            from .development_asset import publish
            result["development_asset"]=publish(result,settings,
                {"tasks_sha256":file_sha256(Path(bindings["tasks"])),
                 "trajectory_revision":bindings.get("trajectory_revision")},root/"task-labels",root/"development-asset")
        atomic_json(root/"selected-development.json",result)
        return result
    require(phase in ("pool","publish"),"Unknown preparation stage")
    selected=read_json(root/"selected-development.json")
    if selected.get("development_asset"):
        from .development_asset import load
        checked=load(selected["development_asset"],expected_ontology_hash=digest(settings))
        require(checked=={k:v for k,v in selected.items() if k!="development_asset"},"Development reference changed")
    require(selected["ontology_hash"]==digest(settings) and selected["per_skill"]==50,"Development ontology/quota changed")
    for task in selected["tasks"]: index.add(task)
    model=model_ref(read_json(bindings["model"]))
    if phase=="pool":
        renderer=renderer or renderer_from(bindings,model)
        record_path=root/"pool/records.jsonl"
        record_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=record_path.with_suffix(".jsonl.partial")
        stream=temporary.open("w",encoding="utf-8")
        try:
            pool,budget=prepare_records(lambda:rows(bindings["trajectories"]),task_rows,renderer,
            {**settings,"source_id":"nebius/SWE-rebench-openhands-trajectories","source_revision":bindings.get("trajectory_revision") or read_json(bindings["download_lock"])["sources"]["trajectories"]["revision"],
             "task_candidates":selected["candidate_count"],
             "task_source_id":read_json(bindings["source_registry"])["datasets"]["development_tasks"]["repo_id"],
             "task_source_revision":read_json(bindings["source_registry"])["datasets"]["development_tasks"]["revision"],
             "trajectory_grouping":bindings.get("trajectory_grouping",{}),
             "task_labels_root":str(root/"training-task-labels")},
            client,root/"pool",held_out=index,max_context=bindings.get("max_context_tokens",131072),
            review_client=review,backend=bindings.get("backend","real"),
            record_sink=lambda r:stream.write(canonical_json(r.to_dict())+"\n"))
        finally:stream.close()
        __import__("os").replace(temporary,record_path)
        return {"eligible":read_json(root/"pool/admission-report.json")["eligible"],"selection_budget":budget}
    from self_curate.signals import read_records
    from self_curate.validation import TaskSuite
    from self_curate.experiments.exp001.inputs import BenchmarkSuite
    pool=read_records(root/"pool/records.jsonl")
    require(bool(pool),"No prepared training records")
    registrations=read_json(bindings["registered_development"])
    by_id={t.task_id:t for t in map(parse_task_record,registrations)}
    require(set(by_id)=={t["instance_id"] for t in selected["tasks"]},"Register exactly the selected development tasks")
    development=[]
    source={t["instance_id"]:t for t in selected["tasks"]}
    for key in sorted(by_id):
        task=by_id[key]
        require(read_artifact(task.payload)["instance"]==source[key],"Registered development content differs from labelled task")
        development.append(replace(task,skills=tuple(selected["assignments"][key]),split="development"))
    sampling=artifact(bindings["sampling"]) if bindings.get("sampling") else publish_json(root/"task-sampling.json",{"schema_version":"1","temperature":bindings.get("task_generation",{}).get("temperature",1.0),"seed_policy":"request"})
    protocol=artifact(bindings["protocol"]) if bindings.get("protocol") else publish_json(root/"task-protocol.json",read_json(bindings.get("task_protocol_file","configs/experiments/exp001/swe-task-protocol.json")))
    suite=TaskSuite("rebench-development",tuple(development),(bindings.get("seed",42),),sampling,protocol,
                    "task_mean",bindings.get("backend","real"))
    benchmarks=[]
    registered=read_json(bindings["registered_benchmarks"])
    if bindings.get("backend","real")=="real":
        require(set(bindings.get("benchmark_protocol_files",{}))==set(registered),
                "Freeze one explicit protocol file for every registered benchmark")
    for name,values in registered.items():
        benchmark_protocol=protocol
        files=bindings.get("benchmark_protocol_files",{})
        if name in files:
            benchmark_protocol=publish_json(root/(name+"-protocol.json"),read_json(files[name]))
        benchmarks.append(BenchmarkSuite(name,tuple(parse_task_record(v) for v in values),
                          (bindings.get("seed",42),),sampling,benchmark_protocol,bindings.get("backend","real")))
    benchmark_ids={r["instance_id"] for r in test_rows}
    require({t.task_id for b in benchmarks for t in b.tasks} == benchmark_ids,
            "Registered benchmarks differ from the isolation catalog")
    lock=read_json(bindings["source_lock"])
    lock["selection_budget"]=read_json(root/"pool/selection-budget.json")
    lock["source_registry_sha256"]=file_sha256(Path(bindings["source_registry"]))
    isolation=publish_json(root/"isolation.json",{"policy":"id-family-near-text-v1",
        "development":selected,"benchmark_instances_sha256":file_sha256(Path(bindings["benchmark_instances"])),
        "admission":read_json(root/"pool/admission-report.json")})
    ledgers=[annotation_ledger(client,bindings.get("backend","real"),request_keys=set(read_json(root/"primary-requests.json")) if (root/"primary-requests.json").exists() else set())]
    if review is not None:ledgers.append(annotation_ledger(review,bindings.get("backend","real"),request_keys=set(read_json(root/"review-requests.json")) if (root/"review-requests.json").exists() else set()))
    ledger=CostLedger("rebench-preparation",tuple(e for l in ledgers for e in l.events),bindings.get("backend","real"))
    path=publish_prepared(pool,model,suite,tuple(benchmarks),skills,lock,isolation,ledger,root/"prepared")
    return {"prepared_manifest":str(path),"selection_budget":lock["selection_budget"]}



def control_stage(bindings, phase):
    root=Path(bindings["output"]).resolve();root.mkdir(parents=True,exist_ok=True)
    require(bindings.get("execution_location")=="workstation","Control preparation is workstation-only")
    if phase=="registry":
        from self_curate.preparation.rebench import registry_and_receipts
        settings=load_settings(bindings["ontology"])
        skills=ontology(settings)
        registry,lock=registry_and_receipts(read_json(bindings["download_lock"]),bindings["sources"],
                                           read_json(bindings["terminal_source"]),skills)
        atomic_json(bindings["source_registry"],registry)
        atomic_json(bindings["source_lock"],lock)
        return {"source_registry":bindings["source_registry"],"source_lock":bindings["source_lock"]}
    if phase=="model":
        from self_curate.server.bootstrap import prepare_model
        from self_curate.experiments.exp001.workstation import source_inventory
        spec=read_json(bindings["download_lock"])["sources"]["model"]
        folder=Path(bindings["sources"])/"model"
        receipt=read_json(folder/"snapshot-receipt.json")
        require(receipt["identity"]=={k:spec[k] for k in ("repo_id","repo_type","revision")}
                and source_inventory(folder/"files")==receipt["files"],"Model snapshot changed")
        model=prepare_model({"model":spec},{"root":str((folder/"files").resolve())},bindings["lora"],root/"models")
        if Path(bindings["model"]).exists():
            require(read_json(bindings["model"])==model.to_dict(),"Initial model reference changed")
        else:atomic_json(bindings["model"],model.to_dict())
        return {"model":bindings["model"],"initial_adapter":str(root/"models/initial-adapter")}
    require(phase=="config","Unknown control stage")
    from self_curate.experiments.exp001.config import load_config
    from self_curate.experiments.exp001.inputs import load_prepared_manifest
    source=read_json(bindings["server_template"])
    prepared=load_prepared_manifest(root/"prepared/prepared-manifest.json")
    budget=json.loads(prepared.source_lock_json)["selection_budget"]
    require(budget["selected_count"]==budget["raw_success_count"]//10,"Changed success denominator")
    settings=source["adapter_settings"]
    settings.update(prepared_manifest=str(root/"prepared/prepared-manifest.json"),
                    selection_budget=budget,prompts="configs/experiments/exp001/prompts-rebench",
                    source_registry={"path":bindings["source_registry"],"sha256":file_sha256(Path(bindings["source_registry"]))},
                    model_id=prepared.model.base_model_id,model_revision=prepared.model.revision,
                    training_source={"mode":"rebench_success","dataset":"nebius/SWE-rebench-openhands-trajectories",
                                     "revision":read_json(bindings["source_registry"])["datasets"]["rebench_trajectories"]["revision"],"split":"train"})
    from self_curate.contracts import ObjectiveSpec,SkillWeight,EditPolicy
    protocols=root/"protocols";protocols.mkdir(exist_ok=True)
    disabled=publish_json(protocols/"no-inner-evaluation.json",
                         {"enabled":False,"reason":"EXP001 has no NICE/search/acceptance"})
    objective=ObjectiveSpec("exp001-rebench-initial",tuple(SkillWeight(s,1/len(prepared.skills)) for s in prepared.skills),(),disabled,disabled)
    atomic_json(protocols/"objective.json",objective.to_dict())
    atomic_json(protocols/"edit-policy.json",EditPolicy(.1,.01,.99,4,128,128,1,600).to_dict())
    sampling=publish_json(protocols/"agent-sampling.json",{"temperature":0,"seed":source["seed"],"max_tokens":8192})
    base=str((Path(bindings["sources"])/"model/files").resolve())
    initial=str(root/"models/initial-adapter")
    require(len(bindings["gpu_uuids"])==8,"Freeze eight allocated GPU UUIDs")
    gateway=bindings.get("container_gateway")
    require(isinstance(gateway,str) and gateway,"Freeze the actual execution Docker gateway, not a guessed workstation address")
    port=bindings.get("base_port",8100)
    slots=bindings.get("inference_slots_per_gpu",2)
    require(type(slots) is int and 1<=slots<=8,"Invalid declared inference slots")
    entries=[{"base_url":"http://127.0.0.1:"+str(port+i),"port":port+i,"gpu_uuid":u,
              "slots":slots,"base_weights_hash":prepared.model.weights_hash,
              "tokenizer_hash":prepared.model.tokenizer.tokenizer_hash,"api_key_env":"SELF_CURATE_INFERENCE_KEY",
              "container_base_url":"http://"+gateway+":"+str(port+100+i),
              "native_capture_root":str(root/"services/native-tokens"/u),
              "relay_host":gateway,"relay_port":port+100+i,
              "relay_grant_root":str(root/"private/relay-grants")} for i,u in enumerate(bindings["gpu_uuids"])]
    serving=publish_json(protocols/"serving-pool.json",{"schema_version":"1","entries":entries})
    defaults={"model_store":{"base_path":base,"initial_adapter":initial,
                            "base_fingerprint":prepared.model.weights_hash,"loader":"AutoModelForImageTextToText"},
              "task_execution":{"serving_pool":serving.to_dict(),"initial_adapter":initial,
                                "tokenizer_path":base,"harness":bindings.get("harness",{}),"swe_python":bindings.get("swe_python","/opt/graders/verified/bin/python"),
                                "rebench_python":bindings.get("rebench_python","/opt/graders/rebench/bin/python"),
                                "pro_tooling":bindings.get("pro_tooling","/opt/graders/pro/v2/tooling"),
                                "harbor_executable":"harbor","harbor_installed_root":bindings["installed_environments"],
                                "harbor_gpu_slot_root":str(root/"native-gpu-slots"),"slot_root":str(root/"inference-slots")},
              "objective":str(protocols/"objective.json"),"edit_policy":str(protocols/"edit-policy.json"),
              "agent_sampling":sampling.to_dict(),"capacity_receipt":None,
              "runtime_lock":bindings["runtime_lock"],
              "services":{"runtime_root":str(root/"services"),"base_path":base,"max_context_tokens":max([prepared.development.tasks[0].budget.max_context_tokens]+[t.budget.max_context_tokens for b in prepared.benchmarks for t in b.tasks]),
                          "gpu_memory_utilization":.85,"max_num_seqs":slots,"tool_call_parser":"qwen3_coder",
                          "reasoning_parser":"qwen3","lora_rank":bindings["lora"]["rank"],
                          "startup_timeout_seconds":1800,"telemetry":True}}
    for key,value in defaults.items():
        override=bindings.get("server_bindings",{}).get(key)
        settings[key]={**value,**override} if isinstance(value,dict) and isinstance(override,dict) else (value if override is None else override)
    if settings["model_store"].get("base_fingerprint") is None:
        settings["model_store"]["base_fingerprint"]=prepared.model.weights_hash
    settings["lora"]=bindings["lora"]
    if bindings.get("terminal_tasks"):
        from self_curate.execution.harbor_resources import gpu_layout
        required=max((sum(gpu_layout(p.parent).values()) for p in Path(bindings["terminal_tasks"]).glob("*/task.toml")),default=0)
        require(required < len(bindings["gpu_uuids"]), "Native Terminal task GPUs plus target inference exceed the assigned pool")
        reserved=bindings["gpu_uuids"][-required:] if required else []
        settings["terminal_reserved_gpu_uuids"]=reserved
        settings["task_execution"]["harbor_gpu_pool"]=reserved
        require(not required or settings["task_execution"].get("harbor_gpu_slot_root"),
                "Provide the shared native-task GPU reservation directory")
    source["selection_fraction"]=0.1
    source["gpu_uuids"]=bindings["gpu_uuids"]
    source["experiment_id"]="exp001_rebench_success_random_static_v1"
    target=root/("server.resolved.json" if settings.get("capacity_receipt") else "server.preflight.json")
    candidate=root/("server-config-candidate-"+digest(source)[:16]+".json")
    atomic_json(candidate,source)
    cfg=load_config(candidate)
    prepared.validate(cfg)
    if target.exists():require(read_json(target)==source,"Frozen server config changed")
    else:atomic_json(target,source)
    return {"config":str(target),"selection_budget":budget,"research_ready":False,
            "note":"Server-specific capacity and offline receipt verification still runs before compute"}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location",choices=["workstation"],required=True)
    parser.add_argument("--bindings",required=True)
    parser.add_argument("--stage",choices=["registry","model","tasks","pool","publish","config"],required=True)
    args=parser.parse_args(argv)
    __import__("os").environ["CUDA_VISIBLE_DEVICES"]=""
    bindings=read_json(args.bindings)
    root=Path(bindings["output"]).resolve()
    with lease(root/".pipeline.lease"):
        try:
            result=control_stage(bindings,args.stage) if args.stage in ("registry","model","config") else stage(bindings,args.stage)
            atomic_json(root/(args.stage+"-complete.json"),result)
            print(json.dumps(result,ensure_ascii=False))
        except Exception as exc:
            atomic_json(root/(args.stage+"-blocked.json"),{"type":type(exc).__name__,"reason":str(exc)})
            raise
    return 0


if __name__=="__main__":raise SystemExit(main())
