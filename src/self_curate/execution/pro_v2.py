"""SWE-bench Pro V2 official locked agent + pristine patch replay.
No home-made test grading. Version-specific CLI arguments are preflighted on workstation.
"""
from pathlib import Path
import os
import json
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json,read_json
from self_curate.execution.artifacts import read_artifact
from .process import execute


def single_result(job, task_name):
    values=[]
    for file in Path(job).glob("*/result.json"):
        row=read_json(file)
        if row.get("task_name","").split("/")[-1]==task_name:
            values.append((file,row))
    require(len(values)==1,"Expected one exact Pro task result; missing/duplicate is infrastructure failure")
    file,row=values[0]
    require(row.get("exception_info") is None and row.get("reward_valid",True),"Pro environment/verifier failure")
    rewards=(row.get("verifier_result") or {}).get("rewards",{})
    require(set(rewards)=={"reward"} and type(rewards["reward"]) in (int,float) and rewards["reward"] in (0,1),"Invalid official Pro reward")
    return file,row


class ProV2Judge:
    def __init__(self, *, executable="harbor", process=execute, tooling=None):
        self.executable,self.process=executable,process
        self.tooling=Path(tooling).resolve() if tooling else None

    def run(self,request,task_path,binding,output,*,timeout):
        from contextlib import nullcontext
        from self_curate.execution.relay import trial_grant
        require(not binding.get("api_key_env") or binding.get("relay_grant_root"),
                "Pro authenticated model access requires a scoped local relay")
        grant=trial_grant(binding["relay_grant_root"],binding["model_name"],timeout=timeout*2) if binding.get("relay_grant_root") else nullcontext("local-offline-no-secret")
        with grant as scoped_key:
            return self._run(request,task_path,binding,output,timeout=timeout,scoped_key=scoped_key)

    def _run(self,request,task_path,binding,output,*,timeout,scoped_key):
        from self_curate.training.artifacts import checkpoint_fingerprint
        from self_curate.contracts.parsing import artifact
        task_path=Path(task_path).resolve()
        output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
        require(self.tooling is not None and (self.tooling/"patch_replay.py").is_file(),"Pinned official Pro V2 tooling missing")
        environment=read_artifact(request.task.environment)
        protocol=read_artifact(artifact(environment["pro_protocol_receipt"]))
        require(getattr(request,"backend","mock")!="real" or protocol.get("backend")=="real",
                "Synthetic Pro probes cannot authorize real evaluation")
        require(protocol.get("schema_version")=="pro-v2-claude-sdk-offline-v2"
                and protocol.get("offline_agent_verified") is True
                and protocol.get("offline_verifier_verified") is True
                and protocol.get("fresh_sandbox_verified") is True,
                "Workstation must verify offline Pro phases and pristine regrade")
        from .claude_sdk import policy_config
        effective=policy_config(request.task.budget,read_artifact(request.protocol),seed=request.seed,temperature=read_artifact(request.sampling)["temperature"])["generation"]
        required={k:effective[k] for k in ("temperature","top_p","top_k","thinking","preserve_thinking","reasoning_effort","min_p","presence_penalty","repetition_penalty")}
        require(protocol.get("sampling_verified") is True and protocol.get("generation_policy")==required,
                "Pro effective sampling/thinking policy differs from the engine probe receipt")
        require(protocol["tooling_checksum"]==checkpoint_fingerprint(self.tooling)[0],
                "Pro official tooling changed")
        require(protocol["task_checksums"].get(task_path.name)==checkpoint_fingerprint(task_path)[0],
                "Pro task not covered by offline receipt")
        agent_job=output/"jobs"/"agent"
        regrade_job=output/"jobs"/"regrade"
        env=dict(os.environ)
        env["PYTHONPATH"]=str(self.tooling)+os.pathsep+env.get("PYTHONPATH","")
        env["SELF_CURATE_TRIAL_KEY"]=scoped_key
        env["HF_HUB_OFFLINE"]="1"
        endpoint=binding.get("container_base_url",binding["base_url"]).rstrip("/")
        require(protocol.get("environment")=="docker","Pro execution requires verified local Docker")
        require(endpoint in protocol.get("model_endpoints",()),"Model endpoint differs from verified network policy")
        from self_curate.execution.artifacts import local_path
        network_overlay=local_path(artifact(protocol["network_overlay"]))
        prepared=protocol.get("prepared_environments",{}).get(task_path.name)
        require(isinstance(prepared,dict) and prepared.get("agent_image") and prepared.get("verifier_image")
                and prepared.get("harness_backend")=="claude_agent_sdk","Both offline Pro phase environments must be prepared")
        def trial(stage,agent):
            image=prepared["agent_image" if stage=="agent" else "verifier_image"]
            require(__import__("re").fullmatch(r"sha256:[0-9a-f]{64}",image),"Pro cached phase image must be immutable")
            overlay=output/(stage+"-image.yaml")
            overlay.write_text("services:\n  main:\n    image: "+json.dumps(image)+"\n    pull_policy: never\n    build: !reset null\n",encoding="utf-8")
            cfg={"task":{"path":str(task_path)},"trial_name":task_path.name+"__trial",
                 "trials_dir":str(agent_job if stage=="agent" else regrade_job),
                 "agent":agent,"environment":{"type":"docker","delete":False,
                     "extra_docker_compose":[str(overlay),str(network_overlay)]},"verifier":{"disable":False}}
            path=output/(stage+"-trial.json")
            atomic_json(path,cfg)
            result=self.process([self.executable,"trials","start","--config",str(path)],
                                cwd=output,output=output/(stage+"-process"),timeout=timeout,env=env)
            require(result.get("returncode")==0 and not result.get("timed_out"),"Official Pro phase failed")
        from .claude_sdk import policy_config,native_agent_seconds
        agent={"import_path":"self_curate.execution.harbor_claude:ClaudeSDKAgent",
               "model_name":binding["model_name"],"base_url":endpoint,
               "model_api_key_name":"SELF_CURATE_TRIAL_KEY",
               "kwargs":{"harness":binding.get("harness",{}),
                         "gate":policy_config(request.task.budget,read_artifact(request.protocol),seed=request.seed, temperature=read_artifact(request.sampling)["temperature"],
                             wall_seconds=native_agent_seconds(task_path,request.task.budget,read_artifact(request.protocol))),
                         "max_turns":request.task.budget.max_steps,
                         "tool_seconds":read_artifact(request.protocol)["tool_timeout_seconds"],
                         "capture_patch":True},
               "env":{"UV_OFFLINE":"1"}}
        atomic_json(output/"dispatch.json",{"request_hash":request.record_hash,
                    "task":task_path.name,"protocol":protocol,"stages":["agent","pristine_regrade"]})
        trial("agent",agent)
        agent_file,agent_result=single_result(agent_job,task_path.name)
        patch=agent_file.parent/"agent"/"model.patch"
        require(patch.is_file(),"Missing captured patch is infrastructure failure")
        trial("regrade",{"import_path":"patch_replay:PatchReplayAgent","model_name":"replay",
                          "kwargs":{"source_job":str(agent_job)}})
        final_file,result=single_result(regrade_job,task_path.name)
        replay_path=final_file.parent/"agent"/"replay.json"
        require(replay_path.is_file(),"Missing official replay receipt")
        replay_info=read_json(replay_path)
        require(not patch.read_text(encoding="utf-8").strip() or replay_info.get("apply_rc")==0,
                "Nonempty patch failed to apply in pristine sandbox")
        usage=agent_result.get("agent_result") or {}
        return {"status":"completed","reward":float(result["verifier_result"]["rewards"]["reward"]),
                "generated_tokens":usage.get("n_output_tokens"),"input_tokens":usage.get("n_input_tokens"),
                "raw":{"agent":agent_result,"authoritative_regrade":result,
                       "replay":replay_info,"protocol":"official-pro-v2-pristine"}}


def register_pro_task(task_path, *, task_id, skills, budget, runtime_lock, protocol_receipt, output):
    from dataclasses import replace
    from self_curate.execution.registration import register_harbor_task
    from self_curate.execution.artifacts import publish_json,read_artifact
    task=register_harbor_task(task_path,task_id=task_id,skills=skills,budget=budget,
                              runtime_lock=runtime_lock,output=output)
    # Use new artifacts: never overwrite already published environment bytes.
    environment={**read_artifact(task.environment),"harness":"swe_pro_v2",
                 "pro_protocol_receipt":protocol_receipt.to_dict()}
    scorer={**read_artifact(task.scorer),"harness":"swe_pro_v2"}
    return replace(task,environment=publish_json(Path(output)/"pro-environment.json",environment),
                   scorer=publish_json(Path(output)/"pro-scorer.json",scorer),
                   family_ids=("swe-pro-v2:"+task_id,))
