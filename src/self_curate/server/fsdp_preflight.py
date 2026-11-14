"""FSDP-specific interrupt/resume acceptance, isolated from experiment branches."""
from pathlib import Path
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json,read_json
from self_curate.training.fsdp import AccelerateFSDPBackend,stages_from_disk,settings
from self_curate.contracts import digest


def inspect_request(request, config=None):
    spec=settings(config or {})
    require(request.recipe.accumulation_steps==8 and request.recipe.epochs==1,
            "FSDP acceptance uses the frozen global batch8, one epoch")
    require(request.training_id.startswith("preflight-"),"Use an isolated preflight request, never a research branch")
    return {"schema_version":"fsdp-acceptance-v1","backend":"cpu_contract",
            "passed":False,"fsdp_resume_passed":False,"gpu_capacity_verified":False,
            "request_hash":request.record_hash,"settings":spec,
            "max_input_tokens":max(len(e.input_ids) for e in request.data.examples)}


def verify_resume(request,store,gpu_uuids,output,*,config=None):
    import torch
    from safetensors.torch import load_file
    from urllib.parse import urlsplit
    from urllib.request import url2pathname
    inspect_request(request,config)
    require(request.backend=="real" and torch.cuda.is_available(),"Real FSDP acceptance requires allocated server GPUs")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    cfg={**(config or {}),"save_every_steps":1}
    reference=AccelerateFSDPBackend(store,gpu_uuids=gpu_uuids,config=cfg)
    interrupted=AccelerateFSDPBackend(store,gpu_uuids=gpu_uuids,config=cfg,preflight_stop_after_steps=1)
    require(len(request.data.examples)>=24,"FSDP preflight needs three full global batches")
    reference.train(request,root/"reference",emit=lambda stage:None,resume=(root/"reference/job.json").exists())
    if not (root/"resumed/complete/COMMITTED.json").exists():
        if not (root/"resumed/latest.json").exists():
            try:interrupted.train(request,root/"resumed",emit=lambda stage:None)
            except ValueError:pass
            else:raise RuntimeError("Explicit interruption was not exercised")
        require((root/"resumed/latest.json").exists(),"No committed checkpoint after forced interruption")
        interrupted.train(request,root/"resumed",emit=lambda stage:None,resume=True)
    a=stages_from_disk(root/"reference",request);b=stages_from_disk(root/"resumed",request)
    require(len(a)==len(b)==3,"Missing preflight stages")
    for left,right in zip(a,b):
        require((left.examples_seen,left.optimizer_steps,left.supervised_tokens)==
                (right.examples_seen,right.optimizer_steps,right.supervised_tokens),"Recovery changed exposure or steps")
    def weights(checkpoint):
        path=Path(url2pathname(urlsplit(checkpoint.uri).path))
        files=list(path.rglob("adapter_model.safetensors"))
        require(len(files)==1,"Expected one exported PEFT adapter")
        return load_file(str(files[0]))
    x,y=weights(a[-1].checkpoint),weights(b[-1].checkpoint)
    require(set(x)==set(y),"Recovery adapter tensor keys changed")
    require(all(torch.allclose(x[k],y[k],rtol=1e-4,atol=1e-7) for k in x),"Resumed adapter differs from continuous control")
    result={"schema_version":"fsdp-acceptance-v1","backend":"real","passed":True,
            "fsdp_resume_passed":True,"model_hash":request.starting_model.model_hash,"gpu_uuids":list(gpu_uuids),"request_hash":request.record_hash,
            "settings":settings(cfg),"max_input_tokens":max(len(e.input_ids) for e in request.data.examples),
            "gpu_capacity_verified":False,"note":"Separate full-context capacity and inference-reload checks still required"}
    atomic_json(root/"acceptance.json",result)
    return result


def main(argv=None):
    import argparse
    from self_curate.orchestration.io import parse_outer_contract
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",required=True)
    parser.add_argument("--config")
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=["inspect","server"],default="inspect")
    args=parser.parse_args(argv)
    request=parse_outer_contract("TrainingRequest",read_json(args.request))
    if args.mode=="inspect":
        result=inspect_request(request)
        atomic_json(Path(args.output)/"cpu-contract.json",result)
    else:
        require(args.config is not None,"Explicit resolved server config required")
        from self_curate.experiments.exp001.config import load_config
        from self_curate.experiments.exp001.server import build
        cfg=load_config(args.config)
        require(cfg.execution=="server" and cfg.backend=="real","No real preflight on the workstation")
        ports=build(cfg,Path(args.output)/"ports",Path(args.output)/"unused-assets")
        try:
            ports.enter_phase("training")
            result=verify_resume(request,ports.store,cfg.gpu_uuids,args.output,config=__import__("json").loads(cfg.adapter_settings_json).get("fsdp"))
        finally:ports.close()
    print(__import__("json").dumps(result))
    return 0


if __name__=="__main__":raise SystemExit(main())
