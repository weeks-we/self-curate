"""Claude SDK SWE solving; official grading remains a separate fresh environment."""
from pathlib import Path
from contextlib import nullcontext
import io,json,os,shlex,tarfile
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json,read_json
from .claude_sdk import settings,policy_config
from .relay import trial_grant

def archive(name, data, uid=1000, gid=1000):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as tar:
        item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o600;item.uid=uid;item.gid=gid
        tar.addfile(item,io.BytesIO(data))
    return stream.getvalue()

def run_claude(client,tokenizer,request,environment,policy,output,*,docker_factory=None):
    output=Path(output).resolve()
    if output.exists():
        from .sealed import load_sealed
        row=load_sealed(output,request.record_hash)
        client.calls=[{'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens']} for r in row['calls']]
        return row
    output.mkdir(parents=True,exist_ok=False)
    cfg=settings(client.binding.get('harness'))
    binding=client.binding
    require(environment['image'].find('@sha256:')>0,'Pin original grader image')
    image_map=cfg.get('task_images',{})
    image=image_map.get(environment['image'])
    require(isinstance(image,str) and '@sha256:' in image,'Prepared SDK task image mapping required')
    require(cfg.get('network') and cfg['network'] not in ('host','bridge','none'),
            'Dedicated restricted task network required')
    require(cfg.get('network_policy_verified') is True,'Task network egress policy must be smoke verified')
    require(binding.get('relay_grant_root') and binding.get('container_base_url'),
            'Task needs scoped local inference relay')
    if docker_factory is None:
        import docker
        docker_factory=docker.from_env
    docker=docker_factory();container=None
    if os.name!='nt':
        uid,gid=(int(v) for v in cfg.get('task_user','1000:1000').split(':'))
        require(uid!=0,'Task SDK must run as a non-root user')
        os.chown(output,uid,gid)
    try:
        docker.images.get(image) # No pull/build at execution time.
        container=docker.containers.run(image,['sleep','infinity'],detach=True,
            network=cfg['network'],working_dir=environment['cwd'],
            user=cfg.get('task_user','1000:1000'),
            volumes={str(output):{'bind':'/sc-output','mode':'rw'}},
            cap_drop=['ALL'],security_opt=['no-new-privileges:true'],
            labels={'self-curate-request':request.record_hash})
        setup=container.exec_run(['git','reset','--hard',environment['checkout']])
        require(setup.exit_code==0,'Task repository checkout failed')
        with trial_grant(binding['relay_grant_root'],binding['model_name'],
                         timeout=request.task.budget.max_wall_seconds) as key:
            worker={'prompt':environment['problem_statement'],'binding':{
                'model_name':binding['model_name'],'base_url':binding['container_base_url']},
                'key':key,'harness':cfg,'output':'/sc-output','cwd':environment['cwd'],
                'max_turns':request.task.budget.max_steps,
                'tool_seconds':policy['tool_timeout_seconds'],
                'gate':policy_config(request.task.budget,policy,seed=request.seed)}
            require(container.put_archive('/tmp',archive('self-curate-sdk.json',json.dumps(worker).encode(),*(int(v) for v in cfg.get('task_user','1000:1000').split(':')))),
                    'Could not transfer private SDK config')
            # Docker exec itself has no total deadline: host watchdog stops container.
            import threading
            timer=threading.Timer(request.task.budget.max_wall_seconds+30,
                                  lambda: container.stop(timeout=5))
            timer.start()
            try:
                receipt=container.exec_run([cfg['python'],'-m','self_curate.execution.claude_worker',
                                           '--input','/tmp/self-curate-sdk.json'])
            finally:timer.cancel()
        atomic_json(output/'worker-exit.json',{'returncode':receipt.exit_code})
        require((output/'worker-result.json').is_file(),'SDK worker incomplete; preserve logs and reconcile usage')
        row=read_json(output/'worker-result.json')
        diff=container.exec_run(['bash','-lc','git add -N . && git diff --binary'])
        require(diff.exit_code==0,'Could not extract final patch')
        row.update(patch=diff.output.decode('utf-8'),error=None)
        from .artifacts import publish_json
        from .vllm_capture import capture_path
        native=[]
        capture_root=binding.get('native_capture_root')
        if capture_root:
            for call in row['calls']:
                path=capture_path(capture_root,'chatcmpl-'+call['capture_id'])
                require(path.is_file(),'Engine native capture missing; serving receipt not compatible')
                item=read_json(path)
                require(item.get('model_alias')==binding['model_name'] and item.get('complete') is True,
                        'Wrong native adapter capture')
                require(len(item['generated_token_ids'])==call['output_tokens'],
                        'Native output count differs from SDK gate')
                native.append(item)
            row['native_capture']=publish_json(output/'native-capture.json',{
                'model_ref':request.model_ref.to_dict(),'requests':native}).to_dict()
        row['tokenization_hash']=digest({'model':request.model_ref.to_dict(),
            'native_token_capture':bool(capture_root),'capture':row.get('native_capture'),'requests':row['calls']})
        row['action_mask_hash']=digest({'format':'claude-sdk-semantic-v1','actions':row['actions'],
                                       'native_generated_token_mask_available':bool(capture_root),'capture':row.get('native_capture')})
        client.calls=[{'input_tokens':r['input_tokens'],'output_tokens':r['output_tokens']} for r in row['calls']]
        atomic_json(output/'patch.json',{'patch':row['patch'],'exit_status':row['exit_status']})
        from .sealed import seal
        seal(output, request.record_hash, row)
        return row
    finally:
        # Cleanup is independent evidence: it must never erase a sealed answer.
        failures=[]
        if container is not None:
            try: container.remove(force=True)
            except Exception as exc:
                if getattr(exc, 'status_code', None) != 404:
                    failures.append({'resource':getattr(container,'id',None),
                        'type':type(exc).__name__,'message':str(exc)})
        try: docker.close()
        except Exception as exc:
            failures.append({'resource':'docker-client','type':type(exc).__name__,
                             'message':str(exc)})
        atomic_json(output/'cleanup.json',{'status':'pending' if failures else 'ok',
                                         'failures':failures})
