"""Real offline execution and an explicitly synthetic CPU backend."""
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from .common import digest, read, write, sha, event


class MockBackend:
    kind='mock'
    def initial(self,config):
        return {'identity':'synthetic-initial','synthetic':True}
    def train(self,job,config,output,*,resume):
        value={'identity':digest({'selection':job['data_sha256'],'seed':config['seed'],'learning_rate':config['training']['learning_rate']}),'synthetic':True}
        value['epoch_checkpoints']=[{'epoch':i,'global_step':i,'progress':None,'model':{'identity':digest({'final':value['identity'],'epoch':i}),'synthetic':True}} for i in range(1,config['training']['epochs']+1)]
        if config['training'].get('save_at_fractions'):
            from .epoch_checkpoints import progress_steps
            steps=progress_steps(job['count'],config['training']['global_batch_size'],config['training']['save_at_fractions'])
            value['epoch_checkpoints']=[{'epoch':s/steps[-1],'global_step':s,'progress':s/steps[-1],
                'model':{'identity':digest({'branch':value['identity'],'step':s}),'synthetic':True}} for s in steps]
        write(Path(output)/'mock-training.json',value)
        return value
    def evaluate(self,job,config,model,benchmark,output,*,resume):
        file=Path(output)/(benchmark+'-synthetic.json')
        write(file,{'benchmark':benchmark,'synthetic':True,'score':None,'model':model,'data':job['data_sha256']})
        return {'path':str(file.resolve()),'sha256':sha(file)}
    def verify_result(self,value):
        if value.get('cost_path') and sha(value['cost_path'])!=value['cost_sha256']:
            raise ValueError('Saved evaluation cost changed')
        if sha(value['path'])!=value['sha256']:
            raise ValueError('Saved evaluation changed')


class ServerBackend:
    kind='real'
    def __init__(self,bindings):
        self.bindings=read(bindings) if not isinstance(bindings,dict) else bindings
        self.processes=[];self.serving=None;self.logs=[]
        self.grader_images=self.bindings.get('grader_images',{})
        bcb_image=self.grader_images.get('bigcodebench')
        if 'bigcodebench' in self.bindings.get('benchmarks',{}) and (not bcb_image or not (bcb_image.startswith('sha256:') or '@sha256:' in bcb_image)):
            raise ValueError('BigCodeBench requires its independently built, digest-pinned grader image')
        from .workstation import verify_seal
        seal=verify_seal(self.bindings['storage'])
        if seal['seal_hash']!=self.bindings['asset_seal_hash']:
            raise ValueError('Different shared asset seal')
        if not self.bindings.get('server_smoke_passed'):
            raise ValueError('Real eight-GPU FSDP/vLLM/official-grader smoke receipt required')
        receipt=Path(self.bindings['server_smoke_receipt'])
        if sha(receipt)!=self.bindings['server_smoke_sha256']:
            raise ValueError('Server smoke receipt checksum mismatch')
        if read(receipt).get('passed') is not True:
            raise ValueError('Server smoke did not pass')
        proof=read(receipt)
        if proof.get('asset_seal_hash')!=seal['seal_hash'] or proof.get('grader_image')!=self.bindings['grader_image']:
            raise ValueError('Server smoke used different assets or grader image')
    def initial(self,config):
        value=self.bindings['models'][config['model']]
        return {'identity':value['model_hash'],'base_path':value['path'],'loader':value['loader'],'adapter':None}
    def train(self,job,config,output,*,resume):
        self.stop()
        initial=self.initial(config)
        output=Path(output)/'training'
        spec={'execution':'server','approved':config['approved'],'training':config['training'],
              'model_path':initial['base_path'],'loader':initial['loader'],'seed':config['seed'],
              'data':job['data'],'data_sha256':job['data_sha256'],'output':str(output.resolve()),
              'gpu_uuids':config['hardware']['gpu_uuids'],'base_identity':initial['identity']}
        world=spec['training'].get('world_size',8)
        if len(spec['gpu_uuids'])!=world or len(set(spec['gpu_uuids']))!=world:
            raise ValueError('Training group needs matching distinct GPU UUIDs')
        job_file=output.parent/'training-job.json'
        write(job_file,spec)
        completion=output/'complete.json'
        if completion.exists():
            saved=read(completion)
            if saved['job_hash']!=digest({k:v for k,v in spec.items() if k!='output'}):
                raise ValueError('Completed training identity changed')
        else:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(spec['gpu_uuids']),HF_HUB_OFFLINE='1',
                     TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(world),
                     '-m','self_curate.training.ordinary','--job',str(job_file)]
            if resume and any(output.glob('checkpoint-*')):
                command.append('--resume')
            log=output.parent/('train-resume.log' if resume else 'train.log')
            with log.open('a',encoding='utf-8') as stream:
                self.training_process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                try:
                    while self.training_process.poll() is None:
                        if getattr(self,'cancel_event',None) is not None and self.cancel_event.is_set():
                            self.stop();raise RuntimeError('Training lane cancelled')
                        time.sleep(.25)
                    if self.training_process.returncode!=0:raise RuntimeError('Native FSDP training failed; inspect '+str(log))
                finally:self.training_process=None
        saved=read(completion)
        adapter=Path(saved['adapter'])
        identity=digest({'base_identity':initial['identity'],
            'adapter_files':{str(p.relative_to(adapter)):sha(p) for p in sorted(adapter.rglob('*')) if p.is_file()}})
        stages=[]
        for checkpoint in saved['epoch_checkpoints']:
            path=Path(checkpoint['path'])
            checkpoint_identity=digest({'base_identity':initial['identity'],
                'adapter_files':{str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}})
            stages.append({'epoch':checkpoint['epoch'],'global_step':checkpoint['global_step'],'progress':checkpoint.get('progress'),'model':{**initial,'identity':checkpoint_identity,
                'base_identity':initial['identity'],'adapter':str(path),
                'training_cost':read(path/'training-cost.json')}})
        return {**initial,'identity':identity,'base_identity':initial['identity'],'adapter':str(adapter),
                'epoch_checkpoints':stages}
    def fork_lane(self,index):
        import copy
        lane=copy.copy(self)
        lane.processes=[];lane.logs=[];lane.serving=None;lane.serving_base=None;lane.serving_adapter=None
        lane.training_process=None;lane.port_base=8000+100*index
        return lane

    def stop(self):
        import signal
        active=list(self.processes)
        training=getattr(self,'training_process',None)
        if training is not None:active.append(training)
        for process in active:
            if process.poll() is not None:continue
            try:
                if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
            except ProcessLookupError:pass
        for process in active:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                except ProcessLookupError:pass
                process.wait()
        for log in self.logs:log.close()
        self.processes=[];self.logs=[];self.serving=None;self.serving_base=None;self.serving_adapter=None

    def endpoints(self,config):
        return [f'http://127.0.0.1:{getattr(self,"port_base",8000)+i}/v1' for i in range(config['evaluation']['replicas'])]

    def serve(self,model,config,output):
        if self.serving==model['identity']:
            return
        cfg=config['evaluation'];gpu_ids=config['hardware']['gpu_uuids']
        base=model.get('base_identity',model['identity'])
        if self.processes and getattr(self,'serving_base',None)==base and cfg.get('runtime_lora_switching'):
            import json
            for index in range(cfg['replicas']):
                url=f'http://127.0.0.1:{getattr(self,"port_base",8000)+index}'
                if getattr(self,'serving_adapter',None):
                    body=json.dumps({'lora_name':'selected'}).encode()
                    req=urllib.request.Request(url+'/v1/unload_lora_adapter',data=body,headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=120):pass
                if model['adapter']:
                    body=json.dumps({'lora_name':'selected','lora_path':model['adapter']}).encode()
                    req=urllib.request.Request(url+'/v1/load_lora_adapter',data=body,headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=120):pass
            self.serving=model['identity'];self.serving_adapter=model['adapter']
            return
        self.stop()
        cfg=config['evaluation']; gpu_ids=config['hardware']['gpu_uuids']
        if cfg['replicas'] not in (4,8) or len(gpu_ids)!=cfg['replicas'] or cfg['tensor_parallel_size']!=1:
            raise ValueError('Ordinary 9B serving uses eight single-GPU replicas')
        parser='qwen3_coder' if config['model'].startswith('qwen') else 'llama3_json'
        for index,gpu in enumerate(gpu_ids):
            log=(Path(output)/f'vllm-{index}.log').open('a',encoding='utf-8');self.logs.append(log)
            command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',model['base_path'],
                     '--served-model-name','target-qwen9b','--port',str(getattr(self,'port_base',8000)+index),'--host','127.0.0.1',
                     '--tensor-parallel-size','1','--max-model-len',str(cfg['max_model_len']),
                     '--max-num-seqs',str(cfg['max_num_seqs_per_replica']),
                     '--gpu-memory-utilization',str(cfg['gpu_memory_utilization']),
                     '--enable-auto-tool-choice','--tool-call-parser',parser,
                     '--max-num-batched-tokens',str(cfg.get('max_num_batched_tokens',8192)),
                     '--enable-lora','--max-lora-rank',str(cfg.get('max_lora_rank',64))]
            if cfg.get('enable_chunked_prefill',True):command+=['--enable-chunked-prefill']
            if model['adapter']:
                command+=['--lora-modules','selected='+model['adapter']]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ALLOW_RUNTIME_LORA_UPDATING='True')
            self.processes.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt'))
        deadline=time.monotonic()+900
        remaining=set(range(cfg['replicas']))
        while remaining and time.monotonic()<deadline:
            if getattr(self,'cancel_event',None) is not None and self.cancel_event.is_set():
                self.stop();raise RuntimeError('Serving lane cancelled')
            for index in tuple(remaining):
                if self.processes[index].poll() is not None:
                    self.stop();raise RuntimeError('vLLM startup failed; inspect per-replica logs')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{getattr(self,'port_base',8000)+index}/health',timeout=2):
                        remaining.remove(index)
                except OSError:pass
            time.sleep(1)
        if remaining:
            self.stop();raise TimeoutError('vLLM replica startup timeout')
        self.serving=model['identity'];self.serving_base=base;self.serving_adapter=model['adapter']
    def evaluate(self,job,config,model,benchmark,output,*,resume):
        return self.finish_evaluation(self.prepare_evaluation(job,config,model,benchmark,output,resume=resume))

    def generation_tokenizer(self,model):
        from transformers import AutoTokenizer
        if getattr(self,'length_tokenizer_path',None)!=model['base_path']:
            self.length_tokenizer=AutoTokenizer.from_pretrained(model['base_path'],local_files_only=True)
            self.length_tokenizer_path=model['base_path']
        return self.length_tokenizer

    def prepare_evaluation(self,job,config,model,benchmark,output,*,resume):
        from .evaluation import generate
        output=Path(output)/benchmark;output.mkdir(parents=True,exist_ok=True)
        evaluation_started=time.monotonic()
        dataset=self.bindings['benchmarks'][benchmark]
        tasks=Path(dataset['path'])
        if sha(tasks)!=dataset['sha256']:
            raise ValueError('Frozen benchmark changed')
        if dataset.get('evalplus_data') and sha(dataset['evalplus_data'])!=dataset['evalplus_sha256']:
            raise ValueError('Complete EvalPlus enhanced tests changed')
        self.serve(model,config,output)
        generation=generate(tasks,output/'generation',endpoints=self.endpoints(config),
                 model_id='selected' if model['adapter'] else 'target-qwen9b',model_hash=model['identity'],protocol=config['evaluation'],tokenizer=self.generation_tokenizer(model))
        # Grade in a distinct CPU-only container with no network, credentials, home, or GPU mounts.
        project=Path(self.bindings['project']).resolve(); assets=Path(self.bindings['storage']).resolve()
        results=Path(self.bindings['run_root']).resolve()
        def inside(path):
            path=Path(path).resolve()
            for root,prefix in ((assets,'/assets'),(results,'/results'),(project,'/app')):
                try:return prefix+'/'+path.relative_to(root).as_posix()
                except ValueError:pass
            raise ValueError('Artifact is outside explicit container mounts: '+str(path))
        grading=output/'grading';grading.mkdir(exist_ok=True)
        result=grading/'scores.json'
        worker='/opt/olympiad/bin/python' if benchmark=='olympiad' else 'python'
        command=['docker','run','--rm','--network','none','--read-only','--pids-limit','1024',
                 '--memory',str(config['runtime'].get('pipeline',{}).get('grader_memory_gb',48))+'g',
                 '--cpus',str(config['runtime'].get('pipeline',{}).get('grader_cpus_per_job',8)),'--tmpfs','/tmp:rw,size=8g','--tmpfs','/root/.cache:rw,size=2g',
                 '-e','SELF_CURATE_GRADER_SANDBOX=1','-e','HF_HUB_OFFLINE=1','-e','PYTHONPATH=/app/src',
                 '--mount',f'type=bind,src={project},dst=/app,readonly',
                 '--mount',f'type=bind,src={assets},dst=/assets,readonly',
                 '--mount',f'type=bind,src={results},dst=/results,readonly',
                 '--mount',f'type=bind,src={grading.resolve()},dst={inside(grading)}',self.grader_images.get(benchmark) or self.bindings['grader_image'],worker,
                 '-m','self_curate.experiments.ordinary.grade','--benchmark',benchmark,'--tasks',inside(tasks),
                 '--predictions',inside(generation/'predictions'),'--output',inside(result)]
        if dataset.get('grader_root'):command+=['--grader-root',inside(dataset['grader_root'])]
        if dataset.get('evalplus_data'):command+=['--evalplus-data',inside(dataset['evalplus_data'])]
        return {'command':command,'output':str(output),'result':str(result),'generation':str(generation),
                'model_identity':model['identity'],'evaluation_started':evaluation_started}

    def finish_evaluation(self,ticket):
        output=Path(ticket['output']);result=Path(ticket['result']);generation=Path(ticket['generation'])
        command=ticket['command'];evaluation_started=ticket['evaluation_started']
        grading_started=time.monotonic()
        from .grader_process import run as run_grader
        run_grader(command,output/'grader.log',cancel=getattr(self,'cancel_event',None))
        usage=read(generation/'generation.json')
        cost=output/'evaluation-cost.json'
        write(cost,{'schema':'evaluation-cost-v1','synthetic':False,'model_identity':ticket['model_identity'],
            'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens'],
            'generation_wall_seconds':usage['wall_seconds'],'summed_request_seconds':usage['summed_request_seconds'],
            'grading_wall_seconds':time.monotonic()-grading_started,
            'evaluation_wall_seconds':time.monotonic()-evaluation_started,'monetary_cost':None,
            'generation_receipt':str((generation/'generation.json').resolve()),
            'generation_receipt_sha256':sha(generation/'generation.json'),
            'wall_scope':'this evaluation attempt including serving startup when required; cached work not charged again'})
        return {'path':str(result.resolve()),'sha256':sha(result),'cost_path':str(cost.resolve()),'cost_sha256':sha(cost)}
    verify_result=MockBackend.verify_result

    def verify_model(self,model):
        if model['adapter']:
            root=Path(model['adapter'])
            if model['base_identity'] not in {v['model_hash'] for v in self.bindings['models'].values()}:
                raise ValueError('Adapter base identity changed')
            observed=digest({'base_identity':model['base_identity'],
                'adapter_files':{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}})
            if observed!=model['identity']:raise ValueError('Saved model adapter changed')
        else:
            if model['identity'] not in {v['model_hash'] for v in self.bindings['models'].values()}:
                raise ValueError('Saved initial model changed')
