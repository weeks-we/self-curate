"""EXP001 NICE control: 20x4 reference rollouts, 3000 gradients, four SFT branches."""
import copy
import csv
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from .common import read,write,rows,jsonl,sha,digest,run_id,event
from self_curate.training.recovery import lease


def learning_rates(main):
    values=main['training'].get('learning_rates')
    if not isinstance(values,list) or len(values)!=2 or len(set(values))!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('NICE needs two explicit shared exp001.training.learning_rates')
    return values


def config_for(main):
    path=Path(main['nice'].get('config','configs/ordinary/exp001_nice.json'))
    if not path.is_absolute():path=Path(__file__).parents[4]/path
    return read(path)


def as_task(record):
    return {'id':record['id'],'benchmark':'taco_dev','question_hash':record['question_hash'],
            'prompt':record['messages'][0]['content'],'turns':None,'reference':None,
            'raw':{'task_id':record['id'],'input_output':record['metadata']['tests']},'record':record}


def prepare(main,pool,output,*,synthetic=False):
    from .workflow import code_hash
    if main.get('model')!='qwen9b':raise ValueError('NICE requires the main Qwen3.5-9B model')
    cfg=config_for(main);rates=learning_rates(main)
    from .configuration import nice as validate_nice_config
    validate_nice_config(main,cfg)
    if cfg['schema']!='ordinary-nice-v1' or cfg['tasks']!=20 or cfg['rollouts_per_task']!=4 or cfg['candidates']!=3000 or cfg['select_per_group']!=1000:
        raise ValueError('NICE frozen 20/4/3000/1000 design mismatch')
    if cfg['epochs']!=1 or cfg['model']!='qwen9b':raise ValueError('NICE requires Qwen9B and one epoch')
    root=Path(output)
    if root.exists():raise FileExistsError('Create a fresh NICE plan')
    pool=Path(pool);manifest=read(pool/'manifest.json')
    if sha(pool/'pool.jsonl')!=manifest['pool_sha256']:raise ValueError('NICE source pool changed')
    if not synthetic and (manifest['preview'] or not manifest['tokenized']):raise ValueError('Real NICE needs fully tokenized non-preview pool')
    records=list(rows(pool/'pool.jsonl'))
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate pool IDs')
    if not synthetic and any(r.get('domain')!='coding' or r.get('source',{}).get('split')!='train' for r in records):
        raise ValueError('NICE may only draw TACO train tasks')
    records=sorted(records,key=lambda r:r['id']);random.Random(cfg['seed']).shuffle(records)
    tasks=records[:20]
    ids={r['id'] for r in tasks};questions={r['question_hash'] for r in tasks}
    families={r.get('metadata',{}).get('url') for r in tasks}-{None,''}
    remaining=[r for r in records[20:] if r['id'] not in ids and r['question_hash'] not in questions
               and (not r.get('metadata',{}).get('url') or r['metadata']['url'] not in families)]
    candidates=remaining[:3000]
    if len(tasks)!=20 or len(candidates)!=3000:raise ValueError('Need at least 3020 isolated train tasks')
    root.mkdir(parents=True)
    jsonl(root/'tasks.jsonl',(as_task(r) for r in tasks));jsonl(root/'candidates.jsonl',candidates)
    plan={'schema':'ordinary-nice-plan-v1','main':main,'nice':cfg,'learning_rates':rates,'synthetic':synthetic,
          'tasks':str((root/'tasks.jsonl').resolve()),'tasks_sha256':sha(root/'tasks.jsonl'),
          'candidates':str((root/'candidates.jsonl').resolve()),'candidates_sha256':sha(root/'candidates.jsonl'),
          'pool_sha256':manifest['pool_sha256'],'source_code_hash':code_hash(),'run_id':run_id()}
    plan['plan_hash']=digest(plan);write(root/'plan.json',plan)
    return plan


def _check(plan,backend):
    from .workflow import code_hash
    if plan['plan_hash']!=digest({k:v for k,v in plan.items() if k!='plan_hash'}):raise ValueError('NICE plan changed')
    if plan['source_code_hash']!=code_hash():raise ValueError('NICE code changed after freeze')
    for key in ('tasks','candidates'):
        if sha(plan[key])!=plan[key+'_sha256']:raise ValueError('NICE data changed')
    if backend.kind=='real' and plan['synthetic']:raise ValueError('Synthetic NICE cannot run on real backend')


def worker_stage(plan,backend,records,output,mode,*,probe=False):
    import numpy as np
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if backend.kind=='mock':
        result=[];d=plan['nice']['score']['projection_dimension']
        for record in records:
            if mode=='rollout':
                value={'task_id':record['id'],'parent_task_id':record['parent_task_id'],'id':record['id'],
                    'answers':['synthetic answer'],'model_hash':'synthetic-initial','tokenizer_identity':'synthetic',
                    'initial_adapter_hash':'synthetic','input_ids':[1,2,3],'labels':[-100,2,3],
                    'prompt_token_ids':[1],'generated_token_ids':[2,3],'proxy_test':False,'synthetic':True}
            else:
                rng=np.random.default_rng(int(digest(record['id'])[:8],16));v=rng.normal(size=d)
                file=root/(digest(record['id'])+'.npz');np.savez_compressed(file,raw=v*record.get('weight',1),adam=v)
                value={'id':record['id'],'parent_task_id':record.get('parent_task_id'),'kind':record['kind'],
                    'vectors':str(file.resolve()),'vectors_sha256':sha(file),'grad_norm':float(np.linalg.norm(v)),
                    'coordinates':'synthetic','initial_adapter_hash':'synthetic','synthetic':True}
            result.append(value)
        return result
    backend.stop();main=plan['main'];initial=backend.initial(main)
    uuids=main['hardware']['gpu_uuids']
    if len(uuids)!=8 or len(set(uuids))!=8 or not all(u.startswith('GPU-') for u in uuids):raise ValueError('NICE needs eight authorized GPU UUIDs')
    children=[];streams=[]
    try:
        for rank,gpu in enumerate(uuids[:min(8,len(records))]):
            shard=records[rank::min(8,len(records))]
            file=root/f'shard-{rank}.jsonl';jsonl(file,shard)
            job={'execution':'server','approved':True if probe else main['approved'],
                'training':{**main['training'],**plan['nice']['training_overrides']},
                'rollout':{**plan['nice']['rollout'],'max_new_tokens':plan['nice']['gradient_smoke']['max_new_tokens'] if probe else plan['nice']['rollout']['max_new_tokens']},
                'score':plan['nice']['score'],'seed':plan['nice']['seed'],'mode':mode,'gpu_uuid':gpu,
                'model_path':initial['base_path'],'model_hash':initial['identity'],'loader':initial['loader'],
                'records':str(file.resolve()),'records_sha256':sha(file),'source_code_hash':plan['source_code_hash'],
                'output':str((root/f'rank-{rank}').resolve()),'gradient_probe_only':probe}
            job_file=root/f'job-{rank}.json';write(job_file,job)
            stream=(root/f'rank-{rank}-{run_id()}.log').open('a',encoding='utf-8');streams.append(stream)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):env.pop(key,None)
            children.append(subprocess.Popen([sys.executable,'-m','self_curate.experiments.ordinary.nice_worker','--job',str(job_file)],env=env,stdout=stream,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('NICE shard failed; see rank logs')
            time.sleep(.5)
        if any(p.returncode for p in children):raise RuntimeError('NICE worker failed')
        complete=[read(root/f'rank-{i}/complete.json') for i in range(len(children))]
        if len({v['parameter_coordinates'] for v in complete})!=1 or len({v['initial_adapter_hash'] for v in complete})!=1:
            raise ValueError('NICE shards used different coordinates/initial adapters')
        values=[r for v in complete for r in v['records']]
        write(root/'complete.json',{'records':values,'shards':complete,'mode':mode,'probe':probe})
        return values
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()


def grade_records(backend,tasks,records,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if backend.kind=='mock':return [{'id':r['id'],'correct':i%2==0} for i,r in enumerate(records)]
    from .nice_sandbox import grade
    task_file=root/'tasks.jsonl';jsonl(task_file,tasks)
    for record in records:write(root/'predictions'/(digest(record['id'])+'.json'),record)
    result=grade(backend,task_file,root/'predictions',root/'grading')
    return read(result['path'])['per_task']


def score_and_select(plan,gradient_records,output):
    import numpy as np
    from self_curate.valuation.nice import normalize
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    tasks=list(rows(plan['tasks']));candidates=list(rows(plan['candidates']))
    targets={t['id']:np.zeros(plan['nice']['score']['projection_dimension']) for t in tasks}
    by_id={}
    for record in gradient_records:
        if sha(record['vectors'])!=record['vectors_sha256']:raise ValueError('NICE projection changed')
        with np.load(record['vectors']) as data:
            if record['kind']=='target':targets[record['parent_task_id']]+=data['raw']
            else:
                if record['id'] in by_id:raise ValueError('Duplicate candidate gradient')
                by_id[record['id']]=(data['raw'].copy(),data['adam'].copy())
    if set(by_id)!={r['id'] for r in candidates}:raise ValueError('Incomplete candidate gradients')
    task_matrix=np.asarray([normalize(v) for v in targets.values()])
    direction=task_matrix.mean(axis=0)  # NEVER renormalize this mean; zero tasks retain fixed denominator.
    scores=[]
    for record in candidates:
        raw,adam=by_id[record['id']]
        scores.append({'id':record['id'],'score':float(normalize(adam)@direction),
                       'raw_score':float(normalize(raw)@direction)})
    scores.sort(key=lambda r:(r['score'],r['id']))
    if not all(math.isfinite(r['score']) for r in scores):raise ValueError('Nonfinite NICE score')
    if not any(np.linalg.norm(v)>1e-12 for v in targets.values()):
        raise ValueError('All 20 NICE target directions are zero; training would not test a ranking')
    count=plan['nice']['select_per_group'];members={'low':[r['id'] for r in scores[:count]],'high':[r['id'] for r in scores[-count:]]}
    if set(members['low'])&set(members['high']):raise ValueError('High/low overlap')
    mapping={r['id']:r for r in candidates};files={}
    for group,ids in members.items():
        random.Random(plan['nice']['seed']).shuffle(ids)
        path=root/(group+'.jsonl');jsonl(path,(mapping[i] for i in ids))
        files[group]={'data':str(path.resolve()),'data_sha256':sha(path),'ids':ids,'count':len(ids)}
    write(root/'scores.json',{'scores':scores,'target_norms':{k:float(np.linalg.norm(v)) for k,v in targets.items()},
          'fixed_task_denominator':len(tasks),'zero_tasks':int(sum(np.linalg.norm(v)<=1e-12 for v in targets.values())),'synthetic':plan['synthetic']})
    write(root/'selections.json',files)
    return files


def probe_identity(plan,backend):
    settings={**plan['main']['training'],**plan['nice']['training_overrides']}
    keys=('lora_rank','lora_alpha','lora_dropout','lora_target_modules','max_length','gradient_checkpointing')
    return digest({'model':backend.initial(plan['main'])['identity'],'pool':plan['pool_sha256'],
                   'training':{k:settings[k] for k in keys},'score':plan['nice']['score'],
                   'seed':plan['nice']['seed'],'source':plan['source_code_hash']})


def validate_gradient_receipt(plan,backend):
    if backend.kind=='mock':return
    bindings=backend.bindings
    path=bindings.get('nice_gradient_smoke_receipt')
    if not path or sha(path)!=bindings.get('nice_gradient_smoke_sha256'):
        raise ValueError('NICE requires a local-environment gradient-smoke receipt with matching checksum')
    receipt=read(path)
    if not receipt.get('passed') or receipt.get('synthetic') or receipt.get('probe_identity')!=probe_identity(plan,backend):
        raise ValueError('NICE gradient-smoke did not verify these model/data/LoRA/source identities')
    if receipt.get('asset_seal_hash')!=bindings['asset_seal_hash']:
        raise ValueError('NICE gradient-smoke used different assets')


def branch_config(plan):
    base=copy.deepcopy(plan['main'])
    base['training'].update(plan['nice']['training_overrides'])
    # These are optimizer-progress nodes of ONE epoch, not main experiment epoch3.
    base['evaluation'].pop('epochs',None);base['evaluation'].pop('primary_epoch',None)
    base['evaluation'].update(checkpoint='all_progress_after_one_epoch',
        progress_fractions=plan['nice']['checkpoint_fractions'],primary_progress=1.0)
    return base


def execute(plan_file,backend,output,*,resume=False,gradient_smoke=False):
    from .epoch_checkpoints import progress_steps
    plan=read(plan_file);_check(plan,backend)
    if backend.kind=='real' and not gradient_smoke and not plan['main']['approved']:raise ValueError('Approve formal experiment parameters first')
    if not gradient_smoke:validate_gradient_receipt(plan,backend)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    with lease(root/'.lease'):
        identity={'plan_hash':plan['plan_hash'],'backend':backend.kind,'gradient_smoke':gradient_smoke}
        if (root/'identity.json').exists():
            if read(root/'identity.json')!=identity:raise ValueError('NICE resume identity mismatch')
            if not resume:raise ValueError('NICE output exists; use --resume')
        write(root/'identity.json',identity);write(root/'frozen-plan.json',plan)
        state=read(root/'state.json') if (root/'state.json').exists() else {'branches':{},'phase':'prepared'}
        if (root/'complete.json').exists():
            if read(root/'complete.json')['plan_hash']!=plan['plan_hash']:raise ValueError('NICE completion identity changed')
            for saved in state['branches'].values():
                if hasattr(backend,'verify_model'):
                    for stage in saved['model']['epoch_checkpoints']:backend.verify_model(stage['model'])
                for result in saved['evaluations'].values():backend.verify_result(result)
            for selection in state['selections'].values():
                if sha(selection['data'])!=selection['data_sha256']:raise ValueError('NICE subset changed')
            return root
        def phase(value):state['phase']=value;write(root/'state.json',state);event(root,'nice_phase',phase=value)
        tasks=list(rows(plan['tasks']));candidates=list(rows(plan['candidates']))
        if gradient_smoke:
            tasks=tasks[:plan['nice']['gradient_smoke']['tasks']]
            candidates=sorted(candidates,key=lambda r:r.get('length',0),reverse=True)[:plan['nice']['gradient_smoke']['candidates']]
        expanded=[]
        for task in tasks:
            for sample in range(plan['nice']['gradient_smoke']['rollouts_per_task'] if gradient_smoke else plan['nice']['rollouts_per_task']):
                expanded.append({**task,'id':task['id']+':r'+str(sample),'parent_task_id':task['id'],
                    'seed':plan['nice']['seed']+int(digest([task['id'],sample])[:8],16)})
        if not gradient_smoke and not state.get('canonical_verified'):
            phase('canonical_grader_preflight')
            references=[{'id':t['id'],'task_id':t['id'],'answers':[t['record']['messages'][1]['content']],
                         'proxy_test':False,'synthetic':plan['synthetic']} for t in tasks]
            graded=grade_records(backend,tasks,references,root/'canonical')
            # Mock deliberately alternates rewards; canonical control must always pass in real runs.
            if backend.kind=='real' and not all(r['correct'] for r in graded):raise ValueError('Frozen NICE tasks failed canonical grader preflight; do not replace by model scores')
            state['canonical_verified']=True;write(root/'state.json',state)
        phase('rollouts')
        rollouts=worker_stage(plan,backend,expanded,root/'rollouts','rollout',probe=gradient_smoke)
        if len(rollouts)!=len(expanded):raise ValueError('Missing or duplicate NICE rollouts')
        rewards={r['id']:1 for r in rollouts} if gradient_smoke else {r['id']:int(r['correct']) for r in grade_records(backend,expanded,rollouts,root/'rollout-grading')}
        if set(rewards)!={r['id'] for r in expanded}:raise ValueError('Incomplete fixed rollout rewards')
        gradients=[{'id':r['id'],'kind':'target','parent_task_id':r['parent_task_id'],'replay':r,
                    'weight':rewards[r['id']]/(1 if gradient_smoke else 4)} for r in rollouts if rewards[r['id']]]
        gradients += [{'id':r['id'],'kind':'candidate','record':r} for r in candidates]
        phase('gradients')
        vectors=worker_stage(plan,backend,gradients,root/'gradients','gradient',probe=gradient_smoke)
        if gradient_smoke:
            if any(not math.isfinite(v['grad_norm']) or v['grad_norm']<=0 for v in vectors):raise ValueError('Backward probe has invalid/zero gradients')
            receipt={'passed':True,'mode':'gradient_smoke_only','tasks':len(tasks),'candidates':len(candidates),
                'unit_reward_is_not_success':True,'synthetic':plan['synthetic'],'evaluated':False,'trained':False,
                'probe_identity':probe_identity(plan,backend),'asset_seal_hash':getattr(backend,'bindings',{}).get('asset_seal_hash'),
                'note':'Real HF token replay sum-NLL and candidate mean-CE backward; no benchmark/selection/training'}
            write(root/'gradient-acceptance.json',receipt);phase('gradient_smoke_complete');return root
        if 'selections' not in state:
            phase('rank_and_select');state['selections']=score_and_select(plan,vectors,root/'selection');write(root/'state.json',state)
        base=branch_config(plan)
        checkpoint_steps=progress_steps(1000,base['training']['global_batch_size'],base['training']['save_at_fractions'])
        if base['runtime'].get('pipeline',{}).get('enabled'):
            from .parallel_run import nice_jobs
            phase('parallel_nice_branches')
            nice_jobs(plan,base,backend,root,state,root/'state.json',resume)
        else:
            for group,selection in state['selections'].items():
                if sha(selection['data'])!=selection['data_sha256']:raise ValueError('Frozen NICE subset changed')
                for lr in plan['learning_rates']:
                    key=group+'-lr'+format(lr,'.8g');cfg=copy.deepcopy(base);cfg['training']['learning_rate']=lr
                    job={**selection,'job_id':'nice-'+key,'domain':'coding','method':group,'benchmarks':plan['nice']['evaluation']['benchmarks']}
                    branch=root/'branches'/key;branch.mkdir(parents=True,exist_ok=True)
                    saved=state['branches'].setdefault(key,{})
                    saved.update(group=group,learning_rate=lr)
                    phase('train:'+key)
                    if 'model' not in saved:
                        saved['model']=backend.train(job,cfg,branch,resume=resume);write(root/'state.json',state)
                    stages=saved['model']['epoch_checkpoints']
                    if len(stages)!=3 or [v['global_step'] for v in stages]!=checkpoint_steps:
                        raise ValueError('Expected exact native checkpoint steps 42/84/125')
                    for index,stage in enumerate(stages,1):
                        name='progress-'+str(index);model=stage['model']
                        if hasattr(backend,'verify_model'):backend.verify_model(model)
                        folder=branch/name;folder.mkdir(exist_ok=True)
                        for benchmark in ['taco_dev']+job['benchmarks']:
                            eval_key=name+'/'+benchmark
                            if eval_key in saved.get('evaluations',{}):backend.verify_result(saved['evaluations'][eval_key]);continue
                            phase('evaluate:'+key+':'+eval_key)
                            if benchmark=='taco_dev':
                                if backend.kind=='mock':result=backend.evaluate(job,cfg,model,benchmark,folder,resume=resume)
                                else:
                                    from .nice_sandbox import evaluate_tasks
                                    result=evaluate_tasks(backend,cfg,model,plan['tasks'],folder/'taco_dev')
                            else:result=backend.evaluate(job,cfg,model,benchmark,folder,resume=resume)
                            saved.setdefault('evaluations',{})[eval_key]=result;write(root/'state.json',state)
        report(plan,state,root)
        write(root/'complete.json',{'plan_hash':plan['plan_hash'],'branches':4,'training_checkpoints':12,
              'evaluation_results':60,'optimizer_steps_per_branch':checkpoint_steps[-1],
              'checkpoint_steps':checkpoint_steps,'same_task_validation_is_diagnostic':True,'synthetic':plan['synthetic']})
        phase('complete')
    return root


def report(plan,state,root):
    from .reporting import table
    costs=[];scores=[]
    for key,branch in state['branches'].items():
        previous={}
        for i,stage in enumerate(branch['model']['epoch_checkpoints'],1):
            cost=stage['model'].get('training_cost',{})
            base={'branch':key,'group':branch['group'],'learning_rate':branch['learning_rate'],'checkpoint':'progress-'+str(i),'global_step':stage.get('global_step'),
                  'progress':stage.get('progress'),'model_identity':stage['model']['identity'],
                  'synthetic':plan['synthetic'],'unique_examples':1000,**cost}
            if not stage.get('synthetic') and 'global_step' in cost and cost['global_step']!=stage.get('global_step'):
                raise ValueError('Checkpoint cost step mismatch')
            for field in ('sample_exposures','input_tokens','supervised_tokens','train_wall_seconds'):
                current=cost.get(field);prior=previous.get(field,0)
                base['increment_'+field]=None if current is None or prior is None else current-prior
            previous=cost
            costs.append(base)
            for benchmark in ['taco_dev']+plan['nice']['evaluation']['benchmarks']:
                result=branch['evaluations'][base['checkpoint']+'/'+benchmark];grade=read(result['path'])
                usage=read(result['cost_path']) if result.get('cost_path') else {}
                scores.append({**base,'eval_input_tokens':usage.get('input_tokens'),
                               'eval_output_tokens':usage.get('output_tokens'),
                               'generation_wall_seconds':usage.get('generation_wall_seconds'),
                               'grading_wall_seconds':usage.get('grading_wall_seconds'),
                               'evaluation_wall_seconds':usage.get('evaluation_wall_seconds'),'benchmark':benchmark,'metric':grade.get('metric'),'score':grade.get('score'),
                               'count':grade.get('count'),'correct':grade.get('correct'),'result_path':result['path']})
    fields=list(dict.fromkeys(k for r in scores for k in r));table(root/'summary.csv',scores,fields)
    table(root/'training_costs.csv',costs,list(dict.fromkeys(k for r in costs for k in r)))
    (root/'REPORT.md').write_text('# NICE validation\n\n20 held-out training tasks, 4 rollouts each, 3000 candidates, bottom/top1000; '+
        '4 independent one-epoch SFT branches, 12 progress checkpoints, 60 evaluation groups.\n'+
        'Original 20-task scores are diagnostic and not independent generalization. No checkpoint selection by benchmark.\n'+
        'See summary.csv and training_costs.csv; synthetic values are not research measurements.\n',encoding='utf-8')


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--config',default='configs/ordinary/exp001.json');q.add_argument('--pool',required=True);q.add_argument('--output',required=True)
    for name in ('run','gradient-smoke'):
        q=sub.add_parser(name);q.add_argument('--plan',required=True);q.add_argument('--bindings',required=True);q.add_argument('--output',required=True);q.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':prepare(read(a.config),a.pool,a.output)
    else:
        from .backend import ServerBackend
        backend=ServerBackend(a.bindings)
        try:
            result=execute(a.plan,backend,a.output,resume=a.resume,gradient_smoke=a.command=='gradient-smoke')
            if a.command=='gradient-smoke':
                bindings=read(a.bindings);receipt=result/'gradient-acceptance.json'
                bindings.update(nice_gradient_smoke_receipt=str(receipt.resolve()),nice_gradient_smoke_sha256=sha(receipt))
                write(a.bindings,bindings)
        finally:backend.stop()

if __name__=='__main__':main()
