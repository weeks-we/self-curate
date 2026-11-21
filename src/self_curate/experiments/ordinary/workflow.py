"""EXP001 matrix, fixed nested random subsets, durable train/evaluate state machine."""
import math
import random
from pathlib import Path
from .common import digest, read, write, rows, jsonl, sha, run_id, event
from self_curate.training.recovery import lease


def validate_config(cfg):
    from .configuration import ordinary
    ordinary(cfg)
    if cfg.get('schema')!='ordinary-exp001-v1' or cfg['model']!='qwen9b':
        raise ValueError('EXP001 uses Qwen3.5-9B only')
    if cfg['domains']!=['coding','math','if'] or sorted(cfg['random_fractions'])!=[0.05,0.1,0.2,0.3]:
        raise ValueError('EXP001 requires three independent domains and random 5/10/20/30%')
    if cfg['training']['epochs']!=3 or cfg['evaluation'].get('checkpoint')!='each_epoch_after_training':
        raise ValueError('EXP001 requires three epochs and benchmark evaluation of all three epochs')
    if cfg['evaluation'].get('epochs')!=[1,2,3]:
        raise ValueError('EXP001 must evaluate epochs1/2/3')
    if cfg['training']['micro_batch_size']*cfg['training']['world_size']*cfg['training']['gradient_accumulation_steps']!=cfg['training']['global_batch_size']:
        raise ValueError('Invalid effective global batch')
    if cfg['evaluation']['thinking'] or cfg['evaluation']['selection_on_test']:
        raise ValueError('Thinking and test-based selection are disabled')
    if cfg.get('approved') and cfg.get('domain_pool_policy',{}).get('math',{}).get('status')!='confirmed':
        raise ValueError('Math candidate-pool subsampling must be agreed before formal planning')
    return cfg


def plan(config, pools, output, *, synthetic=False):
    cfg=validate_config(config)
    root=Path(output)
    if root.exists():
        raise FileExistsError('Create a new immutable experiment plan')
    root.mkdir(parents=True)
    jobs=[]
    for domain in cfg['domains']:
        folder=Path(pools[domain])
        manifest=read(folder/'manifest.json')
        if sha(folder/'pool.jsonl')!=manifest['pool_sha256']:
            raise ValueError('Pool checksum mismatch')
        records=list(rows(folder/'pool.jsonl'))
        if not records or len({r['id'] for r in records})!=len(records):
            raise ValueError('Pool must contain nonempty, unique question IDs')
        permutation=list(records)
        random.Random(cfg['seed']).shuffle(permutation)
        modes=[('initial',[]),('full',permutation)]
        for fraction in cfg['random_fractions']:
            count=math.floor(len(records)*fraction)
            if count<1:
                raise ValueError('Pool too small for requested fraction')
            modes.append(('random'+str(round(fraction*100)).zfill(2),permutation[:count]))
        for mode,selected in modes:
            job_id=domain+'-'+mode
            file=root/'selections'/(job_id+'.jsonl')
            jsonl(file,selected)
            selected_manifest={'job_id':job_id,'model':cfg['model'],'domain':domain,'method':mode,
                'seed':cfg['seed'],'pool_sha256':manifest['pool_sha256'],'count':len(selected),
                'ids':[r['id'] for r in selected], 'data':str(file.resolve()),'data_sha256':sha(file),
                'preview':manifest['preview'],'tokenized':manifest['tokenized'],
                'pool_max_length':manifest.get('max_length'),
                'benchmarks':cfg['evaluation']['benchmarks'][domain],
                'supervised_tokens':sum(r.get('supervised_tokens',0) for r in selected)}
            write(file.with_suffix('.manifest.json'),selected_manifest)
            jobs.append(selected_manifest)
    expanded=[]
    rates=cfg['training'].get('learning_rates') or [cfg['training']['learning_rate']]
    if len(set(rates))!=len(rates) or any(not math.isfinite(v) or v<=0 for v in rates):raise ValueError('Invalid shared learning rates')
    for job in jobs:
        for lr in ([None] if job['method']=='initial' else rates):
            value={**job,'learning_rate':lr}
            if lr is not None and len(rates)>1:value['job_id']+='-lr'+format(lr,'.8g')
            expanded.append(value)
    jobs=expanded
    nice_plan=None
    if cfg.get('nice',{}).get('enabled'):
        from .nice_experiment import prepare
        prepare(cfg,pools['coding'],root/'nice-plan',synthetic=synthetic)
        nice_plan=str((root/'nice-plan/plan.json').resolve())
    value={'schema':'ordinary-plan-v1','synthetic':synthetic,'config':cfg,'config_hash':digest(cfg),'jobs':jobs,
           'run_id':run_id(),'nice_plan':nice_plan,'nice_plan_sha256':sha(nice_plan) if nice_plan else None,'independent_branches':True,'source_code_hash':code_hash()}
    value['plan_hash']=digest(value)
    write(root/'plan.json',value)
    return value


def code_hash():
    source=Path(__file__).parents[2]
    return digest({str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))})


def execute(plan_file,backend,*,resume=False):
    plan_file=Path(plan_file)
    value=read(plan_file)
    if value['plan_hash']!=digest({k:v for k,v in value.items() if k!='plan_hash'}):
        raise ValueError('Plan was modified')
    if value['source_code_hash']!=code_hash():
        raise ValueError('Source changed after plan freeze; create a new plan')
    config=value['config']
    if value.get('nice_plan'):
        if sha(value['nice_plan'])!=value['nice_plan_sha256']:raise ValueError('Integrated NICE plan changed')
        from .nice_experiment import validate_gradient_receipt
        validate_gradient_receipt(read(value['nice_plan']),backend)
    if backend.kind=='real':
        if value.get('synthetic'):raise ValueError('Synthetic plan cannot run on server')
        if not config['approved'] or any(j['preview'] or not j['tokenized'] for j in value['jobs']):
            raise ValueError('Formal execution needs approved parameters and fully tokenized non-preview pools')
        if any(j['pool_max_length']!=config['training']['max_length'] for j in value['jobs']):
            raise ValueError('Training length changed after common pool admission')
    output=plan_file.parent/('mock-results' if backend.kind=='mock' else 'results')/value['run_id']
    output.mkdir(parents=True,exist_ok=True)
    state_file=output/'state.json'
    write(output/'frozen-plan.json',value)
    with lease(output/'.lease'):
        if state_file.exists():
            if not resume:
                raise ValueError('Run exists; request explicit resume')
            state=read(state_file)
            if state['plan_hash']!=value['plan_hash'] or state['backend']!=backend.kind:
                raise ValueError('Resume identity mismatch')
        else:
            state={'plan_hash':value['plan_hash'],'backend':backend.kind,'completed':{},'pending':None}
        nice_results=None
        if value.get('nice_plan'):
            from .nice_experiment import execute as execute_nice
            execute_nice(value['nice_plan'],backend,output/'nice',resume=resume)
            nice_results=read(output/'nice/complete.json')
        if config['runtime'].get('pipeline',{}).get('enabled'):
            from .parallel_run import ordinary_jobs
            ordinary_jobs(value,backend,output,state,state_file,resume)
        else:
            for index,job in enumerate(value['jobs']):
                if sha(job['data'])!=job['data_sha256']:
                    raise ValueError('Selection changed')
                key=job['job_id']
                folder=output/key
                folder.mkdir(exist_ok=True)
                prior=state['completed'].get(key,{})
                event(output,'progress',job=key,index=index+1,total=len(value['jobs']),status='starting')
                print(f'[{index+1}/{len(value["jobs"])}] {key}',flush=True)
                if 'model' not in prior:
                    state['pending']={'job':key,'phase':'train'}
                    write(state_file,state)
                    import copy
                    branch_config=copy.deepcopy(config)
                    if job.get('learning_rate') is not None:branch_config['training']['learning_rate']=job['learning_rate']
                    model=backend.initial(config) if job['method']=='initial' else backend.train(job,branch_config,folder,resume=resume)
                    prior['model']=model
                    state['completed'][key]=prior
                    write(state_file,state)
                if hasattr(backend,'verify_model'):backend.verify_model(prior['model'])
                if job['method']=='initial':
                    stage_models=[('initial',prior['model'])]
                else:
                    stages=prior['model'].get('epoch_checkpoints',[])
                    expected=list(range(1,config['training']['epochs']+1))
                    if [s['epoch'] for s in stages]!=expected:
                        raise ValueError('Every epoch checkpoint is required for evaluation')
                    stage_models=[('epoch-'+str(s['epoch']).zfill(2),s['model']) for s in stages if s['epoch'] in config['evaluation']['epochs']]
                for stage_name,stage_model in stage_models:
                    if hasattr(backend,'verify_model'):backend.verify_model(stage_model)
                    stage_folder=folder/stage_name;stage_folder.mkdir(exist_ok=True)
                    for benchmark in job['benchmarks']:
                        evaluation_key=stage_name+'/'+benchmark
                        if evaluation_key in prior.get('evaluations',{}):
                            backend.verify_result(prior['evaluations'][evaluation_key])
                            continue
                        state['pending']={'job':key,'phase':'evaluate','checkpoint':stage_name,'benchmark':benchmark}
                        write(state_file,state)
                        result=backend.evaluate(job,config,stage_model,benchmark,stage_folder,resume=resume)
                        prior.setdefault('evaluations',{})[evaluation_key]=result
                        write(state_file,state)
                state['pending']=None
                write(state_file,state)
        write(output/'complete.json',{'plan_hash':value['plan_hash'],'backend':backend.kind,'jobs':len(value['jobs']),
                                     'results':state['completed'],'evaluation_results':sum(len(r.get('evaluations',{})) for r in state['completed'].values()),'phase':'training_evaluation_complete',
                                     'nice':nice_results,'label_finalization':'pending_external_labels_and_split'})
        from .reporting import publish
        publish(output,value,state)
    return output


def development(records,per_skill,seed):
    """Multi-label tasks may overlap groups; execute each task once. Never duplicate to fill shortages."""
    by_skill={}
    unique={r['id']:r for r in records}
    for record in unique.values():
        for skill in record['skills']:
            by_skill.setdefault(skill,[]).append(record['id'])
    shortages={skill:len(ids) for skill,ids in by_skill.items() if len(ids)<per_skill}
    if not by_skill or shortages:
        raise ValueError('Insufficient task coverage for 50/skill: '+str(shortages))
    groups={}
    for skill,ids in sorted(by_skill.items()):
        random.Random(str(seed)+':'+skill).shuffle(ids)
        groups[skill]=ids[:per_skill]
    return {'groups':groups,'unique_ids':sorted({i for ids in groups.values() for i in ids}),
            'per_skill':per_skill,'seed':seed,'source_count':len(unique)}


def estimate(plan_value):
    cfg=plan_value['config']
    lo,hi=cfg['estimate']['aggregate_training_tokens_per_second']
    tokens=sum(sum(len(r['messages'][0]['content'])+len(r['messages'][1]['content']) for r in rows(job['data']))
               for job in plan_value['jobs'] if job['method']!='initial')
    # Character counts are not tokenizer evidence. Estimate only after exact lengths are prepared.
    exact=sum(sum(r.get('length',0) for r in rows(job['data'])) for job in plan_value['jobs'] if job['method']!='initial')
    per_pass=exact
    exact*=cfg['training']['epochs']
    return {'epochs':cfg['training']['epochs'],'input_tokens_per_pass_across_branches':per_pass,'exact_input_tokens':exact,'characters_per_pass':tokens,
            'training_hours_unpadded_assumption':[exact/hi/3600,exact/lo/3600] if exact else None,
            'excludes':['padding','checkpoint_io','evaluation','download','source_preparation'],
            'warning':'Planning throughput assumptions; replace with server measurements. No claimed GPU benchmark.'}
