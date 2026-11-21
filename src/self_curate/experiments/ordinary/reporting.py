"""Publish score and resource tables without inventing unmeasured/missing values."""
import csv
from pathlib import Path
from .common import read, write


def table(path, records, fields):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)


def publish(output, plan, state):
    output=Path(output); long=[];training=[];wide=[]
    for job in plan['jobs']:
        record=state['completed'][job['job_id']]
        models={'initial':record['model']} if job['method']=='initial' else {
            'epoch-'+str(s['epoch']).zfill(2):s['model'] for s in record['model']['epoch_checkpoints']}
        previous={}
        for stage,model in models.items():
            measured=model.get('training_cost',{})
            initial=stage=='initial'
            base={'job':job['job_id'],'domain':job['domain'],'method':job['method'],
                'checkpoint':stage,'model_identity':model['identity'],'synthetic':model.get('synthetic',False),
                'unique_examples':job['count'],'planned_sample_exposures':0 if initial else job['count']*int(stage[-2:]),
                'sample_exposures':0 if initial else measured.get('sample_exposures'),
                'train_input_tokens':0 if initial else measured.get('input_tokens'),
                'train_supervised_tokens':0 if initial else measured.get('supervised_tokens'),
                'train_wall_seconds':0 if initial else measured.get('train_wall_seconds'),
                'train_gpu_hours':0 if initial else measured.get('gpu_hours')}
            epoch_cost={**base}
            for dest,source in (('epoch_sample_exposures','sample_exposures'),('epoch_input_tokens','input_tokens'),
                                ('epoch_supervised_tokens','supervised_tokens'),('epoch_wall_seconds','train_wall_seconds')):
                current=measured.get(source); prior=previous.get(source,0)
                epoch_cost[dest]=None if current is None or prior is None else current-prior
            if not initial:training.append(epoch_cost)
            previous=measured
            if not initial and int(stage[-2:]) not in plan['config']['evaluation']['epochs']:
                continue
            wide_row={**base}
            for benchmark in job['benchmarks']:
                saved=record['evaluations'][stage+'/'+benchmark]
                score=read(saved['path']);cost=read(saved['cost_path']) if saved.get('cost_path') else {}
                long.append({**base,'benchmark':benchmark,'count':score.get('count'),'correct':score.get('correct'),
                    'score':score.get('score'),'metric':score.get('metric'),'eval_input_tokens':cost.get('input_tokens'),
                    'eval_output_tokens':cost.get('output_tokens'),
                    'generation_wall_seconds':cost.get('generation_wall_seconds'),
                    'grading_wall_seconds':cost.get('grading_wall_seconds'),
                    'evaluation_wall_seconds':cost.get('evaluation_wall_seconds'),
                    'score_path':saved['path'],'cost_path':saved.get('cost_path')})
                wide_row[benchmark]=score.get('score')
            wide.append(wide_row)
    fields=list(long[0]);table(output/'summary.csv',long,fields)
    table(output/'training_costs.csv',training,list(training[0]))
    bench=list(dict.fromkeys(r['benchmark'] for r in long))
    table(output/'scores_wide.csv',wide,list(wide[0])+[b for b in bench if b not in wide[0]])
    lines=['# EXP001 results','',
        'Epochs 1/2/3 are evaluated; epoch3 is primary. Never select checkpoints using benchmark scores.',
        'Blank = unmeasured, never zero. Synthetic scores/costs are not research measurements.',
        'Training costs are cumulative at checkpoint. Do not sum them across benchmarks or epochs.',
        'Per-epoch deltas: training_costs.csv. Generation/CPU grading costs: summary.csv.',
        'Wall time excludes unrecorded work lost before a recovery checkpoint; no fabricated currency cost.',
        '', '| Job | Checkpoint | Benchmark | Score | Train input tokens | Exposures | Train seconds | Eval output tokens |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in long:
        lines.append('| '+' | '.join('' if r.get(k) is None else str(r[k]) for k in
            ('job','checkpoint','benchmark','score','train_input_tokens','sample_exposures','train_wall_seconds','eval_output_tokens'))+' |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(output/'report-manifest.json',{'schema':'ordinary-report-v1','rows':len(long),
        'training_checkpoints':len(training),'plan_hash':plan['plan_hash'],
        'primary_epoch':3,'missing_measurements_are_null':True})
