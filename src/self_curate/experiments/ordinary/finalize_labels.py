"""Publish post-EXP001 skill distributions and future development/training splits."""
from collections import Counter
from pathlib import Path
from .common import read,write,rows,jsonl,sha
from .workflow import development


def finalize(pool,output,*,completed_run,per_skill=50,seed=42,coding=False):
    complete=read(Path(completed_run)/'complete.json')
    if complete.get('jobs')!=18 or complete.get('evaluation_results')!=72:
        raise ValueError('EXP001 six-by-three checkpoint matrix is not complete')
    root=Path(output)
    if root.exists():raise FileExistsError('Label/split assets are immutable')
    manifest=read(Path(pool)/'manifest.json');file=Path(pool)/'pool.jsonl'
    if sha(file)!=manifest['pool_sha256']:raise ValueError('Pool changed')
    records=list(rows(file));counts=Counter(s for r in records for s in r['skills'])
    if coding and any(r.get('domain')!='coding' or r.get('source',{}).get('split')!='train' for r in records):
        raise ValueError('Coding development must come exclusively from the TACO train split')
    domain=records[0]['domain'] if records else None
    taxonomy=read(Path(__file__).parents[4]/'configs/ordinary/skills.json')['domains'].get(domain,{})
    for skill in taxonomy:counts.setdefault(skill,0)
    unknown=[r['id'] for r in records if not r['skills'] or r.get('label_status')=='missing' or (taxonomy and set(r['skills'])-set(taxonomy))]
    def eligible(record):
        metadata=record.get('metadata',{})
        if metadata.get('evaluation_eligible') is False:return False
        return not (record.get('domain')=='math' and (metadata.get('is_proof') is True or str(metadata.get('question_type','')).lower()=='proof' or str(metadata.get('answer','')).lower()=='proof'))
    development_records=[r for r in records if eligible(r)]
    development_counts=Counter(s for r in development_records for s in r['skills'])
    for skill in taxonomy:development_counts.setdefault(skill,0)
    root.mkdir(parents=True)
    def lengths(field):
        values=sorted(r[field] for r in records if r.get(field) is not None)
        if not values:return {'count':0,'missing':len(records)}
        import math
        return {'count':len(values),'missing':len(records)-len(values),'min':values[0],'max':values[-1],
            'median':values[(len(values)-1)//2],
            'p90':values[max(0,math.ceil(len(values)*.9)-1)],
            'p95':values[max(0,math.ceil(len(values)*.95)-1)]}
    write(root/'distribution.json',{'unique_examples':len(records),'skill_memberships':dict(counts),
        'missing_label_ids':unknown,'multi_label_examples':sum(len(r['skills'])>1 for r in records),
        'skill_percentages':{k:v/len(records) for k,v in counts.items()} if records else {},
        'lengths':{field:lengths(field) for field in ('answer_chars','answer_tokens','prompt_tokens','length')},
        'development_eligible_count':len(development_records),'training_only_count':len(records)-len(development_records),
        'development_skill_memberships':dict(development_counts),
        'shortages':{k:v for k,v in development_counts.items() if v<per_skill},
        'note':'Multi-label memberships can sum above 100%; length tokens require a frozen tokenizer.'})
    if unknown:raise ValueError('Complete skill labels before splitting; distribution saved')
    if any(v<per_skill for v in development_counts.values()):raise ValueError('Skill coverage shortage; distribution saved, do not duplicate tasks')
    result=development(development_records,per_skill,seed);ids=set(result['unique_ids'])
    jsonl(root/'development.jsonl',(r for r in records if r['id'] in ids))
    jsonl(root/'future-training.jsonl',(r for r in records if r['id'] not in ids))
    write(root/'manifest.json',{**result,'source_pool_sha256':sha(file),
        'development_sha256':sha(root/'development.jsonl'),
        'future_training_sha256':sha(root/'future-training.jsonl'),
        'exp001_training_overlap_possible':True,'synthetic':complete['backend']=='mock',
        'use':'future experiments starting from the original base; not independent validation of EXP001 checkpoints'})
    return result
