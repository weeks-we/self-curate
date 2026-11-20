"""Exact source join for the 20k prompt-only pool; no fuzzy or generated answers."""
from pathlib import Path
from .common import rows,jsonl,write,sha,digest,read
from self_curate.preparation.ordinary import normalize_question

PREFIX='You are a knowledgeable math assistant. Answer the following questions and think step by step. '
SUFFIX='Please output the final answer within \\boxed{}.'


def join(raw,reference,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new enriched source path')
    sidecar=read(str(raw)+'.source.json')
    if sidecar['sha256']!=sha(raw):raise ValueError('20k input checksum mismatch')
    targets={}
    for row in rows(raw):
        prompt=row['prompt']
        if len(prompt)!=1 or prompt[0]['role']!='user':raise ValueError('Unexpected 20k conversation')
        problem=prompt[0]['content'].strip()
        if problem.startswith(PREFIX):problem=problem[len(PREFIX):]
        if problem.endswith(SUFFIX):problem=problem[:-len(SUFFIX)].strip()
        key=normalize_question(problem)
        if key in targets:raise ValueError('Duplicate normalized 20k question; review source before joining')
        targets[key]=(row,problem)
    matches={}
    for ref in rows(reference):
        key=normalize_question(ref['problem'])
        if key not in targets or not isinstance(ref.get('solution'),str) or not ref['solution'].strip():continue
        if key not in matches or digest(ref)<digest(matches[key]):matches[key]=ref
    reference_sha=sha(reference)
    missing=[];values=[]
    for key,(source,problem) in targets.items():
        if key not in matches:
            missing.append({'source_row_hash':digest(source),'question_hash':digest(key),'reason':'no_exact_reference'})
            continue
        ref=matches[key]
        values.append({'problem':problem,'solution':ref['solution'],
            'answer':source['reward_model']['ground_truth'],'problem_type':ref.get('problem_type'),
            'source':'exact-20k-reference-join',
            'reference_provenance':{'raw_row_hash':digest(source),'reference_row_hash':digest(ref),
                'reference_file_sha256':reference_sha,'matching':'exact_normalized_problem',
                'fresh_solution_verification':False}})
    jsonl(output,values);jsonl(str(output)+'.unmatched.jsonl',missing)
    write(str(output)+'.source.json',{'source':sidecar['source'],'sha256':sha(output),
        'preview':sidecar.get('preview',False),'reference_file_sha256':reference_sha,
        'original_20k_sha256':sha(raw),'matched':len(values),'unmatched':len(missing),
        'label_completion':'native_if_available_else_required_after_EXP001'})
    return {'matched':len(values),'unmatched':len(missing)}
