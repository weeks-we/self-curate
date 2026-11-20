"""Prepare sealed ordinary-task sets, preserving official raw rows for graders."""
import csv
import json
from pathlib import Path
from .common import digest, jsonl, rows, write, sha, read
from self_curate.preparation.ordinary import normalize_question


def decode(value):
    if not isinstance(value,str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Multi-IF CSV stores escaped quotes inside JSON-valued cells.
        return json.loads(value.replace('\\"','"'))


def task(key, row):
    turns=None
    if key in ('math500','aime25','aime26'):
        question=row['problem']; gold=row['answer']
    elif key=='gsm_plus':
        question=row['question']; gold=row['answer']
    elif key=='olympiad':
        if row.get('modality') not in (None,'Text-only','text-only','TO') and any(row.get('image_'+str(i)) for i in range(1,9)):
            raise ValueError('Image task in text-only Olympiad configuration')
        question=row['question']; gold=row['final_answer']
    elif key in ('ifeval','ifbench'):
        question=row['prompt']; gold=None
        if len(row['instruction_id_list'])!=len(row['kwargs']):
            raise ValueError('IF checker mismatch')
    elif key=='multi_if':
        if row.get('language')!='English':
            return None
        turns=[decode(row[f'turn_{i}_prompt']) for i in (1,2,3)]
        if any(t.get('role')!='user' for t in turns):
            raise ValueError('Invalid Multi-IF user turn')
        question=turns[0]['content']; gold=None
    elif key in ('humaneval_plus','mbpp_plus'):
        question=row['prompt']; gold=None
    elif key=='livecodebench':
        question=row['question_content']; gold=None
    elif key=='bigcodebench':
        question=row['instruct_prompt']; gold=None
    else:
        raise ValueError('Unknown benchmark: '+key)
    prompt=question
    if key in ('math500','aime25','aime26','gsm_plus','olympiad'):
        prompt+='\n\nShow your solution and put the final answer in \\boxed{}.'
    if key in ('humaneval_plus','mbpp_plus','livecodebench','bigcodebench'):
        if key=='livecodebench' and row.get('starter_code'):
            prompt+='\n\nStarter code:\n'+row['starter_code']
        prompt+='\n\nReturn the complete Python code in a fenced code block.'
    identifier=str(row.get('task_id',row.get('unique_id',row.get('id',row.get('key',row.get('question_id',digest(question)))))))
    if key=='livecodebench':
        identifier=str(row['platform'])+':'+identifier
    return {'id':key+':'+identifier,'benchmark':key,'question_hash':digest(normalize_question(question)),
            'prompt':prompt,'turns':turns,'reference':gold,'raw':row}


def load_local(root, spec):
    root=Path(root)
    if spec.get('jsonl_files'):
        for file in spec['jsonl_files']:
            yield from rows(root/file)
    elif spec.get('csv_file'):
        with (root/spec['csv_file']).open(encoding='utf-8') as stream:
            yield from csv.DictReader(stream)
    else:
        import pyarrow.parquet as pq
        files=sorted(root.glob(spec['parquet_glob']))
        if not files:
            raise ValueError('Pinned benchmark split missing: '+spec['parquet_glob'])
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
                yield from batch.to_pylist()


def prepare_benchmarks(lock, bindings, output):
    from .sources import verify_lock
    verify_lock(lock)
    if bindings['lock_hash']!=lock['lock_hash']:
        raise ValueError('Source bindings mismatch')
    output=Path(output)
    if output.exists():
        raise FileExistsError('Benchmark manifests are immutable')
    output.mkdir(parents=True)
    result={}
    for key,spec in lock['sources'].items():
        if spec.get('purpose')!='final_test':
            continue
        values=[]
        evalplus=None
        if key in ('humaneval_plus','mbpp_plus'):
            asset=read(Path(bindings['evalplus_root'])/'manifest.json')[key]
            if sha(asset['path'])!=asset['sha256']:raise ValueError('EvalPlus cache checksum mismatch')
            source_rows=rows(asset['path']);evalplus=asset['path']
        else:source_rows=load_local(bindings['paths'][key],spec)
        for row in source_rows:
            value=task(key,row)
            if value is not None:
                values.append(value)
        if not values or len({v['id'] for v in values})!=len(values):
            raise ValueError('Empty/duplicate benchmark IDs: '+key)
        file=output/(key+'.jsonl')
        jsonl(file,values)
        result[key]={'path':str(file.resolve()),'sha256':sha(file),'count':len(values),
                     'source':spec,'ids':[v['id'] for v in values]}
        if evalplus:
            result[key]['evalplus_data']=evalplus
            result[key]['evalplus_sha256']=sha(evalplus)
    write(output/'manifest.json',{'schema':'ordinary-benchmarks-v1','lock_hash':lock['lock_hash'],'benchmarks':result})
    return result
