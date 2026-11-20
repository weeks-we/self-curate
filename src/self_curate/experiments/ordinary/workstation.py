"""Networked preparation; outputs are reusable, checksummed assets, never results."""
import json
import subprocess
from pathlib import Path
from .common import read,write,rows,jsonl,sha,digest


def cache_evalplus(storage):
    """Download the installed pinned EvalPlus release, not its simplified HF mirror."""
    from evalplus.data.humaneval import _ready_human_eval_plus_path
    from evalplus.data.mbpp import _ready_mbpp_plus_path
    root=Path(storage)/'evalplus';root.mkdir(parents=True,exist_ok=True)
    result={}
    import shutil,gzip
    for key,loader in [('humaneval_plus',_ready_human_eval_plus_path),('mbpp_plus',_ready_mbpp_plus_path)]:
        source=Path(loader());target=root/(key+'.jsonl')
        opener=gzip.open if source.suffix=='.gz' else open
        with opener(source,'rt',encoding='utf-8') as stream:
            values=[json.loads(line) for line in stream if line.strip()]
        if not values or any('plus_input' not in r for r in values):
            raise ValueError('EvalPlus enhanced tests are missing')
        if target.exists():
            if list(rows(target))!=values:raise ValueError('Existing EvalPlus release changed')
        else:jsonl(target,values)
        result[key]={'path':str(target.resolve()),'sha256':sha(target),'count':len(values)}
    write(root/'manifest.json',result)
    return result


def seal(storage,lock_file,bindings_file):
    """Run after all downloads/preparation. Hash even large model files once."""
    root=Path(storage).resolve()
    lock=read(lock_file);bindings=read(bindings_file)
    if lock['lock_hash']!=bindings['lock_hash']:raise ValueError('Source lock mismatch')
    files={}
    for key,path in bindings['paths'].items():
        path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('Assets must remain under the shared storage root')
        for file in sorted(path.rglob('*')):
            if file.is_file() and '.git' not in file.relative_to(path).parts:
                files[str(file.relative_to(root))]=sha(file)
    if not files:raise ValueError('No assets to seal')
    value={'schema':'ordinary-asset-seal-v1','lock_hash':lock['lock_hash'],'files':files}
    value['seal_hash']=digest(value)
    write(root/'asset-seal.json',value)
    return value


def verify_seal(storage):
    root=Path(storage);value=read(root/'asset-seal.json')
    if value['seal_hash']!=digest({k:v for k,v in value.items() if k!='seal_hash'}):
        raise ValueError('Invalid asset seal')
    for path,expected in value['files'].items():
        target=(root/path).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target)!=expected:
            raise ValueError('Shared asset checksum mismatch: '+path)
    return value


def bindings(lock_file,source_bindings,benchmarks_file,storage,project,run_root,grader_image,output):
    """Portable paths require the same mount point on workstation and compute node."""
    lock=read(lock_file);sources=read(source_bindings);benchmarks=read(benchmarks_file)
    if not (lock['lock_hash']==sources['lock_hash']==benchmarks['lock_hash']):
        raise ValueError('Prepared assets have different locks')
    if not grader_image.startswith('sha256:') and '@sha256:' not in grader_image:
        raise ValueError('Freeze the built grader image ID/digest, not a mutable tag')
    models={}
    for key in ('qwen9b','qwen4b','llama8b'):
        spec=lock['sources'][key];path=Path(sources['paths'][key])
        indexes=list(path.glob('*.safetensors.index.json'))
        if not indexes:raise ValueError('Complete model snapshot required: '+key)
        index=read(indexes[0])
        for file in set(index['weight_map'].values()):
            if not (path/file).is_file():raise ValueError('Missing model shard: '+file)
        models[key]={'path':str(path.resolve()),'loader':spec['loader'],
                     'model_hash':digest({'revision':spec['revision'],'index':sha(indexes[0])})}
    grader_keys={'livecodebench':'lcb_grader','humaneval_plus':'evalplus_grader',
        'mbpp_plus':'evalplus_grader','bigcodebench':'bcb_grader',
        'ifeval':'ifeval_grader','ifbench':'ifbench_grader','multi_if':'multi_if_grader',
        'olympiad':'olympiad_grader'}
    prepared=benchmarks['benchmarks']
    for key,entry in prepared.items():
        if key in grader_keys:entry['grader_root']=sources['paths'][grader_keys[key]]
    value={'schema':'ordinary-server-bindings-v1','lock_hash':lock['lock_hash'],
        'models':models,'benchmarks':prepared,'storage':str(Path(storage).resolve()),
        'project':str(Path(project).resolve()),'run_root':str(Path(run_root).resolve()),
        'grader_image':grader_image,'grader_images':{'bigcodebench':None},'server_smoke_passed':False,
        'asset_seal_hash':read(Path(storage)/'asset-seal.json')['seal_hash']}
    write(output,value)
    return value


def reserve(raw_pool,output,*,per_skill=50,seed=42):
    """Freeze native-label development IDs before ANY baseline training."""
    from .workflow import development
    root=Path(output)
    if root.exists():raise FileExistsError('Development assets are immutable')
    values=list(rows(Path(raw_pool)/'pool.jsonl'))
    result=development(values,per_skill,seed)
    selected=set(result['unique_ids'])
    root.mkdir(parents=True)
    jsonl(root/'tasks.jsonl',(r for r in values if r['id'] in selected))
    write(root/'manifest.json',{**result,'pool_sha256':sha(Path(raw_pool)/'pool.jsonl'),
                               'tasks_sha256':sha(root/'tasks.jsonl')})
    return result
