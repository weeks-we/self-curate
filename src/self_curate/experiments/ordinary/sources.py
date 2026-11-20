"""Workstation-only source resolution/download; compute jobs never use the network."""
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from .common import digest, read, write, jsonl, sha


def resolve(registry, output):
    from huggingface_hub import HfApi
    api = HfApi()
    def one(item):
        name, spec = item
        if spec['kind']=='git' and re.fullmatch('[0-9a-f]{40}',spec.get('revision','')):
            # ls-remote does not advertise arbitrary SHA object IDs. Fetch verifies
            # availability at download time; do not accidentally replace a pin by HEAD.
            commit=spec['revision']
        elif spec['kind'] == 'git':
            result = subprocess.run(['git', 'ls-remote', spec['url'], spec.get('revision', 'HEAD')],
                                    capture_output=True, text=True, check=True, timeout=90)
            commit = result.stdout.split()[0]
        else:
            info = api.repo_info(spec['repo'], repo_type=spec['kind'], revision=spec.get('revision', 'main'))
            commit = info.sha
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Source did not resolve to a commit: ' + name)
        return name, {**spec, 'revision': commit}
    sources = dict(ThreadPoolExecutor(6).map(one, registry['sources'].items()))
    lock = {'schema': 'ordinary-source-lock-v1', 'sources': sources, 'registry_hash': digest(registry)}
    lock['lock_hash'] = digest(lock)
    if Path(output).exists() and read(output) != lock:
        raise ValueError('Never overwrite an existing source lock; use a new path')
    write(output, lock)
    return lock


def verify_lock(lock):
    if lock.get('lock_hash') != digest({k:v for k,v in lock.items() if k != 'lock_hash'}):
        raise ValueError('Source lock hash mismatch')


def download(lock, storage, *, include_models=False):
    from huggingface_hub import snapshot_download
    verify_lock(lock)
    root = Path(storage)
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in lock['sources'].items():
        if spec['kind'] == 'model' and not include_models:
            continue
        if spec['kind'] == 'git':
            folder = root/'graders'/name
            if not folder.exists():
                folder.parent.mkdir(parents=True, exist_ok=True)
                folder.mkdir()
                subprocess.run(['git','-C',str(folder),'init'],check=True)
                subprocess.run(['git','-C',str(folder),'fetch','--depth','1',spec['url'],spec['revision']],check=True)
                if spec.get('sparse_paths'):
                    subprocess.run(['git','-C',str(folder),'sparse-checkout','init','--cone'],check=True)
                    subprocess.run(['git','-C',str(folder),'sparse-checkout','set',*spec['sparse_paths']],check=True)
                subprocess.run(['git','-C',str(folder),'checkout','--detach',spec['revision']],check=True)
            commit = subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
            if commit != spec['revision']:
                raise ValueError('Grader checkout mismatch: '+name)
            results[name] = str(folder.resolve())
        else:
            path = snapshot_download(spec['repo'], repo_type=spec['kind'], revision=spec['revision'],
                                     cache_dir=root/'hf', allow_patterns=spec.get('allow_patterns'))
            results[name] = str(Path(path).resolve())
    write(root/'source-bindings.json', {'lock_hash':lock['lock_hash'],'paths':results})
    return results


def preview(lock, key, output, *, count=3):
    """Viewer API is a preview only: verify SHA before/after, never authorize formal training."""
    from huggingface_hub import HfApi
    verify_lock(lock)
    source = lock['sources'][key]
    api = HfApi()
    if not 1 <= count <= 100:
        raise ValueError('Preview limited to 100 records')
    before = api.dataset_info(source['repo']).sha
    if before != source['revision']:
        raise ValueError('Viewer latest no longer matches lock; use pinned snapshot')
    url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
        'dataset':source['repo'],'config':source['config'],'split':source['split'],
        'offset':0,'length':count})
    with urlopen(url, timeout=90) as response:
        value = json.load(response)
    if api.dataset_info(source['repo']).sha != before:
        raise ValueError('Source changed during preview')
    jsonl(output, (r['row'] for r in value['rows']))
    write(str(output)+'.source.json', {'source':source,'preview':True,'count':len(value['rows']),
                                     'sha256':sha(output)})


def export_training(lock, bindings, key, output):
    """Read actual local Parquet from the pinned repository; never load a remote dataset script."""
    import pyarrow.parquet as pq
    import fnmatch
    verify_lock(lock)
    if bindings['lock_hash'] != lock['lock_hash']:
        raise ValueError('Bindings belong to a different source lock')
    spec = lock['sources'][key]
    root = Path(bindings['paths'][key])
    files = sorted(p for p in root.rglob('*.parquet')
                   if fnmatch.fnmatch(p.relative_to(root).as_posix(),spec['parquet_glob']))
    if not files:
        raise ValueError('No files matched pinned split: '+key)
    def records():
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                yield from batch.to_pylist()
    jsonl(output, records())
    write(str(output)+'.source.json', {'source':spec,'preview':False,'lock_hash':lock['lock_hash'],
                                      'sha256':sha(output),'files':[{ 'name':p.name,'sha256':sha(p)} for p in files]})
