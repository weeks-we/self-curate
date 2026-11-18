"""Immutable task answer publication, verified before grade-only recovery."""
from pathlib import Path
from datetime import datetime, timezone
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json, read_json

def seal(root, request_hash, row):
    root=Path(root)
    require(not (root/'AGENT_SEALED.json').exists(),'Answer already sealed')
    names=['worker-result.json','sdk-events.jsonl','inference-usage.json','patch.json']
    require(row.get('usage_complete') is True,'Cannot seal incomplete usage')
    atomic_json(root/'answer.json',row)
    names.append('answer.json')
    hashes={name:file_sha256(root/name) for name in names}
    atomic_json(root/'AGENT_SEALED.json',{'schema':'sc-agent-sealed-v1',
        'request_hash':request_hash,'sealed_at':datetime.now(timezone.utc).isoformat(),
        'checksums':hashes})

def load_sealed(root, request_hash):
    root=Path(root).resolve(); marker=root/'AGENT_SEALED.json'
    require(marker.is_file() and not marker.is_symlink(),'Unsealed attempt: reconcile, never replay automatically')
    saved=read_json(marker)
    require(saved.get('schema')=='sc-agent-sealed-v1' and saved.get('request_hash')==request_hash,
            'Sealed request identity mismatch')
    required={'worker-result.json','sdk-events.jsonl','inference-usage.json','patch.json','answer.json'}
    require(set(saved.get('checksums',{}))==required,'Incomplete sealed evidence')
    for name,sha in saved['checksums'].items():
        p=root/name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root), 'Unsafe evidence path')
        require(file_sha256(p)==sha,'Sealed evidence changed')
    row=read_json(root/'answer.json')
    require(row.get('usage_complete') is True and row['patch']==read_json(root/'patch.json')['patch'],
            'Sealed patch or usage mismatch')
    return row

def seal_result(root, request_hash):
    root=Path(root).resolve()
    require(not (root/'RESULT_SEALED.json').exists(),'Result already sealed')
    require(all(not p.is_symlink() and p.resolve().is_relative_to(root) for p in root.rglob('*')),
            'Unsafe result evidence')
    require(all((root/n).is_file() for n in ('task-result.json','request.json','actions.json','observations.json','timing-tokens.json','lifecycle-costs.json')),
            'Missing final result evidence')
    hashes={p.relative_to(root).as_posix():file_sha256(p) for p in root.rglob('*') if p.is_file()}
    atomic_json(root/'RESULT_SEALED.json',{'schema':'sc-result-sealed-v1','request_hash':request_hash,'checksums':hashes})

def load_result(root, request_hash):
    from self_curate.contracts import ComputeCost,ResourceUsage
    from self_curate.contracts.parsing import artifact
    from self_curate.validation import TaskRunResult
    root=Path(root).resolve();mark=read_json(root/'RESULT_SEALED.json')
    require(mark.get('schema')=='sc-result-sealed-v1' and mark.get('request_hash')==request_hash,'Result identity mismatch')
    require(bool(mark.get('checksums')) and set(mark['checksums'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='RESULT_SEALED.json'},'Result evidence set changed')
    for name,sha in mark['checksums'].items():
        path=root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and file_sha256(path)==sha,'Result evidence corrupted')
    row=read_json(root/'task-result.json')
    require(row['request_hash']==request_hash,'Result request mismatch')
    usage=dict(row['usage']);usage['compute']=ComputeCost(**usage['compute'])
    row.update(usage=ResourceUsage(**usage),actions=artifact(row['actions']),observations=artifact(row['observations']))
    return TaskRunResult(**row)
