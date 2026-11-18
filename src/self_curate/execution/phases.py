"""Durable infrastructure phases. Model generation must not be retried here."""
from pathlib import Path
from datetime import datetime,timezone
import time
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import atomic_json,read_json,lease

def stamp(): return datetime.now(timezone.utc).isoformat()

def phase(root, identity, operation, *, valid, retries=2):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with lease(root/'phase.lock'):
        expected=digest(identity)
        if (root/'identity.json').exists():
            require(read_json(root/'identity.json')==identity,'Phase identity changed')
        else: atomic_json(root/'identity.json',identity)
        for index in range(retries+1):
            work=root/('attempt-%03d'%index)
            if work.exists():
                if (work/'SEALED.json').exists():
                    mark=read_json(work/'SEALED.json')
                    require(mark.get('identity')==expected and bool(mark.get('checksums')),'Invalid phase seal')
                    require(set(mark['checksums'])=={p.relative_to(work).as_posix() for p in work.rglob('*') if p.is_file() and p.name!='SEALED.json'},'Phase evidence set changed')
                    for name,sha in mark['checksums'].items():
                        p=work/name
                        require(not p.is_symlink() and p.resolve().is_relative_to(work.resolve()) and file_sha256(p)==sha,'Phase evidence changed')
                    value=read_json(work/'value.json')
                    require(valid(value),'Sealed phase is invalid')
                    return value
                require((work/'FAILED.json').exists(),'Uncertain in-flight phase: reconcile before retry')
                continue
            work.mkdir();start=time.perf_counter()
            atomic_json(work/'STARTED.json',{'started_at':stamp(),'identity':expected})
            try:
                value=operation(work)
                require(valid(value),'Infrastructure phase did not produce a valid result')
                atomic_json(work/'value.json',value)
                atomic_json(work/'timing.json',{'finished_at':stamp(),'active_seconds':time.perf_counter()-start})
                hashes={p.relative_to(work).as_posix():file_sha256(p) for p in work.rglob('*') if p.is_file()}
                atomic_json(work/'SEALED.json',{'identity':expected,'checksums':hashes})
                return value
            except Exception as exc:
                atomic_json(work/'FAILED.json',{'finished_at':stamp(),'active_seconds':time.perf_counter()-start,
                    'error_type':type(exc).__name__,'message':str(exc)})
        raise RuntimeError('Infrastructure retries exhausted; all attempts retained')

def lifecycle(root):
    root=Path(root); stages=[]
    for p in sorted(root.rglob('STARTED.json')):
        start=read_json(p);folder=p.parent
        end=folder/('timing.json' if (folder/'timing.json').exists() else 'FAILED.json')
        row=read_json(end) if end.exists() else {}
        stages.append({'phase':folder.relative_to(root).as_posix(),**start,**row,'failed':(folder/'FAILED.json').exists()})
    segments=[]
    for p in root.glob('timing-segments/*.json'):
        if p.name.endswith('.finished.json'):continue
        end=p.with_suffix('.finished.json')
        segments.append(read_json(end if end.exists() else p))
    selected=segments or stages
    times=[datetime.fromisoformat(x[k]) for x in selected for k in ('started_at','finished_at') if x.get(k)]
    return {'stages':stages,'segments':segments,'active_seconds':sum(x.get('active_seconds') or 0 for x in selected),
        'failed_active_seconds':sum(x.get('active_seconds') or 0 for x in selected if x.get('failed')),
        'infrastructure_failed_seconds':sum(x.get('active_seconds',0) for x in stages if x['failed']),
        'clock_anomalies':[x for x in selected if x.get('finished_at') and x['finished_at']<x['started_at']],
        'wall_span_seconds':(max(times)-min(times)).total_seconds() if times else None,
        'incomplete_phases':sum(not x.get('finished_at') for x in selected),
        'historical_missing_costs_inferred':False}

def timed_call(root, stage, function, *args, **kwargs):
    import uuid
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/(stage+'-'+uuid.uuid4().hex+'.json')
    value={'stage':stage,'started_at':stamp(),'finished_at':None,'active_seconds':None,'failed':None}
    atomic_json(path,value);start=time.perf_counter()
    try:
        result=function(*args,**kwargs);value['failed']=False;return result
    except BaseException as exc:
        value.update(failed=True,error_type=type(exc).__name__);raise
    finally:
        value.update(finished_at=stamp(),active_seconds=time.perf_counter()-start)
        # Separate immutable completion; an interrupted start remains visible.
        atomic_json(path.with_suffix('.finished.json'),value)
