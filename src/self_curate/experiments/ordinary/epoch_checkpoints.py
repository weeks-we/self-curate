"""Collect native HF epoch checkpoints without loading weights or optimizer state."""
from pathlib import Path
from .common import read


def collect(output,epochs):
    root=Path(output).resolve();found={}
    for folder in root.glob('checkpoint-*'):
        if not folder.is_dir():continue
        if not folder.resolve().is_relative_to(root):raise ValueError('Checkpoint escapes training output')
        state=read(folder/'trainer_state.json')
        epoch=state.get('epoch');step=state.get('global_step')
        if not isinstance(epoch,(int,float)) or abs(epoch-round(epoch))>1e-6:
            raise ValueError('Expected whole-epoch native checkpoint')
        epoch=int(round(epoch))
        if epoch in found or folder.name!='checkpoint-'+str(step):raise ValueError('Ambiguous checkpoint identity')
        if not (folder/'adapter_config.json').is_file() or not any((folder/name).is_file() for name in ('adapter_model.safetensors','adapter_model.bin')):
            raise ValueError('Epoch adapter files are missing')
        found[epoch]={'epoch':epoch,'global_step':step,'path':str(folder.resolve())}
    if set(found)!=set(range(1,epochs+1)):raise ValueError('Missing or unexpected epoch checkpoints')
    return [found[i] for i in range(1,epochs+1)]


def progress_steps(count,global_batch,fractions):
    import math
    if not fractions:return []
    if count<1 or global_batch<1 or sorted(fractions)!=list(fractions) or fractions[-1]!=1.0:
        raise ValueError('Invalid progress saving schedule')
    total=math.ceil(count/global_batch)
    steps=[math.ceil(total*f) for f in fractions]
    if len(set(steps))!=len(steps) or any(s<1 for s in steps):
        raise ValueError('Too few optimizer updates for distinct checkpoints')
    return steps


def collect_progress(output,steps):
    root=Path(output).resolve();result=[]
    for step in steps:
        folder=root/('checkpoint-'+str(step))
        state=read(folder/'trainer_state.json')
        if state['global_step']!=step:raise ValueError('Progress checkpoint step mismatch')
        if not (folder/'adapter_config.json').is_file() or not any((folder/n).exists() for n in ('adapter_model.safetensors','adapter_model.bin')):
            raise ValueError('Missing progress adapter')
        result.append({'epoch':state['epoch'],'global_step':step,'path':str(folder),
                       'progress':step/steps[-1]})
    return result


def resumable_checkpoint(output,*,world_size=8):
    from .common import run_id
    root=Path(output).resolve();valid=[]
    for folder in sorted(root.glob('checkpoint-*')):
        if not folder.is_dir():continue
        if not folder.resolve().is_relative_to(root):raise ValueError('Resume checkpoint escapes output')
        marker=folder/'training-cost.json'
        if not marker.exists():
            folder.rename(root/('incomplete-'+folder.name+'-'+run_id()))
            continue
        state=read(folder/'trainer_state.json');cost=read(marker)
        step=state['global_step']
        if folder.name!='checkpoint-'+str(step) or cost['global_step']!=step:raise ValueError('Checkpoint accounting identity mismatch')
        optimizer=any(p.name.startswith('optimizer') for p in folder.iterdir())
        if not optimizer or not (folder/'scheduler.pt').exists() or not all((folder/f'rng_state_{i}.pth').exists() for i in range(world_size)):
            raise ValueError('Published checkpoint lost optimizer/scheduler/matching-rank RNG state')
        if not (folder/'adapter_config.json').exists() or not any((folder/n).exists() for n in ('adapter_model.safetensors','adapter_model.bin')):
            raise ValueError('Published checkpoint adapter is missing')
        valid.append((step,str(folder)))
    if not valid:raise ValueError('No complete native checkpoint; incomplete saves retained separately. Restart branch explicitly, never resume adapter only.')
    return max(valid)[1]
