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
