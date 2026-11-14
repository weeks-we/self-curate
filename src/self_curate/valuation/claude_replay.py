"""Bridge native Claude/vLLM task evidence into the unchanged NICE contract."""
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.training import TrainingExample
from self_curate.execution.artifacts import read_artifact
from .contracts import ReplayTrajectory

def replay_from_rollout(rollout):
    trace=read_artifact(rollout.actions)['trace']
    captures=trace.get('native_capture')
    require(captures is not None,'No engine-native capture: SDK text is not a NICE replay')
    evidence=__import__('self_curate.contracts.parsing',fromlist=['artifact']).artifact(captures)
    packet=read_artifact(evidence)
    require(packet['model_ref']==rollout.model_ref.to_dict(),'Replay model identity changed')
    examples=[]
    for index,row in enumerate(packet['requests']):
        require(row['complete'] is True and row['token_origin']=='engine_request_output'
                and row['model_alias']=='sc-'+rollout.model_ref.model_hash,
                'Wrong adapter or incomplete native capture')
        prefix=tuple(row['prompt_token_ids']);generated=tuple(row['generated_token_ids'])
        if not generated:continue
        require(bool(prefix),'Missing causal prefix')
        ids=prefix+generated
        examples.append(TrainingExample('sdk-native-'+str(index),digest(row),ids,
                                         (-100,)*len(prefix)+generated))
    require(sum(e.supervised_tokens for e in examples)==rollout.generated_tokens,
            'Native generated-token coverage differs from measured rollout usage')
    return ReplayTrajectory(rollout.record_hash,tuple(examples),evidence,rollout.model_ref.tokenizer)
