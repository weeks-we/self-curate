"""Independent GPU shards: exact HF rollouts or frozen-LoRA backward, never DDP/step."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from .common import read,write,rows,sha,digest,event


def parameter_hash(parameters):
    import torch
    h=hashlib.sha256()
    for name,p in parameters:
        h.update(name.encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def token_loss(model,ids,labels,reduction,device):
    import torch
    import torch.nn.functional as F
    x=torch.tensor([ids],dtype=torch.long,device=device)
    y=torch.tensor([labels],dtype=torch.long,device=device)
    if len(ids)!=len(labels) or len(ids)<2 or not any(v!=-100 for v in labels[1:]):
        raise ValueError('Invalid causal replay supervision')
    logits=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),y[:,1:].reshape(-1),
                         ignore_index=-100,reduction=reduction)
    if not torch.isfinite(loss):raise ValueError('Nonfinite replay loss')
    return loss


def projected_gradients(parameters,dimension,seed,epsilon,clip):
    """Original global coordinates, Adam BEFORE projection, bounded per-layer copies."""
    import math
    import numpy as np
    # Existing project's CountSketch hash, with stable global offsets.
    norm=math.sqrt(sum(float(p.grad.detach().float().norm())**2 for _,p in parameters if p.grad is not None))
    if not math.isfinite(norm):raise ValueError('Nonfinite gradient norm')
    factor=min(1.,clip/(norm+1e-6));raw=np.zeros(dimension);adam=np.zeros(dimension);offset=0
    for _,p in parameters:
        flat=None if p.grad is None else p.grad.detach().float().cpu().numpy().reshape(-1)
        for start in range(0,p.numel(),262144):
            size=min(262144,p.numel()-start)
            values=np.zeros(size) if flat is None else flat[start:start+size].astype(np.float64)
            if not np.isfinite(values).all():raise ValueError('Nonfinite gradient coordinates')
            idx=np.arange(offset+start,offset+start+size,dtype=np.uint64)
            with np.errstate(over='ignore'):
                bits=idx+np.uint64(seed)+np.uint64(0x9E3779B97F4A7C15)
                bits=(bits^(bits>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
                bits=(bits^(bits>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
                bits^=bits>>np.uint64(31)
            buckets=(bits%np.uint64(dimension)).astype(np.int64)
            signs=np.where((bits>>np.uint64(63))==0,1.,-1.)
            raw+=np.bincount(buckets,weights=values*signs,minlength=dimension)
            transformed=values*factor;transformed/=np.abs(transformed)+epsilon
            adam+=np.bincount(buckets,weights=transformed*signs,minlength=dimension)
        offset+=p.numel()
    if not np.isfinite(raw).all() or not np.isfinite(adam).all():raise ValueError('Nonfinite projection')
    return raw,adam,norm


def run(job):
    import torch
    import numpy as np
    from transformers import AutoTokenizer,set_seed
    from self_curate.training.lora import build
    from self_curate.training.ordinary import encode_example
    from self_curate.preparation.ordinary_tokenization import NonThinkingTokenizer,renderer
    stage_started=time.monotonic()
    job=read(job)
    if job.get('execution')!='server' or not job['approved'] or not torch.cuda.is_available():
        raise ValueError('NICE worker needs explicit approved server/GPU execution')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=job['gpu_uuid'] or not job['gpu_uuid'].startswith('GPU-'):
        raise ValueError('NICE GPU UUID mismatch')
    if sha(job['records'])!=job['records_sha256']:raise ValueError('NICE shard changed')
    output=Path(job['output']);output.mkdir(parents=True,exist_ok=True)
    identity=digest({k:v for k,v in job.items() if k!='output'})
    if (output/'identity.json').exists() and read(output/'identity.json')['hash']!=identity:
        raise ValueError('NICE shard resume identity mismatch')
    write(output/'identity.json',{'hash':identity})
    set_seed(job['seed'])
    tokenizer=AutoTokenizer.from_pretrained(job['model_path'],local_files_only=True)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model,_=build(job);model.to('cuda:0')
    parameters=tuple((n,p) for n,p in model.named_parameters() if p.requires_grad)
    coordinates=digest([(n,list(p.shape),str(p.dtype)) for n,p in parameters])
    before=parameter_hash(parameters)
    tokenizer_identity=digest({'tokenizer':tokenizer.backend_tokenizer.to_str(),'template':tokenizer.chat_template,'thinking':False})
    write(output/'coordinates.json',{'hash':coordinates,'initial_adapter_hash':before,'model_hash':job['model_hash'],
        'tokenizer_identity':tokenizer_identity,'parameters':sum(p.numel() for _,p in parameters)})
    results=[];started=time.monotonic()
    event(output,'worker_session_start',mode=job['mode'],model_load_seconds=started-stage_started)
    if job['mode']=='gradient':
        model.train()
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
        if job['training']['gradient_checkpointing']:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:model.eval()
    for index,record in enumerate(rows(job['records'])):
        file=output/(digest(record['id'])+'.json')
        record_hash=digest(record)
        if file.exists():
            saved=read(file)
            if saved['identity_hash']!=identity or saved['record_hash']!=record_hash:raise ValueError('Cached NICE record changed')
            if saved.get('vectors') and sha(saved['vectors'])!=saved['vectors_sha256']:raise ValueError('NICE vectors changed')
            results.append(saved);continue
        tick=time.monotonic()
        if job['mode']=='rollout':
            settings=job['rollout']
            prompt=NonThinkingTokenizer(tokenizer).apply_chat_template([{'role':'user','content':record['prompt']}],
                tokenize=True,add_generation_prompt=True)
            if len(prompt)+settings['max_new_tokens']>settings['max_model_len']:raise ValueError('Rollout context overflow')
            torch.manual_seed(record['seed']);torch.cuda.manual_seed_all(record['seed'])
            x=torch.tensor([prompt],device='cuda:0',dtype=torch.long)
            kwargs={'do_sample':settings['temperature']>0,'max_new_tokens':settings['max_new_tokens'],
                    'pad_token_id':tokenizer.pad_token_id,'use_cache':True,'return_dict_in_generate':False,
                    'output_scores':False,'repetition_penalty':1.0,'typical_p':1.0}
            if kwargs['do_sample']:kwargs.update(temperature=settings['temperature'],top_p=settings['top_p'],top_k=settings['top_k'])
            with torch.inference_mode():sequence=model.generate(input_ids=x,attention_mask=torch.ones_like(x),**kwargs)
            generated=sequence[0,len(prompt):].tolist()
            if not generated:raise ValueError('Empty NICE rollout')
            value={'id':record['id'],'task_id':record['id'],'parent_task_id':record['parent_task_id'],
                'answers':[tokenizer.decode(generated,skip_special_tokens=True)],'input_ids':prompt+generated,
                'labels':[-100]*len(prompt)+generated,'prompt_token_ids':prompt,'generated_token_ids':generated,
                'usage':[{'input_tokens':len(prompt),'output_tokens':len(generated)}],
                'stop_reason':'length' if len(generated)==settings['max_new_tokens'] else 'eos',
                'tokenizer_identity':tokenizer_identity,'model_hash':job['model_hash'],
                'initial_adapter_hash':before,'proxy_test':False}
        else:
            if record['kind']=='candidate':
                encoded=encode_example(record['record'],tokenizer,job['training']['max_length'])
                reduction='mean'
            else:
                encoded=record['replay'];reduction='sum'
                if encoded['model_hash']!=job['model_hash'] or encoded['tokenizer_identity']!=tokenizer_identity or encoded['initial_adapter_hash']!=before:
                    raise ValueError('Replay model/tokenizer mismatch')
                if encoded['input_ids']!=encoded['prompt_token_ids']+encoded['generated_token_ids'] or encoded['labels']!=[-100]*len(encoded['prompt_token_ids'])+encoded['generated_token_ids']:
                    raise ValueError('Actual generated token replay was changed')
            model.zero_grad(set_to_none=True)
            loss=token_loss(model,encoded['input_ids'],encoded['labels'],reduction,'cuda:0')
            loss.backward()
            cfg=job['score']
            raw,adam,norm=projected_gradients(parameters,cfg['projection_dimension'],cfg['projection_seed'],cfg['adam_epsilon'],cfg['max_grad_norm'])
            if record['kind']=='target' and record['weight']!=1:raw*=record['weight']
            vector=output/(digest(record['id'])+'.npz')
            np.savez_compressed(vector,raw=raw,adam=adam if record['kind']=='candidate' else np.zeros_like(adam))
            value={'id':record['id'],'kind':record['kind'],'parent_task_id':record.get('parent_task_id'),
                'vectors':str(vector.resolve()),'vectors_sha256':sha(vector),'grad_norm':norm,
                'input_tokens':len(encoded['input_ids']),'supervised_tokens':sum(v!=-100 for v in encoded['labels'][1:]),
                'loss':float(loss.detach()),'reduction':reduction,'coordinates':coordinates,
                'initial_adapter_hash':before,'optimizer_step_called':False}
        value.update(identity_hash=identity,record_hash=record_hash,wall_seconds=time.monotonic()-tick,synthetic=False)
        write(file,value);results.append(value)
        event(output,'nice_record',id=record['id'],mode=job['mode'],index=index+1,
              elapsed_seconds=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated())
    after=parameter_hash(parameters)
    if before!=after:raise ValueError('NICE changed adapter weights')
    write(output/'complete.json',{'identity_hash':identity,'records':results,'parameter_coordinates':coordinates,
          'initial_adapter_hash':before,'after_adapter_hash':after,'optimizer_step_called':False,
          'wall_seconds':time.monotonic()-started,'model_load_seconds':started-stage_started,
          'record_wall_seconds_including_cached':sum(r.get('wall_seconds',0) for r in results),
          'gpu_peak_bytes':torch.cuda.max_memory_allocated()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);run(p.parse_args().job)

if __name__=='__main__':main()
