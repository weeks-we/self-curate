"""HF Trainer + Accelerate FSDP/PEFT, response-only supervision, native checkpoint resume."""
import os
from pathlib import Path
import time
from self_curate.experiments.ordinary.common import digest, read, rows, write, sha, event


def encode_example(record, tokenizer, max_length):
    messages = record['messages']
    if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
        raise ValueError('Expected a single user/assistant pair')
    from self_curate.preparation.ordinary_tokenization import renderer
    from self_curate.contracts.parsing import message
    encoded=renderer(tokenizer).encode(tuple(message(m) for m in messages))
    full=list(encoded.input_ids)
    if max_length is None or len(full) > max_length:
        raise ValueError('over_length')
    labels=[token if mask else -100 for token,mask in zip(full,encoded.assistant_mask)]
    if not any(x!=-100 for x in labels):
        raise ValueError('empty_supervision')
    return {'input_ids':full,'attention_mask':[1]*len(full),'labels':labels}


def train(job, *, resume=False):
    import torch
    from datasets import load_dataset
    from accelerate import PartialState
    from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText,
                              Trainer, TrainingArguments, TrainerCallback, set_seed)
    from peft import LoraConfig, get_peft_model
    from transformers.trainer_utils import get_last_checkpoint
    from self_curate.training.recovery import lease
    job = read(job) if not isinstance(job, dict) else job
    if job.get('execution') != 'server' or not job.get('approved'):
        raise ValueError('Formal training requires approved parameters and execution=server')
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE','1')) != 8:
        raise ValueError('Use torchrun with eight explicitly authorized GPUs; no CPU fallback')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible) != 8 or visible != job['gpu_uuids'] or not all(x.startswith('GPU-') for x in visible):
        raise ValueError('GPU UUID authorization mismatch')
    cfg, output = job['training'], Path(job['output'])
    if job['loader']=='AutoModelForImageTextToText' and cfg.get('require_optimized_linear_kernels',True):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn
        if not callable(chunk_gated_delta_rule) or not callable(causal_conv1d_fn):
            raise ValueError('Optimized Qwen3.5 linear-attention kernels are unavailable')
    if sha(job['data']) != job['data_sha256']:
        raise ValueError('Selected training data changed')
    effective = cfg['micro_batch_size']*cfg['gradient_accumulation_steps']*8
    if effective != cfg['global_batch_size']:
        raise ValueError('micro × accumulation × world != global batch')
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK','0'))
    # Rank zero owns publication; HF/Accelerate owns distributed barriers/state.
    identity = {k:v for k,v in job.items() if k != 'output'}
    if rank == 0:
        if (output/'job.json').exists() and read(output/'job.json') != identity:
            raise ValueError('Resume model/data/config identity mismatch')
        write(output/'job.json',identity)
    checkpoint = get_last_checkpoint(str(output)) if resume else None
    if resume and checkpoint is None:
        raise ValueError('No HF training checkpoint to resume; never silently restart')
    if not resume and any(output.glob('checkpoint-*')):
        raise ValueError('Existing checkpoints require --resume')
    set_seed(job['seed'])
    tokenizer = AutoTokenizer.from_pretrained(job['model_path'],local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # Memory-mapped Arrow cache: do not materialize a million-tokenized-record
    # Python list on EACH of eight ranks. Rank zero prepares once, others reuse.
    state=PartialState()
    cache=output/'tokenized';cache.mkdir(exist_ok=True)
    def encode(record):
        value=encode_example(record,tokenizer,cfg['max_length'])
        return {**value,'length':len(value['input_ids']),
                'supervised_tokens':sum(x!=-100 for x in value['labels'])}
    with state.main_process_first():
        raw=load_dataset('json',data_files=job['data'],split='train',cache_dir=str(cache/'raw'))
        encoded=raw.map(encode,remove_columns=raw.column_names,
             cache_file_name=str((cache/'encoded.arrow').resolve()),
             num_proc=cfg['dataloader_num_workers'],load_from_cache_file=True,
             desc='Encode frozen ordinary subset')
    model_cls = AutoModelForImageTextToText if job['loader']=='AutoModelForImageTextToText' else AutoModelForCausalLM
    model = model_cls.from_pretrained(job['model_path'],local_files_only=True,dtype=torch.bfloat16)
    model.config.use_cache = False
    # Explicit text-linear targets; exclude vision/projector modules from multimodal Qwen.
    suffixes=set(cfg['lora_target_modules'])
    targets=[name for name,module in model.named_modules()
             if isinstance(module,torch.nn.Linear) and name.rsplit('.',1)[-1] in suffixes
             and not any(p in name.lower() for p in ('visual','vision','projector'))]
    if not targets:
        raise ValueError('No compatible LoRA targets in pinned model architecture')
    model = get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],
                  lora_dropout=cfg['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    if rank==0:write(output/'lora-targets.json',{'names':targets,'count':len(targets),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    if cfg['gradient_checkpointing']:
        model.enable_input_require_grads()
    class Collator:
        def __call__(self, batch):
            size=max(len(v['input_ids']) for v in batch)
            multiple=cfg.get('pad_to_multiple_of',8)
            size=((size+multiple-1)//multiple)*multiple
            return {key:torch.tensor([v[key]+[(-100 if key=='labels' else tokenizer.pad_token_id if key=='input_ids' else 0)]*(size-len(v[key]))
                                      for v in batch],dtype=torch.long) for key in ('input_ids','attention_mask','labels')}
    supervised=sum(encoded['supervised_tokens'])
    previous=read(Path(checkpoint)/'training-cost.json') if checkpoint else {}
    if checkpoint and not previous:
        raise ValueError('Resume requires checkpoint accounting receipt')
    # Device counters are reduced only at publication: no per-microbatch .item() sync.
    counts=torch.zeros(4,dtype=torch.int64,device=state.device)
    session_id=__import__('uuid').uuid4().hex[:12]
    receipt_base={k:int(previous.get(k,0)) for k in
                  ('sample_exposures','input_tokens','supervised_tokens','micro_batches')}
    previous_elapsed=float(previous.get('train_wall_seconds',0))
    class AccountedTrainer(Trainer):
        def training_step(self, model, inputs, *args, **kwargs):
            # The first label cannot be predicted by causal shift; count loss targets.
            loss=super().training_step(model,inputs,*args,**kwargs)
            counts[0]+=inputs['input_ids'].shape[0]
            counts[1]+=inputs['attention_mask'].sum().to(counts.device)
            counts[2]+=(inputs['labels'][...,1:]!=-100).sum().to(counts.device)
            counts[3]+=1
            return loss
    class Logging(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            totals=counts.clone()
            torch.distributed.all_reduce(totals,op=torch.distributed.ReduceOp.SUM)
            elapsed=torch.tensor(time.monotonic()-started,device=counts.device,dtype=torch.float64)
            torch.distributed.all_reduce(elapsed,op=torch.distributed.ReduceOp.MAX)
            if state.is_world_process_zero:
                values={k:receipt_base[k]+int(v) for k,v in zip(receipt_base,totals.tolist())}
                value={**values,'schema':'checkpoint-training-cost-v1','synthetic':False,
                    'session_id':session_id,'epoch':state.epoch,'global_step':state.global_step,
                    'unique_examples':len(encoded),'train_wall_seconds':previous_elapsed+elapsed.item(),
                    'gpu_count':8,'gpu_hours':8*(previous_elapsed+elapsed.item())/3600,
                    'native_input_tokens_seen':state.num_input_tokens_seen,
                    'counting':'successful consumed training microbatches across all ranks; includes sampler repeats',
                    'wall_scope':'trainer.train plus native epoch checkpoint IO; excludes model/tokenizer startup',
                    'replayed_or_lost_work_before_resume_not_in_checkpoint':bool(checkpoint),
                    'monetary_cost':None,'resume_checkpoint':checkpoint}
                write(output/f'checkpoint-{state.global_step}'/'training-cost.json',value)
                event(output,'checkpoint_cost',**value)
            torch.distributed.barrier()

        def on_log(self,args,state,control,logs=None,**kwargs):
            event(output/'ranks'/str(rank),'consumed_work',session_id=session_id,step=state.global_step,
                  rank_session_counts=dict(zip(receipt_base,counts.tolist())),
                  elapsed_seconds=time.monotonic()-started,resume_checkpoint=checkpoint)
            event(output/'ranks'/str(rank),'gpu_log',step=state.global_step,
                  allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                  peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
            if state.is_world_process_zero:
                event(output,'train_log',step=state.global_step,epoch=state.epoch,metrics=logs or {},
                      gpu_peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started)
    args=TrainingArguments(output_dir=str(output),num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],
         per_device_train_batch_size=cfg['micro_batch_size'],gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
         bf16=True,gradient_checkpointing=False,
         fsdp=True,fsdp_config={'version':2,'reshard_after_forward':cfg.get('fsdp_reshard_after_forward',False),
              'auto_wrap_policy':'TRANSFORMER_BASED_WRAP','state_dict_type':'FULL_STATE_DICT',
              'cpu_ram_efficient_loading':False,'activation_checkpointing':cfg['gradient_checkpointing']},
         save_strategy=cfg['save_strategy'],save_total_limit=cfg['save_total_limit'],save_only_model=False,
         optim=cfg.get('optimizer','adamw_torch_fused'),dataloader_pin_memory=True,
         dataloader_persistent_workers=cfg['dataloader_num_workers']>0,
         logging_steps=cfg['logging_steps'],report_to='none',seed=job['seed'],data_seed=job['seed'],
         warmup_steps=cfg['warmup_ratio'],weight_decay=cfg['weight_decay'],dataloader_drop_last=False,
         train_sampling_strategy='group_by_length',include_num_input_tokens_seen='non_padding',
         length_column_name='length',
         dataloader_num_workers=cfg['dataloader_num_workers'],remove_unused_columns=False)
    started=time.monotonic()
    trainer=AccountedTrainer(model=model,args=args,train_dataset=encoded,data_collator=Collator(),callbacks=[Logging()])
    event(output/'ranks'/str(rank),'train_session_start',session_id=session_id,resume_checkpoint=checkpoint)
    try:
        result=trainer.train(resume_from_checkpoint=checkpoint)
    finally:
        event(output/'ranks'/str(rank),'train_session_exit',session_id=session_id,
              rank_session_counts=dict(zip(receipt_base,counts.tolist())),elapsed_seconds=time.monotonic()-started)

    trainer.save_model(str(output/'final'))
    trainer.save_state()
    if rank == 0:
        tokenizer.save_pretrained(output/'final')
        from self_curate.experiments.ordinary.epoch_checkpoints import collect
        checkpoints=collect(output,int(cfg['epochs']))
        write(output/'complete.json',{'epochs':cfg['epochs'],'epoch_checkpoints':checkpoints,'job_hash':digest(identity),'examples':len(encoded),
              'supervised_tokens_per_epoch':supervised,'input_tokens_per_epoch':sum(encoded['length']),
              'elapsed_seconds':time.monotonic()-started,'metrics':result.metrics,'adapter':str((output/'final').resolve()),
              'resume_checkpoint':checkpoint,'global_batch_size':effective})


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    train(args.job,resume=args.resume)


if __name__ == '__main__':
    main()
