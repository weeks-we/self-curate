"""Validate executable configuration against supported ordinary/NICE code paths."""
import math


def integer(value,label,minimum=1):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError('Invalid '+label)


def positive(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('Invalid '+label)


def ordinary(cfg):
    if type(cfg.get('approved')) is not bool:raise ValueError('approved must be a JSON boolean')
    t=cfg['training'];e=cfg['evaluation'];h=cfg['hardware']
    for key in ('epochs','world_size','micro_batch_size','gradient_accumulation_steps','global_batch_size',
                'max_length','lora_rank','lora_alpha','logging_steps','pad_to_multiple_of'):
        integer(t[key],'training.'+key)
    integer(t['dataloader_num_workers'],'dataloader workers',0)
    if t['world_size'] not in (4,8) or h['gpu_count']!=8:raise ValueError('Host needs eight GPUs; training groups support four/eight')
    ids=h['gpu_uuids']
    if cfg['approved'] and (len(ids)!=8 or len(set(ids))!=8 or not all(isinstance(v,str) and v.startswith('GPU-') for v in ids)):
        raise ValueError('Approved real config needs eight distinct GPU UUIDs')
    runtime=cfg['runtime']
    if runtime['execution']!='server' or runtime['offline'] is not True or runtime['automatic_parameter_changes'] is not False or runtime['require_server_smoke'] is not True:
        raise ValueError('Server/offline/frozen-parameter/preflight settings are required')
    if t['method']!='lora' or t['fsdp']!='native_transformers_fsdp2' or t['bf16'] is not True:
        raise ValueError('Supported training is BF16 LoRA with native FSDP2')
    if t['packing'] is not False or t['length_grouping'] is not True:
        raise ValueError('Packing/ungrouped sampling are not implemented in this experiment')
    if t['save_strategy']!='epoch' or t['save_only_model'] is not False or t['save_total_limit']<3:
        raise ValueError('Keep all three complete native epoch checkpoints')
    if t['max_length']%t['pad_to_multiple_of']:
        raise ValueError('Max length must align with padding multiple')
    if t['lora_rank']>64:raise ValueError('Current vLLM adapter service caps rank at64')
    if t['global_batch_size']!=t['world_size']*t['micro_batch_size']*t['gradient_accumulation_steps']:
        raise ValueError('Global batch differs from world times micro times accumulation')
    positive(t['learning_rate'],'default learning rate')
    rates=t.get('learning_rates') or [t['learning_rate']]
    for rate in rates:positive(rate,'shared learning rate')
    if len(set(rates))!=len(rates):raise ValueError('Duplicate learning rates')
    if cfg.get('nice',{}).get('enabled') and len(rates)!=2:
        raise ValueError('Integrated NICE requires exactly two shared learning rates')
    for key in ('replicas','tensor_parallel_size','max_num_seqs_per_replica','max_model_len','max_tokens','math_max_tokens'):
        integer(e[key],'evaluation.'+key)
    if e['replicas']!=8 or e['tensor_parallel_size']!=1:raise ValueError('Use eight single-GPU replicas')
    if e['temperature']!=0 or e['top_p']!=1 or e['thinking'] is not False:
        raise ValueError('Greedy non-thinking final evaluation is frozen')
    if max(e['max_tokens'],e['math_max_tokens'])>=e['max_model_len']:
        raise ValueError('Generation budget must leave input context')
    from .evaluation import resolve_protocol
    if e.get('benchmark_budgets'):
        expected={b for values in e['benchmarks'].values() for b in values}
        if set(e['benchmark_budgets'])!=expected:raise ValueError('Benchmark budget registry is incomplete or unknown')
        for benchmark in expected:resolve_protocol(e,benchmark)
    if not 0<e['gpu_memory_utilization']<1:raise ValueError('Invalid vLLM memory fraction')
    if cfg.get('nice',{}).get('enabled') and cfg['nice'].get('order')!='before_ordinary_baseline_training':
        raise ValueError('Integrated NICE executes before expensive baseline training')
    pipeline=runtime.get('pipeline',{})
    if not pipeline.get('enabled') and t['world_size']!=8:
        raise ValueError('Four-GPU training requires the explicit two-lane scheduler')
    if pipeline.get('enabled'):
        if pipeline['lanes']!=2 or pipeline['gpus_per_lane']!=4 or t['world_size']!=4:
            raise ValueError('EXP001 pipeline uses two four-GPU groups')
        integer(pipeline['cpu_grading_workers'],'CPU grader workers')
    return cfg


def nice(main,cfg):
    if main.get('model')!='qwen9b' or cfg['model']!='qwen9b' or cfg['domain']!='coding':
        raise ValueError('NICE is Qwen9B coding only')
    if cfg['selection']!='bottom_and_top_no_middle' or cfg['learning_rates_from']!='exp001.training.learning_rates':
        raise ValueError('NICE selection/LR source mismatch')
    if cfg['seed']!=main['seed']:raise ValueError('NICE and main sampling seed differ')
    t={**main['training'],**cfg['training_overrides']};s=cfg['score'];r=cfg['rollout'];e=cfg['evaluation'];b=cfg['gradient_smoke']
    integer(t['lora_rank'],'NICE LoRA rank');integer(t['lora_alpha'],'NICE LoRA alpha')
    if t['lora_rank']>64 or t['world_size'] not in (4,8) or t['bf16'] is not True or t['packing'] is not False or t['length_grouping'] is not True:
        raise ValueError('Unsupported NICE adapter/distributed/data configuration')
    if t['epochs']!=1 or t['global_batch_size']!=8 or t['micro_batch_size']*t['world_size']*t['gradient_accumulation_steps']!=8:
        raise ValueError('NICE one epoch/effective batch8 mismatch')
    fractions=[1/3,2/3,1.0]
    if cfg['checkpoint_fractions']!=fractions or t['save_at_fractions']!=fractions:
        raise ValueError('NICE three progress schedules differ')
    if t['save_strategy']!='no' or t['save_only_model'] is not False or t['save_total_limit']<3:
        raise ValueError('NICE uses custom native progress saves, not epoch saves')
    if t['loss_reduction']!='sample_mean' or t['lora_dropout']!=0:
        raise ValueError('NICE requires sample mean and zero dropout')
    if s['adam_epsilon']!=t['adam_epsilon'] or s['max_grad_norm']!=t['max_grad_norm']:
        raise ValueError('NICE fresh-Adam eps/clip differ from target SFT')
    positive(s['adam_epsilon'],'NICE epsilon');positive(s['max_grad_norm'],'NICE clipping')
    integer(s['projection_dimension'],'projection dimension');integer(s['projection_seed'],'projection seed',0)
    if s['projection_seed']>=2**64:raise ValueError('Projection seed exceeds uint64')
    if s['optimizer_step'] is not False or s['candidate_reduction']!='assistant_token_mean' or s['target_reduction']!='generated_token_sum_reward_divided_by_4':
        raise ValueError('NICE gradient formula mismatch')
    if s['zero_target']!='keep_zero_in_fixed_20_task_denominator':raise ValueError('Fixed20 denominator required')
    if r['engine']!='local_huggingface_exact_token_ids' or r['thinking'] is not False or (r['temperature'],r['top_p'],r['top_k'])!=(1.,1.,0):
        raise ValueError('NICE exact HF rollout sampling protocol mismatch')
    integer(r['max_new_tokens'],'NICE output budget');integer(r['max_model_len'],'NICE context')
    if t['max_length']+r['max_new_tokens']>r['max_model_len']:raise ValueError('NICE context cannot cover admitted input plus output')
    if e['benchmarks']!=main['evaluation']['benchmarks']['coding'] or e['checkpoints']!='all_three_progress_points' or e['task_set']!='same_20_development_tasks_diagnostic_only' or e['checkpoint_selection']!='none_report_all_final_primary':
        raise ValueError('NICE evaluation scope disagrees with implemented workflow')
    if b['train'] is not False or b['evaluate'] is not False or b['reward']!='unit_weight_for_backward_probe_not_task_success' or b['rollouts_per_task']!=1:
        raise ValueError('B must remain a unit-weight backward-only probe')
    for key in ('tasks','candidates','max_new_tokens'):integer(b[key],'gradient_smoke.'+key)
    if b['tasks']>20 or b['candidates']>3000 or b['max_new_tokens']>r['max_new_tokens']:
        raise ValueError('Gradient smoke exceeds formal bounds')
    return cfg
