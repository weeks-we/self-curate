"""Shared deterministic text LoRA construction for ordinary SFT and NICE gradients."""
def build(job):
    import torch
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText
    from peft import LoraConfig,get_peft_model
    cfg=job['training']
    if job['loader']=='AutoModelForImageTextToText' and cfg.get('require_optimized_linear_kernels',True):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn
        if not callable(chunk_gated_delta_rule) or not callable(causal_conv1d_fn):
            raise ValueError('Optimized Qwen3.5 kernels unavailable')
    cls=AutoModelForImageTextToText if job['loader']=='AutoModelForImageTextToText' else AutoModelForCausalLM
    model=cls.from_pretrained(job['model_path'],local_files_only=True,dtype=torch.bfloat16)
    model.config.use_cache=False
    suffixes=set(cfg['lora_target_modules'])
    targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
             and n.rsplit('.',1)[-1] in suffixes and not any(x in n.lower() for x in ('visual','vision','projector'))]
    if not targets:raise ValueError('No compatible LoRA text targets')
    model=get_peft_model(model,LoraConfig(r=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    return model,targets
