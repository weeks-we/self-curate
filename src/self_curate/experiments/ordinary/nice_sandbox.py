"""NICE reuses the ordinary CPU grader mounts and vLLM benchmark generation."""
from pathlib import Path
import subprocess
import time
from .common import read,write,sha,rows


def grade(backend,tasks,predictions,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    bindings=backend.bindings
    grader=bindings.get('nice_grader_root')
    if not grader:raise ValueError('Bindings need nice_grader_root for fixed TACO checker')
    def inside(path):
        path=Path(path).resolve()
        for directory,prefix in ((bindings['storage'],'/assets'),(bindings['run_root'],'/results'),(bindings['project'],'/app')):
            try:return prefix+'/'+path.relative_to(Path(directory).resolve()).as_posix()
            except ValueError:pass
        raise ValueError('NICE artifact outside explicit sandbox mounts')
    result=root/'scores.json'
    identity={'tasks_sha256':sha(tasks),'predictions':{p.name:sha(p) for p in sorted(Path(predictions).glob('*.json'))},
              'grader_root':grader,'grader_image':bindings['grader_image']}
    if (root/'identity.json').exists() and read(root/'identity.json')!=identity:raise ValueError('NICE grade resume identity mismatch')
    write(root/'identity.json',identity)
    if result.exists():
        if not (root/'cost.json').exists():write(root/'cost.json',{'wall_seconds':None,'scope':'timing receipt lost after successful grading; not reconstructed'})
        return {'path':str(result.resolve()),'sha256':sha(result)}
    command=['docker','run','--rm','--network','none','--read-only','--pids-limit','1024',
             '--memory','48g','--cpus','8','--tmpfs','/tmp:rw,size=8g','--tmpfs','/root/.cache:rw,size=2g',
             '-e','SELF_CURATE_GRADER_SANDBOX=1','-e','HF_HUB_OFFLINE=1','-e','PYTHONPATH=/app/src']
    for directory,prefix in ((bindings['storage'],'/assets'),(bindings['run_root'],'/results'),(bindings['project'],'/app')):
        command+=['--mount',f'type=bind,src={Path(directory).resolve()},dst={prefix},readonly']
    command+=['--mount',f'type=bind,src={root.resolve()},dst={inside(root)}',bindings['grader_image'],
              'python','-m','self_curate.experiments.ordinary.grade','--benchmark','taco_dev',
              '--tasks',inside(tasks),'--predictions',inside(predictions),'--output',inside(result),'--grader-root',inside(grader)]
    tick=time.monotonic()
    from .grader_process import run as run_grader
    run_grader(command,root/'grader.log',cancel=getattr(backend,'cancel_event',None))
    write(root/'cost.json',{'wall_seconds':time.monotonic()-tick,'scope':'official TACO CPU grading'})
    return {'path':str(result.resolve()),'sha256':sha(result)}


def evaluate_tasks(backend,cfg,model,tasks,output):
    return finish_tasks(prepare_tasks(backend,cfg,model,tasks,output))


def prepare_tasks(backend,cfg,model,tasks,output):
    from .evaluation import generate
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    backend.serve(model,cfg,root)
    tick=time.monotonic()
    generation=generate(tasks,root/'generation',endpoints=backend.endpoints(cfg),
        model_id='selected' if model['adapter'] else 'target-qwen9b',model_hash=model['identity'],protocol=cfg['evaluation'],tokenizer=backend.generation_tokenizer(model))
    return (backend,model,tasks,root,generation,tick)


def finish_tasks(ticket):
    backend,model,tasks,root,generation,tick=ticket
    result=grade(backend,tasks,generation/'predictions',root/'grading')
    usage=read(generation/'generation.json');cost=root/'evaluation-cost.json'
    write(cost,{'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens'],
               'generation_wall_seconds':usage['wall_seconds'],'evaluation_wall_seconds':time.monotonic()-tick,
               'grading_wall_seconds':read(root/'grading/cost.json')['wall_seconds'],
               'same_task_diagnostic':True,'model_identity':model['identity']})
    return {**result,'cost_path':str(cost.resolve()),'cost_sha256':sha(cost)}
