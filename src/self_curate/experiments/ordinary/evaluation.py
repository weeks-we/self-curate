"""Concurrent direct generation: no agent harness, isolated answers and resumable per-task files."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time
from threading import BoundedSemaphore
from .common import digest, read, write, rows, sha, event


def generate(tasks_file,output,*,endpoints,model_id,model_hash,protocol,api_key_env='VLLM_API_KEY',proxy_test=False):
    from openai import OpenAI
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    values=list(rows(tasks_file))
    if not values or not endpoints:
        raise ValueError('Generation requires nonempty tasks and endpoints')
    batch_started=time.monotonic()
    identity={'tasks_sha256':sha(tasks_file),'model_hash':model_hash,'model_id':model_id,
              'protocol':protocol,'proxy_test':proxy_test}
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity:
        raise ValueError('Generation resume identity mismatch')
    write(output/'identity.json',identity)
    max_tokens=protocol['math_max_tokens'] if values[0]['benchmark'] in ('math500','aime25','aime26','gsm_plus','olympiad') else protocol['max_tokens']
    clients=[OpenAI(base_url=url,api_key=os.environ.get(api_key_env,'local-unused'),max_retries=0,timeout=300) for url in endpoints]
    slots=[BoundedSemaphore(protocol['max_num_seqs_per_replica']) for _ in endpoints]
    def one(item):
        index,task=item
        path=output/'predictions'/(digest(task['id'])+'.json')
        if path.exists():
            value=read(path)
            if value['identity_hash']!=digest(identity) or value['task_id']!=task['id']:
                raise ValueError('Cached prediction mismatch')
            return value
        started=time.monotonic()
        messages=[]; answers=[]; usage=[]
        turns=task['turns'] or [{'role':'user','content':task['prompt']}]
        for turn_index,turn in enumerate(turns):
            messages.append(turn)
            body={'thinking':{'type':'disabled'}} if proxy_test else ({'chat_template_kwargs':{'enable_thinking':False}} if protocol.get('model_family','qwen3_5')=='qwen3_5' else {})
            turn_file=output/'turns'/(digest(task['id'])+'-'+str(turn_index)+'.json')
            if turn_file.exists():
                cached=read(turn_file)
                if cached['identity_hash']!=digest(identity):
                    raise ValueError('Turn identity mismatch')
                if cached['status']!='completed':
                    raise ValueError('Uncertain in-flight generation; reconcile explicitly before retrying')
                answer=cached['answer'];one_usage=cached['usage']
            else:
                write(turn_file,{'status':'pending','identity_hash':digest(identity),'task_id':task['id']})
                with slots[index%len(clients)]:
                    response=clients[index%len(clients)].chat.completions.create(
                        model=model_id,messages=messages,temperature=0,top_p=1,max_tokens=max_tokens,extra_body=body)
                answer=response.choices[0].message.content or ''
                one_usage={'input_tokens':response.usage.prompt_tokens,'output_tokens':response.usage.completion_tokens,
                           'finish_reason':response.choices[0].finish_reason}
                write(turn_file,{'status':'completed','identity_hash':digest(identity),'task_id':task['id'],
                                 'answer':answer,'usage':one_usage})
            answers.append(answer)
            usage.append(one_usage)
            messages.append({'role':'assistant','content':answer})
        result={'task_id':task['id'],'identity_hash':digest(identity),'answers':answers,'usage':usage,
                'wall_seconds':time.monotonic()-started,'model_hash':model_hash,'proxy_test':proxy_test}
        write(path,result)
        return result
    # A worker slot maps deterministically to a replica. Actual engine concurrency is bounded separately.
    with ThreadPoolExecutor(max_workers=len(clients)*protocol['max_num_seqs_per_replica']) as executor:
        futures=[executor.submit(one,item) for item in enumerate(values)]
        results=[]
        for index,future in enumerate(as_completed(futures),1):
            results.append(future.result())
            if index%25==0 or index==len(values):
                print(f'  generation {index}/{len(values)}',flush=True)
    write(output/'generation.json',{'identity':identity,'count':len(results),
          'input_tokens':sum(u['input_tokens'] for r in results for u in r['usage']),
          'output_tokens':sum(u['output_tokens'] for r in results for u in r['usage']),
          'summed_request_seconds':sum(r['wall_seconds'] for r in results),
          'wall_seconds':time.monotonic()-batch_started,
          'truncated_answers':sum(u['finish_reason']=='length' for r in results for u in r['usage'])})
    return output
