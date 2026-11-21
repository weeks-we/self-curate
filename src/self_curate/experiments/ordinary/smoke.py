"""CPU acceptance entrypoint. All data/model replies are explicitly synthetic."""
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .common import read,write,jsonl,sha,digest
from .workflow import plan,execute,development
from .backend import MockBackend


def smoke(output):
    root=Path(output)
    if root.exists():raise FileExistsError('Use a fresh smoke directory')
    root.mkdir(parents=True)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    pools={}
    for domain in ('coding','math','if'):
        folder=root/domain;folder.mkdir()
        values=[{'id':domain+str(i),'question_hash':digest([domain,i]),'skills':['a','b'],
            'messages':[{'role':'user','content':'Synthetic '+str(i)},
                        {'role':'assistant','content':'Synthetic answer'}]} for i in range(100)]
        jsonl(folder/'pool.jsonl',values)
        write(folder/'manifest.json',{'pool_sha256':sha(folder/'pool.jsonl'),'preview':True,'tokenized':False})
        pools[domain]=str(folder)
    cfg=read(Path(__file__).parents[4]/'configs/ordinary/exp001.json')
    cfg=copy.deepcopy(cfg);cfg['nice']['enabled']=False;cfg['training']['learning_rates']=[cfg['training']['learning_rate']]
    value=plan(cfg,pools,root/'plan',synthetic=True)
    check('eighteen independent branches',len(value['jobs'])==18)
    for domain in pools:
        jobs={j['method']:set(j['ids']) for j in value['jobs'] if j['domain']==domain}
        check(domain+' nested random fractions',jobs['random05']<jobs['random10']<jobs['random20']<jobs['random30']<jobs['full'])
    class Interrupted(MockBackend):
        calls=0
        def evaluate(self,*args,**kwargs):
            self.calls+=1
            if self.calls==10:raise RuntimeError('injected interruption')
            return super().evaluate(*args,**kwargs)
    try:execute(root/'plan/plan.json',Interrupted())
    except RuntimeError as exc:
        if str(exc)!='injected interruption':raise
    else:raise AssertionError('Interruption was not exercised')
    result=execute(root/'plan/plan.json',MockBackend(),resume=True)
    check('all branches recovered',read(result/'complete.json')['jobs']==18)
    check('all 192 checkpoint-benchmark results',read(result/'complete.json')['evaluation_results']==192)
    state=read(result/'state.json')
    for job in value['jobs']:
        completed=state['completed'][job['job_id']]
        expected=1 if job['method']=='initial' else len(value['config']['evaluation']['epochs'])
        check(job['job_id']+' checkpoint isolation',len(completed['evaluations'])==expected*len(job['benchmarks']))
    before=sha(result/'complete.json')
    execute(root/'plan/plan.json',MockBackend(),resume=True)
    check('completed run idempotent',before==sha(result/'complete.json'))
    try:development([{'id':str(i),'skills':['a']} for i in range(49)],50,42)
    except ValueError:checks.append('skill shortage fails explicitly')
    else:raise AssertionError('Shortage silently duplicated tasks')
    import csv
    table=list(csv.DictReader((result/'summary.csv').open(encoding='utf-8')))
    check('192 score/cost rows',len(table)==192)
    check('missing mock costs are blank',all(not r['train_input_tokens'] for r in table if r['checkpoint']!='initial'))
    check('45 checkpoint training rows',len(list(csv.DictReader((result/'training_costs.csv').open(encoding='utf-8'))))==45)
    check('report tables published',(result/'REPORT.md').exists() and (result/'scores_wide.csv').exists())
    _roles(root,checks)
    receipt={'passed':True,'checks':checks,'count':len(checks),'synthetic':True,
        'gpu_verified':False,'real_weights_verified':False,'official_graders_verified':False}
    write(root/'acceptance.json',receipt)
    return receipt


def _roles(root,checks):
    from self_curate.agents.smol_client import SmolSharedRoleClient,bind_role_tool
    from self_curate.agents.runtime import RoleRuntime
    from self_curate.smoke.mock import initial_model
    calls={r:0 for r in ('lexec','leval','lopt')}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            observed=any(m['role'] not in ('system','assistant') and 'Observation:' in json.dumps(m['content']) for m in body['messages'])
            name='final_answer' if observed else next(t['function']['name'] for t in body['tools'] if t['function']['name']!='final_answer')
            arguments={'answer':{'ok':True}} if observed else {'request':'{}'}
            data=json.dumps({'id':'synthetic','object':'chat.completion','created':0,'model':body['model'],
                'choices':[{'index':0,'finish_reason':'tool_calls','message':{'role':'assistant','content':None,
                'tool_calls':[{'id':'call1','type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}]}}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    def callback(role):
        def observe(request):calls[role]+=1;return {'role':role}
        return observe
    try:
        model=initial_model()
        tools={r:[bind_role_tool('observe_'+r,'Read the synthetic role summary.',callback(r))] for r in calls}
        client=SmolSharedRoleClient(model,endpoint=f'http://127.0.0.1:{server.server_port}/v1',model_id='synthetic',output=root/'roles',proxy_test=True,tools_by_role=tools,max_steps=3)
        runtime=RoleRuntime(client,model=model,session_id='cpu-smoke',backend='mock')
        for role in calls:
            reply=runtime.generate(role,{'instruction':'Observe then return JSON.'},timeout_seconds=20)
            if calls[role]!=1 or reply.content!={'ok':True} or reply.usage.agent_input_tokens!=20:
                raise AssertionError('Tool calling / cost mismatch: '+role)
            if len(runtime.history(role))!=2:raise AssertionError('Role history isolation failed')
            checks.append(role+' actual ToolCallingAgent / isolated history / token accounting')
    finally:server.shutdown();server.server_close()
