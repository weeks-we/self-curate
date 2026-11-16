"""Shared weights, separate SDK sessions; role tools remain deterministic services."""
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import asyncio,json,time
from self_curate.contracts import ResourceUsage,ComputeCost
from self_curate.contracts.types import require
from self_curate.execution.claude_sdk import gated_session,settings

class ClaudeSharedRoleClient:
    def __init__(self,router,model,output,seed,*,harness=None,checkpoint=None,session_fn=gated_session):
        self.router,self.model_ref,self.output,self.seed=router,model,Path(output),seed
        self.harness=settings(harness);self.session_fn=session_fn;self.checkpoint=checkpoint

    def generate_json(self,*,role,system_prompt,payload,timeout_seconds):
        require(role in ('lopt','lexec','leval','Lopt','Lexec','Leval'), 'Unknown isolated role')
        started=time.perf_counter()
        work=self.output/(role+'-'+uuid4().hex);work.mkdir(parents=True,exist_ok=False)
        # Only this role's explicit payload/history is supplied. Never SDK resume/fork.
        with self.router.bind(SimpleNamespace(model_ref=self.model_ref,checkpoint=self.checkpoint),work) as binding:
            cfg={'agent':{'wall_seconds':timeout_seconds,'total_output_tokens':8192},
                'generation':{'context_length':self.harness.get('role_context_tokens',131072),
                    'max_new_tokens':8192,'temperature':0,'top_p':1,'top_k':0,'seed':self.seed,
                    'thinking':True,'preserve_thinking':True,'reasoning_effort':'xhigh'}}
            row=asyncio.run(self.session_fn(json.dumps(payload,ensure_ascii=False),binding,
                self.harness,work,cfg,cwd=work,system_prompt=system_prompt,
                tools=(),max_turns=1))
        value=(row['result'] or {}).get('result')
        # Strict JSON only; do not silently strip fences or salvage partial text.
        from self_curate.contracts.parsing import load_json
        try:
            require(row['exit_status']=='completed','Role SDK budget exhausted; no valid proposal')
            content=load_json(value)
            require(isinstance(content,dict),'Role must return a JSON object')
        except Exception as exc:
            exc.usage=ResourceUsage(ComputeCost(time.perf_counter()-started,None,None),
                agent_input_tokens=sum(c['input_tokens'] for c in row['calls']),
                agent_output_tokens=sum(c['output_tokens'] for c in row['calls']))
            raise
        return {'content':content,'usage':ResourceUsage(
            ComputeCost(time.perf_counter()-started,None,None),
            agent_input_tokens=sum(c['input_tokens'] for c in row['calls']),
            agent_output_tokens=sum(c['output_tokens'] for c in row['calls'])).to_dict()}
