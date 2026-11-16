"""Version-bound engine-side token evidence, before text/tool parsing.

Start via python -m self_curate.execution.vllm_capture <api_server arguments>.
This hook observes tokens; it does not reconstruct them from SDK text.
"""
from pathlib import Path
from functools import wraps
import hashlib,importlib.metadata,os,runpy,time,json
from contextvars import ContextVar
REQUEST_SEED=ContextVar("self_curate_seed",default=None)
REQUEST_GENERATION=ContextVar("self_curate_generation",default=None)

class SeedMiddleware:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http":
            return await self.app(scope,receive,send)
        value=dict(scope.get("headers",())).get(b"x-self-curate-seed")
        try:
            seed=None if value is None else int(value)
            if seed is not None and not 0<=seed<2**63:raise ValueError('seed outside int64')
        except (ValueError,TypeError):
            await send({'type':'http.response.start','status':400,
                        'headers':[(b'content-type',b'application/json')]})
            await send({'type':'http.response.body','body':b'{"error":"invalid request seed"}'})
            return
        raw=dict(scope.get("headers",())).get(b"x-self-curate-generation")
        try:
            policy=None if raw is None else json.loads(raw)
            if policy is not None:
                require(isinstance(policy,dict) and policy.get("thinking") is True and policy.get("preserve_thinking") is True,"Invalid trusted generation policy")
        except (ValueError,TypeError):
            await send({"type":"http.response.start","status":400,"headers":[]})
            await send({"type":"http.response.body","body":b"invalid generation policy"});return
        token=REQUEST_SEED.set(seed);generation=REQUEST_GENERATION.set(policy)
        try:await self.app(scope,receive,send)
        finally:
            REQUEST_SEED.reset(token);REQUEST_GENERATION.reset(generation)
from self_curate.training.recovery import atomic_json
from self_curate.contracts.types import require

def capture_path(root,request_id):
    return Path(root)/(hashlib.sha256(request_id.encode()).hexdigest()+'.json')

def install_capture(engine_class,root):
    original=engine_class.generate
    require(not getattr(original,'_self_curate_capture',False),'Token capture already installed')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if os.name!='nt':root.chmod(0o700)
    @wraps(original)
    async def generate(self,prompt,sampling_params,request_id,*args,**kwargs):
        started=time.time();prompt_ids=None;tokens={};complete=False
        seed=REQUEST_SEED.get()
        if seed is None:seed=getattr(sampling_params,"seed",None)
        require(type(seed) is int and 0<=seed<2**63,'Missing fixed request seed; use SDK gateway/middleware')
        sampling_params.seed=seed
        delta=getattr(getattr(sampling_params,'output_kind',None),'name',None)=='DELTA'
        lora=kwargs.get('lora_request')
        try:
            async for result in original(self,prompt,sampling_params,request_id,*args,**kwargs):
                if result.prompt_token_ids is not None:
                    incoming=list(result.prompt_token_ids)
                    require(prompt_ids is None or prompt_ids==incoming,'Engine input tokens changed mid-request')
                    prompt_ids=incoming
                for output in result.outputs:
                    ids=list(output.token_ids)
                    if delta:tokens.setdefault(output.index,[]).extend(ids)
                    else:tokens[output.index]=ids
                if result.finished:
                    require(prompt_ids is not None and set(tokens)=={0},
                            'Capture requires one text generation with actual prompt IDs')
                    path=capture_path(root,request_id)
                    require(not path.exists(),'Engine request ID reused')
                    atomic_json(path,{'schema_version':'vllm-native-capture-v1','request_id':request_id,
                        'model_alias':getattr(lora,'lora_name',None),'prompt_token_ids':prompt_ids,
                        'generated_token_ids':tokens[0],'complete':True,'started_at':started,
                        'seed':seed,'effective_generation':REQUEST_GENERATION.get(),
                        'effective_sampling':{k:getattr(sampling_params,k,None) for k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty')},'vllm_version':'0.22.0','token_origin':'engine_request_output'})
                    complete=True
                yield result
        finally:
            if not complete:
                atomic_json(capture_path(root,request_id+'.incomplete'),{
                    'request_id':request_id,'complete':False,'usage_unknown':True})
    generate._self_curate_capture=True
    engine_class.generate=generate


def install_anthropic_generation(serving_class):
    """Set real ChatCompletionRequest fields before Anthropic conversion drops extras."""
    original=serving_class._build_base_request.__func__
    require(not getattr(original,'_self_curate_generation',False),'Generation policy installed twice')
    def convert(cls,request,messages):
        result=original(cls,request,messages)
        policy=REQUEST_GENERATION.get()
        if policy is not None:
            result.chat_template_kwargs={'enable_thinking':policy['thinking'],
                'preserve_thinking':policy['preserve_thinking'],'reasoning_effort':policy['reasoning_effort']}
            for field in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty'):
                setattr(result,field,policy.get(field,{'min_p':0.,'presence_penalty':0.,'repetition_penalty':1.}.get(field)))
        return result
    convert._self_curate_generation=True
    serving_class._build_base_request=classmethod(convert)

def main():
    require(importlib.metadata.version('vllm')=='0.22.0','Native capture requires pinned vLLM 0.22.0')
    root=os.environ.get('SELF_CURATE_NATIVE_CAPTURE_ROOT')
    require(bool(root),'Set a private native token evidence directory')
    from vllm.v1.engine.async_llm import AsyncLLM
    # -m executes this file as __main__; middleware imports its canonical name.
    # Share the same ContextVar rather than installing a second module instance.
    module=__import__('importlib').import_module('self_curate.execution.vllm_capture')
    module.install_capture(AsyncLLM,root)
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    module.install_anthropic_generation(AnthropicServingMessages)
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')

if __name__=='__main__':main()
