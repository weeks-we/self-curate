"""Server-only native compaction diagnostic; never a research task."""
import asyncio,json,os,time,uuid
from pathlib import Path
from dataclasses import asdict,is_dataclass
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json
from .claude_sdk import options_for,settings
from .claude_gateway import Gateway
from .context_acceptance import verify_compaction

async def run_probe(binding, harness, gate, output, *, execution, pressure_prompt, minimum_long_input):
    require(execution=='server','Native compaction requires explicit server execution')
    require(isinstance(pressure_prompt,str) and pressure_prompt,'Full pressure prompt required')
    from claude_agent_sdk import ClaudeSDKClient
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=settings(harness);key=os.environ.get(binding.get('api_key_env',''),'')
    require(bool(key),'Scoped inference key required')
    gateway=Gateway(('127.0.0.1',0),binding.get('sdk_upstream',binding['base_url']),gate,upstream_key=key).start()
    token,budget=gateway.register(binding['model_name'],output/'inference-usage.json')
    marker=uuid.uuid4().hex;events=[];turns=0;session_ids=set()
    atomic_json(output/'intent.json',{'diagnostic_only':True,'pressure_prompt':pressure_prompt,
        'minimum_long_input':minimum_long_input,'marker':marker,'gate':gate})
    env={**os.environ,'ANTHROPIC_BASE_URL':'http://127.0.0.1:'+str(gateway.server.server_port),
        'ANTHROPIC_API_KEY':token,'CLAUDE_CONFIG_DIR':str(output/'claude-home'),
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW':str(gate['generation']['context_length']),
        'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE':'80'}
    options=options_for(binding['model_name'],output,env,cfg,tools=('Bash',),max_turns=250)
    async def conversation():
        nonlocal turns
        async with ClaudeSDKClient(options=options) as client:
            for prompt in ['Reply briefly with READY.','Remember this diagnostic marker: '+marker,
                           pressure_prompt,'Use Bash to print '+marker+' and then finish.']:
                await client.query(prompt)
                terminal=False
                async for message in client.receive_response():
                    row=asdict(message) if is_dataclass(message) else vars(message)
                    event={'sdk_type':type(message).__name__,'data':row,'monotonic':time.monotonic()}
                    events.append(event)
                    with (output/'sdk-events.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(event)+'\n')
                    if row.get('session_id'):session_ids.add(row['session_id'])
                    if event['sdk_type']=='ResultMessage':
                        terminal=True;turns+=row.get('num_turns',0)
                        require(not row.get('is_error') and turns<=250,'Diagnostic SDK failure/turn limit')
                require(terminal,'Missing diagnostic terminal result')
    try:
        await asyncio.wait_for(conversation(),timeout=gate['agent']['wall_seconds'])
        settled=await asyncio.to_thread(budget.snapshot)
        require(len(session_ids)==1 and not settled['incomplete_usage'],'Session changed or usage incomplete')
        result=verify_compaction(events,settled['records'],minimum_long_input=minimum_long_input)
        result.update(diagnostic_only=True,sdk_session_id=next(iter(session_ids)))
        atomic_json(output/'diagnostic-result.json',result)
        return result
    finally:
        budget.cancel.set();gateway.close()
