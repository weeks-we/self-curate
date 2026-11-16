"""Pinned Claude Agent SDK harness. Model identity belongs to AdapterRouter.

Roles use fresh, tool-free sessions; task tools run only in task containers.
Neither SDK events nor re-tokenized text are native NICE replay evidence.
"""
import asyncio
from contextlib import aclosing
from dataclasses import asdict, is_dataclass
from pathlib import Path
from urllib.parse import urlsplit
import importlib.metadata
import json
import os
import subprocess
import time
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json

SDK_VERSION = '0.2.87'
CLI_VERSION = '2.1.153'
TOOLS = ['Bash', 'Read', 'Edit', 'Write', 'Glob', 'Grep']
DEFAULTS = {'backend':'claude_agent_sdk', 'sdk_version':SDK_VERSION,
    'cli_version':CLI_VERSION, 'cli_path':'/opt/self-curate-sdk/bin/claude-pinned',
    'python':'/opt/self-curate-sdk/bin/python', 'context_management':'claude_native'}

def settings(value=None):
    cfg={**DEFAULTS, **(value or {})}
    require(cfg['backend']=='claude_agent_sdk' and cfg['sdk_version']==SDK_VERSION
            and cfg['cli_version']==CLI_VERSION, 'Pinned Claude harness required')
    require(cfg['context_management']=='claude_native', 'Do not combine native compaction with observation clipping')
    require(str(cfg['cli_path']).startswith('/'), 'Explicit absolute Claude runtime required')
    return cfg

def options_for(model, cwd, env, config, *, system_prompt=None, tools=(), max_turns=1):
    from claude_agent_sdk import ClaudeAgentOptions
    cfg=settings(config)
    require(isinstance(model,str) and model.startswith('sc-') and '/' not in model
            and not any(c.isspace() for c in model), 'Use the router-owned adapter alias')
    url=urlsplit(env.get('ANTHROPIC_BASE_URL',''))
    require(url.scheme in ('http','https') and url.hostname and not url.username
            and not url.password and not url.query and not url.fragment
            and url.path in ('','/'), 'Explicit inference root URL required')
    require(bool(env.get('ANTHROPIC_API_KEY')), 'Scoped inference credential required')
    require(type(max_turns) is int and max_turns>0, 'Positive max_turns required')
    require(set(tools)<=set(TOOLS), 'Unsupported Claude tool')
    # An allow-list prevents inherited OAuth, cloud-provider or user settings.
    allowed={'PATH','HOME','USER','LANG','LC_ALL','TMPDIR','SSL_CERT_FILE','SSL_CERT_DIR',
             'ANTHROPIC_BASE_URL','ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN','CLAUDE_CONFIG_DIR',
             'CLAUDE_CODE_AUTO_COMPACT_WINDOW','CLAUDE_AUTOCOMPACT_PCT_OVERRIDE'}
    clean={k:v for k,v in env.items() if k in allowed}
    clean.update(ANTHROPIC_MODEL=model, CLAUDE_CODE_SUBAGENT_MODEL=model,
        ANTHROPIC_AUTH_TOKEN=env['ANTHROPIC_API_KEY'], DISABLE_AUTOUPDATER='1',
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1',
        CLAUDE_CODE_USE_BEDROCK='0', CLAUDE_CODE_USE_VERTEX='0', CLAUDE_CODE_USE_FOUNDRY='0',
        CLAUDE_CODE_OAUTH_TOKEN='', CLAUDE_FORCE_OAUTH='0',
        **{'ANTHROPIC_DEFAULT_'+tier+'_MODEL':model for tier in ('OPUS','SONNET','HAIKU')})
    return ClaudeAgentOptions(model=model, cwd=str(cwd), cli_path=cfg['cli_path'],
        system_prompt=(system_prompt if system_prompt is not None else
                       {'type':'preset','preset':'claude_code'}),
        tools=list(tools), allowed_tools=list(tools),
        disallowed_tools=['Task','Agent','AskUserQuestion','WebSearch','WebFetch'],
        permission_mode='bypassPermissions' if tools else 'dontAsk',
        max_turns=max_turns, setting_sources=[], mcp_servers={}, strict_mcp_config=True,
        env=clean, max_buffer_size=16*1024*1024)

async def session(prompt, model, cwd, env, config, output, *, system_prompt=None,
                  tools=(), max_turns=1, wall_seconds=600, tool_seconds=300,
                  query_fn=None, verify_versions=True):
    cfg=settings(config)
    require(wall_seconds>0 and tool_seconds>0, 'Positive harness deadlines required')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if verify_versions:
        require(importlib.metadata.version('claude-agent-sdk')==SDK_VERSION,'SDK version changed')
        version=subprocess.check_output([cfg['cli_path'],'--version'],text=True,timeout=30).split()[0]
        require(version==CLI_VERSION,'Claude executable version changed')
    if query_fn is None:
        from claude_agent_sdk import query as query_fn
    opts=options_for(model,cwd,env,cfg,system_prompt=system_prompt,tools=tools,max_turns=max_turns)
    from claude_agent_sdk import HookMatcher
    async def supervise(data, tool_use_id, context):
        if data.get('hook_event_name')=='PreToolUse' and data.get('tool_name')=='Bash':
            updated=dict(data.get('tool_input',{}))
            updated['timeout']=min(updated.get('timeout') or tool_seconds*1000,tool_seconds*1000)
            updated['run_in_background']=False
            return {'hookSpecificOutput':{'hookEventName':'PreToolUse','updatedInput':updated}}
        return {}
    opts.hooks={'PreToolUse':[HookMatcher(hooks=[supervise])]}
    started=time.perf_counter(); final=None; actions=[]; observations=[]
    with (output/'sdk-events.jsonl').open('x',encoding='utf-8') as log:
        async def consume():
            nonlocal final
            async with aclosing(query_fn(prompt=prompt,options=opts)) as messages:
                async for message in messages:
                    name=type(message).__name__
                    row=asdict(message) if is_dataclass(message) else vars(message)
                    log.write(json.dumps({'sdk_type':name,'data':row},ensure_ascii=False)+'\n');log.flush()
                    if name=='AssistantMessage':
                        require(not row.get('error'),'Assistant API error')
                        actions.append({'role':'assistant',**row})
                    elif name=='UserMessage': observations.append(row)
                    elif name=='ResultMessage':
                        require(final is None,'Duplicate SDK result');final=row
        try:
            await asyncio.wait_for(consume(),timeout=wall_seconds)
            require(final is not None,'SDK omitted terminal result')
            stop='step_limit' if final.get('subtype')=='error_max_turns' else 'completed'
            require(stop=='step_limit' or (not final.get('is_error') and final.get('subtype')=='success'),
                    'SDK infrastructure/API failure: '+str(final.get('subtype')))
        except asyncio.TimeoutError:
            stop='wall_time_limit'
        except BaseException as exc:
            atomic_json(output/'sdk-failure.json',{'type':type(exc).__name__,
                'wall_seconds':time.perf_counter()-started,'usage_unknown':True})
            raise
    result={'result':final,'exit_status':stop,'actions':actions,'observations':observations,
        'wall_seconds':time.perf_counter()-started,'sdk_version':SDK_VERSION,'cli_version':CLI_VERSION,
        'native_gradient_masks_available':False}
    atomic_json(output/'sdk-result.json',result)
    return result

def policy_config(budget, policy, *, seed=0, temperature=0, wall_seconds=None):
    wall_seconds=budget.max_wall_seconds if wall_seconds is None else wall_seconds
    require(type(wall_seconds) in (int,float) and 0<wall_seconds<=budget.max_wall_seconds,"Invalid effective agent deadline")
    return {'agent':{'wall_seconds':wall_seconds,
                     'total_output_tokens':budget.max_output_tokens},
        'generation':{'context_length':budget.max_context_tokens,
            'max_new_tokens':policy['per_call_output_tokens'], 'temperature':policy.get('temperature',policy.get('sampling_temperature',temperature)),
            'seed':seed,
            'top_p':policy.get('top_p',1.0),'top_k':policy.get('top_k',0),
            'thinking':policy.get('thinking',True),'preserve_thinking':policy.get('preserve_thinking',True),
            'min_p':policy.get('min_p',0.0),'presence_penalty':policy.get('presence_penalty',0.0),
            'repetition_penalty':policy.get('repetition_penalty',1.0),
            'reasoning_effort':policy.get('reasoning_effort','xhigh')}}

async def gated_session(prompt, binding, config, output, gate_config, *, cwd,
                        system_prompt=None, tools=(), max_turns=1, tool_seconds=300):
    """Private per-session gate enforces real budgets, route and accounts requests."""
    from .claude_gateway import Gateway
    from .relay import trial_grant
    from contextlib import nullcontext
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    upstream=binding.get('sdk_upstream',binding['base_url']).rstrip('/')
    grant=(trial_grant(binding['relay_grant_root'],binding['model_name'],
            timeout=gate_config['agent']['wall_seconds']) if binding.get('use_scoped_relay') else
            nullcontext(os.environ.get(binding.get('api_key_env',''),'')))
    with grant as key:
        require(bool(key),'No inference credential; cloud fallback prohibited')
        gateway=Gateway(('127.0.0.1',0),upstream,gate_config,upstream_key=key).start()
        token,budget=gateway.register(binding['model_name'],output/'inference-usage.json')
        budget.persist()
        try:
            env={**os.environ,'ANTHROPIC_BASE_URL':'http://127.0.0.1:'+str(gateway.server.server_port),
                 'ANTHROPIC_API_KEY':token,'CLAUDE_CONFIG_DIR':str(output/'claude-home'),
                 'CLAUDE_CODE_AUTO_COMPACT_WINDOW':str(gate_config['generation']['context_length']),
                 'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE':'80'}
            try:
                row=await session(prompt,binding['model_name'],cwd,env,config,output,
                    system_prompt=system_prompt,tools=tools,max_turns=max_turns,
                    wall_seconds=gate_config['agent']['wall_seconds'],tool_seconds=tool_seconds)
            except Exception:
                if not budget.stop_reason or not budget.stop_reason.endswith('LIMIT') or budget.incomplete_usage:
                    raise
                # A hard budget stop is an episode result. Preserve any available patch.
                row={'result':None,'actions':[],'observations':[],
                     'exit_status':budget.stop_reason.lower(),'native_gradient_masks_available':False}
            settled=await asyncio.to_thread(budget.snapshot)
            require(settled['requests']>0 or settled['stop_reason'] is not None,
                    'SDK returned without accounted inference work')
            row['calls']=settled['records']
            row['usage_complete']=not settled['incomplete_usage']
            if settled['stop_reason'] and settled['stop_reason'].endswith('LIMIT'):
                row['exit_status']=settled['stop_reason'].lower()
            require(not settled['incomplete_usage'],'Unknown/in-flight inference usage; reconcile before retry')
            return row
        finally:
            budget.cancel.set();gateway.close()


def native_agent_seconds(task_path, budget, policy):
    """Harbor phase timeouts own agent time; TaskBudget wall is an outer watchdog."""
    import tomllib
    native=tomllib.loads((Path(task_path)/'task.toml').read_text(encoding='utf-8'))
    value=native.get('agent',{}).get('timeout_sec')
    if type(value) not in (int,float):
        from harbor.models.task.config import TaskConfig
        value=TaskConfig.model_validate(native).agent.timeout_sec
    cap=policy.get('agent_wall_seconds',value)
    require(type(value) in (int,float) and value>0 and type(cap) in (int,float) and cap>0,
            'Invalid native agent timeout')
    effective=min(value,cap)
    require(effective<=budget.max_wall_seconds,'Outer watchdog shortens native agent phase')
    return effective
