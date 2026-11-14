"""Runs inside an isolated prepared task container, never on a host workspace."""
import argparse
import asyncio
import os
from pathlib import Path
from self_curate.training.recovery import read_json,atomic_json
from .claude_sdk import gated_session,TOOLS

async def run(cfg):
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    # Config is transported privately; credentials must not appear in evidence.
    os.environ['SELF_CURATE_WORKER_KEY']=cfg.pop('key')
    binding={**cfg['binding'],'api_key_env':'SELF_CURATE_WORKER_KEY'}
    try:
        result=await gated_session(cfg['prompt'],binding,cfg['harness'],output,cfg['gate'],
            cwd=cfg['cwd'],tools=TOOLS,max_turns=cfg['max_turns'],
            tool_seconds=cfg['tool_seconds'])
        atomic_json(output/'worker-result.json',result)
    finally:
        os.environ.pop('SELF_CURATE_WORKER_KEY',None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True)
    args=parser.parse_args();path=Path(args.input)
    cfg=read_json(path);path.unlink()
    asyncio.run(run(cfg))

if __name__=='__main__':main()
