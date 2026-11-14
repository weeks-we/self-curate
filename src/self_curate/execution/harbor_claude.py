"""Harbor lifecycle/verifier with the same SDK harness used by SWE tasks.

No online installation. The task image must include the isolated SDK runtime.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,os,shlex
from harbor.agents.base import BaseAgent
from self_curate.contracts.types import require
from self_curate.execution.claude_sdk import settings,SDK_VERSION,CLI_VERSION

class ClaudeSDKAgent(BaseAgent):
    def __init__(self,*args,harness=None,gate=None,max_turns=75,tool_seconds=300,
                 base_url=None,model_api_key_name='SELF_CURATE_TRIAL_KEY',
                 capture_patch=False,**kwargs):
        self.harness=settings(harness);self.gate=gate;self.turns=max_turns
        self.tool_seconds=tool_seconds;self.endpoint=base_url;self.key_name=model_api_key_name
        self.capture_patch=capture_patch
        super().__init__(*args,**kwargs)

    @staticmethod
    def name():return 'self-curate-claude-sdk'

    def version(self):return SDK_VERSION+'+claude-'+CLI_VERSION

    async def setup(self,environment):
        cmd=shlex.join([self.harness['python'],'-c',
            "import importlib.metadata as m; import self_curate.execution.claude_worker; "
            "assert m.version('claude-agent-sdk') == '"+SDK_VERSION+"'"])
        row=await environment.exec(command=cmd,timeout_sec=30)
        require(row.return_code==0,'Prepared isolated SDK runtime missing; workstation rebuild required')

    async def run(self,instruction,environment,context):
        require(self.gate is not None and self.endpoint,'Explicit task budgets and scoped endpoint required')
        key=os.environ.get(self.key_name)
        require(bool(key),'Scoped model credential missing')
        logs=str(self.environment_logs_dir)
        cwd_result=await environment.exec(command='pwd',timeout_sec=10)
        require(cwd_result.return_code==0,'Cannot resolve native task workspace')
        cwd=cwd_result.stdout.strip()
        worker={'prompt':instruction,'binding':{'base_url':self.endpoint.rstrip('/'),
            'model_name':self.model_name},'key':key,'harness':self.harness,
            'gate':self.gate,'output':logs+'/sdk','cwd':cwd,
            'max_turns':self.turns,'tool_seconds':self.tool_seconds}
        # Credentials only in a short-lived private file, never command/evidence JSON.
        with TemporaryDirectory(prefix='self-curate-sdk-') as folder:
            path=Path(folder)/'input.json';path.write_text(json.dumps(worker),encoding='utf-8')
            path.chmod(0o600)
            await environment.upload_file(source_path=path,target_path='/tmp/self-curate-sdk.json')
        command=shlex.join([self.harness['python'],'-m','self_curate.execution.claude_worker',
                            '--input','/tmp/self-curate-sdk.json'])
        result=await environment.exec(command=command,
            timeout_sec=self.gate['agent']['wall_seconds']+30)
        # Logs mirror to host through Harbor; retrieve only the small usage receipt.
        usage=await environment.exec(command='cat '+shlex.quote(logs+'/sdk/inference-usage.json'),timeout_sec=10)
        require(usage.return_code==0,'SDK omitted usage receipt')
        usage=json.loads(usage.stdout)
        context.n_input_tokens=usage['input_tokens']
        context.n_output_tokens=usage['output_tokens']
        require(not usage.get('incomplete_usage'),'Incomplete inference usage; reconcile before retry')
        require(result.return_code==0,'SDK worker failed; preserve native logs')
        if self.capture_patch:
            result=await environment.exec(command='git add -N . && git diff --binary > '+
                shlex.quote(logs+'/model.patch'),timeout_sec=20)
            require(result.return_code==0,'Pro SDK patch capture failed')
