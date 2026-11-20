"""Factory preserving RoleRuntime and all existing role-adapter contracts."""
from pathlib import Path
from .runtime import RoleRuntime
from .prompts import load_prompts
from .smol_client import SmolSharedRoleClient


def build_runtime(model_ref,*,endpoint,model_id,output,session_id,identity_check,
                  tools_by_role,prompts_directory=None,proxy_test=False):
    prompts_directory=prompts_directory or Path(__file__).parents[3]/'configs/ordinary/prompts'
    client=SmolSharedRoleClient(model_ref,endpoint=endpoint,model_id=model_id,output=output,
        identity_check=identity_check,tools_by_role=tools_by_role,proxy_test=proxy_test)
    return RoleRuntime(client,model=model_ref,session_id=session_id,
        prompts=load_prompts(prompts_directory),backend='mock' if proxy_test else 'real')
