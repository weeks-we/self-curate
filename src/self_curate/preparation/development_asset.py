"""Immutable, reusable selected-task snapshots; no task environments or inference."""
from pathlib import Path
from self_curate.contracts import digest
from self_curate.contracts.types import require
from self_curate.execution.artifacts import publish_json,read_artifact
from self_curate.contracts.parsing import artifact
from self_curate.training.recovery import atomic_json,read_json,commit_directory,verify_directory
from self_curate.data.pool import file_sha256

def publish(result,settings,source_identity,labels_root,output):
    output=Path(output)
    identity=digest({'selection':result,'ontology':digest(settings),'source':source_identity})
    if (output/'COMMITTED.json').exists():
        verify_directory(output);require(read_json(output/'identity.json')['identity']==identity,
                                          'Frozen development asset changed')
    else:
        output.mkdir(parents=True,exist_ok=True)
        for key in ('identity.json','manifest.json'):
            if (output/key).exists():
                require(read_json(output/'identity.json')['identity']==identity,'Partial task asset identity changed')
        atomic_json(output/'identity.json',{'identity':identity})
        atomic_json(output/'selection.json',result)
        atomic_json(output/'ontology.json',settings['ontology'])
        public=[{k:t.get(k) for k in ('instance_id','repo','base_commit','problem_statement')}
                for t in result['tasks']]
        atomic_json(output/'solver-tasks.json',public)
        # Full source task rows (including reference materials) remain preparation/scorer-only.
        atomic_json(output/'private-reference-tasks.json',result['tasks'])
        labels={}
        for task in result['tasks']:
            path=Path(labels_root)/(digest(task['instance_id'])+'.json')
            require(path.is_file(),'Selected task lacks Teacher label evidence')
            labels[task['instance_id']]=read_json(path)
        atomic_json(output/'task-labels.json',labels)
        manifest={'schema_version':'development-asset-v1','identity':identity,
            'ontology_hash':digest(settings),'ontology_version':settings['version'],
            'source':source_identity,'task_ids':[t['instance_id'] for t in result['tasks']],
            'groups':result['groups'],'per_skill':result['per_skill'],
            'private_reference_materials_must_not_enter_solver':True,
            'files':{name:file_sha256(output/name) for name in (
                'identity.json','selection.json','ontology.json','solver-tasks.json',
                'private-reference-tasks.json','task-labels.json')}}
        atomic_json(output/'manifest.json',manifest);commit_directory(output)
    path=output/'manifest.json'
    return {'uri':path.resolve().as_uri(),'sha256':file_sha256(path)}

def load(reference,*,expected_ontology_hash=None,expected_source=None):
    from self_curate.execution.artifacts import local_path
    ref=artifact(reference) if isinstance(reference,dict) else reference
    path=local_path(ref);verify_directory(path.parent)
    value=read_artifact(ref)
    require(value['schema_version']=='development-asset-v1','Unknown development asset schema')
    if expected_ontology_hash is not None:
        require(value['ontology_hash']==expected_ontology_hash,'Development ontology changed')
    if expected_source is not None:require(value['source']==expected_source,'Development source changed')
    for name,sha in value['files'].items():
        p=(path.parent/name).resolve()
        require(p.is_relative_to(path.parent.resolve()) and file_sha256(p)==sha,'Development asset changed')
    result=read_json(path.parent/'selection.json')
    require(value['task_ids']==[t['instance_id'] for t in result['tasks']],'Task asset IDs mismatch')
    return result
