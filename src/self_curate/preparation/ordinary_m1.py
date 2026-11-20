"""Bridge ordinary examples to existing M1/operator/signal contracts without evidence spans."""
from dataclasses import replace
from self_curate.contracts import ArtifactRef,DataLabels,DataTag
from self_curate.data.normalize import normalize_trajectory
from self_curate.data.features import attach_metadata_record,make_metadata_record
from self_curate.experiments.ordinary.common import digest
from self_curate.data.task_links import task_identity,task_tags


def to_m1(example,tokenization_backend,*,declared_skills=()):
    source=example['source']
    descriptor=source.get('source',source)
    source_id=descriptor.get('repo',descriptor.get('source_id','ordinary-local-preview'))
    revision=descriptor.get('revision','preview')
    raw={'schema_version':'1','data_id':example['id'],'messages':example['messages'],
         'source':{'source_id':source_id,'revision':revision,'record_id':example['id'],
                   'family_ids':[example['question_hash']],
                   'artifact':ArtifactRef('ordinary://raw/'+example['content_hash'],example['content_hash']).to_dict()},
         'quality':{'status':'pass','method':'executable','verifier_id':'ordinary-static-admission-v1',
                    'evidence':ArtifactRef('ordinary://static-admission/'+digest(example),digest(example)).to_dict()}}
    record=normalize_trajectory(raw,tokenization_backend,tuple(declared_skills))
    present=set(example['skills'])
    tags=[DataTag('skill_label_basis','task_required'),DataTag('skill_label_version',example['label_version']),DataTag('label_source',example['label_status']),
          DataTag('question_hash',example['question_hash']),
          DataTag('fresh_solution_execution','not_performed')]
    tags.append(DataTag('evaluation_eligible',str(example.get('metadata',{}).get('evaluation_eligible',True)).lower()))
    tags.append(DataTag('annotation_provenance_hash',digest(example.get('label_provenance',{}))))
    tags.extend(task_tags(task_identity({'instance_id':example['id'],'question_hash':example['question_hash']},
                                       source_id=source_id,revision=revision)))
    tags.extend(DataTag('skill_status:'+skill,'present' if skill in present else 'not_assessed')
                for skill in sorted(present|set(declared_skills)))
    record=attach_metadata_record(record,make_metadata_record(record,DataLabels(task_type=example['domain'],tags=tuple(tags)),
                           provenance=(ArtifactRef('ordinary://task-labels/'+digest(example['raw_labels']),digest(example['raw_labels'])),)))
    return replace(record,normalization_version='m1-v4')
