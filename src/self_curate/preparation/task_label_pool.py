"""Task-required labels inherited by trajectories; no observed-skill evidence claims."""
from pathlib import Path
from dataclasses import replace
from self_curate.contracts import digest,canonical_json
from self_curate.contracts.parsing import message
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json,read_json
from self_curate.execution.artifacts import publish_json
from self_curate.data.pool import file_sha256
from .teacher import task_packet,annotate,validate_task


def label_tasks(tasks,settings,client,output,review=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={}
    for task in sorted(tasks,key=lambda t:t['instance_id']):
        packet=task_packet(task,settings)
        identity=digest({'packet':packet,'prompt':settings['task_prompt'],
            'mode':settings.get('review_mode'),'teacher':{k:client.settings.get(k) for k in ('model','base_url','thinking','reasoning_effort','max_output_tokens')}})
        path=output/(digest(task['instance_id'])+'.json')
        if path.exists():
            row=read_json(path)
            require(row['identity']==identity,'Task label identity changed; use a new preparation root')
            response=Path(row['response'])
            require(file_sha256(response)==row['response_sha256'],'Task label response changed')
            validate_task(row['annotation'],packet,settings)
        else:
            value,response,provenance=annotate('task',packet,settings,client,review)
            row={'identity':identity,'instance_id':task['instance_id'],'task_hash':digest(task),
                 'packet':packet,'annotation':value,'response':str(Path(response).resolve()),
                 'response_sha256':file_sha256(response),'provenance':provenance,
                 'label_basis':'task_required','trajectory_evidence_assessed':False}
            atomic_json(path,row)
        result[task['instance_id']]=(row,path)
    return result


def prepare_task_label_pool(rows,tasks,renderer,settings,client,output,*,held_out,max_context,
                            review_client=None,record_sink=None):
    from .rebench import success_budget,source_messages
    from .rendering import canonical_messages
    from .isolation import family_ids
    from self_curate.experiments.exp001.workstation import light_gate
    from self_curate.data import normalize_trajectory
    from self_curate.contracts.features import DataLabels,DataTag
    from self_curate.data.features import make_metadata_record,attach_metadata_record
    from self_curate.data.task_links import task_identity,task_tags,validate_task_join
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=lambda:rows() if callable(rows) else iter(rows)
    budget=success_budget(rows);require(budget['selected_count']>=3,'Insufficient success budget')
    lookup={t['instance_id']:t for t in tasks};require(len(lookup)==len(tasks),'Duplicate task IDs')
    successful={x['instance_id'] for x in source() if type(x.get('resolved')) in (int,bool) and x['resolved']==1}
    require(successful<=set(lookup),'Successful trajectory/task join incomplete')
    labelled=label_tasks([lookup[k] for k in successful],settings,client,
                         settings.get('task_labels_root',output/'task-labels'),review_client)
    def count(messages,tools):
        encoded=renderer.encode(tuple(message(m) for m in canonical_messages(messages)))
        return len(encoded.input_ids),sum(encoded.assistant_mask)
    records=[];audit=[];decisions=[];state={};retained=0;eligible=0
    for raw in source():
        task=lookup.get(raw.get('instance_id'))
        if raw.get('instance_id') in successful:validate_task_join(raw,task)
        excluded={raw.get('instance_id')} if task is not None and held_out.conflicts(task) else set()
        try:messages=source_messages(raw)
        except (ValueError,TypeError,KeyError):messages=None
        passed,checks=light_gate([{**raw,'trajectory':messages}],excluded,count,max_context,
            fingerprint_func=lambda m,t:digest({'messages':[message(v).to_dict() for v in canonical_messages(m)]}),state=state)
        audit.extend(checks)
        if not passed:continue
        eligible+=1
        row,label_path=labelled[raw['instance_id']];value=row['annotation']
        no_skill=(value['classification_state']=='uncovered' and not value['missing_context']
                  and not value['task_risks'] and value['confidence']!='low'
                  and all(x['status']=='not_required' for x in value['skills']))
        decision={'trajectory_id':raw['trajectory_id'],'instance_id':raw['instance_id'],
                  'decision':'exclude_task_uncovered' if no_skill else 'retain',
                  'label_basis':'task_required','task_label_hash':digest(row)}
        decisions.append(decision)
        if no_skill:continue
        item=output/'items'/digest(raw['trajectory_id']);item.mkdir(parents=True,exist_ok=True)
        original=publish_json(item/'raw.json',raw)
        proof=publish_json(item/'task-label.json',row)
        claim=publish_json(item/'admission-evidence.json',{'source_resolved':1,
             'light_static_checks':'pass','fresh_execution':'not_performed','raw_record_hash':digest(raw)})
        candidate={'schema_version':'1','data_id':raw['trajectory_id'],'messages':canonical_messages(messages),
          'source':{'source_id':settings['source_id'],'revision':settings['source_revision'],
                    'record_id':raw['instance_id'],'artifact':original.to_dict(),'family_ids':list(family_ids(task))},
          'quality':{'status':'pass','method':'executable','verifier_id':'source-resolved-and-light-static-v1','evidence':claim.to_dict()}}
        record=normalize_trajectory(candidate,renderer,tuple(settings['skills']))
        relation=task_identity(task,source_id=settings.get('task_source_id',settings['source_id']),
                               revision=settings.get('task_source_revision',settings['source_revision']))
        mapping={'core':'present','supporting':'present','not_required':'absent','uncertain':'uncertain'}
        tags=task_tags(relation)+(DataTag('skill_label_basis','task_required'),)+tuple(
             DataTag('skill_status:'+x['skill_id'],mapping[x['status']]) for x in value['skills'])
        record=attach_metadata_record(record,make_metadata_record(record,
            DataLabels(repository=task.get('repo'),tags=tags),provenance=(proof,)))
        record=replace(record,normalization_version='m1-v4')
        atomic_json(item/'record.json',record.to_dict());atomic_json(item/'task-link.json',relation)
        atomic_json(item/'annotation-decision.json',decision)
        atomic_json(item/'task-skills.json',{'data_id':record.data_id,'label_basis':'task_required',
            'skills':{x['skill_id']:mapping[x['status']] for x in value['skills']}})
        retained+=1
        if record_sink is None:records.append(record)
        else:record_sink(record)
    atomic_json(output/'selection-budget.json',budget)
    atomic_json(output/'admission-report.json',{'decisions':audit,'raw_success':budget['raw_success_count'],
        'eligible':eligible,'fresh_execution':'not_performed','policy':'source-resolved-and-light-static-v1'})
    atomic_json(output/'annotation-decisions.json',{'records':decisions})
    atomic_json(output/'annotation-summary.json',{'hard_screened':eligible,'retained':retained,
        'unique_success_tasks':len(successful),'trajectory_teacher_calls':0,'label_basis':'task_required',
        'excluded_task_uncovered':sum(x['decision']=='exclude_task_uncovered' for x in decisions),
        'budget_denominator_changed':False})
    require(retained>=budget['selected_count'],'Task-labelled pool cannot satisfy original raw-success 10% budget')
    return tuple(records),budget
