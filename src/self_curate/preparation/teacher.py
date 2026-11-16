"""Versioned task/trajectory Teacher contracts, review and evidence-to-M1 adapter."""
from pathlib import Path
from copy import deepcopy
import json
from self_curate.contracts import AnnotationSet,SkillAnnotation,EvidenceSpan,digest
from self_curate.contracts.types import require
from self_curate.contracts.parsing import load_json
from self_curate.data.pool import file_sha256
from self_curate.training.recovery import read_json,atomic_json
from self_curate.execution.artifacts import publish_json

class AnnotationNeedsContext(ValueError):
    """Unresolved material/annotation problem; never a no-skill quality rejection."""

def load_settings(path):
    path=Path(path).resolve();value=read_json(path)
    if value.get('annotation_contract')!='coding-teacher-v1':return value
    for kind in ('task','trajectory','review'):
        value[kind+'_prompt']=(path.parent/value[kind+'_prompt_file']).read_text(encoding='utf-8')
    value['_schemas']={kind:read_json(path.parent/name) for kind,name in value['schemas'].items()}
    if value.get('trajectory_group_prompt_file'):
        value['trajectory_group_prompt']=(path.parent/value['trajectory_group_prompt_file']).read_text(encoding='utf-8')
    return value

def validate_schema(value,kind,settings):
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError
    try:Draft202012Validator(settings['_schemas'][kind]).validate(value)
    except ValidationError as exc:raise ValueError('Teacher schema mismatch: '+exc.json_path) from exc

def pointer(value,path):
    require(isinstance(path,str) and path.startswith('/'),'Explicit JSON pointer required')
    for part in path.split('/')[1:]:
        part=part.replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value

def ref_text(ref,packet):
    source=ref['source'];require(source in packet['source_catalog'],'Unknown evidence source')
    if source in ('trajectory','helper_trajectory'):
        rows=packet[source]['messages'];i=ref['message_id']
        require(type(i) is int and 0<=i<len(rows) and rows[i]['message_id']==i,'Unknown stable message ID')
        value=rows[i]
    else:
        require(ref['message_id'] is None,'Non-message source must not claim a message ID')
        value=packet[source]
    text=pointer(value,ref['field']);quote=ref['quote']
    count=text.count(quote) if type(text) is str and type(quote) is str and quote else 0
    require(count==1,
            'Evidence quote must uniquely match: source='+source+', message_id='+str(ref['message_id'])+
            ', field='+ref['field']+', occurrences='+str(count)+'; select an exact longer unique substring')
    return text

def identity(value,packet):
    for name in ('request_id','instance_id','ontology_version'):
        require(value[name]==packet[name],'Teacher response identity mismatch: '+name)

def task_packet(task,settings,helper=None):
    public={'instance_id':task['instance_id'],'repo':task.get('repo'),
        'base_commit':task.get('base_commit'),'problem_statement':task['problem_statement']}
    material={**public,'gold_patch':task.get('patch',''),'test_patch':task.get('test_patch',''),
        'provided_source':task.get('provided_source',{}),
        'FAIL_TO_PASS':task.get('FAIL_TO_PASS',[]),'PASS_TO_PASS':task.get('PASS_TO_PASS',[])}
    packet={'instance_id':task['instance_id'],'ontology_version':settings['version'],
        'ontology':settings['ontology'],'source_catalog':['task'],
        'solver_allowed_inputs':['problem_statement','initial_repository'],
        'task':material,'helper_trajectory':None,
        'coverage':{'provided_fields':list(material),'helper_optional':True},
        'task_content_hash':digest(task),'output_schema':settings['_schemas']['task']}
    if settings.get('task_labels_only'):helper=None
    if helper is not None:
        from self_curate.data.task_links import validate_task_join
        validate_task_join(helper,task)
        require(helper['instance_id']==task['instance_id'] and helper.get('resolved')==1,
                'Helper must be a matching source-success trajectory')
        raw=helper['trajectory'];raw=load_json(raw) if isinstance(raw,str) else raw
        rows=[{'message_id':i,**m} for i,m in enumerate(raw)]
        packet['helper_trajectory']={'trajectory_id':helper['trajectory_id'],
            'messages':rows,'model_patch':helper.get('model_patch'),
            'source_resolved':helper['resolved'],'content_hash':digest(helper)}
        packet['source_catalog'].append('helper_trajectory')
    packet['request_id']='task-'+digest(packet)
    return packet

def trajectory_packet(raw,task,messages,renderer,settings):
    from self_curate.data.task_links import validate_task_join
    validate_task_join(raw,task)
    _,rendered,_,chars=renderer.align(messages)
    rows=[{'message_id':i,'role':m.role,'content':m.content,
           'rendered':rendered[chars[i][0]:chars[i][1]]} for i,m in enumerate(messages)]
    original=raw['trajectory'];original=load_json(original) if isinstance(original,str) else original
    # Message IDs refer to the exact normalized M1 sequence; retain the original mapping.
    shift=len(messages)-len(original)
    require(shift in (0,1),'Unexpected source message normalization')
    packet={'instance_id':task['instance_id'],'trajectory_id':raw['trajectory_id'],
        'ontology_version':settings['version'],'ontology':settings['ontology'],
        'source_catalog':['trajectory','task'],'task':{'instance_id':task['instance_id'],
            'repo':task.get('repo'),'base_commit':task.get('base_commit'),
            'problem_statement':task['problem_statement'],'gold_patch':task.get('patch',''),
            'test_patch':task.get('test_patch','')},
        'trajectory':{'messages':rows,'model_patch':raw.get('model_patch'),
            'source_resolved':raw.get('resolved'),'source_tools':raw.get('tools',[]),
            'message_id_namespace':'normalized-M1-message-index-v1',
            'original_message_mapping':[{'message_id':i,'original_index':None if i<shift else i-shift}
                                        for i in range(len(messages))]},
        'coverage':{'state':'complete','message_count':len(messages),'silently_truncated':False},
        'trajectory_content_hash':digest(raw),'output_schema':settings['_schemas']['trajectory']}
    packet['request_id']='trajectory-'+digest(packet)
    return packet

def validate_task(value,packet,settings):
    validate_schema(value,'task',settings);identity(value,packet)
    rows=value['skills'];skills=tuple(settings['skills'])
    require(len(rows)==len(skills) and {r['skill_id'] for r in rows}==set(skills),'Exact skill coverage required')
    for row in rows:
        require(settings.get('task_labels_only') or row['status'] not in ('core','supporting') or bool(row['evidence']),'Positive task skill needs evidence')
        for ref in row.get('evidence',[]):ref_text(ref,packet)
    for risk in value['task_risks']:
        for ref in risk['evidence']:ref_text(ref,packet)
    primary=value['primary_skill']
    if value['classification_state']=='classified':
        require(primary in skills and next(r for r in rows if r['skill_id']==primary)['status']=='core',
                'Classified task needs a core primary skill')
    else:require(primary is None,'Composite/uncertain/uncovered tasks cannot force a primary')
    return primary

def validate_trajectory(value,packet,settings):
    validate_schema(value,'trajectory',settings);identity(value,packet)
    require(value['trajectory_id']==packet['trajectory_id'],'Wrong trajectory ID')
    skills=tuple(settings['skills']);rows=value['skills']
    require(len(rows)==len(skills) and {r['skill_id'] for r in rows}==set(skills),'Exact trajectory skill coverage required')
    demonstrated=False;uncertain=False;events=set()
    for row in rows:
        status=row['status'];demonstrated|=status=='demonstrated';uncertain|=status=='uncertain'
        if status=='not_observed':require(not row['events'],'Not-observed skill cannot contain events')
        if status=='demonstrated':require(bool(row['events']),'Demonstrated skill needs events')
        for event in row['events']:
            require(event['event_id'] not in events,'Duplicate evidence event ID');events.add(event['event_id'])
            if status=='demonstrated':
                require(bool(event['assistant_refs']) and bool(event['outcome_refs']),
                        'Demonstrated event needs assistant action and supporting outcome')
            for group in ('context_refs','assistant_refs','outcome_refs'):
                for ref in event[group]:
                    ref_text(ref,packet)
                    if group=='outcome_refs':
                        require(ref['source']=='trajectory','Task reference material cannot establish a trajectory outcome')
                    if group=='assistant_refs':
                        require(ref['source']=='trajectory' and
                            packet['trajectory']['messages'][ref['message_id']]['role']=='assistant',
                            'Assistant evidence must belong to this trajectory assistant')
                        require(ref['field'] in ('/content','/rendered'),'Use declared assistant payload field')
                        rendered=packet['trajectory']['messages'][ref['message_id']]['rendered']
                        require(rendered.count(ref['quote'])==1,'Quote ambiguous in rendered assistant payload')
    require(value['coverage_state']=='complete','Partial annotation cannot authorize retention/exclusion')
    expected='retain' if demonstrated else ('needs_review' if uncertain or value['missing_context'] else 'exclude_no_skill')
    require(value['annotation_decision']==expected,'Annotation disposition conflicts with skill evidence')
    if expected=='exclude_no_skill':require(bool(value['decision_reason']),'No-skill exclusion needs a reason')
    return expected


def review_mode(settings):
    mode=settings.get('review_mode','sampled')
    require(mode in ('off','flagged','sampled','all'),'Unknown Teacher review mode')
    return mode


def review_client_for(settings, review):
    mode=review_mode(settings)
    if mode=='off':return None
    # Old direct integrations without this control retain their prior optional-client behavior.
    if 'review_mode' in settings:
        require(review is not None,'Enabled Teacher review requires a configured reviewer')
    return review

def annotate(kind,packet,settings,client,review=None,*,random_review=False):
    mode=review_mode(settings);review=review_client_for(settings,review)
    validate=validate_task if kind=='task' else validate_trajectory
    value=None;proof=None;error=None;primary_proof=None
    try:
        value,proof=client.ask(settings[kind+'_prompt'],packet);primary_proof=proof
        validate(value,packet,settings)
    except (ValueError,TypeError,KeyError) as exc:
        error=type(exc).__name__+': '+str(exc)
    need_review=error is not None
    if value is not None and error is None:
        if kind=='task':need_review=value['classification_state']=='uncertain' or bool(value['task_risks']) or value['confidence']=='low' or any(r['status']=='uncertain' for r in value['skills'])
        else:need_review=value['annotation_decision']=='needs_review' or value['confidence']=='low' or any(r['status']=='uncertain' for r in value['skills'])
    review_record=None
    selected=(mode=="all" or need_review or (mode=="sampled" and random_review))
    if review is not None and selected:
        review_packet={'request_id':'review-'+digest((packet,value,error)),
            'data_id':packet.get('trajectory_id',packet['instance_id']),
            'annotation_kind':kind,'ontology':settings['ontology'],'original_packet':packet,
            'candidate_annotation':value,'program_issues':[] if error is None else [error],
            'annotation_instructions':settings[kind+'_prompt'],
            'annotation_output_schema':settings['_schemas'][kind],
            'review_output_schema':settings['_schemas']['review']}
        reviewed,review_proof=review.ask(settings['review_prompt'],review_packet)
        validate_schema(reviewed,'review',settings)
        require(reviewed['request_id']==review_packet['request_id'] and reviewed['data_id']==review_packet['data_id'],
                'Review response identity mismatch')
        verdict=reviewed['verdict']
        if verdict=='approve':
            require(error is None and reviewed['corrected_annotation'] is None,'Cannot approve invalid annotation')
        elif verdict=='revise':
            require(type(reviewed['corrected_annotation']) is dict,'Revision needs complete annotation')
            value=reviewed['corrected_annotation'];validate(value,packet,settings);proof=review_proof;error=None
        else:raise AnnotationNeedsContext('Review unresolved: '+verdict)
        review_record={'response_path':str(review_proof),'verdict':verdict,'response_sha256':file_sha256(review_proof)}
    elif error is not None:raise AnnotationNeedsContext('Invalid annotation without successful review: '+error)
    require(value is not None and proof is not None,'Missing annotation')
    validate(value,packet,settings)
    if kind=='trajectory' and value['annotation_decision']=='needs_review':
        raise AnnotationNeedsContext('No demonstrated skill, unresolved uncertainty; not a no-skill exclusion')
    return value,Path(proof),{'primary_response':None if primary_proof is None else str(primary_proof),
        'final_response_sha256':file_sha256(proof),'review':review_record,
        'review_mode':mode,'review_status':'reviewed' if review_record else ('disabled' if mode=='off' else 'not_selected'),
        'review_trigger':{'flagged':need_review,'random_sample':bool(random_review)}}

def to_m1(value,packet,messages,renderer,response_path):
    encoded,rendered,offsets,chars=renderer.align(messages)
    result=[]
    for row in value['skills']:
        if row['status']=='uncertain':
            result.append(SkillAnnotation(row['skill_id'],None,file_sha256(response_path)));continue
        spans=[]
        for event in row['events']:
            if row['status']!='demonstrated':continue
            for ref in event['assistant_refs']:
                i=ref['message_id'];payload=packet['trajectory']['messages'][i]['rendered']
                require(payload==rendered[chars[i][0]:chars[i][1]],'Evidence payload changed after Teacher')
                a=chars[i][0]+payload.index(ref['quote']);b=a+len(ref['quote'])
                tokens=[j for j,(x,y) in enumerate(offsets) if x<y and x<b and y>a
                        and encoded.message_spans[i].start<=j<encoded.message_spans[i].end]
                require(tokens and all(encoded.assistant_mask[j] for j in tokens),
                        'Evidence must map to real assistant supervision')
                require(tokens==list(range(tokens[0],tokens[-1]+1)),'Noncontiguous evidence tokens')
                spans.append(EvidenceSpan(i,tokens[0],tokens[-1]+1))
        if row['status']=='demonstrated':require(bool(spans),'Positive skill lost all token evidence')
        result.append(SkillAnnotation(row['skill_id'],tuple(spans),file_sha256(response_path),
                                      len(row['events']),()))
    return AnnotationSet(encoded.identity.tokenizer_hash,encoded.identity.template_hash,tuple(result))

def annotation_receipt(packet,value,path,provenance,output):
    return publish_json(output,{'packet':packet,'annotation':value,'provenance':provenance,
        'annotation_hash':digest(value),'response_sha256':file_sha256(path)})
