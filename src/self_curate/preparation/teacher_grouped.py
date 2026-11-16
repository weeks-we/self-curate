"""Bounded same-task Teacher groups with per-trajectory validation and recovery.

Spool source records once, load at most max_group_size complete conversations.
The shared task is context; only the member's own trajectory proves its skills.
"""
from pathlib import Path
import json
from collections import defaultdict
from self_curate.contracts import digest,canonical_json
from self_curate.contracts.types import require
from self_curate.training.recovery import read_json,atomic_json
from self_curate.data.pool import file_sha256
from self_curate.data.task_links import task_identity,validate_task_join
from self_curate.contracts.parsing import message
from .rendering import canonical_messages
from .teacher import trajectory_packet,annotate,AnnotationNeedsContext


GROUP_INSTRUCTIONS='''
批量模式：输入的 task 和 ontology 由同一任务的若干轨迹共享，成员 packet 中未重复这些字段。
对每个成员分别返回完整 teacher-trajectory-v1 annotation，保留其 request_id、instance_id、trajectory_id。
不能给整个任务一个共用技能标签；成功与否、质量及展示技能均逐条判断。
每个成员的 message_id 命名空间独立，只能引用该成员 trajectory 的分析/动作/观察。
任务/gold/test材料仅帮助理解目标，不构成任何成员的实际动作证据；不能借用其他成员的证据。
输出严格为 {schema_version:teacher-trajectory-group-v1,group_id:原值,items:[
 {trajectory_id:原值,annotation:完整annotation或null,error:null或具体无法标注原因}]}。
每个请求成员恰好一项。无法标注的成员返回error，不要把失败或缺材料标成exclude_no_skill。
'''


def grouping_settings(settings):
    cfg={'enabled':False,'max_group_size':2,'max_input_bytes':600000,
         'output_tokens_per_member':6000,'output_overhead_tokens':2048,
         **settings.get('trajectory_grouping',{})}
    require(set(cfg)=={'enabled','max_group_size','max_input_bytes','output_tokens_per_member','output_overhead_tokens'},
            'Unknown trajectory grouping controls')
    require(type(cfg['enabled']) is bool,'Explicit grouping mode required')
    for key in ('max_group_size','max_input_bytes','output_tokens_per_member','output_overhead_tokens'):
        require(type(cfg[key]) is int and cfg[key]>0,'Positive grouping '+key+' required')
    require(cfg['max_group_size']<=3,'Start with groups of at most three complete trajectories')
    return cfg


def group_packet(packets, settings):
    require(bool(packets),'Empty trajectory group')
    require(len({p['instance_id'] for p in packets})==1,'Cross-task group forbidden')
    require(len({p['trajectory_id'] for p in packets})==len(packets),'Duplicate group member')
    require(all(p['task']==packets[0]['task'] for p in packets),'Conflicting shared task snapshots')
    members=[{k:v for k,v in p.items() if k not in ('task','ontology','output_schema')} for p in packets]
    value={'schema_version':'teacher-trajectory-group-v1','instance_id':packets[0]['instance_id'],
           'task':packets[0]['task'],'ontology':settings['ontology'],'members':members,
           'annotation_output_schema':settings['_schemas']['trajectory'],
           'output_schema':{'type':'object','required':['schema_version','group_id','items'],
                            'description':'one independently identified annotation or error per member'}}
    value['group_id']='trajectory-group-'+digest(value)
    return value


def _members(value, packet):
    require(isinstance(value,dict) and set(value)=={'schema_version','group_id','items'},'Invalid group envelope')
    require(value['schema_version']=='teacher-trajectory-group-v1' and value['group_id']==packet['group_id'],
            'Wrong group identity')
    require(type(value['items']) is list,'Group items must be a list')
    expected={p['trajectory_id'] for p in packet['members']};result={}
    for row in value['items']:
        require(isinstance(row,dict) and set(row)=={'trajectory_id','annotation','error'},'Invalid member envelope')
        key=row['trajectory_id']
        require(key in expected and key not in result,'Unknown or duplicate group trajectory ID')
        require((isinstance(row['annotation'],dict) and row['error'] is None) or
                (row['annotation'] is None and isinstance(row['error'],str) and bool(row['error'])),
                'Member must contain exactly an annotation or error')
        result[key]=row
    # Missing members block only themselves; unrelated successful members are retained.
    return result


class _SavedAnswer:
    def __init__(self, value, proof, error=None):self.value,self.proof,self.error=value,proof,error
    def ask(self, *args):
        if self.error:raise AnnotationNeedsContext(self.error)
        return self.value,self.proof


def annotate_packet_group(packets,settings,client,review,output,*,review_ids=()):
    """Returns completed members; blocked members are durable and independently retryable."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    shared=group_packet(packets,settings)
    identity=digest({'packets':packets,'settings':settings,'primary':client.settings,
                     'review':None if review is None else review.settings,'review_ids':sorted(review_ids)})
    intent=output/'intent.json'
    if intent.exists():require(read_json(intent)=={'hash':identity},'Group inputs changed; use a new output root')
    else:atomic_json(intent,{'hash':identity})
    done={};pending=[]
    for packet in packets:
        path=output/(digest(packet['trajectory_id'])+'.json')
        if path.exists():
            saved=read_json(path)
            require(saved['packet_hash']==digest(packet) and file_sha256(Path(saved['response']))==saved['response_sha256'],
                    'Completed group member evidence changed')
            from .teacher import validate_trajectory
            validate_trajectory(saved['annotation'],packet,settings)
            done[packet['trajectory_id']]=saved
        else:pending.append(packet)
    if not pending:return done,{}
    from .teacher import review_client_for
    review=review_client_for(settings,review)
    failures={};responses={};proof=None
    if len(packets)>1:
        atomic_json(output/'group-packet.json',shared)
        value,proof=client.ask(settings['trajectory_prompt']+'\n'+settings.get('trajectory_group_prompt',GROUP_INSTRUCTIONS),shared)
        atomic_json(output/'group-response-ref.json',{'path':str(proof),'sha256':file_sha256(proof)})
        responses=_members(value,shared)
    for packet in pending:
        key=packet['trajectory_id'];path=output/(digest(key)+'.json')
        try:
            if len(packets)==1:
                value,response,provenance=annotate('trajectory',packet,settings,client,review,random_review=key in review_ids)
            else:
                member=responses.get(key)
                seeded=_SavedAnswer(None,proof,'Missing member response' if member is None else member['error']) if member is None or member['error'] else _SavedAnswer(member['annotation'],proof)
                value,response,provenance=annotate('trajectory',packet,settings,seeded,review,random_review=key in review_ids)
                provenance={**provenance,'group_id':shared['group_id'],'group_response':str(proof),
                            'group_response_sha256':file_sha256(proof)}
            saved={'packet_hash':digest(packet),'annotation':value,'response':str(response),
                   'response_sha256':file_sha256(response),'provenance':provenance}
            atomic_json(path,saved);done[key]=saved
        except Exception as exc:
            failures[key]={'type':type(exc).__name__,'message':str(exc),'not_a_no_skill_exclusion':True}
            atomic_json(output/(digest(key)+'-blocked.json'),failures[key])
    atomic_json(output/'completion.json',{'completed':sorted(done),'blocked':failures})
    return done,failures


def prepare_group_annotations(source,lookup,renderer,settings,client,review,output,accepted,review_ids):
    """Source factory -> per-item raw files -> bounded groups -> individual receipts."""
    from .rebench import source_messages
    cfg=grouping_settings(settings);output=Path(output)
    groups=defaultdict(list);manifest={};paths={}
    task_source=settings.get('task_source_id',settings['source_id'])
    task_revision=settings.get('task_source_revision',settings['source_revision'])
    for raw in source():
        if raw['trajectory_id'] not in accepted:continue
        task=lookup[raw['instance_id']];validate_task_join(raw,task)
        identity=task_identity(task,source_id=task_source,revision=task_revision)
        key=identity['task_key'];manifest[key]={'identity':identity,'trajectory_ids':[]}
        item=output/'items'/digest(raw['trajectory_id']);item.mkdir(parents=True,exist_ok=True)
        path=item/'raw.json'
        if path.exists():require(read_json(path)==raw,'Source trajectory changed during recovery')
        else:atomic_json(path,raw)
        require(raw['trajectory_id'] not in paths,'Duplicate grouped trajectory ID')
        groups[key].append(raw['trajectory_id']);paths[raw['trajectory_id']]=path
    for key,ids in groups.items():manifest[key]['trajectory_ids']=sorted(ids)
    atomic_json(output/'task-relations.json',{'schema_version':'task-relations-v1','tasks':manifest,
               'scope':'hard-screened isolated candidates; provenance only, not an operator query cache'})
    references={};group_count=0
    def flush(packets):
        nonlocal group_count
        if not packets:return
        group_count+=1
        group_root=output/'teacher-groups'/digest([p['request_id'] for p in packets])
        done,failures=annotate_packet_group(packets,settings,client,review,group_root,review_ids=set(review_ids)&{p['trajectory_id'] for p in packets})
        references.update({key:group_root/(digest(key)+'.json') for key in done})
        if failures:raise AnnotationNeedsContext('Group members blocked; successful member receipts preserved: '+','.join(failures))
    def packets_for(key):
        for data_id in sorted(groups[key]):
            raw=read_json(paths[data_id]);task=lookup[raw['instance_id']]
            messages=tuple(message(m) for m in canonical_messages(source_messages(raw)))
            yield trajectory_packet(raw,task,messages,renderer,settings)
    for key in sorted(groups):
        for packets in bounded_packet_groups(packets_for(key),settings,client.settings.get('max_output_tokens',16384)):
            flush(packets)
    atomic_json(output/'grouping-summary.json',{'groups':group_count,'trajectories':len(references),
        'task_count':len(groups),'grouping':cfg,'per_trajectory_validation':True})
    return references


def bounded_packet_groups(packets, settings, output_cap):
    """Shared planner for formal preparation and paid small pilots; never truncate."""
    cfg=grouping_settings(settings);pending=[]
    for packet in packets:
        if pending and pending[0]['instance_id']!=packet['instance_id']:
            yield pending;pending=[]
        trial=pending+[packet]
        fit=(len(trial)<=cfg['max_group_size'] and
             len(canonical_json(group_packet(trial,settings)).encode('utf-8'))<=cfg['max_input_bytes'] and
             cfg['output_overhead_tokens']+cfg['output_tokens_per_member']*len(trial)<=output_cap)
        if pending and not fit:
            yield pending;pending=[]
        pending.append(packet)
    if pending:yield pending
