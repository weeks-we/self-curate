"""Small workstation-only Teacher trial; never starts training or task containers."""
from pathlib import Path
import argparse,json,os
from self_curate.contracts import digest,canonical_json
from self_curate.contracts.parsing import message
from self_curate.contracts.types import require
from self_curate.training.recovery import atomic_json,read_json,lease
from self_curate.preparation.provider import AnnotationClient
from self_curate.preparation.teacher import load_settings,task_packet,trajectory_packet,annotate,to_m1
from self_curate.preparation.rebench import source_messages
from self_curate.preparation.rendering import canonical_messages
from self_curate.experiments.exp001.workstation import rows

class TextEvidencePreview:
    """Payload quote validation only. This supplies NO tokenizer or token metrics."""
    def align(self,messages):
        text='';spans=[]
        for msg in messages:
            payload=msg.content
            if msg.tool_calls:
                payload+= ('\n' if payload else '')+canonical_json([t.to_dict() for t in msg.tool_calls])
            spans.append((len(text),len(text)+len(payload)));text+=payload+'\n'
        return None,text,None,tuple(spans)

def run(config,*,client=None,review=None,renderer=None):
    require(config.get('execution_location')=='workstation','Pilot is workstation-only')
    root=Path(config['output']).resolve();root.mkdir(parents=True,exist_ok=True)
    settings={**load_settings(config['ontology']),'trajectory_grouping':config.get('trajectory_grouping',{})}
    if 'review_mode' in config:settings['review_mode']=config['review_mode']
    from .teacher import review_mode
    mode=review_mode(settings)
    if settings.get('task_labels_only'):
        require(config.get('trajectory_limit',0)==0,'Task-only pilot must not annotate trajectories')
    tasks=list(rows(config['tasks']));trajectories=list(rows(config['trajectories']))
    lookup={t['instance_id']:t for t in tasks}
    require(len(lookup)==len(tasks),'Duplicate pilot task IDs')
    require(all(t['instance_id'] in lookup for t in trajectories),'Pilot trajectory/task join failed')
    require(all(type(config.get(k,100)) is int and 0<=config.get(k,100)<=100
                for k in ('task_limit','trajectory_limit')),
            'Pilot caps are 100 tasks and 100 trajectories; full preparation is separate')
    tasks=sorted(tasks,key=lambda t:digest((config.get('seed',42),t['instance_id'])))[:config.get('task_limit',100)]
    trajectories=sorted((t for t in trajectories if t.get('resolved')==1),
        key=lambda t:digest((config.get('seed',42),t['trajectory_id'])))[:config.get('trajectory_limit',100)]
    require(bool(tasks or trajectories),'Pilot selected no records')
    intent={'config':config,'ontology_hash':digest(settings),
        'tasks':[digest(t) for t in tasks],'trajectories':[digest(t) for t in trajectories],
        'implementation':{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path(__file__).with_name('teacher.py'),Path(__file__).with_name('provider.py'),
                      Path(__file__).with_name('teacher_grouped.py'))}}
    if (root/'intent.json').exists():require(read_json(root/'intent.json')==intent,'Pilot identity changed; use a new root')
    else:atomic_json(root/'intent.json',intent)
    require(config.get('backend') in ('real','mock'),'Declare pilot domain')
    client=client or AnnotationClient({**config['teacher'],'backend':config['backend'],
        'execution':'workstation'},root/'primary')
    if mode=='off':review=None
    if mode!='off' and review is None and config.get('review_teacher'):
        review=AnnotationClient({**config['review_teacher'],'backend':config['backend'],
            'execution':'workstation'},root/'review')
    helpers={}
    for row in ([] if settings.get('task_labels_only') else trajectories):
        key=row['instance_id']
        if key not in helpers or row['trajectory_id']<helpers[key]['trajectory_id']:helpers[key]=row
    if renderer is None and config.get('tokenizer_path'):
        from self_curate.preparation.rebench_workstation import renderer_from
        from self_curate.contracts.parsing import model_ref
        renderer=renderer_from(config,model_ref(read_json(config['model'])))
    preview=renderer is None
    renderer=renderer or TextEvidencePreview()
    summary={'backend':config['backend'],'review_mode':mode,'representative_sample':False,
        'development_selection_performed':False,'task_annotations':[], 'trajectory_annotations':[],
        'token_metrics':'awaiting_locked_tokenizer' if preview else 'computed_with_declared_renderer',
        'primary_counts':{s:0 for s in settings['skills']},'no_skill_exclusions':[]}
    for kind,items in (('task',tasks),('trajectory',trajectories)):
        grouped_results={};grouped_errors={}
        if kind=='trajectory' and settings.get('trajectory_grouping',{}).get('enabled'):
            from .teacher_grouped import annotate_packet_group,bounded_packet_groups
            review_ids={items[0]['trajectory_id']} if items else set()
            def packets():
                for item in sorted(items,key=lambda t:(t['instance_id'],t['trajectory_id'])):
                    try:
                        messages=tuple(message(m) for m in canonical_messages(source_messages(item)))
                        yield trajectory_packet(item,lookup[item['instance_id']],messages,renderer,settings)
                    except Exception as exc:grouped_errors[item['trajectory_id']]=str(exc)
            for group in bounded_packet_groups(packets(),settings,client.settings.get('max_output_tokens',16384)):
                try:
                    done,errors=annotate_packet_group(group,settings,client,review,
                        root/'trajectory-groups'/digest([p['request_id'] for p in group]),review_ids=review_ids)
                    grouped_results.update(done)
                    grouped_errors.update({k:v['message'] for k,v in errors.items()})
                except Exception as exc:
                    grouped_errors.update({p['trajectory_id']:str(exc) for p in group})
        for i,item in enumerate(items):
            identifier=item['instance_id'] if kind=='task' else item['trajectory_id']
            folder=root/kind/digest(identifier);folder.mkdir(parents=True,exist_ok=True)
            atomic_json(folder/'raw.json',item)
            try:
                if kind=='task':packet=task_packet(item,settings,helpers.get(item['instance_id']))
                else:
                    messages=tuple(message(m) for m in canonical_messages(source_messages(item)))
                    packet=trajectory_packet(item,lookup[item['instance_id']],messages,renderer,settings)
                atomic_json(folder/'packet.json',packet)
                # Sample the first annotation of each kind only when sampled review is enabled.
                if identifier in grouped_errors:raise ValueError(grouped_errors[identifier])
                if identifier in grouped_results:
                    saved=grouped_results[identifier]
                    require(saved['packet_hash']==digest(packet),'Pilot group packet mismatch')
                    value,path,provenance=saved['annotation'],Path(saved['response']),saved['provenance']
                else:
                    value,path,provenance=annotate(kind,packet,settings,client,review,random_review=i==0)
                atomic_json(folder/'annotation.json',value);atomic_json(folder/'provenance.json',provenance)
                result={'id':identifier,'status':'validated','annotation_hash':digest(value)}
                if kind=='task':
                    result['primary_skill']=value['primary_skill']
                    if value['primary_skill']:summary['primary_counts'][value['primary_skill']]+=1
                else:
                    result['decision']=value['annotation_decision']
                    if value['annotation_decision']=='exclude_no_skill':summary['no_skill_exclusions'].append(identifier)
                    elif not preview:
                        annotations=to_m1(value,packet,messages,renderer,path)
                        from self_curate.data.normalize import evidence_metrics
                        metrics,union=evidence_metrics(renderer.encode(messages),annotations,tuple(settings['skills']))
                        atomic_json(folder/'m1-annotations.json',annotations.to_dict())
                        atomic_json(folder/'skill-features.json',{'metrics':[m.to_dict() for m in metrics],
                            'supervised_evidence_union':union})
                summary[kind+'_annotations'].append(result)
            except Exception as exc:
                result={'id':identifier,'status':'blocked','error_type':type(exc).__name__,
                    'not_a_no_skill_exclusion':True,'error_message':str(exc)}
                atomic_json(folder/'blocked.json',result);summary[kind+'_annotations'].append(result)
            atomic_json(root/'summary.json',summary)
            print(json.dumps({'phase':kind,'done':i+1,'total':len(items),**result},ensure_ascii=False),flush=True)
    summary['passed']=all(r['status']=='validated' for k in ('task_annotations','trajectory_annotations') for r in summary[k])
    summary['ready_for_full_annotation']=False # Human semantic review + coverage/distribution pilot still required.
    atomic_json(root/'summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read_json(args.config)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    with lease(Path(config['output'])/'.pilot.lease'):result=run(config)
    print(json.dumps({'passed':result['passed'],'output':config['output'],
        'ready_for_full_annotation':False},ensure_ascii=False))
    return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
