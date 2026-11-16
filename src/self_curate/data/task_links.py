"""Version-scoped task provenance and ephemeral one-to-many trajectory lookups.

Task links never merge labels, signals, gradient caches or NICE valuations.
"""
from types import MappingProxyType
from collections import Counter, defaultdict
from self_curate.contracts import digest
from self_curate.contracts.types import require, sha256


def validate_task_join(raw, task):
    require(raw.get('instance_id')==task.get('instance_id'), 'Task/trajectory ID mismatch')
    for key in ('repo','base_commit'):
        left,right=raw.get(key),task.get(key)
        if left is not None and right is not None:
            require(str(left).casefold()==str(right).casefold(), 'Task/trajectory '+key+' mismatch')


def task_identity(task, *, source_id, revision):
    require(isinstance(source_id,str) and source_id and isinstance(revision,str) and revision,
            'Explicit task source and revision required')
    require(isinstance(task.get('instance_id'),str) and task['instance_id'], 'Task ID required')
    base={'instance_id':task['instance_id'],'repository':task.get('repo'),
          'base_commit':task.get('base_commit'),'task_source_id':source_id,'task_source_revision':revision}
    return {**base,'task_key':digest(base),'task_content_hash':digest(task)}


def task_tags(identity):
    from self_curate.contracts.features import DataTag
    return tuple(DataTag(k,str(v)) for k,v in identity.items() if v is not None)


def record_task_identity(record):
    labels=record.metadata
    wanted={'task_key','instance_id','repository','base_commit','task_source_id','task_source_revision','task_content_hash'}
    tags={}
    for tag in (() if labels is None else labels.tags):
        if tag.name in wanted:
            require(tag.name not in tags or tags[tag.name]==tag.value,'Conflicting task identity tags')
            tags[tag.name]=tag.value
    if 'task_key' in tags:
        sha256(tags['task_key'],'task_key')
        require({'instance_id','task_source_id','task_source_revision','task_content_hash'}<=set(tags),
                'Incomplete explicit task identity')
        require(tags['instance_id']==record.source.record_id,'Task identity contradicts source record ID')
        base={k:tags.get(k) for k in ('instance_id','repository','base_commit','task_source_id','task_source_revision')}
        require(digest(base)==tags['task_key'],'Task key contradicts source/repository/base commit')
        sha256(tags['task_content_hash'],'task_content_hash')
        return {**tags,'identity_mode':'explicit-task-v1'}
    # Old datasets lack task release metadata. Keep their source namespaces separate.
    base={'source_id':record.source.source_id,'source_revision':record.source.revision,
          'instance_id':record.source.record_id}
    return {**base,'task_key':digest(base),'identity_mode':'legacy-source-scoped'}


class TaskRelationView:
    """O(N) construction, O(1) key and sibling lookup; no persistent query cache."""
    def __init__(self, records):
        identities={};groups=defaultdict(list);descriptors={}
        for record in records:
            require(record.data_id not in identities,'Duplicate trajectory in task view')
            item=record_task_identity(record);key=item['task_key']
            require(key not in descriptors or descriptors[key]==item,'Conflicting task snapshots under one task key')
            identities[record.data_id]=MappingProxyType(item);descriptors[key]=item
            groups[key].append(record.data_id)
        self._identities=MappingProxyType(identities)
        self._groups=MappingProxyType({k:tuple(v) for k,v in groups.items()})

    def identity(self, data_id):
        require(data_id in self._identities,'Unknown trajectory ID in task view')
        return self._identities[data_id]

    def task_key(self, data_id):
        return self.identity(data_id)['task_key']

    def for_task(self, key):
        require(key in self._groups,'Unknown task key')
        return self._groups[key]

    def related(self, data_id, *, within=None):
        key=self.task_key(data_id)
        if within is None:return self._groups[key]
        ids=tuple(within)
        require(len(ids)==len(set(ids)),'Duplicate query IDs')
        return tuple(i for i in ids if self.task_key(i)==key)

    def summary(self, data_ids=None):
        ids=tuple(self._identities) if data_ids is None else tuple(data_ids)
        require(len(ids)==len(set(ids)),'Duplicate task summary IDs')
        counts=Counter(self.task_key(i) for i in ids)
        return {'trajectories':len(ids),'unique_tasks':len(counts),
                'trajectories_per_task_histogram':{str(k):v for k,v in sorted(Counter(counts.values()).items())},
                'capacity_under_per_task_limit':{str(k):sum(min(k,n) for n in counts.values()) for k in (1,2,3)},
                'identity_modes':dict(Counter(self.identity(i)['identity_mode'] for i in ids))}
