"""Chronological native compaction evidence, independent from research episodes."""
from self_curate.contracts.types import require

def verify_compaction(events, records, *, minimum_long_input, minimum_drop=1000):
    require(records and len({r.get('session_id') for r in records})==1
            and bool(records[0].get('session_id')),'Missing/mixed gateway session')
    require([r.get('request_sequence') for r in records]==list(range(1,len(records)+1)),
            'Missing/duplicate/out-of-order requests')
    for r in records:
        require(type(r.get('input_tokens')) is int and r['input_tokens']>=0
            and type(r.get('output_tokens')) is int and r['output_tokens']>=0
            and type(r.get('started_monotonic')) in (int,float)
            and r['started_monotonic']<=r['finished_monotonic'],'Incomplete request timing/usage')
    boundaries=[e['monotonic'] for e in events if e.get('sdk_type')=='SystemMessage'
        and e.get('data',{}).get('subtype')=='compact_boundary']
    for boundary in boundaries:
        before=[r for r in records if r['finished_monotonic']<=boundary and r['input_tokens']>=minimum_long_input]
        after=[r for r in records if r['started_monotonic']>boundary]
        tools=[e for e in events if e.get('monotonic',0)>boundary and e.get('sdk_type')=='UserMessage'
               and any(b.get('type')=='tool_result' for b in e.get('data',{}).get('content',[]) if isinstance(b,dict))]
        if before and after and tools:
            high=before[-1];low=after[0]
            if high['input_tokens']-low['input_tokens']>=minimum_drop:
                return {'passed':True,'boundary_monotonic':boundary,'before':high['request_sequence'],
                    'after':low['request_sequence'],'input_drop':high['input_tokens']-low['input_tokens']}
    raise ValueError('No native compaction followed by lower-input main request and actual tool result')
