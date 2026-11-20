"""Two-level proportional sampling, exact total, bounded-memory row selection."""
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
from .common import read,write,jsonl,sha,digest


def quotas(counts,total):
    population=sum(counts.values())
    if total<0 or total>population:raise ValueError('Requested quota exceeds population')
    if not population:return {k:0 for k in counts}
    result={k:total*n//population for k,n in counts.items()}
    order=sorted(counts,key=lambda k:(-(total*counts[k]%population),k))
    for key in order[:total-sum(result.values())]:result[key]+=1
    assert sum(result.values())==total and all(result[k]<=counts[k] for k in counts)
    return result


def sample(raw,output,*,count=50000,seed=42):
    if type(count) is not int or count<=0 or type(seed) is not int:raise ValueError('Positive count and integer seed required')
    raw=Path(raw);root=Path(output)
    if root.exists():raise FileExistsError('Use a new immutable sample directory')
    provenance=read(str(raw)+'.source.json');input_sha=sha(raw)
    if provenance['sha256']!=input_sha:raise ValueError('Source sidecar checksum mismatch')
    def group(row):
        values=[row['source'],row['problem_type']]
        if any(x is not None and not isinstance(x,str) for x in values):raise ValueError('Stratum fields must be strings or null')
        return tuple(json.dumps(x,ensure_ascii=False) for x in values)
    counts=Counter();source_counts=Counter()
    with raw.open('rb') as stream:
        for line in stream:
            if not line.strip():continue
            g=group(json.loads(line));counts[g]+=1;source_counts[g[0]]+=1
    if sum(counts.values())<count:raise ValueError('Not enough source rows; never sample with replacement')
    source_quotas=quotas(source_counts,count);allocated={}
    for source in sorted(source_counts):
        inner={g[1]:n for g,n in counts.items() if g[0]==source}
        allocated.update({(source,k):v for k,v in quotas(inner,source_quotas[source]).items()})
    heaps={g:[] for g,q in allocated.items() if q}
    with raw.open('rb') as stream:
        index=0
        while True:
            offset=stream.tell();line=stream.readline()
            if not line:break
            if not line.strip():continue
            row=json.loads(line);g=group(row);capacity=allocated[g]
            if capacity:
                score=int(hashlib.sha256(str(seed).encode()+b'\0'+str(index).encode()+b'\0'+line).hexdigest(),16)
                item=(-score,index,offset,digest(row))
                heap=heaps[g]
                if len(heap)<capacity:heapq.heappush(heap,item)
                elif score < -heap[0][0]:heapq.heapreplace(heap,item)
            index+=1
    chosen=sorted((item for heap in heaps.values() for item in heap),key=lambda x:x[1])
    if len(chosen)!=count:raise ValueError('Source changed or quota selection incomplete')
    root.mkdir(parents=True)
    def selected():
        with raw.open('rb') as stream:
            for _,_,offset,expected in chosen:
                stream.seek(offset);row=json.loads(stream.readline())
                if digest(row)!=expected:raise ValueError('Selected source row changed')
                yield row
    target=root/'sample.jsonl';jsonl(target,selected())
    if sha(raw)!=input_sha:raise ValueError('Source changed during sampling; sample not committed')
    report={'schema':'numina-hierarchical-sample-v1','algorithm':'largest_remainder_then_hash_priority_v1',
        'count':count,'seed':seed,'input_sha256':input_sha,'sample_sha256':sha(target),
        'population':sum(counts.values()),'sampling_unit':'source row before admission/deduplication',
        'sources':[{'source':json.loads(k),'population':n,'quota':source_quotas[k],
                    'original_ratio':n/sum(source_counts.values()),'sample_ratio':source_quotas[k]/count}
                   for k,n in sorted(source_counts.items())],
        'strata':[{'source':json.loads(g[0]),'problem_type':json.loads(g[1]),'population':n,
                   'quota':allocated[g],'original_within_source_ratio':n/source_counts[g[0]],
                   'sample_within_source_ratio':allocated[g]/source_quotas[g[0]] if source_quotas[g[0]] else None}
                  for g,n in sorted(counts.items())],
        'selected_rows':[{'row_index':i,'content_hash':h} for _,i,_,h in chosen],
        'qualification':'Subsequent admission/isolation/length filtering may reduce count and change proportions.'}
    report['sampling_hash']=digest(report)
    write(root/'manifest.json',report)
    write(str(target)+'.source.json',{**provenance,'sha256':sha(target),'parent_sha256':input_sha,
        'sampling_hash':report['sampling_hash'],'sampling_count':count,'sampling_seed':seed})
    return report
