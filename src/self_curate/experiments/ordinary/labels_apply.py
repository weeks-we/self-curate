"""Attach source-bound external task labels without computing paid annotations."""
from collections import Counter
from pathlib import Path
from .common import rows,read,write,jsonl,sha,digest


def apply_labels(pool,annotations,output):
    root=Path(output)
    if root.exists():raise FileExistsError('Labeled assets are immutable')
    source=Path(pool)/'pool.jsonl';manifest=read(Path(pool)/'manifest.json')
    if sha(source)!=manifest['pool_sha256']:raise ValueError('Pool checksum mismatch')
    taxonomy=read(Path(__file__).parents[4]/'configs/ordinary/skills.json')['domains']
    annotations=list(rows(annotations));by_id={a['id']:a for a in annotations}
    if len(by_id)!=len(annotations):raise ValueError('Duplicate annotation IDs')
    seen=set();counts=Counter();missing=0
    def converted():
        nonlocal missing
        for record in rows(source):
            a=by_id.get(record['id'])
            if a:
                if a.get('needs_review'):raise ValueError('Resolve uncertain annotations before publication')
                if a['content_hash']!=record['content_hash']:raise ValueError('Annotation content mismatch')
                if not a['skills'] or len(set(a['skills']))!=len(a['skills']) or set(a['skills'])-set(taxonomy[record['domain']]):
                    raise ValueError('Unknown/empty/duplicate skills')
                if not a.get('model') or not a.get('revision'):raise ValueError('Label model/version required')
                record={**record,'skills':sorted(a['skills']),'label_status':'teacher',
                    'label_version':'ordinary-skills-v2','raw_labels':a['skills'],
                    'label_provenance':{'annotation_hash':digest(a),'model':a['model'],'revision':a['revision']}}
                seen.add(record['id'])
            counts.update(record['skills']);missing+=not bool(record['skills'])
            yield record
        if seen!=set(by_id):raise ValueError('Annotation IDs not present in this pool')
    root.mkdir(parents=True)
    jsonl(root/'pool.jsonl',converted())
    write(root/'manifest.json',{**manifest,'pool_sha256':sha(root/'pool.jsonl'),
        'parent_pool_sha256':sha(source),'coverage':dict(counts),'missing_skills':missing,
        'label_version':'ordinary-skills-v2'})
