"""The single active ordinary-task experiment entrypoint."""
import argparse
from pathlib import Path
import shutil
import zipfile
from .common import read,write,rows,sha


def main(argv=None):
    parser=argparse.ArgumentParser(prog='python -m self_curate.experiments.ordinary')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('lock');p.add_argument('--registry',default='configs/ordinary/sources.json');p.add_argument('--output',required=True)
    p=sub.add_parser('download');p.add_argument('--lock',required=True);p.add_argument('--storage',required=True);p.add_argument('--include-models',action='store_true')
    p=sub.add_parser('preview');p.add_argument('--lock',required=True);p.add_argument('--key',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=3)
    p=sub.add_parser('export');p.add_argument('--lock',required=True);p.add_argument('--bindings',required=True);p.add_argument('--key',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('benchmarks');p.add_argument('--lock',required=True);p.add_argument('--bindings',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('prepare');p.add_argument('--domain',choices=('coding','math','if'),required=True);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokenizer');p.add_argument('--max-length',type=int,default=8192);p.add_argument('--blocked',action='append',default=[]);p.add_argument('--preview',action='store_true')
    p=sub.add_parser('plan');p.add_argument('--config',default='configs/ordinary/exp001.json');p.add_argument('--pools',required=True);p.add_argument('--output',required=True);p.add_argument('--synthetic',action='store_true')
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--bindings');p.add_argument('--mock',action='store_true');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('estimate');p.add_argument('--plan',required=True)
    p=sub.add_parser('development');p.add_argument('--pool',required=True);p.add_argument('--per-skill',type=int,default=50);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',required=True)
    p=sub.add_parser('bundle');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('evalplus-cache');p.add_argument('--storage',required=True);p.add_argument('--bindings',required=True)
    p=sub.add_parser('seal');p.add_argument('--storage',required=True);p.add_argument('--lock',required=True);p.add_argument('--bindings',required=True)
    p=sub.add_parser('reserve');p.add_argument('--pool',required=True);p.add_argument('--output',required=True);p.add_argument('--per-skill',type=int,default=50)
    p=sub.add_parser('server-bindings')
    for key in ('lock','sources','benchmarks','storage','project','run-root','grader-image','output'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('smoke');p.add_argument('--output',required=True)
    p=sub.add_parser('finalize-labels')
    for key in ('pool','output','completed-run'):p.add_argument('--'+key,required=True)
    p.add_argument('--coding',action='store_true');p.add_argument('--per-skill',type=int,default=50)
    p=sub.add_parser('math-reference-join')
    for key in ('raw','reference','output'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('apply-labels')
    for key in ('pool','annotations','output'):p.add_argument('--'+key,required=True)
    p=sub.add_parser('sample-math')
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=50000);p.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(argv)
    if args.command=='sample-math':
        from .stratified import sample
        result=sample(args.raw,args.output,count=args.count,seed=args.seed)
        print({'count':result['count'],'sampling_hash':result['sampling_hash']})
    elif args.command=='apply-labels':
        from .labels_apply import apply_labels
        apply_labels(args.pool,args.annotations,args.output)
    elif args.command=='math-reference-join':
        from .math_reference import join
        print(join(args.raw,args.reference,args.output))
    elif args.command=='finalize-labels':
        from .finalize_labels import finalize
        finalize(args.pool,args.output,completed_run=args.completed_run,per_skill=args.per_skill,coding=args.coding)
    elif args.command=='smoke':
        from .smoke import smoke
        print(smoke(args.output))
    elif args.command=='evalplus-cache':
        from .workstation import cache_evalplus
        cache_evalplus(args.storage)
        value=read(args.bindings);value['evalplus_root']=str((Path(args.storage)/'evalplus').resolve());write(args.bindings,value)
    elif args.command=='seal':
        from .workstation import seal
        seal(args.storage,args.lock,args.bindings)
    elif args.command=='reserve':
        from .workstation import reserve
        reserve(args.pool,args.output,per_skill=args.per_skill)
    elif args.command=='server-bindings':
        from .workstation import bindings
        bindings(args.lock,args.sources,args.benchmarks,args.storage,args.project,args.run_root,args.grader_image,args.output)
    elif args.command=='lock':
        from .sources import resolve
        resolve(read(args.registry),args.output)
    elif args.command=='download':
        from .sources import download
        download(read(args.lock),args.storage,include_models=args.include_models)
    elif args.command=='preview':
        from .sources import preview
        preview(read(args.lock),args.key,args.output,count=args.count)
    elif args.command=='export':
        from .sources import export_training
        export_training(read(args.lock),read(args.bindings),args.key,args.output)
    elif args.command=='benchmarks':
        from .benchmarks import prepare_benchmarks
        prepare_benchmarks(read(args.lock),read(args.bindings),args.output)
    elif args.command=='prepare':
        from self_curate.preparation.ordinary import prepare
        sidecar=read(str(args.raw)+'.source.json')
        if sidecar['sha256']!=sha(args.raw):
            raise ValueError('Raw source sidecar checksum mismatch')
        if sidecar.get('preview') and not args.preview:
            raise ValueError('Preview data cannot become a formal pool')
        if not args.preview and (not args.tokenizer or not args.blocked):
            raise ValueError('Formal admission requires the frozen tokenizer and benchmark/development isolation inputs')
        tokenizer=None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
        blocked={r['question_hash'] for file in args.blocked for r in rows(file)}
        prepare(args.domain,args.raw,args.output,source=sidecar,blocked_questions=blocked,
                tokenizer=tokenizer,max_length=args.max_length,preview=args.preview)
    elif args.command=='plan':
        from .workflow import plan
        plan(read(args.config),read(args.pools),args.output,synthetic=args.synthetic)
    elif args.command=='run':
        from .workflow import execute
        from .backend import MockBackend,ServerBackend
        if not args.mock and not args.bindings:
            raise ValueError('Real execution requires workstation/server bindings')
        backend=MockBackend() if args.mock else ServerBackend(args.bindings)
        try:print(execute(args.plan,backend,resume=args.resume))
        finally:
            if hasattr(backend,'stop'):backend.stop()
    elif args.command=='estimate':
        from .workflow import estimate
        print(estimate(read(args.plan)))
    elif args.command=='development':
        from .workflow import development
        from .common import jsonl
        records=list(rows(Path(args.pool)/'pool.jsonl'))
        result=development(records,args.per_skill,args.seed)
        if Path(args.output).exists():raise FileExistsError('Development set already exists')
        write(Path(args.output)/'manifest.json',result)
        ids=set(result['unique_ids'])
        jsonl(Path(args.output)/'tasks.jsonl',(r for r in records if r['id'] in ids))
    elif args.command=='bundle':
        root=Path(args.run).resolve();output=Path(args.output).resolve()
        if not (root/'complete.json').exists():raise ValueError('Only completed runs can be bundled')
        allowed=('.json','.jsonl','.csv','.log','.md','.txt')
        with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in allowed and 'training' not in path.relative_to(root).parts:
                    archive.write(path,path.relative_to(root))
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:raise ValueError('ZIP CRC verification failed')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
