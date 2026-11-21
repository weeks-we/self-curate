"""Offline grading worker. Generated Python is evaluated only inside the CPU grader image."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from .common import digest, read, write, rows


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def code(answer):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.DOTALL)
    return blocks[-1].strip() if blocks else answer.strip()


def math_grade(reference,answer,*,timeout_seconds=5):
    from math_verify import parse,verify,LatexExtractionConfig,ExprExtractionConfig
    # Prefer explicit final boxes, with expression fallback. Never ask a paid judge.
    if isinstance(reference,list):
        if len(reference)!=1:
            raise ValueError('Multi-answer Olympiad requires its official judging protocol')
        reference=reference[0]
    gold_text=str(reference).strip()
    if '\\boxed' not in gold_text:gold_text='\\boxed{'+gold_text+'}'
    gold=parse(gold_text,extraction_config=[LatexExtractionConfig()], parsing_timeout=timeout_seconds,raise_on_error=True)
    if not gold:
        raise ValueError('Reference answer is not parseable under the frozen math grading protocol')
    predicted=parse(answer,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()], parsing_timeout=timeout_seconds,raise_on_error=True)
    return {'correct':bool(gold and predicted and verify(gold,predicted,timeout_seconds=timeout_seconds,raise_on_error=True)),
            'gold_parseable':bool(gold),'prediction_parseable':bool(predicted)}


def if_grade(key,task,answers,grader_root):
    from .benchmarks import decode
    if key=='multi_if':
        sys.path.insert(0,str(grader_root))
        metrics=module(Path(grader_root)/'metrics.py','ordinary_multi_if_metrics')
        values=[]
        for i,answer in enumerate(answers,1):
            ids=decode(task['raw'][f'turn_{i}_instruction_id_list'])
            kwargs=[decode(x) for x in decode(task['raw'][f'turn_{i}_kwargs'])]
            strict=metrics.gen_acc_strict({'response':answer,'instruction_id_list':ids,'kwargs':kwargs})
            loose=metrics.gen_acc_loose({'response':answer,'instruction_id_list':ids,'kwargs':kwargs})
            values.append({'strict':all(strict['follow_instruction_list']),
                           'loose':all(loose['follow_instruction_list']),
                           'strict_instructions':strict['follow_instruction_list'],
                           'loose_instructions':loose['follow_instruction_list']})
        return {'correct':all(v['strict'] for v in values),'turns':values,'last_turn_strict':values[-1]['strict']}
    if key=='ifeval':
        sys.path.insert(0,str(grader_root))
        from instruction_following_eval import evaluation_lib
    else:
        # Use the commit-pinned installed package: its import-time cache directory
        # is prepared at image build, not written under read-only source mounts.
        evaluation_lib=module(Path(grader_root)/'evaluation_lib.py','ordinary_ifbench_evaluation')
    raw=task['raw']
    kwargs=[{k:v for k,v in kw.items() if v is not None} for kw in raw['kwargs']]
    example=evaluation_lib.InputExample(key=raw['key'],instruction_id_list=raw['instruction_id_list'],
                                       prompt=raw['prompt'],kwargs=kwargs)
    strict=evaluation_lib.test_instruction_following_strict(example,{raw['prompt']:answers[0]})
    loose=evaluation_lib.test_instruction_following_loose(example,{raw['prompt']:answers[0]})
    return {'correct':strict.follow_all_instructions,'strict':strict.follow_all_instructions,'loose':loose.follow_all_instructions,
            'strict_instructions':strict.follow_instruction_list,'loose_instructions':loose.follow_instruction_list}


def bcb_grade(tasks,predictions,output):
    import importlib
    from .common import jsonl,run_id
    native=importlib.import_module('bigcodebench.evaluate')
    problems={t['raw']['task_id']:t['raw'] for t in tasks}
    folder=Path(output).parent/('bcb-native-'+run_id());folder.mkdir()
    samples=folder/'samples.jsonl'
    jsonl(samples,({'task_id':t['raw']['task_id'],
                   'solution':code(read(Path(predictions)/(digest(t['id'])+'.json'))['answers'][0])} for t in tasks))
    original=(native.get_bigcodebench,native.get_bigcodebench_hash)
    native.get_bigcodebench=lambda **kwargs:problems
    native.get_bigcodebench_hash=lambda **kwargs:digest(problems)
    try:
        native.evaluate(split='instruct',subset='full',samples=str(samples),execution='local',
                        pass_k='1',calibrated=True,parallel=8,no_gt=False)
    finally:
        native.get_bigcodebench,native.get_bigcodebench_hash=original
    evaluated=read(folder/'samples_eval_results.json')['eval']
    rates=read(folder/'samples_pass_at_k.json')
    if rates.get('failed_tasks'):
        raise RuntimeError('BCB canonical solutions failed environment verification: '+str(rates['failed_tasks']))
    if set(evaluated)!=set(problems):raise ValueError('Incomplete official BCB evaluation')
    return [{'id':t['id'],'correct':evaluated[t['raw']['task_id']][0]['status']=='pass'} for t in tasks]


def summarize(key,results):
    n=len(results)
    summary={'benchmark':key,'metric':'strict_prompt_accuracy' if key in ('ifeval','ifbench','multi_if') else
             'pass@1' if key in ('livecodebench','humaneval_plus','mbpp_plus','bigcodebench','taco_dev') else 'math_verify_accuracy',
             'count':n,'correct':sum(bool(r['correct']) for r in results),
             'score':sum(bool(r['correct']) for r in results)/n,'per_task':results}
    if key=='olympiad':summary['metric']='official_mathjudger_accuracy'
    if key in ('ifeval','ifbench'):
        summary['strict_prompt_accuracy']=sum(r.get('strict',r['correct']) for r in results)/n
        summary['loose_prompt_accuracy']=sum(r['loose'] for r in results)/n
        for mode in ('strict','loose'):
            values=[x for r in results for x in r[mode+'_instructions']]
            summary[mode+'_instruction_accuracy']=sum(values)/len(values)
        if key=='ifbench':
            summary.update(metric='loose_prompt_accuracy',correct=sum(r['loose'] for r in results),
                           score=summary['loose_prompt_accuracy'])
    if key=='multi_if':
        if any(len(r['turns'])!=3 for r in results):raise ValueError('Multi-IF requires three turns')
        turns=[v for r in results for v in r['turns']]
        summary['all_turns_strict_accuracy']=summary['score']
        summary['conversation_count']=n
        for mode in ('strict','loose'):
            summary[mode+'_prompt_accuracy']=sum(v[mode] for v in turns)/len(turns)
            values=[x for v in turns for x in v[mode+'_instructions']]
            summary[mode+'_instruction_accuracy']=sum(values)/len(values)
            summary[mode+'_prompt_accuracy_per_turn']=[sum(r['turns'][i][mode] for r in results)/n for i in range(3)]
        for mode in ('strict','loose'):
            summary[mode+'_instruction_accuracy_per_turn']=[
                sum(x for r in results for x in r['turns'][i][mode+'_instructions']) /
                sum(len(r['turns'][i][mode+'_instructions']) for r in results) for i in range(3)]
        four=[sum(summary[mode+'_'+level+'_accuracy_per_turn'][i]
                  for mode in ('strict','loose') for level in ('prompt','instruction'))/4 for i in range(3)]
        summary['four_metric_mean_per_turn']=four
        summary.update(metric='english_four_metric_mean_across_turns',correct=None,
                       score=sum(four)/3,scope='English subset, not the multilingual paper average')
    return summary


def lcb_parallel(tasks,predictions,grader_root):
    sys.path.insert(0,str(grader_root))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
    samples=[CodeGenerationProblem(**t['raw']).get_evaluation_sample() for t in tasks]
    generated=[[code(read(Path(predictions)/(digest(t['id'])+'.json'))['answers'][0])] for t in tasks]
    results,metadata=evaluate_generations(samples,generated,debug=False,num_process_evaluate=8,timeout=6)
    output=[]
    for i,t in enumerate(tasks):
        tests=results[i][0];info=metadata[i][0]
        if info.get('error_code')==-5:raise RuntimeError('Official LCB grader infrastructure failed: '+str(info))
        output.append({'id':t['id'],'correct':bool(tests) and all(v is True or v==1 for v in tests),
                       'tests':[bool(v is True or v==1) for v in tests],'metadata':info})
    return output


def grade(key,tasks_file,predictions,output,*,grader_root=None,evalplus_data=None):
    tasks=list(rows(tasks_file))
    predictions=Path(predictions)
    if key in ('livecodebench','humaneval_plus','mbpp_plus','bigcodebench','taco_dev'):
        if not Path('/.dockerenv').exists() or os.environ.get('SELF_CURATE_GRADER_SANDBOX')!='1':
            raise ValueError('Generated code requires the isolated, network-disabled grader container')
    results=[]
    for task in tasks:
        prediction=read(predictions/(digest(task['id'])+'.json'))
        if prediction.get('proxy_test') or prediction.get('task_id')!=task['id']:
            raise ValueError('Proxy or mismatched prediction cannot enter formal metrics')
    if key=='livecodebench':
        results=lcb_parallel(tasks,predictions,grader_root)
    elif key=='bigcodebench':
        results=bcb_grade(tasks,predictions,output)
    elif key in ('humaneval_plus','mbpp_plus'):
        if not evalplus_data:
            raise ValueError('Pinned complete EvalPlus data path required')
        os.environ['HUMANEVAL_OVERRIDE_PATH' if key=='humaneval_plus' else 'MBPP_OVERRIDE_PATH']=str(evalplus_data)
        from evalplus.evaluate import evaluate
        from .common import jsonl
        samples=Path(output).parent/'evalplus-samples.jsonl'
        jsonl(samples,({'task_id':task['raw']['task_id'],
                        'solution':code(read(predictions/(digest(task['id'])+'.json'))['answers'][0])} for task in tasks))
        native=Path(output).parent/'evalplus-native.json'
        evaluate(dataset='humaneval' if key=='humaneval_plus' else 'mbpp',samples=str(samples),parallel=8,output_file=str(native))
        result=read(native)
        for task in tasks:
            item=result['eval'][task['raw']['task_id']][0]
            results.append({'id':task['id'],'correct':item['base_status']=='pass' and item['plus_status']=='pass'})
    else:
        for task in tasks:
            prediction=read(predictions/(digest(task['id'])+'.json'))
            if prediction.get('proxy_test'):
                raise ValueError('Proxy API generations cannot enter formal benchmark results')
            answers=prediction['answers']
            if key in ('ifeval','ifbench','multi_if'):
                value=if_grade(key,task,answers,grader_root)
            elif key=='taco_dev':
                from .taco_grade import check
                value=check(task,code(answers[0]),grader_root)
            elif key=='livecodebench':
                sys.path.insert(0,str(grader_root))
                from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
                if key=='livecodebench':
                    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
                    problem=CodeGenerationProblem(**task['raw'])
                    sample=problem.get_evaluation_sample()
                else:
                    sample={'input_output':json.dumps(task['raw']['tests'])}
                test_results,metadata=check_correctness(sample,code(answers[0]),timeout=6,debug=False)
                if metadata.get('error_code')==-5:
                    raise RuntimeError('Official grader infrastructure failed: '+str(metadata))
                value={'correct':bool(test_results) and all(x is True or x==1 for x in test_results),
                       'tests':[bool(x is True or x==1) for x in test_results],'metadata':metadata}
            elif key=='olympiad':
                judger=module(Path(grader_root)/'inference/code/math_judger.py','ordinary_olympiad_judger').MathJudger()
                reference=task['reference']
                gold=','.join(reference) if isinstance(reference,list) else str(reference)
                precision=task['raw'].get('error')
                if precision is None or precision=='':
                    precision=1e-8
                elif isinstance(precision,str):
                    precision=float(precision)
                value={'correct':bool(judger.judge(gold,answers[0],precision=precision))}
            else:
                value=math_grade(task['reference'],answers[0])
            results.append({'id':task['id'],**value})
    if len(results)!=len(tasks) or not tasks:
        raise ValueError('Incomplete grading')
    summary=summarize(key,results)
    write(output,summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    for name in ('benchmark','tasks','predictions','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--grader-root')
    parser.add_argument('--evalplus-data')
    args=parser.parse_args()
    grade(args.benchmark,args.tasks,args.predictions,args.output,grader_root=args.grader_root,evalplus_data=args.evalplus_data)


if __name__=='__main__':
    main()
