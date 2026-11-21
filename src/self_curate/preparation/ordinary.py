"""Official-field adapters: one problem/one fixed answer, native labels, no paid teacher."""
import ast
import json
import re
from collections import Counter
from pathlib import Path
from self_curate.experiments.ordinary.common import digest, jsonl, rows, write

CODING_SKILLS = {'amortized_analysis','bit_manipulation','complete_search','data_structures',
                 'dynamic_programming','greedy_algorithms','range_queries','sorting'}

MATH = {'Algebra': 'algebra', 'Geometry': 'geometry', 'Number Theory': 'number_theory',
        'Combinatorics': 'combinatorics', 'Calculus': 'calculus', 'Inequalities': 'algebra',
        'Logic and Puzzles': 'logic', 'Probability': 'probability', 'Statistics': 'probability'}
IF_GROUP = {'keywords': 'lexical', 'language': 'language', 'length_constraints': 'length',
            'detectable_content': 'required_content', 'detectable_format': 'format',
            'combination': 'composition', 'startend': 'boundary', 'change_case': 'case',
            'punctuation': 'punctuation', 'format': 'format', 'length': 'length',
            'content': 'required_content', 'style': 'style'}


# Native Personas IF constraint names differ from IFEval's underscored IDs.
IF_ALIASES = {'case': ('case',), 'specific_ending': ('boundary',),
              'give_two_responses': ('composition',), 'repeat_the_prompt': ('composition',),
              'include_keywords': ('lexical',), 'use_quotation': ('format',),
              'in_english_and_capital': ('language','case'), 'response_language': ('language',)}


def native_if_skills(labels):
    skills=set();unknown=[]
    for label in labels:
        prefix=re.sub(r'[\s_-]+','_',str(label).split(':',1)[0].strip().lower())
        groups=IF_ALIASES.get(prefix)
        if groups is None and prefix in IF_GROUP:groups=(IF_GROUP[prefix],)
        if groups is None:unknown.append(str(label))
        else:skills.update(groups)
    return sorted(skills),unknown


def sequence(value):
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        raise ValueError('Label/list field has unsupported type')
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(value)
            if isinstance(result, (tuple, list)):
                return list(result)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return [value.strip()]


def normalize_question(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def adapt(domain, row, source):
    descriptor=source.get('source',source)
    source={k:descriptor[k] for k in ('repo','revision','split','source_id') if k in descriptor}
    if domain=='math' and 'prompt' in row and 'solution' not in row:
        raise RuntimeError('Math20k has no SFT solution or skill. Enrich by source-bound reference join; never use final ground_truth as a worked solution.')
    if domain == 'coding':
        prompt = row['question'].strip()
        if str(row.get('picture_num', '0')).strip() not in ('', '0', '0.0', 'None'):
            raise ValueError('image_required')
        solutions = sequence(row['solutions'])
        solutions = [s for s in solutions if isinstance(s, str) and s.strip()]
        if not solutions:
            raise ValueError('no_reference_solution')
        # Canonical answer is invariant to file order and model/selection method.
        answer = min(solutions, key=lambda s: digest(s)).strip()
        ast.parse(answer)
        tests = json.loads(row['input_output']) if isinstance(row['input_output'], str) else row['input_output']
        if not isinstance(tests, dict) or not tests.get('inputs') or len(tests['inputs']) != len(tests.get('outputs', [])):
            raise ValueError('invalid_tests')
        raw_labels = sequence(row.get('skill_types'))
        normalized={str(label):re.sub(r'[^a-z0-9]+','_',str(label).lower()).strip('_') for label in raw_labels if str(label).strip()}
        skills=sorted({skill for skill in normalized.values() if skill in CODING_SKILLS})
        unmapped=[label for label,skill in normalized.items() if skill not in CODING_SKILLS]
        starter = row.get('starter_code', '').strip()
        model_prompt = prompt + ('\n\nStarter code:\n' + starter if starter else '') + '\n\nReturn a complete Python solution in a fenced code block.'
        extra = {'url': row.get('url'), 'tags': sequence(row.get('tags')), 'tests': tests,
                 'starter_code': starter, 'difficulty': row.get('difficulty'),'unmapped_skill_types':unmapped}
    elif domain == 'math':
        if not isinstance(row.get('problem'),str) or not isinstance(row.get('solution'),str):
            raise ValueError('empty_supervision')
        if row.get('problem_is_valid') not in (None, 'Yes', True) or row.get('solution_is_valid') not in (None, 'Yes', True):
            raise ValueError('source_invalid')
        final_answer=str(row.get('answer') or '').strip().lower()
        is_proof=str(row.get('question_type','')).strip().lower()=='proof' or final_answer=='proof'
        if not is_proof and final_answer in ('','notfound'):
            raise ValueError('not_automatically_verifiable')
        prompt, answer = row['problem'].strip(), row['solution'].strip()
        raw_labels = [row.get('problem_type')]
        skills = [MATH[row['problem_type']]] if row.get('problem_type') in MATH else []
        model_prompt = prompt + ('\n\nProvide a complete proof.' if is_proof else '\n\nShow your solution and put the final answer in \\boxed{}.')
        extra = {'reference_provenance':row.get('reference_provenance'), 'answer': row.get('answer'), 'is_proof':is_proof, 'evaluation_eligible':not is_proof, 'question_type': row.get('question_type'), 'source': row.get('source')}
    elif domain == 'if':
        messages = row['messages']
        if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
            raise ValueError('unsupported_conversation')
        prompt, answer = row['prompt'].strip(), messages[1]['content'].strip()
        if messages[0]['content'].strip() != prompt:
            raise ValueError('prompt_mismatch')
        raw_labels = sequence(row.get('constraints'))
        # These are taxonomy IDs, not reconstructed checker kwargs.
        skills,unmapped = native_if_skills(raw_labels)
        model_prompt = prompt
        extra = {'constraints': raw_labels, 'source_id': row.get('id'),
                 'checker_kwargs_available': False,'unmapped_constraints':unmapped}
    else:
        raise ValueError('Unknown domain')
    if not prompt or not answer:
        raise ValueError('empty_supervision')
    qhash = digest(normalize_question(prompt))
    return {'schema': 'ordinary-example-v1', 'id': domain + '-' + qhash,
            'answer_chars':len(answer), 'prompt_chars':len(prompt),
            'domain': domain, 'question_hash': qhash, 'source': source,
            'content_hash': digest(row), 'skills': skills, 'raw_labels': raw_labels,
            'label_status': 'native' if skills and not extra.get('unmapped_constraints') and not extra.get('unmapped_skill_types') else 'missing',
            'label_version': 'native-map-v2' if domain in ('coding','if') else 'native-map-v1',
            'messages': [{'role': 'user', 'content': model_prompt}, {'role': 'assistant', 'content': answer}],
            'metadata': extra}


def prepare(domain, raw_file, output, *, source, blocked_questions=(), tokenizer=None,
            max_length=None, preview=False):
    if tokenizer is not None and (isinstance(max_length,bool) or not isinstance(max_length,int) or max_length<=0):
        raise ValueError('Tokenized admission requires a positive integer max_length')
    length_counts=Counter()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Prepared pools are immutable; use a new output directory')
    output.mkdir(parents=True)
    seen, excluded, coverage = set(blocked_questions), Counter(), Counter()
    accepted = 0
    def converted():
        nonlocal accepted
        for index, row in enumerate(rows(raw_file)):
            try:
                record = adapt(domain, row, source)
                if record['question_hash'] in seen:
                    raise ValueError('duplicate_or_heldout_question')
            except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                excluded[str(exc)] += 1
                with (output/'excluded.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'row': index, 'source_hash': digest(row), 'reason': str(exc)})+'\n')
                continue
            if tokenizer is not None:
                from self_curate.training.ordinary import encode_example
                try:
                    encoded = encode_example(record, tokenizer, 2**63-1)
                    length_counts[len(encoded['input_ids'])]+=1
                    if len(encoded['input_ids'])>max_length:raise ValueError('over_length')
                except ValueError as exc:
                    if str(exc) not in ('over_length','empty_supervision'):raise
                    excluded[str(exc)]+=1
                    with (output/'excluded.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'row':index,'source_hash':digest(row),'reason':str(exc)})+'\n')
                    continue
                record['length'] = len(encoded['input_ids'])
                record['supervised_tokens'] = sum(x != -100 for x in encoded['labels'])
                record['answer_tokens']=record['supervised_tokens']
                record['prompt_tokens']=len(tokenizer.encode(record['messages'][0]['content'],add_special_tokens=False))
                from self_curate.preparation.ordinary_tokenization import renderer
                record['length_tokenization']=renderer(tokenizer).identity.to_dict()
                record['tokenizer_hash']=record['length_tokenization']['tokenizer_hash']
            seen.add(record['question_hash'])
            coverage.update(record['skills'])
            accepted += 1
            yield record
    jsonl(output/'pool.jsonl', converted())
    from self_curate.experiments.ordinary.common import sha
    summary = {'domain': domain, 'source': source, 'raw_sha256': sha(raw_file), 'count': accepted,
               'coverage': dict(coverage), 'excluded': dict(excluded), 'preview': preview,
               'tokenized': tokenizer is not None, 'max_length': max_length,
               'pool_sha256': sha(output/'pool.jsonl'), 'label_policy': 'retain_missing_for_random_and_full',
               'fresh_solution_execution': False}
    if tokenizer is not None:
        summary['length_audit_before_length_filter']=length_distribution(length_counts)
    write(output/'manifest.json', summary)
    return summary


def length_distribution(counts):
    """Exact whole-pool lengths after structural/isolation checks, before length rejection."""
    total=sum(counts.values())
    result={'count':total,'scope':'after structural and question-isolation checks, before max_length filtering',
            'tokens':'full nonthinking chat template including prompt and answer',
            'over_limit':{str(limit):sum(n for length,n in counts.items() if length>limit) for limit in (4096,8192,12288,16384,32768)}}
    if not total:return {**result,'min':None,'max':None,'p50':None,'p95':None,'p99':None,'mean':None}
    import math
    def quantile(q):
        cumulative=0
        for length,n in sorted(counts.items()):
            cumulative+=n
            if cumulative>=math.ceil(q*total):return length
    return {**result,'min':min(counts),'max':max(counts),'p50':quantile(.5),'p95':quantile(.95),'p99':quantile(.99),
            'mean':sum(length*n for length,n in counts.items())/total}
