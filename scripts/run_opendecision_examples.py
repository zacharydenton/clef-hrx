#!/usr/bin/env python3
"""Adapt pinned OpenDecision examples and run sequential guarded CLEF inference."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def literals(path):
    return {n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


def prepare(source):
    cases = []

    def add(name, group, request, expected=None, **extra):
        request['model'] = 'clef'
        cases.append(dict(id=name, group=group, request=request,
                          expected=expected or {}, **extra))

    for filename in ['m1_smoke_test.py', 'm1_smoke_test1.py']:
        tree = ast.parse((source / 'examples' / filename).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('choice', 'noul', 'score'):
                args = {kw.arg: ast.literal_eval(kw.value) for kw in n.keywords}
                state = args.pop('state')
                add(f'{filename}:{len(cases)}', 'text',
                    dict(state=state, questions={'decision': dict(type=n.func.attr, **args)}))
    # Extract constructor arguments without importing or executing upstream examples.
    tree = ast.parse((source / 'examples/m3_typesafe_sdk_demo.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'system_one')
    args = {k.arg: k.value for k in call.keywords}
    questions = {}
    for key, value in zip(args['questions'].keys, args['questions'].values):
        questions[ast.literal_eval(key)] = dict(type=value.func.id.lower(),
            **{k.arg: ast.literal_eval(k.value) for k in value.keywords})
    add('sdk_customer', 'text', dict(state=ast.literal_eval(args['state']), questions=questions))
    timing_tree = ast.parse((source / 'demos/images/measure_latency.py').read_text())
    assignment = next(n for n in ast.walk(timing_tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'requests' for t in n.targets))
    add('customer_routing', 'text', ast.literal_eval(assignment.value)['customer_routing'],
        {'department': 'billing', 'urgent': True})

    mapping = dict(zip(
        ['IMG_3718 Medium.jpeg', 'IMG_3751 Medium.jpeg', 'IMG_3975 Medium.jpeg',
         'IMG_4085 Medium.jpeg', 'IMG_4259 Medium.jpeg', 'IMG_4323.PNG',
         'IMG_4339 Medium.jpeg', 'IMG_4367 Medium.jpeg', 'PXL_20250524_124127738.jpg', 'binn.jpg', 'fenn.jpg'],
        ['bird', 'moon', 'cat-patio', 'bread', 'wooded-stream', 'patio-animals',
         'parked-cars', 'cat-window', 'bear', 'overflowing-bin', 'damaged-fence']))
    manifest = json.loads((source / 'demos/images/questions.json').read_text())
    for case in manifest['cases']:
        name = mapping[case['image']]
        image = source / f'docs/assets/images/{name}.jpg'
        assert image.is_file()
        add(name, 'images', dict(state='', images=[str(image.resolve())],
            questions={k: v['question'] for k, v in case['questions'].items()}),
            {k: v['expected'] for k, v in case['questions'].items()},
            upstream_image=case['image'], decision=case.get('decision'),
            notes={k: v['note'] for k, v in case['questions'].items() if v['note']})
    image_tree = ast.parse((source / 'examples/image_decisions.py').read_text())
    payload = next(n.value for n in ast.walk(image_tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
    image_questions = next(v for k, v in zip(payload.keys, payload.values)
                           if ast.literal_eval(k) == 'questions')
    add('image_api_bear', 'image_api', dict(state='',
        images=[str((source / 'docs/assets/images/bear.jpg').resolve())],
        questions=ast.literal_eval(image_questions)))

    gdpr = literals(source / 'benchmarks/gdpr_wiki/questions.py')
    questions, expected, score_expected = {}, {}, {}
    for kind in ['noul', 'choice', 'score']:
        for key, value in gdpr[kind.upper() + '_QUESTIONS'].items():
            questions[key] = dict(type=kind, **{k: value[k] for k in ('instructions', 'criteria') if k in value})
            if kind == 'score':
                score_expected[key] = value['expected_index']
            else:
                expected[key] = value['expected']
    add('gdpr', 'gdpr', dict(state=(source / 'tests/GDPR_wiki_test.txt').read_text(),
        questions=questions), expected, score_expected=score_expected)

    facts = [json.loads(line) for line in (source / 'benchmarks/insurance_claim/fact_cases.jsonl').read_text().splitlines()]
    questions, expected = {}, {}
    for fact in facts:
        question = dict(type='choice', instructions=(
            'Classify the evidence relation for the following proposition. Use only the supplied evidence; '
            'missing evidence is unknown, not a contradiction.\nSource identifiers: '
            + ', '.join(fact['source_ids']) + '\nProposition: ' + fact['proposition']
            + '\nExplicit opposite: ' + fact['contradiction']), criteria={
                'supports': 'Evidence supports the proposition and does not support its opposite.',
                'contradicts': 'Evidence supports the opposite and does not support the proposition.',
                'unknown': 'Evidence establishes neither the proposition nor its opposite.',
                'conflicted': 'Evidence supports both the proposition and its opposite.'})
        questions[fact['id']] = question
        expected[fact['id']] = fact['expected']
        add('insurance_oracle:' + fact['id'], 'insurance_oracle',
            dict(state=fact['evidence'], questions={fact['id']: question}),
            {fact['id']: fact['expected']})
    add('insurance_document', 'insurance_document',
        dict(state=(source / 'tests/sample_claim.txt').read_text(), questions=questions), expected)
    return cases


def predictions(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and 'questions' in row and 'timings' in row:
            rows.append(row)
    return rows


def summarize(cases, raw, source):
    assert len(raw) == 2 * len(cases), (len(raw), len(cases))
    spec = importlib.util.spec_from_file_location('od_rules', source / 'src/opendecision/rules.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    results, groups = [], {}
    for index, case in enumerate(cases):
        warm, row = raw[2 * index:2 * index + 2]
        assert warm['questions'] == row['questions'], case['id']
        assert warm['input_tokens'] == row['input_tokens']
        answers = {}
        assert {q['question_id'] for q in row['questions']} == set(case['request']['questions'])
        for q in row['questions']:
            probabilities = dict(zip(q['option_ids'], q['probabilities']))
            kind = case['request']['questions'][q['question_id']]['type']
            winner = max(probabilities, key=probabilities.get)
            answer = dict(winner=winner, probabilities=probabilities)
            if kind == 'noul':
                answer['answer'] = probabilities['true'] >= 0.5
            elif kind == 'score':
                answer['answer'] = sum(float(k) * v for k, v in probabilities.items())
            else:
                answer['answer'] = winner
            expected = case['expected'].get(q['question_id'])
            answer.update(expected=expected, correct=None if expected is None else answer['answer'] == expected)
            answers[q['question_id']] = answer
        ms = sum(row['timings'][k] for k in ['encode_ms', 'prepare_ms', 'inference_ms', 'readback_ms'])
        result = dict(id=case['id'], group=case['group'], answers=answers,
                      tokens=row['input_tokens'], resident_ms=ms, tok_s=row['input_tokens'] * 1000 / ms,
                      repeat_exact=True, timings=row['timings'])
        decision = case.get('decision')
        if 'animal_present' in answers and 'animal_type' in answers:
            decision = dict(presence_question='animal_present', choice_question='animal_type',
                            target_choice='bear', action='sound_alarm', otherwise='no_alarm')
        if decision:
            choice = answers[decision['choice_question']]
            trigger = (answers[decision['presence_question']]['answer']
                       and choice['answer'] == decision['target_choice']
                       and choice['probabilities'][choice['winner']] >= 0.6)
            result['action'] = decision['action'] if trigger else decision['otherwise']
        if case['id'] == 'insurance_document':
            facts = {k: rules.relation_status(v['answer']) for k, v in answers.items()}
            # Same deterministic inputs as upstream run_decisions.py.
            facts.update(treatment_within_24_hours='established', rental_necessity_established='unknown')
            result['composed'] = []
            for d in json.loads((source / 'benchmarks/insurance_claim/decisions.json').read_text()):
                actual = rules.evaluate_rule(d['rule'], facts)['status']
                result['composed'].append(dict(id=d['id'], answer=actual,
                    expected=d['expected'], correct=actual == d['expected']))
        results.append(result)
        group = groups.setdefault(case['group'], dict(correct=0, scored=0, tokens=0, resident_ms=0, requests=0))
        group['correct'] += sum(a['correct'] is True for a in answers.values())
        group['scored'] += sum(a['correct'] is not None for a in answers.values())
        group['tokens'] += result['tokens']
        group['resident_ms'] += ms
        group['requests'] += 1
    for group in groups.values():
        group['tok_s'] = group['tokens'] * 1000 / group['resident_ms']
    return dict(groups=groups, cases=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/OpenDecision'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['clef', 'clef-flash', 'clef-omni'])
    parser.add_argument('--binary', type=Path, default=Path('target/release/clef'))
    parser.add_argument('--precision', choices=['bf16', 'int4-mlp', 'int4'], default='bf16')
    parser.add_argument('--drop-checkpoint-cache', action='store_true',
                        help='Use the bench-internals loader cache advice; kernel selection is unchanged')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--cases-json', type=Path,
                        help='Run a prepared case list instead of the standard examples')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases_json.read_text()) if args.cases_json else prepare(args.source)
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    requests = args.output / 'requests.jsonl'
    requests.write_text(''.join(json.dumps(c['request']) + '\n' for c in cases for _ in range(2)))
    state = dict(source_revision=subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
                 cases=len(cases), questions=sum(len(c['request']['questions']) for c in cases),
                 method='One warmup then one measured call per example; sequential guarded runs; no truncation.',
                 precision=args.precision,
                 models={}, started=time.time())
    state['source_files_sha256'] = {
        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.source.rglob('*'))
        if p.is_file() and '.git' not in p.relative_to(args.source).parts
        and (p.suffix in ('.py', '.json', '.jsonl', '.txt')
             or 'docs/assets/images' in str(p))}
    state['implementation_revision'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip()
    state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for model in ['clef', 'clef-flash', 'clef-omni']:
        saved = args.output / (model + '.json')
        if saved.exists():
            state['models'][model] = json.loads(saved.read_text())['groups']
    print(json.dumps({k: v for k, v in state.items() if k not in ('models', 'source_files_sha256')}), flush=True)
    if args.prepare_only:
        return
    binary = args.output / ('clef-cache-loader' if args.drop_checkpoint_cache else 'clef')
    if not binary.exists():
        shutil.copy2(args.binary, binary)
    state['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLEF_')}
    if args.drop_checkpoint_cache:
        env['CLEF_BENCH_CHECKPOINT_DROP_CACHE'] = '1'
    for model in args.models:
        requests = args.output / (model + '-requests.jsonl')
        requests.write_text(''.join(json.dumps(dict(c['request'], model=model)) + '\n'
                                   for c in cases for _ in range(2)))
        budget = (40 if model == 'clef-flash' else 70) if args.precision == 'bf16' else 28
        command = [str(binary.resolve()), '--model', model, '--precision', args.precision,
                   '--memory-gib', str(budget), '--max-length', '65536', '--strict-context']
        if model != 'clef-flash':
            command += ['--model-dir', 'artifacts/model' if model == 'clef' else 'artifacts/model-omni-tiled']
        command += ['decide', str(requests), '--jsonl', '--raw']
        guard = args.output / (model + '-guard')
        full_command = ['python3', 'scripts/memory_guard.py', '--output', str(guard),
                        '--memory-gib', str(budget), '--', *command]
        state.update(current_model=model, command=full_command)
        (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
        print('Starting ' + model, flush=True)
        code = subprocess.call(full_command, env=env)
        gs = json.loads((guard / 'status.json').read_text())
        if code or gs['state'] != 'complete':
            state.update(state='stopped', guard_state=gs['state'], returncode=code)
            (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')
            raise RuntimeError(f'{model}: guard {gs["state"]}, exit {code}')
        summary = summarize(cases, predictions(guard / 'run.log'), args.source)
        summary.update(model=model, precision=args.precision)
        (args.output / (model + '.json')).write_text(json.dumps(summary, indent=2) + '\n')
        state['models'][model] = summary['groups']
        print(model + ': ' + json.dumps(summary['groups']), flush=True)
    state.update(state='complete', finished=time.time())
    (args.output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
