#!/usr/bin/env python3
"""Write a compact report from saved OpenDecision example runs."""
import argparse
import json
from pathlib import Path
import re
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--source', type=Path, default=Path('artifacts/OpenDecision'))
    p.add_argument('--output', type=Path, default=Path('docs/opendecision-examples.md'))
    p.add_argument('--json-output', type=Path, default=Path('docs/results/opendecision-20261016.json'))
    args = p.parse_args()
    cases = json.loads((args.results / 'cases.json').read_text())
    images = {c['id']: c for c in cases if c['group'] == 'images'}
    upstream = []
    for section in (args.source / 'docs/image-examples.md').read_text().split('\n### ')[1:]:
        name = re.search(r'assets/images/(.*?)\.jpg', section)[1]
        answers = {}
        for key, kind, blob in re.findall(r'\*\*`([^`]+)`\*\* \((noul|choice)\).*?```json\s*(.*?)\s*```', section, re.S):
            raw = json.loads(blob)
            actual = raw['noul'] >= .5 if kind == 'noul' else raw['choice']
            expected = images[name]['expected'][key]
            answers[key] = dict(answer=actual, expected=expected,
                                correct=None if expected is None else actual == expected)
        assert set(answers) == set(images[name]['request']['questions'])
        upstream.append(dict(id=name, answers=answers,
            median_ms=int(re.search(r'API call: (\d+) ms', section)[1])))
    models = {m: json.loads((args.results / (m + '.json')).read_text())
              for m in ['clef', 'clef-flash', 'clef-omni']
              if (args.results / (m + '.json')).exists()}
    assert models, 'No completed runs'
    status = json.loads((args.results / 'status.json').read_text())
    guards = {}
    for model in models:
        guard = args.results / (model + '-guard')
        guard_status = json.loads((guard / 'status.json').read_text())
        models[model]['precision'] = guard_status['command'][guard_status['command'].index('--precision') + 1]
        clients = set()
        for line in (guard / 'telemetry.jsonl').read_text().splitlines():
            row = json.loads(line)
            clients.update(pid for pid, info in row['process_memory'].items() if not info['owned'])
        guards[model] = dict(status=guard_status, other_gpu_clients=sorted(clients))
    payload = dict(provenance=status, guards=guards, models=models, upstream_images=upstream)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + '\n')
    names = {'clef': 'CLEF', 'clef-flash': 'Flash', 'clef-omni': 'Omni'}
    out = ['# OpenDecision examples on CLEF\n',
           f"[OpenDecision source](https://github.com/deepanwadhwa/OpenDecision/tree/{status['source_revision']}). "
           '38 requests, 91 questions per model on Radeon 8060S.\n',
           '## Results\n',
           '| Model | Precision | Images | GDPR | Insurance: selected evidence | Insurance: full document | Composed decisions |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for model, report in models.items():
        g = report['groups']
        composed = next(c['composed'] for c in report['cases'] if c['id'] == 'insurance_document')
        out.append(f"| {names[model]} | {report.get('precision', 'bf16').upper()} | {g['images']['correct']}/26 | {g['gdpr']['correct']}/10 | "
                   f"{g['insurance_oracle']['correct']}/17 | {g['insurance_document']['correct']}/17 | "
                   f"{sum(c['correct'] for c in composed)}/10 |")
    original_correct = sum(a['correct'] is True for c in upstream for a in c['answers'].values())
    out += ['', f"OpenDecision's saved gallery matches **{original_correct}/26** labelled answers. "
            'Its published document results are 9/10 GDPR, 17/17 insurance facts and 10/10 composed decisions. '
            'These are saved upstream results, not a local rerun of OpenDecision.\n',
            'Five image questions have no verified label and are unscored. All 17 insurance fact labels '
            'are `supports`; that track does not test discrimination between relation classes. '
            'The composed rules retain the two upstream deterministic facts (treatment within 24 hours '
            'and unknown rental necessity). Subjective scores and unlabelled text examples are shown below.\n',
            '## Speed\n',
            '| Model | Overall input tok/s | Text median ms | Image median ms | Image input tok/s | GDPR ms | Full insurance ms |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, report in models.items():
        c = report['cases']
        text_ms = statistics.median(x['resident_ms'] for x in c if x['group'] == 'text')
        image_ms = statistics.median(x['resident_ms'] for x in c if x['group'] == 'images')
        gdpr = next(x['resident_ms'] for x in c if x['id'] == 'gdpr')
        insurance = next(x['resident_ms'] for x in c if x['id'] == 'insurance_document')
        overall = sum(x['tokens'] for x in c) * 1000 / sum(x['resident_ms'] for x in c)
        out.append(f"| {names[model]} | {overall:,.1f} | {text_ms:,.1f} | {image_ms:,.1f} | "
                   f"{report['groups']['images']['tok_s']:,.1f} | {gdpr:,.1f} | {insurance:,.1f} |")
    out += ['', 'One warmup and one timed repeat per request in a resident process. Timing includes '
            'encoding/media preparation, GPU execution and readback; loading is excluded. '
            'Every timed output matches its warmup exactly. Runs include intermittent GPU contention. '
            'GPU clients and memory telemetry are saved '
            'with each run. Single repeats are illustrative latency measurements.\n',
            'OpenDecision reports median HTTP times from five requests on Apple M3; the gallery uses '
            'a small SigLIP2 image model. Its hardware, timing boundary and image preprocessing differ '
            'from these native CLEF runs.\n',
            '## Answers\n',
            '| Example / question | Expected | ' + ' | '.join(names[m] for m in models) + ' |',
            '| --- | --- | ' + ' | '.join('---' for _ in models) + ' |']
    indexed = {m: {c['id']: c for c in r['cases']} for m, r in models.items()}

    def show(value):
        if value is None:
            return '—'
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '\\|')

    for case in cases:
        if case['group'] == 'insurance_oracle':
            continue  # Full-document results below cover the same propositions.
        for key in case['request']['questions']:
            values = []
            for model in models:
                answer = indexed[model][case['id']]['answers'][key]
                value = show(answer['answer'])
                if answer['correct'] is False:
                    value = '**' + value + '**'
                values.append(value)
            out.append(f"| {case['id']} / {key} | {show(case['expected'].get(key))} | " + ' | '.join(values) + ' |')
    out += ['', 'Bold answers differ from the supplied labels. Raw probabilities, timings, '
            'composed decisions and all selected-evidence results are in the '
            f'[JSON report](results/{args.json_output.name}).\n',
            '## Reproduction\n',
            'The adapter reads literal upstream examples without running their model code. '
            'Images use the published gallery assets with unchanged question captions. '
            'GDPR and insurance receive the complete documents, without retrieval or truncation. '
            'Insurance `Relation` is represented as a four-option CLEF `choice`. '
            'The generic image API example uses the supplied bear photo. The interactive Doom '
            'game has no fixed request corpus and is outside this run.\n',
            '```sh',
            'git clone https://github.com/deepanwadhwa/OpenDecision.git artifacts/OpenDecision',
            f"git -C artifacts/OpenDecision checkout {status['source_revision']}",
            'cargo build --release --locked --features bench-internals',
            'python3 scripts/run_opendecision_examples.py --models clef clef-flash --drop-checkpoint-cache --output tests/results/opendecision-repro',
            'python3 scripts/run_opendecision_examples.py --models clef-omni --precision int4 --drop-checkpoint-cache --output tests/results/opendecision-repro',
            'python3 scripts/report_opendecision_examples.py tests/results/opendecision-repro',
            '```\n',
            'The runner uses local CLEF and Omni checkpoints at `artifacts/model` and '
            '`artifacts/model-omni-tiled`, and the pinned Flash Hugging Face cache. '
            'GPU jobs run sequentially under the existing memory guard. The cache-releasing loader '
            'uses `CLEF_BENCH_CHECKPOINT_DROP_CACHE=1`; this advises away consumed checkpoint pages '
            'without changing weights, precision or kernel selection. Flash used the ordinary build; '
            'CLEF and Omni use this loader option. Omni BF16 attempts stopped at the unchanged '
            '1 GiB additional-swap guard before producing predictions; the reported Omni run uses INT4. '
            'Raw outputs and telemetry remain in the chosen output directory.']
    args.output.write_text('\n'.join(out) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
