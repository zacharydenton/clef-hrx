#!/usr/bin/env python3
"""Gate an optimization on the same complete JevBench suite and report speed."""
import argparse
import json
import math
from pathlib import Path


def load(path):
    metadata = json.loads((path / 'metadata.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    records = [json.loads(line) for line in (path / 'scored.jsonl').read_text().splitlines()]
    tasks = {r['task_id']: r for r in records}
    assert len(tasks) == len(records) == metadata['count'] == summary['n_planned'] == 231
    assert summary['complete'] and summary['n_attempted'] == 231
    assert summary['n_correct'] == sum(bool(r['correct']) for r in records)
    return metadata, summary, tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--min-tps', type=float, help='Optional resident end-to-end throughput gate')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    bm, bs, br = load(a.baseline)
    cm, cs, cr = load(a.candidate)
    for key in ['dataset_sha256', 'benchmark_commit', 'checkpoint_revision']:
        assert bm[key] == cm[key], f'{key} differs'
    assert br.keys() == cr.keys()
    assert all(br[k]['input_tokens'] == cr[k]['input_tokens'] for k in br), 'token counts differ'
    assert all(set(br[k]['probs']) == set(cr[k]['probs']) for k in br), 'answer choices differ'
    finite = all(math.isfinite(p) for r in cr.values() for p in r['probs'].values())
    valid = finite and all(r['ok'] and r['strict_valid'] and not r['renormalized'] for r in cr.values())
    gained = [k for k in br if not br[k]['correct'] and cr[k]['correct']]
    lost = [k for k in br if br[k]['correct'] and not cr[k]['correct']]
    bt = bs['speed']['resident']['input_tokens_per_second']
    ct = cs['speed']['resident']['input_tokens_per_second']
    passed = valid and cs['n_correct'] >= bs['n_correct'] and (a.min_tps is None or ct >= a.min_tps)
    report = {
        'passed': passed, 'baseline_correct': bs['n_correct'], 'candidate_correct': cs['n_correct'],
        'cases': len(cr), 'strict_valid': valid, 'gained': gained, 'lost': lost,
        'max_probability_change': max(abs(br[k]['probs'][label] - cr[k]['probs'][label]) for k in br for label in br[k]['probs']),
        'baseline_resident_tok_s': bt, 'candidate_resident_tok_s': ct, 'speedup': ct / bt,
        'required_tok_s': a.min_tps, 'baseline_brier': bs['brier_mean'], 'candidate_brier': cs['brier_mean'],
        'baseline_ece': bs['ece']['ece'], 'candidate_ece': cs['ece']['ece'],
        'baseline_binary_sha256': bm['binary_sha256'], 'candidate_binary_sha256': cm['binary_sha256'],
        'note': 'Accuracy gates on total correct; probability drift and individual changed decisions are reported. Speed includes preparation and compilation, excluding only model loading with the first request.',
    }
    text = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    print(text, end='')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
