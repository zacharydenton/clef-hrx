#!/usr/bin/env python3
"""Gate same-model, same-precision optimizations on complete WorkflowEvals runs."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    def load(root):
        manifest=json.loads((root/'native-manifest.json').read_text())
        report=json.loads((root/'summary.json').read_text())
        if not report['complete'] or report['limited_smoke_run']:
            raise ValueError('only complete, unlimited evaluations may be compared')
        if sum(w['planned_cases'] for w in report['workflows'].values())!=705:
            raise ValueError('all 705 cases are required')
        return manifest,report
    bm,br=load(baseline);cm,cr=load(candidate)
    for key in ['workflow_commit','dependency_lock_sha256','datasets']:
        if bm[key]!=cm[key]: raise ValueError(f'{key} differs')
    for key in ['model','checkpoint_revision','precision','max_length']:
        if bm['configuration'][key]!=cm['configuration'][key]: raise ValueError(f'{key} differs')
    changes=[];passed=True
    if br['workflows'].keys()!=cr['workflows'].keys(): raise ValueError('workflow selection differs')
    for name,old in br['workflows'].items():
        new=cr['workflows'][name]
        if old['cases']!=new['cases'] or old['planned_cases']!=new['planned_cases']:
            raise ValueError(f'{name}: case counts differ')
        passed &= new['failed_cases']<=old['failed_cases']
        old_scores={s['label_set_id']:s for s in old['scores']}
        new_scores={s['label_set_id']:s for s in new['scores']}
        if old_scores.keys()!=new_scores.keys() or 'labels-consensus' not in old_scores:
            raise ValueError('reference label sets differ or consensus is missing')
        for label,previous in old_scores.items():
            current=new_scores[label]
            if previous['metric_denominators']!=current['metric_denominators']:
                raise ValueError(f'{name}/{label}: scoring denominators differ')
            if previous['metrics'].keys()!=current['metrics'].keys(): raise ValueError('metrics differ')
            for metric,value in previous['metrics'].items():
                next_value=current['metrics'][metric]
                if value is None or next_value is None: raise ValueError('unscored metric')
                gated=label=='labels-consensus'
                if gated: passed &= next_value>=value
                changes.append({'workflow':name,'label_set':label,'metric':metric,
                                'baseline':value,'candidate':next_value,'gated':gated})
    return {'passed':bool(passed),'metrics':changes,'baseline_speed':br['speed'],
            'candidate_speed':cr['speed'],'note':'Consensus metrics are gated per workflow; all label sets are reported. Dynamic branches may change the native request mix.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args(); result=compare(args.baseline,args.candidate)
    text=json.dumps(result,indent=2)+'\n'
    if args.output: args.output.write_text(text)
    print(text,end='')
    return int(not result['passed'])

if __name__=='__main__': raise SystemExit(main())
