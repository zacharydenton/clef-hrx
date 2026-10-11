#!/usr/bin/env python3
"""Run both models and all precision modes sequentially with an immutable binary.

WorkflowEvals resumes through upstream case checkpoints. Failed context runs get
new attempt directories; successful jobs are never repeated on resume.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=Path('target/release/clef'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clef-dir', type=Path, default=Path('artifacts/model'))
    p.add_argument('--flash-dir', type=Path, default=Path('artifacts/model-flash'))
    p.add_argument('--workflow-python', type=Path, default=Path('artifacts/WorkflowEvals/.venv/bin/python'))
    p.add_argument('--suites', nargs='+', choices=['context', 'workflowevals'], default=['context','workflowevals'])
    p.add_argument('--resume', action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    binary=args.output/'clef'
    runners={name:Path(__file__).with_name(f'{name}.py') for name in ['qualify_context','eval_workflowevals','bench_workload']}
    manifest={'binary_sha256':digest(args.binary),'runners':{k:digest(v) for k,v in runners.items()},
              'models':{'clef':str(args.clef_dir.resolve()),'clef-flash':str(args.flash_dir.resolve())},
              'suites':args.suites,'precisions':['bf16','int4-mlp','int4'],'memory_gib':80,
              'workflow_python':str(args.workflow_python.absolute())}
    manifest_path=args.output/'manifest.json'
    status_path=args.output/'status.json'
    if args.resume:
        if json.loads(manifest_path.read_text())!=manifest or digest(binary)!=manifest['binary_sha256']:
            p.error('resume requires identical binary, runners, model paths and settings')
        status=json.loads(status_path.read_text())
    else:
        shutil.copy2(args.binary,binary)
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        status={'complete':False,'jobs':{}}
    def save():
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
    save()
    for suite in args.suites:
        for model in ['clef-flash','clef']:
            for precision in manifest['precisions']:
                key=f'{suite}-{model}-{precision}'
                job=status['jobs'].get(key,{})
                if job.get('state')=='complete': continue
                if digest(binary)!=manifest['binary_sha256'] or any(digest(runners[k])!=v for k,v in manifest['runners'].items()):
                    raise RuntimeError('binary or runner changed during qualification')
                attempt=job.get('attempt',0)+1
                destination=args.output/key
                runner='eval_workflowevals' if suite=='workflowevals' else 'qualify_context'
                interpreter=str(args.workflow_python.absolute()) if suite=='workflowevals' else sys.executable
                command=[interpreter,str(runners[runner]),'--binary',str(binary),'--model',model,
                         '--model-dir',manifest['models'][model],'--precision',precision,'--output',str(destination)]
                if destination.exists():
                    if suite=='workflowevals': command+=['--resume']
                    else:
                        destination=args.output/f'{key}-attempt-{attempt}'
                        command[-1]=str(destination)
                job={'state':'running','attempt':attempt,'command':command,'started':time.time(),'output':str(destination)}
                status['jobs'][key]=job;save()
                print('Starting',key,flush=True)
                with (args.output/f'{key}-{attempt}.log').open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                complete=result.returncode==0
                if complete and suite=='workflowevals':
                    report=json.loads((destination/'summary.json').read_text())
                    complete=report['complete'] and not report['limited_smoke_run']
                job.update(state='complete' if complete else 'failed',exit_code=result.returncode,finished=time.time())
                save();print(job['state'],key,flush=True)
    status['complete']=len(status['jobs'])==len(args.suites)*6 and all(j['state']=='complete' for j in status['jobs'].values())
    save()
    return 0 if status['complete'] else 1

if __name__=='__main__':
    raise SystemExit(main())
