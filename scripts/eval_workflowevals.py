#!/usr/bin/env python3
"""Run pinned WorkflowEvals workflows/scoring against a resident local CLEF CLI.

Use artifacts/WorkflowEvals/.venv/bin/python after syncing its pinned uv.lock.
No API calls or reference answers are used by the inference adapter.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Literal

WORKFLOW_COMMIT = '0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb'
DATASETS = {
    'invoice_processing': '6beeb2d2acd65c086c835022f5f4d7434114cafc',
    'customer_service': 'b1342f5a704587dbc465867c38c2694348ff86e4',
    'agent_trace_observability': '8635540973910a92465fe2bc53e195375aa6e1a8',
    'security_incidents': 'fbe1ea5c69cf494157fd23f2002a0d9d9a418443',
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def response_from_raw(request, raw):
    """Preserve native probabilities and expected scores without decimal rounding."""
    from typesafe_sdk import SystemOneResponse
    predictions = {q['question_id']: q for q in raw['questions']}
    if len(predictions) != len(raw['questions']) or predictions.keys() != request['questions'].keys():
        raise ValueError('Native question IDs differ from the request')
    answers = {}
    for qid, question in request['questions'].items():
        prediction = predictions[qid]
        keys, values = prediction['option_ids'], prediction['probabilities']
        kind = question['type']
        expected = ({'true', 'false'} if kind == 'noul' else
                    set(question['criteria']) if kind == 'choice' else
                    {str(i) for i in range(len(question['criteria']))})
        if len(keys) != len(set(keys)) or set(keys) != expected or len(values) != len(keys):
            raise ValueError(f'{qid}: native options differ from the requested schema')
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > 1e-5:
            raise ValueError(f'{qid}: invalid native probability distribution')
        probs = dict(zip(keys, values, strict=True))
        if kind == 'noul':
            answer = {'type': kind, 'noul': probs['true']}
        elif kind == 'choice':
            winner = max(probs, key=probs.get)
            answer = {'type': kind, 'choice': winner, 'confidence': probs[winner], 'probabilities': probs}
        elif kind == 'score':
            answer = {'type': kind, 'score': sum(int(k) * p for k, p in probs.items()),
                      'confidence': max(values), 'probabilities': {int(k): p for k, p in probs.items()},
                      'legend': dict(enumerate(question['criteria']))}
        else:
            raise ValueError(f'Unsupported question kind: {kind}')
        answers[qid] = answer
    return SystemOneResponse.model_validate({
        'model': request.get('model', 'clef') + '-hrx', 'usage': {'input_tokens': raw['input_tokens'], 'output_tokens': 0},
        'answers': answers,
    })


class NativeCLI:
    def __init__(self, command, output, max_length):
        self.command, self.max_length = command, max_length
        self.workflow = None
        self.first = True
        self.failed = False
        self.calls = []
        self.stderr = (output / 'native-stderr.log').open('a')
        self.log = (output / 'native-calls.jsonl').open('a')
        self.preflight_log = (output / 'preflight.jsonl').open('a')
        self.encode_command = command[:command.index('decide')] + ['encode']
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1)
        self.lock = threading.Lock()

    def system_one(self, *, model, state, questions):
        from core.question_types import question_record
        request = {'model': model, 'state': state,
                   'questions': {key: question_record(q) for key, q in questions.items()}}
        with self.lock:
            # Upstream Session recursively halves sets rejected with this exact
            # capacity marker. Preflight protects the resident process from a
            # schema overflow and avoids silently truncating the workflow state.
            from core.session import GRAMMAR_LIMIT
            wire = json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n'
            preflight_started = time.perf_counter()
            encoded = subprocess.run(self.encode_command, input=wire, text=True, capture_output=True)
            preflight_seconds = time.perf_counter() - preflight_started
            token_count = len(json.loads(encoded.stdout)['input_ids']) if encoded.returncode == 0 else None
            capacity = 'context capacity exceeded:' in encoded.stderr or 'schema and media require' in encoded.stderr
            self.preflight_log.write(json.dumps({'workflow': self.workflow, 'question_count': len(questions),
                                                 'input_tokens': token_count, 'capacity_rejected': capacity,
                                                 'wall_seconds': preflight_seconds, 'error': encoded.stderr,
                                                 'request_sha256': hashlib.sha256(wire.encode()).hexdigest()}) + '\n')
            self.preflight_log.flush()
            if capacity:
                raise ValueError(f'{GRAMMAR_LIMIT}: native input exceeds {self.max_length} tokens')
            if encoded.returncode:
                raise ValueError(f'Native encoding failed: {encoded.stderr.strip()}')
            if self.failed or self.process.poll() is not None:
                self.failed = True
                raise RuntimeError('Native CLI stopped; see native-stderr.log')
            started = time.perf_counter()
            try:
                self.process.stdin.write(wire)
                self.process.stdin.flush()
                line = self.process.stdout.readline()
                elapsed = time.perf_counter() - started
                if not line:
                    self.failed = True
                    raise RuntimeError('Native CLI returned EOF; see native-stderr.log')
                raw = json.loads(line)
                if raw['input_tokens'] != token_count:
                    raise ValueError('Native input length differs from preflight')
                response = response_from_raw(request, raw)
                record = {'workflow': self.workflow, 'first_request_including_load': self.first,
                          'wall_seconds': elapsed, 'input_tokens': raw['input_tokens'],
                          'preflight_seconds': preflight_seconds,
                          'at_context_limit': raw['input_tokens'] == self.max_length,
                          'request': request, 'prediction': raw}
                self.log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                self.log.flush()
                self.calls.append(record)
                self.first = False
                return response
            except (BrokenPipeError, OSError):
                self.failed = True
                raise

    def close(self):
        # Upstream closes its client after each case. The owner shuts down the
        # shared process only after all selected workflows have finished.
        pass

    def shutdown(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=15)
        finally:
            self.log.close()
            self.preflight_log.close()
            self.stderr.close()


def summarize_calls(calls):
    resident = [r for r in calls if not r['first_request_including_load']]
    elapsed = sum(r['wall_seconds'] for r in resident)
    return {'calls': len(calls), 'resident_calls': len(resident),
            'resident_input_tokens': sum(r['input_tokens'] for r in resident),
            'resident_wall_seconds': elapsed,
            'resident_input_tokens_per_second': sum(r['input_tokens'] for r in resident) / elapsed if elapsed else None,
            'at_context_limit_calls': sum(r['at_context_limit'] for r in calls),
            'resident_phase_seconds': {phase: sum(r['prediction']['timings'][phase + '_ms'] for r in resident) / 1000
                                       for phase in ('encode', 'prepare', 'inference', 'readback')}}


def next_case_ids(planned, completed, budget):
    """Bound scheduling without changing the full manifest or completed results."""
    if budget is not None and budget < 1:
        raise ValueError('new-case budget must be positive')
    return [case_id for case_id in planned if case_id not in completed][:budget]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('artifacts/WorkflowEvals'))
    parser.add_argument('--binary', type=Path, default=Path('target/release/clef'))
    parser.add_argument('--model', choices=('clef', 'clef-flash', 'clef-omni'), default='clef')
    parser.add_argument('--model-dir', type=Path, default=Path('artifacts/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workflows', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--limit', type=int, help='Smoke run only; omit for the complete suite')
    parser.add_argument('--max-length', type=int, default=65536)
    parser.add_argument('--precision', choices=('bf16', 'int4-mlp', 'int4'), default='bf16')
    parser.add_argument('--memory-gib', type=int, default=80)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--new-cases-per-workflow', type=int,
                        help='Run at most this many additional cases per workflow; retain the full suite plan for resume')
    parser.add_argument('--dry-run', action='store_true', help='Validate pinned datasets without GPU inference')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.new_cases_per_workflow is not None and args.new_cases_per_workflow < 1:
        parser.error('--new-cases-per-workflow must be positive')
    if not 1 <= args.max_length <= 65536:
        parser.error('--max-length must be 1..65536')
    benchmark = args.benchmark.resolve()
    commit = subprocess.check_output(['git', '-C', str(benchmark), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != WORKFLOW_COMMIT or subprocess.check_output(['git', '-C', str(benchmark), 'diff', 'HEAD']):
        parser.error('WorkflowEvals source must match the pinned, unmodified commit')
    sys.path.insert(0, str(benchmark))
    from core.contract import ModelIdentity
    from core.io import write_record
    from core.providers import ModelConfig
    from core.runner import execute, plan, load_checkpoints
    from core.scoring import build_score_data
    from core.workflow import load
    from bench_workload import hardware

    class NativeConfig(ModelConfig):
        provider: Literal['native'] = 'native'
        binary_sha256: str
        checkpoint_revision: str
        runner_sha256: str
        max_length: int
        precision: str
        experiment_environment: dict[str, str]
        batching: str = 'joint; recursively halve over-capacity schemas before inference'
        native_probability_normalization: Literal[False] = False

        @property
        def identity(self):
            return ModelIdentity(name='native:' + self.model, lab='other')

    environment = {k: v for k, v in os.environ.items() if k.startswith(('CLEF_EXPERIMENT_', 'CLEF_BENCH_'))}
    inspection = json.loads(subprocess.check_output([str(args.binary.resolve()), '--model', args.model,
        '--model-dir', str(args.model_dir.resolve()), 'inspect'], text=True))
    config = NativeConfig(model=args.model + '-hrx', thinking='off', question_mode='all',
                          binary_sha256=file_hash(args.binary), runner_sha256=file_hash(__file__),
                          checkpoint_revision=inspection['revision'],
                          max_length=args.max_length, precision=args.precision, experiment_environment=environment)
    workflows = {name: load(name, DATASETS[name]) for name in args.workflows}
    jobs = {name: plan(w, config, limit=args.limit) for name, w in workflows.items()}
    command = [str(args.binary.resolve()), '--model', args.model, '--strict-context', '--model-dir', str(args.model_dir.resolve()),
               '--memory-gib', str(args.memory_gib), '--max-length', str(args.max_length),
               'decide', '-', '--jsonl', '--raw']
    if args.precision != 'bf16':
        command[1:1] = ['--precision', args.precision]
    manifest = {'workflow_commit': commit, 'dependency_lock_sha256': file_hash(benchmark / 'uv.lock'),
                'configuration': config.model_dump(mode='json'), 'command': command,
                'datasets': {name: {'source': w.bundle.dataset_source, 'case_ids': jobs[name].case_ids}
                             for name, w in workflows.items()}}
    if args.dry_run:
        print(json.dumps({name: {'cases': len(jobs[name].case_ids), 'dataset': w.bundle.dataset_source}
                          for name, w in workflows.items()}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output / 'native-manifest.json'
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error('Resume requires identical binary, runner, settings and pinned cases')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        (args.output / 'runner.py').write_bytes(Path(__file__).read_bytes())
        (args.output / 'working-tree.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
    stop = threading.Event()
    def monitor():
        with (args.output / 'telemetry.jsonl').open('a') as f:
            while not stop.is_set():
                f.write(json.dumps(hardware()) + '\n'); f.flush(); stop.wait(1)
    telemetry = threading.Thread(target=monitor, daemon=True)
    telemetry.start()
    native = NativeCLI(command, args.output, args.max_length)
    summary = {}
    try:
        def progress(message):
            print(message, flush=True)
            if native.failed:
                raise RuntimeError('Stopping suite after native process failure; inspect the saved case and stderr')
        for name, workflow in workflows.items():
            native.workflow = name
            output = args.output / name
            completed = load_checkpoints(output, {c.case_id: c.content_hash for c in workflow.bundle.cases})
            selected = next_case_ids(jobs[name].case_ids, completed, args.new_cases_per_workflow)
            batch = jobs[name].model_copy(update={'case_ids': selected})
            results = execute(workflow, batch, output, prices={}, workers=1, question_workers=1,
                              run_id='clef-hrx', client_factory=lambda _: native,
                              model_factory=lambda _: args.model, progress=progress)
            scores = build_score_data(results, policy_ids=jobs[name].policy_ids)
            write_record(output / 'scores.json', scores)
            cases = results.runs[0].cases
            summary[name] = {'cases': len(cases), 'planned_cases': len(jobs[name].case_ids),
                             'failed_cases': sum(c.execution.status != 'ok' for c in cases),
                             'scores': [s.model_dump(mode='json') for s in scores.series]}
            print(name, json.dumps(summary[name]), flush=True)
    finally:
        native.shutdown()
        stop.set(); telemetry.join()
        calls = [json.loads(line) for line in (args.output / 'native-calls.jsonl').read_text().splitlines()]
        for name in summary:
            summary[name]['speed'] = summarize_calls([r for r in calls if r['workflow'] == name])
        report = {'workflows': summary, 'speed': summarize_calls(calls),
                  'complete': len(summary) == len(workflows) and all(s['cases'] == s['planned_cases'] for s in summary.values()),
                  'limited_smoke_run': args.limit is not None,
                  'new_cases_per_workflow_this_invocation': args.new_cases_per_workflow,
                  'timing_note': 'Native request wall time includes encoding, compilation, preparation, GPU and readback; excludes the first request of each CLI process and the separate CPU preflight. Workflow case timing includes preflight, capacity-driven splitting and Python workflow execution. Requests exceeding the context limit are split before inference; no accepted request is truncated.'}
        preflight = [json.loads(line) for line in (args.output / 'preflight.jsonl').read_text().splitlines()]
        report['preflight'] = {'attempts': len(preflight), 'capacity_splits': sum(r['capacity_rejected'] for r in preflight),
                               'wall_seconds': sum(r['wall_seconds'] for r in preflight)}
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(any(s['failed_cases'] for s in summary.values()))


if __name__ == '__main__':
    raise SystemExit(main())
