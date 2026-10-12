#!/usr/bin/env python3
"""Run pinned WorkflowEvals through a private, memory-guarded llama.cpp server.

Use the WorkflowEvals venv. CPU preflight checks complete input lengths against
HRX's matching tokenizer; every llama response must report the same token count.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Literal
import urllib.error
import urllib.request

from bench_workload import hardware
from eval_jevbench_llama import sha256, write_json
from eval_workflowevals import DATASETS, WORKFLOW_COMMIT, next_case_ids

GIB = 1 << 30


def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {key: int(fields[key].split()[0]) * 1024 for key in ('MemAvailable', 'SwapFree')}


def required_memory(model_bytes, context, reserve_bytes, pilot_peak_bytes=0):
    # Admission estimate, not a measured allocator bound. Keep a separate live
    # guard because HIP graph buffers and other clients can change the footprint.
    scratch = max(math.ceil(context / 32768) * 12 * GIB,
                  max(0, pilot_peak_bytes - model_bytes) * (context / 32768) ** 2)
    return math.ceil(model_bytes + scratch + reserve_bytes)


def memory_pressure(before, current, reserve_bytes):
    if current["MemAvailable"] < reserve_bytes:
        return "available memory below reserve"
    if before["SwapFree"] - current["SwapFree"] > GIB:
        return "swap grew by more than 1 GiB"
    return None


def kill_owned(process, sig):
    if process.poll() is None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass


def check_pilot(output, configuration):
    manifest = json.loads((output / "manifest.json").read_text())
    process = json.loads((output / "process.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    prior = manifest["configuration"]
    for key in ("binary_sha256", "model_sha256", "harness_sha256"):
        if prior[key] != configuration[key]:
            raise ValueError(f"pilot {key} differs")
    if (prior["context"] != 32768 or not summary["smoke"] or
            not process.get("replay_exact") or process.get("error") or
            summary["memory_guard_triggered"] or summary["transport_errors"] or
            summary["speed"]["calls"] != 3):
        raise ValueError("a successful 32k invoice pilot is required before 64k")
    rows = [json.loads(line) for line in (output / "llama-calls.jsonl").read_text().splitlines()]
    if len(rows) != 3 or any(row["input_tokens"] < 30000 for row in rows):
        raise ValueError("pilot must exercise a full invoice of at least 30,000 tokens")
    return process["peak_available_memory_drop_bytes"]


def server_command(binary, model, context, port):
    return [str(binary.resolve()), '--model', str(model.resolve()), '--host', '127.0.0.1',
            '--port', str(port), '-ngl', '999', '-np', '1', '-c', str(context),
            '-b', str(context), '-ub', str(context), '-t', '16', '-tb', '16',
            '-fa', 'on', '--no-warmup', '--cache-ram', '0', '--no-context-shift',
            '--fit', 'off', '--log-verbosity', '4']


def validated_response(request, raw, tokens):
    from typesafe_sdk import SystemOneResponse
    if raw.get('usage', {}).get('input_tokens') != tokens:
        raise ValueError('llama input length differs from strict preflight; truncation is forbidden')
    if raw['usage'].get('output_tokens') != 0:
        raise ValueError('unexpected generated tokens')
    if set(raw.get('answers', {})) != set(request['questions']):
        raise ValueError('answer question IDs differ')
    for qid, q in request['questions'].items():
        answer = raw['answers'][qid]
        if answer.get('type') != q['type']:
            raise ValueError('answer type differs')
        if q['type'] == 'noul':
            values = [answer['noul']]
        else:
            probs = answer['probabilities']
            keys = set(q['criteria']) if q['type'] == 'choice' else {str(i) for i in range(len(q['criteria']))}
            if set(probs) != keys:
                raise ValueError('answer options differ')
            values = list(probs.values())
            if abs(sum(values) - 1) > 1e-5:
                raise ValueError('probabilities do not sum to one')
            if q['type'] == 'choice':
                if answer['choice'] not in keys:
                    raise ValueError('choice is outside schema')
                if probs[answer['choice']] < max(values) - 1e-6:
                    raise ValueError('choice differs from maximum probability')
            if q['type'] == 'score':
                expected = sum(int(k) * p for k, p in probs.items())
                if not math.isfinite(answer['score']) or abs(answer['score'] - expected) > 1e-5:
                    raise ValueError('score differs from probability expectation')
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError('invalid probability; no renormalization is allowed')
    return SystemOneResponse.model_validate_json(json.dumps(raw, allow_nan=False))


def speed_summary(rows):
    resident = [r for r in rows if not r['first_request']]
    seconds = sum(r['wall_seconds'] for r in resident)
    tokens = sum(r['input_tokens'] for r in resident)
    return {'calls': len(rows), 'resident_calls': len(resident), 'resident_input_tokens': tokens,
            'resident_wall_seconds': seconds, 'resident_input_tokens_per_second': tokens / seconds if seconds else None}


class LlamaClient:
    def __init__(self, endpoint, encoder, output, context):
        self.endpoint, self.encoder, self.context = endpoint, encoder, context
        self.workflow, self.failed, self.first = None, False, True
        self.lock = threading.Lock()
        self.log = (output / 'llama-calls.jsonl').open('a')
        self.preflight = (output / 'preflight.jsonl').open('a')
        self.errors = (output / 'errors.jsonl').open('a')

    def request(self, request):
        from core.session import GRAMMAR_LIMIT
        with self.lock:
            if self.failed:
                raise RuntimeError('llama transport stopped after a failed request')
            wire = json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n'
            before = time.perf_counter()
            encoded = subprocess.run(self.encoder, input=wire, capture_output=True, text=True)
            seconds = time.perf_counter() - before
            capacity = 'context capacity exceeded:' in encoded.stderr or 'schema and media require' in encoded.stderr
            tokens = len(json.loads(encoded.stdout)['input_ids']) if encoded.returncode == 0 else None
            self.preflight.write(json.dumps({'workflow': self.workflow, 'input_tokens': tokens,
                'capacity_rejected': capacity, 'wall_seconds': seconds, 'error': encoded.stderr}) + '\n')
            self.preflight.flush()
            if capacity:
                raise ValueError(f'{GRAMMAR_LIMIT}: input exceeds {self.context} tokens')
            if encoded.returncode:
                self.failed = True
                raise ValueError(f'CPU preflight failed: {encoded.stderr.strip()}')
            started = time.perf_counter()
            try:
                query = urllib.request.Request(self.endpoint + '/v1/systemone', data=wire.encode(),
                                               headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(query, timeout=1800) as response:
                    raw = json.load(response)
                result = validated_response(request, raw, tokens)
                elapsed = time.perf_counter() - started
                self.log.write(json.dumps({'workflow': self.workflow, 'first_request': self.first,
                    'input_tokens': tokens, 'wall_seconds': elapsed, 'preflight_seconds': seconds,
                    'request': request, 'response': raw}, ensure_ascii=False, allow_nan=False) + '\n')
                self.log.flush(); self.first = False
                return result
            except Exception as error:
                self.failed = True
                detail = error.read().decode(errors='replace') if isinstance(error, urllib.error.HTTPError) else str(error)
                self.errors.write(json.dumps({'workflow': self.workflow, 'error': detail,
                    'wall_seconds': time.perf_counter() - started, 'request': request}) + '\n')
                self.errors.flush()
                raise

    def system_one(self, *, model, state, questions):
        from core.question_types import question_record
        return self.request({'model': 'clef', 'state': state,
                             'questions': {key: question_record(q) for key, q in questions.items()}})

    def close(self):
        pass  # Shared server remains resident across cases.

    def shutdown(self):
        for stream in (self.log, self.preflight, self.errors):
            stream.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--llama-source', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--encoder', type=Path, default=Path('tests/results/flash-support-matrix/clef'))
    p.add_argument('--model-dir', type=Path, default=Path('artifacts/model'))
    p.add_argument('--benchmark', type=Path, default=Path('artifacts/WorkflowEvals'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--context', type=int, choices=(32768, 65536), default=65536)
    p.add_argument('--reserve-gib', type=float, default=16)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pilot-output', type=Path, help='Successful 32k smoke output, required before 64k')
    p.add_argument('--gpu-lock', type=Path, default=Path('tests/results/flash-support-matrix/run.lock'))
    p.add_argument('--gpu-lock-fd', type=int, help=argparse.SUPPRESS)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke-request', type=Path, help='Three identical invoice replays; no full-suite claim')
    p.add_argument('--new-cases-per-workflow', type=int)
    args = p.parse_args()
    with ExitStack() as cleanup:
        return run(args, p, cleanup)


def run(args, p, cleanup):
    if not math.isfinite(args.reserve_gib) or args.reserve_gib < 12:
        p.error('reserve must be at least 12 GiB')
    if args.new_cases_per_workflow is not None and args.new_cases_per_workflow < 1:
        p.error('new-case budget must be positive')
    if args.context != 65536 and not args.smoke_request:
        p.error('full WorkflowEvals requires 65536 context; use 32768 only for the invoice smoke test')
    if args.context == 65536 and not args.pilot_output:
        p.error('64k requires --pilot-output from a successful 32k invoice smoke test')
    # Share the qualification lock; inherited descriptors preserve queue ownership.
    gpu_lock = None
    if not args.dry_run:
        args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
        gpu_lock = cleanup.enter_context(
            os.fdopen(os.dup(args.gpu_lock_fd), 'a') if args.gpu_lock_fd is not None
            else args.gpu_lock.open('a'))
        if os.fstat(gpu_lock.fileno()).st_ino != args.gpu_lock.stat().st_ino:
            p.error('inherited GPU lock does not match --gpu-lock')
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('GPU qualification slot occupied. No server started.', file=sys.stderr)
            return 75
        # Refuse before large file hashing or dataset loading on a busy host.
        admission = required_memory(args.model.stat().st_size, args.context, int(args.reserve_gib * GIB))
        if memory()['MemAvailable'] < admission:
            print(f'Memory admission refused: need {admission / GIB:.1f} GiB available. No server started.', file=sys.stderr)
            return 75
    if args.smoke_request and args.resume:
        p.error('smoke runs use a fresh output directory')
    commit = subprocess.check_output(['git', '-C', str(args.benchmark), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != WORKFLOW_COMMIT or subprocess.check_output(['git', '-C', str(args.benchmark), 'diff', 'HEAD']):
        p.error('WorkflowEvals must match the pinned, unmodified source')
    sys.path.insert(0, str(args.benchmark.resolve()))
    from core.contract import ModelIdentity
    from core.providers import ModelConfig
    from core.runner import execute, load_checkpoints, plan
    from core.scoring import build_score_data
    from core.workflow import load

    class LocalConfig(ModelConfig):
        provider: Literal['llama-local'] = 'llama-local'
        binary_sha256: str
        model_sha256: str
        harness_sha256: str
        context: int

        @property
        def identity(self):
            return ModelIdentity(name='llama.cpp:' + self.model, lab='other')

    source_manifest = json.loads(args.manifest.read_text())
    entry, = [f for f in source_manifest['siblings'] if f['rfilename'] == args.model.name]
    if args.model.stat().st_size != entry['size'] or sha256(args.model) != entry['lfs']['sha256']:
        p.error('GGUF differs from pinned download manifest')
    configuration = LocalConfig(model=args.model.stem, thinking='off', question_mode='all',
        binary_sha256=sha256(args.binary), model_sha256=entry['lfs']['sha256'],
        harness_sha256=sha256(__file__), context=args.context)
    workflows = {name: load(name, revision) for name, revision in DATASETS.items()}
    plans = {name: plan(w, configuration) for name, w in workflows.items()}
    encoder = [str(args.encoder.resolve()), '--model-dir', str(args.model_dir.resolve()),
               '--offline', '--strict-context', '--max-length', str(args.context), 'encode']
    library_text = subprocess.check_output(['ldd', str(args.binary.resolve())], text=True)
    libraries = [Path(line.split('=>', 1)[1].split()[0]) for line in library_text.splitlines() if '=> /' in line]
    manifest = {'configuration': configuration.model_dump(mode='json'), 'workflow_commit': commit,
        'dependency_lock_sha256': sha256(args.benchmark / 'uv.lock'),
        'encoder_sha256': sha256(args.encoder), 'encoder_command': encoder,
        'shared_helper_sha256': {n: sha256(Path(__file__).with_name(n)) for n in
                                ['eval_workflowevals.py', 'eval_jevbench_llama.py', 'bench_workload.py']},
        'model_repository': source_manifest['id'], 'model_revision': source_manifest['sha'],
        'llama_working_tree_sha256': hashlib.sha256(subprocess.check_output(['git', '-C', str(args.llama_source), 'diff', 'HEAD'])).hexdigest(),
        'llama_commit': subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip(),
        'library_sha256': {str(f): sha256(f) for f in libraries},
        'datasets': {n: {'source': w.bundle.dataset_source, 'case_ids': plans[n].case_ids} for n, w in workflows.items()},
        'memory_reserve_gib': args.reserve_gib, 'smoke_request_sha256': sha256(args.smoke_request) if args.smoke_request else None,
        'environment': {k: v for k, v in os.environ.items() if k.startswith(('GGML_', 'HIP_', 'ROCR_', 'HSA_'))}}
    pilot_peak = check_pilot(args.pilot_output, manifest['configuration']) if args.pilot_output else 0
    manifest['pilot_manifest_sha256'] = sha256(args.pilot_output / 'manifest.json') if args.pilot_output else None
    admission = required_memory(args.model.stat().st_size, args.context, int(args.reserve_gib * GIB), pilot_peak)
    if args.dry_run:
        print(json.dumps({'manifest': manifest, 'required_available_gib': admission / GIB,
                          'current_available_gib': memory()['MemAvailable'] / GIB}, indent=2))
        return 0
    if memory()['MemAvailable'] < admission:
        print(f'Memory admission refused: need {admission / GIB:.1f} GiB available before loading; '
              f'have {memory()["MemAvailable"] / GIB:.1f} GiB. No server started.', file=sys.stderr)
        return 75
    args.output.mkdir(parents=True, exist_ok=args.resume)
    old_manifest = args.output / 'manifest.json'
    if args.resume and json.loads(old_manifest.read_text()) != manifest:
        p.error('resume requires identical model, binary, encoder, libraries, datasets and settings')
    write_json(old_manifest, manifest)
    (args.output / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'llama-working-tree.diff').write_bytes(subprocess.check_output(['git', '-C', str(args.llama_source), 'diff', 'HEAD']))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = server_command(args.binary, args.model, args.context, port)
    client = LlamaClient(f'http://127.0.0.1:{port}', encoder, args.output, args.context)
    cleanup.callback(client.shutdown)
    stop, guarded = threading.Event(), threading.Event()
    before = memory(); metadata = {'command': command, 'started': time.time(), 'memory_before': before,
        'required_available_bytes': admission, 'peak_available_memory_drop_bytes': 0, 'guard_error': None}
    summaries = {}
    process = None

    def interrupted(sig, frame):
        # WorkflowEvals drains active worker threads before returning. Close the
        # transport first so an interrupted HTTP request cannot keep the GPU live.
        client.failed = True
        if process is not None:
            kill_owned(process, signal.SIGKILL)
        raise KeyboardInterrupt(f'signal {sig}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        cleanup.callback(signal.signal, sig, signal.getsignal(sig))
        signal.signal(sig, interrupted)
    with (args.output / 'server.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        metadata['server_pid'] = process.pid
        write_json(args.output / 'process.json', metadata)

        def monitor():
            try:
                with (args.output / 'telemetry.jsonl').open('a') as telemetry:
                    last_log = 0
                    while not stop.is_set():
                        mem = memory()
                        metadata['peak_available_memory_drop_bytes'] = max(
                            metadata['peak_available_memory_drop_bytes'], before['MemAvailable'] - mem['MemAvailable'])
                        reason = memory_pressure(before, mem, int(args.reserve_gib * GIB))
                        if reason:
                            raise RuntimeError(reason)
                        if time.monotonic() - last_log >= 1:
                            telemetry.write(json.dumps({'memory': mem, **hardware()}) + '\n')
                            telemetry.flush()
                            last_log = time.monotonic()
                        stop.wait(0.1)
            except Exception as error:
                metadata['guard_error'] = str(error)
                guarded.set()
                client.failed = True
                kill_owned(process, signal.SIGKILL)

        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('llama server stopped; inspect server.log and memory telemetry')
                try:
                    with urllib.request.urlopen(client.endpoint + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup timeout')
                time.sleep(0.25)
            metadata['startup_seconds'] = time.time() - metadata['started']
            if args.smoke_request:
                request = json.loads(args.smoke_request.read_text())
                replies = [client.request(request).model_dump(mode='json') for _ in range(3)]
                if any(r['answers'] != replies[0]['answers'] for r in replies[1:]):
                    raise ValueError('identical-input replay changed answers')
                metadata['replay_exact'] = True
            else:
                def progress(message):
                    print(message, flush=True)
                    if client.failed:
                        raise RuntimeError('Stopping after failed transport or memory guard')
                for name, workflow in workflows.items():
                    client.workflow = name
                    dest = args.output / name
                    completed = load_checkpoints(dest, {c.case_id: c.content_hash for c in workflow.bundle.cases})
                    if any(c.execution.status != 'ok' for c in completed.values()):
                        raise ValueError('Failed checkpoints retained: inspect failures and use a fresh output to rerun')
                    selected = next_case_ids(plans[name].case_ids, completed, args.new_cases_per_workflow)
                    batch = plans[name].model_copy(update={'case_ids': selected})
                    results = execute(workflow, batch, dest, prices={}, workers=1, question_workers=1,
                        run_id='llama-cpp', client_factory=lambda _: client, model_factory=lambda _: 'clef', progress=progress)
                    scores = build_score_data(results, policy_ids=plans[name].policy_ids)
                    write_json(dest / 'scores.json', scores.model_dump(mode='json'))
                    cases = results.runs[0].cases
                    summaries[name] = {'cases': len(cases), 'planned_cases': len(plans[name].case_ids),
                        'failed_cases': sum(c.execution.status != 'ok' for c in cases),
                        'scores': [s.model_dump(mode='json') for s in scores.series]}
        except BaseException as error:
            metadata['error'] = repr(error)
            client.failed = True
            raise
        finally:
            stop.set(); thread.join()
            if process.poll() is None:
                kill_owned(process, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    kill_owned(process, signal.SIGKILL); process.wait()
            client.shutdown()
            rows = [json.loads(s) for s in (args.output / 'llama-calls.jsonl').read_text().splitlines()]
            for name, summary in summaries.items():
                summary['speed'] = speed_summary([r for r in rows if r['workflow'] == name])
            metadata.update(memory_guard_triggered=guarded.is_set(), finished=time.time())
            write_json(args.output / 'process.json', metadata)
            errors = (args.output / 'errors.jsonl').read_text().splitlines()
            complete = not client.failed and not guarded.is_set() and len(summaries) == 4 and all(s['cases'] == s['planned_cases'] and not s['failed_cases'] for s in summaries.values())
            speed = speed_summary(rows)
            if errors or guarded.is_set() or client.failed:
                speed['resident_input_tokens_per_second'] = None
                for summary in summaries.values():
                    summary['speed']['resident_input_tokens_per_second'] = None
            write_json(args.output / 'summary.json', {'workflows': summaries, 'complete': complete,
                'smoke': bool(args.smoke_request), 'speed': speed, 'transport_errors': len(errors),
                'memory_guard_triggered': guarded.is_set(),
                'timing_note': 'Sequential loopback HTTP includes tokenization, GPU execution, response parsing and validation. Server startup, first request of each process and CPU preflight are excluded from resident throughput. Full workflow wall time includes CPU preflight. No accepted input is truncated.'})
    return int(client.failed or guarded.is_set() or any(s['failed_cases'] for s in summaries.values()))


if __name__ == '__main__':
    raise SystemExit(main())
