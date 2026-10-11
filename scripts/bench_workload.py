#!/usr/bin/env python3
"""Record a Criterion run, its exact source, hardware telemetry, and comparisons."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import threading
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hardware():
    devices = {}
    for card in Path('/sys/class/drm').glob('card[0-9]*'):
        if '-' in card.name:
            continue
        device = card / 'device'
        if read(device / 'vendor') != '0x1002':
            continue
        values = {key: read(device / key) for key in (
            'device', 'gpu_busy_percent', 'pp_dpm_sclk', 'pp_dpm_mclk',
            'mem_info_gtt_used', 'mem_info_vram_used', 'power_dpm_force_performance_level')}
        for sensor in device.glob('hwmon/hwmon*/*_input'):
            if sensor.name.startswith(('temp', 'power', 'freq')):
                values[sensor.name] = read(sensor)
        devices[card.name] = values
    return {'time': time.time(), 'devices': devices, 'meminfo': read('/proc/meminfo')}


def paired_summary(samples):
    """Resample paired measurement batches together, preserving their covariance."""
    rng = random.Random(1729)
    ratios = []
    for _ in range(10000):
        batch = [samples[rng.randrange(len(samples))] for _ in samples]
        ratios.append(sum(x['reference_ns'] for x in batch) / sum(x['optimized_ns'] for x in batch))
    ratios.sort()
    iterations = sum(x['iterations'] for x in samples)
    old = sum(x['reference_ns'] for x in samples)
    new = sum(x['optimized_ns'] for x in samples)
    return {'reference_ns_per_iteration': old / iterations, 'candidate_ns_per_iteration': new / iterations,
            'speedup': old / new, 'confidence_level': 0.99,
            'speedup_lower': ratios[49], 'speedup_upper': ratios[9949],
            'method': '10,000 paired-batch bootstrap resamples; warmup excluded; fixed seed 1729'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', help='Unique Criterion baseline name and artifact directory')
    p.add_argument('--suite', choices=['workload', 'workload_inference', 'kernels', 'normalization', 'preprocessing'], default='workload')
    p.add_argument('--profile', choices=['quick', 'full'], default='full')
    p.add_argument('--filter', default='')
    p.add_argument('--compare', help='Existing Criterion baseline name')
    p.add_argument('--test', action='store_true', help='One iteration per case; smoke check, no performance claim')
    args = p.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.name):
        p.error('name must contain only letters, digits, hyphens, and underscores')
    root = Path('benches/results') / args.name
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CLEF_BENCH_PROFILE=args.profile,
               CLEF_BENCH_RUNTIME_RECORD=str((root / 'runtime.json').resolve()))
    files = set(subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0'))
    hashes = {}
    for name in sorted(files):
        path = Path(name)
        if not path.is_file() or path.parts[0] not in ('src', 'kernels', 'benches', 'examples', 'scripts', 'Cargo.toml', 'Cargo.lock'):
            continue
        if path.parts[:2] == ('benches', 'results'):
            continue
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        dest = root / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    metadata = {
        'name': args.name, 'suite': args.suite, 'profile': args.profile,
        'filter': args.filter, 'compare': args.compare, 'smoke_only': args.test,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'system': list(os.uname()), 'cpu': read('/proc/cpuinfo'),
        'hardware_before': hardware(), 'source_sha256': hashes,
        'environment': {k: v for k, v in env.items() if k.startswith('CLEF_BENCH_')
                        or k in ('CLEF_MODEL_DIR', 'CLEF_TRACE_DIR', 'CLEF_COMPILE_REPORT_DIR')},
    }
    if 'CLEF_BENCH_REQUESTS' in env:
        data = Path(env['CLEF_BENCH_REQUESTS']).read_bytes()
        dest = root / 'inputs' / 'requests.jsonl'
        dest.parent.mkdir()
        dest.write_bytes(data)
        metadata['requests_sha256'] = hashlib.sha256(data).hexdigest()
    metadata['candidates'] = {}
    for key in ('CLEF_BENCH_GEMM_SOURCE', 'CLEF_BENCH_DELTA_SOURCE', 'CLEF_BENCH_DELTA_REFERENCE_SOURCE', 'CLEF_BENCH_DELTA_PREPARE_SOURCE', 'CLEF_BENCH_ATTENTION_SOURCE', 'CLEF_BENCH_INT8_SOURCE', 'CLEF_BENCH_INTEGER_SOURCE', 'CLEF_BENCH_INTEGER_REFERENCE_SOURCE', 'CLEF_BENCH_PACKING_SOURCE'):
        if key in env:
            data = Path(env[key]).read_bytes()
            dest = root / 'candidates' / f'{key}.loom'
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(data)
            metadata['candidates'][key] = hashlib.sha256(data).hexdigest()
    (root / 'working-tree.diff').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD']))
    build = ['cargo', 'bench', '--locked', '--features', 'bench-internals', '--bench', args.suite, '--no-run', '--message-format=json']
    with (root / 'build.stderr').open('w') as stderr:
        result = subprocess.run(build, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
    (root / 'build.jsonl').write_text(result.stdout)
    result.check_returncode()
    artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    binaries = [x['executable'] for x in artifacts if x.get('reason') == 'compiler-artifact'
                and x.get('target', {}).get('name') == args.suite and x.get('executable')]
    binary = Path(binaries[-1])
    metadata['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    command = [str(binary), '--bench', args.filter]
    if args.test:
        command += ['--test']
    elif args.compare:
        command += ['--baseline', args.compare]
    else:
        command += ['--save-baseline', args.name]
    metadata['command'] = command
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    stop = threading.Event()
    def monitor():
        with (root / 'telemetry.jsonl').open('w') as f:
            while not stop.is_set():
                f.write(json.dumps(hardware()) + '\n')
                f.flush()
                stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True)
    started = time.time()
    thread.start()
    try:
        with (root / 'criterion.log').open('w') as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        stop.set()
        thread.join()
    metadata.update(exit_code=result.returncode, elapsed_seconds=time.time() - started, hardware_after=hardware())
    if args.test:
        metadata['smoke_cases'] = sum(line.startswith('Testing ') for line in (root / 'criterion.log').read_text().splitlines())
    runtime = root / 'runtime.json'
    if runtime.exists():
        metadata['runtime'] = json.loads(runtime.read_text())
        library = Path(metadata['runtime']['compiler_library'])
        for name in ('manifest.json', 'provenance.json'):
            source = library.parent / name
            if source.exists():
                shutil.copy2(source, root / f'runtime-{name}')
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    criterion = Path('target/criterion')
    rows = []
    for estimates in criterion.rglob('estimates.json'):
        if estimates.parent.name != ('new' if args.compare else args.name) or estimates.stat().st_mtime < started:
            continue
        case = estimates.parent.parent
        if args.compare:
            shutil.copytree(estimates.parent, case / args.name, dirs_exist_ok=True)
        dest = root / 'criterion' / case.relative_to(criterion)
        shutil.copytree(case, dest, dirs_exist_ok=True)
        row = {'benchmark': str(case.relative_to(criterion)), 'estimates': json.loads(estimates.read_text())}
        paired = case / 'paired.json'
        if paired.exists() and paired.stat().st_mtime >= started:
            samples = json.loads(paired.read_text()).get('measurement_samples', [])
            if samples:
                row['paired'] = paired_summary(samples)
        change = case / 'change/estimates.json'
        if change.exists() and change.stat().st_mtime >= started:
            row['change'] = json.loads(change.read_text())
            ci = row['change']['mean']['confidence_interval']
            row['classification'] = ('regression' if ci['lower_bound'] > 0.03 else
                                     'improvement' if ci['upper_bound'] < -0.03 else 'inconclusive')
        rows.append(row)
    phases = criterion / 'workload_phases'
    if phases.exists():
        for path in phases.glob('*.json'):
            if path.stat().st_mtime >= started:
                dest = root / 'phases' / path.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(path, dest)
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    if result.returncode == 0 and not args.test and not rows:
        raise SystemExit(f'{root}: no measured Criterion cases; check the filter')
    if result.returncode == 0 and args.test and not metadata['smoke_cases']:
        raise SystemExit(f'{root}: no smoke cases ran; check the filter and suite settings')
    cases = f"{metadata['smoke_cases']} smoke cases" if args.test else f'{len(rows)} measured cases'
    print(f'{root}: {cases}; exit={result.returncode}', flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
