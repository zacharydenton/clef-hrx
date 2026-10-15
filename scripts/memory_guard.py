#!/usr/bin/env python3
"""Run one GPU job with a shared lock and fixed host-memory reserves.

Admission requires the engine budget plus 32 GiB available. Stop only the owned
process group below 24 GiB available or above 1 GiB additional system swap.
Shared-GPU use is allowed and recorded. Failed jobs are never retried.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

GIB = 1 << 30


def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {key: int(fields[key].split()[0]) * 1024 for key in ('MemAvailable', 'SwapFree')}


def pressure(before, current):
    return current['MemAvailable'] < 24 * GIB or before['SwapFree'] - current['SwapFree'] > GIB


def diagnostics(leader, clients):
    """Best-effort observations only; admission and stop rules do not use them."""
    owned, pending = set(), [str(leader)]
    while pending:
        pid = pending.pop()
        if pid in owned:
            continue
        owned.add(pid)
        try:
            pending.extend(Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except OSError:
            pass  # A short-lived encoder or compiler may have exited.
    processes = {}
    for pid in sorted(owned | set(clients)):
        try:
            fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            values = {key: int(fields[key].split()[0]) * 1024
                      for key in ('VmRSS', 'VmHWM', 'VmSwap', 'RssAnon', 'RssFile') if key in fields}
            processes[pid] = {'owned': pid in owned, **values}
        except (OSError, ValueError):
            pass
    reclaim = {}
    try:
        counters = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        reclaim = {key: int(counters[key]) for key in
                   ('pswpin', 'pswpout', 'pgscan_kswapd', 'pgscan_direct', 'pgsteal_kswapd', 'pgsteal_direct')
                   if key in counters}
    except (OSError, ValueError):
        pass
    return {'process_memory': processes, 'reclaim_counters': reclaim}


def stop_owned(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-gib', type=int, required=True)
    parser.add_argument('--lock', type=Path, default=Path('tests/results/flash-support-matrix/run.lock'))
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.memory_gib <= 0:
        parser.error('provide a positive engine budget and a command after --')
    executable = shutil.which(command[0])
    if executable is None:
        parser.error(f'executable not found: {command[0]}')
    args.output.mkdir(parents=True, exist_ok=False)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    state = {'state': 'pending', 'pid': os.getpid(), 'command': command,
             'started': time.time(), 'memory_gib': args.memory_gib,
             'admission_gib': args.memory_gib + 32, 'reserve_gib': 24,
             'maximum_additional_swap_gib': 1,
             'guard_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'binary_sha256': hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
             'environment': {k: v for k, v in os.environ.items() if k.startswith('CLEF_')}}

    def save():
        temp = args.output / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.output / 'status.json')

    with args.lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state['state'] = 'lock_busy'
            save()
            return 75
        before = memory()
        state['before'] = before
        if before['MemAvailable'] < (args.memory_gib + 32) * GIB:
            state['state'] = 'admission_refused'
            save()
            return 75
        process = None

        def interrupt(signum, _frame):
            state.update(state='interrupted', signal=signum)
            stop_owned(process)

        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (args.output / 'run.log').open('w') as log, (args.output / 'telemetry.jsonl').open('w') as telemetry:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(state='running', child=process.pid)
                save()
                last = 0
                while process.poll() is None:
                    current = memory()
                    if pressure(before, current):
                        clients = [p.name for p in Path('/sys/class/kfd/kfd/proc').glob('*')]
                        state.update(state='held_for_memory', pressure=current,
                                     pressure_diagnostics=diagnostics(process.pid, clients))
                        stop_owned(process)
                        break
                    if time.monotonic() - last >= 2:
                        clients = [p.name for p in Path('/sys/class/kfd/kfd/proc').glob('*')]
                        telemetry.write(json.dumps({'time': time.time(), **current, 'kfd_pids': clients,
                                                    **diagnostics(process.pid, clients)}) + '\n')
                        telemetry.flush()
                        last = time.monotonic()
                    time.sleep(0.1)
                code = process.wait()
                if state['state'] == 'running':
                    state['state'] = 'complete' if code == 0 else 'failed'
                state.update(returncode=code, finished=time.time())
                save()
                return code if state['state'] in ('complete', 'failed') else 75
        except BaseException as error:
            state.update(state='guard_error', error=str(error), finished=time.time())
            save()
            raise
        finally:
            # Also terminate descendants that outlived their group leader.
            stop_owned(process)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
