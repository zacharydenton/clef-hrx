#!/usr/bin/env python3
"""Run a benchmark with no other KFD clients; never stop another user's job.

This checks compute clients, not every possible graphics workload. Existing
benchmark runners still record clocks, temperatures, memory and raw timings.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def clients():
    root = Path('/sys/class/kfd/kfd/proc')
    if not root.is_dir():
        raise RuntimeError('KFD process telemetry is unavailable')
    return sorted(int(p.name) for p in root.iterdir() if p.name.isdigit())


def foreign_clients(pids, process_group):
    foreign = []
    for pid in pids:
        try:
            if os.getpgid(pid) != process_group:
                foreign.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            foreign.append(pid)
    return foreign


def stop_owned_group(process):
    # The child starts a new session: this group belongs only to this command.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--wait', action='store_true', help='Wait for five idle seconds before starting')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('provide a command after --')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('x') as log:
        def record(event, **fields):
            log.write(json.dumps({'time': time.time(), 'event': event, **fields}) + '\n')
            log.flush()

        idle = 0
        previous = None
        while True:
            pids = clients()
            if pids != previous:
                record('waiting' if pids else 'idle', kfd_pids=pids)
                if pids:
                    print(f'GPU is occupied by KFD clients {pids}.', flush=True)
                previous = pids
            if not args.wait:
                if pids:
                    raise SystemExit(75)
                break
            idle = idle + 1 if not pids else 0
            if idle >= 6:
                break
            time.sleep(1)

        process = subprocess.Popen(command, start_new_session=True)
        record('started', command=command, process_group=process.pid)
        contended = False
        try:
            while True:
                pids = clients()
                foreign = foreign_clients(pids, process.pid)
                if pids != previous:
                    record('clients', kfd_pids=pids, foreign_pids=foreign)
                    previous = pids
                if foreign:
                    contended = True
                    record('contention', foreign_pids=foreign)
                    print(f'Foreign KFD clients {foreign}; stopping only the benchmark group.', flush=True)
                    stop_owned_group(process)
                    break
                code = process.poll()
                if code is not None:
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                stop_owned_group(process)
            record('finished', returncode=process.returncode, contended=contended)
        raise SystemExit(75 if contended else process.returncode)


if __name__ == '__main__':
    main()
