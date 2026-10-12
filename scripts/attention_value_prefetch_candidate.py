#!/usr/bin/env python3
"""Prefetch attention values before computation; model evaluation is required."""
import argparse
from pathlib import Path


def generate(original):
    wide = '%v0_stagevc0 =' in original
    phases = [('v0_', '%qk0_qkzero ='), ('v1_', '%pv0_ph0 =')] if wide else [('', '%qkzero =')]
    candidate = original
    for prefix, before in phases:
        start = candidate.index(f'%{prefix}stagevc0 =')
        end = candidate.index('kernel.barrier<workgroup>', start)
        block = candidate[start:end]
        loads = '\n'.join(line for line in block.splitlines() if not line.startswith('vector.store')) + '\n'
        stores = '\n'.join(line for line in block.splitlines() if line.startswith('vector.store')) + '\n'
        if stores.count('vector.store') != 2 or candidate.count(before) != 1:
            raise ValueError('attention staging structure changed')
        candidate = candidate[:start] + stores + candidate[end:]
        candidate = candidate.replace(before, loads + 'scf.schedule.fence\n' + before, 1)
    header = ('// Experimental value prefetch with 64-key softmax tiles.\n' if wide else
              '// Experimental V prefetch before QK; arithmetic and LDS layout unchanged.\n')
    return header + candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(candidate)


if __name__ == '__main__':
    main()
