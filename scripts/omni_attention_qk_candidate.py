#!/usr/bin/env python3
"""Interleave independent QK accumulators before masking and softmax.

The source finishes each 16-key dot product before starting the next. This
variant visits each K-dimension fragment across all key fragments first, so
independent matrix accumulators can overlap. Every accumulator keeps its
original ascending dimension order. Masking, reductions and PV are unchanged.
The input source retains its H3/krea2 attribution.
"""
import argparse
import re
from pathlib import Path


def generate(source):
    match = re.search(r'%keyround = index.add %keys, %c(31|63|127) : index', source)
    assert match, 'Unrecognized key tile'
    parts = (int(match[1]) + 1) // 16
    begin = source.index('%krow0 =')
    end = source.index('%partnermax,', begin)
    old = source[begin:end]
    rows, tail, matrix = [], [], {}
    for part in range(parts):
        first = old.index(f'%krow{part} =')
        last = old.index(f'%krow{part+1} =') if part+1 < parts else len(old)
        block = old[first:last]
        rows.append(block[:block.index('\n')+1])
        tail.append(block[block.index(f'%scaled{part} ='):])
        for dim in range(8):
            matrix[part,dim] = re.search(
                rf'^%kd{part}_{dim} =.*\n%kf{part}_{dim} =.*\n%dot{part}_{dim} =.*\n',
                block, re.M,
            )[0]
    new = ''.join(rows)
    for dim in range(8):
        new += ''.join(matrix[part,dim] for part in range(parts))
    new += ''.join(tail)
    assert sorted(old.splitlines()) == sorted(new.splitlines()), 'Operation sequence changed'
    return '// Experimental interleaved QK scheduling; unchanged arithmetic.\n' + source[:begin] + new + source[end:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()))
