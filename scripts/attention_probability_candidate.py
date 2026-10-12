#!/usr/bin/env python3
"""Generate a single-BF16-probability attention experiment; evaluation is required."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('benches/kernels/attention_shared.loom'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    original = data.decode('utf-8')
    lines = []
    removed = renamed = 0
    for line in original.splitlines():
        if line.startswith(('%probhi =', '%problo =', '%problow =', '%lp',
                            'view.store %lp', '%pl0 =', '%pl1 =')) \
                or (line.startswith('%next') and 'vector.mma %pl' in line):
            removed += 1
            continue
        if line.startswith('%part') and 'vector.mma %ph' in line:
            line = line.replace('%part', '%next', 1)
            renamed += 1
        if not line.startswith('//'):
            lines.append(line)
    if (removed, renamed) != (37, 16):
        raise RuntimeError(f'kernel structure changed: removed={removed}, renamed={renamed}')
    digest = hashlib.sha256(data).hexdigest()
    header = (
        '// Experimental single BF16 probability component; QK and PV accumulate in FP32.\n'
        '// Derived from attention_shared.loom; drops the residual component and changes arithmetic.\n'
        '// Full model evaluation is required. Production dispatch is unchanged.\n'
        f'// Source SHA256: {digest}\n'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(header + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
