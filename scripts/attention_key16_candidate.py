#!/usr/bin/env python3
"""Generate shared attention with production's 16-key online update cadence."""
import argparse
from pathlib import Path


def generate(source):
    """Keep both BF16 probability components; mask the unused score half."""
    for marker in ('%problow =', '%threads = index.constant 256',
                   '%lds_bytes = index.constant 64768'):
        if source.count(marker) != 1:
            raise ValueError(f'expected the two-component shared attention source: {marker}')
    substitutions = [
        ('%keyround = index.add %keys, %c31',
         '%key15 = index.constant 15 : index\n%keyround = index.add %keys, %key15'),
        ('%keytiles = index.div %keyround, %c32',
         '%keytiles = index.div %keyround, %c16'),
        ('%causaltiles = index.div %query_end, %c32',
         '%causaltiles = index.div %query_end, %c16'),
        ('%kb = index.mul %kt, %c32', '%kb = index.mul %kt, %c16'),
        ('%kpresent = index.cmp ult, %kr0, %keys : index',
         '%key_present = index.cmp ult, %kr0, %keys : index\n'
         '%in_tile = index.cmp ult, %kr, %c16 : index\n'
         '%kpresent = scalar.andi %key_present, %in_tile : i1'),
    ]
    for old, new in substitutions:
        if source.count(old) != 1:
            raise ValueError(f'attention structure changed: {old}')
        source = source.replace(old, new)

    # The second value half is unused. Removing its load also avoids reading
    # beyond the packed value buffer on the final aligned 16-key iteration.
    for start_marker in ('%stagevk1a =', '%ph1 ='):
        if source.count(start_marker) != 1:
            raise ValueError(f'attention staging changed: {start_marker}')
        start = source.index(start_marker)
        end = source.index('kernel.barrier<workgroup>', start)
        source = source[:start] + source[end:]
    old_yield = 'scf.yield %nextmax, %nextsum, ' + ', '.join(
        f'%next1_{i}' for i in range(8))
    new_yield = 'scf.yield %nextmax, %nextsum, ' + ', '.join(
        f'%next0_{i}' for i in range(8))
    if source.count(old_yield) != 1:
        raise ValueError('attention accumulator yield changed')
    source = source.replace(old_yield, new_yield)
    return ('// Experimental 16-key online updates with two BF16 probability components.\n'
            '// Preserve production reduction order while reusing shared Q/K/V.\n' + source)


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
