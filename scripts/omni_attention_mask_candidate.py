#!/usr/bin/env python3
"""Skip per-element attention masks for key tiles preceding every query.

Interior key tiles are both within the input and earlier than the first query
of the workgroup. Their scores pass through unchanged; boundary tiles retain
the source's masks. QK, softmax, PV, scaling and rounding are untouched.
The input source retains its H3/krea2 attribution.
"""
import argparse
import re
from pathlib import Path


def generate(source):
    match = re.search(r'%keyround = index.add %keys, %c(31|63|127) : index', source)
    assert match, 'Unrecognized key tile'
    keys = int(match[1]) + 1
    base = f'%kb = index.mul %kt, %c{keys} : index\n'
    assert source.count(base) == 1
    source = source.replace(base, base + (
        f'%mask_tile_end = index.add %kb, %c{keys} : index\n'
        '%mask_in_bounds = index.cmp ule, %mask_tile_end, %keys : index\n'
        '%mask_before_queries = index.cmp ule, %mask_tile_end, %qbase : index\n'
        '%mask_causal_ok = scf.select %is_causal, %mask_before_queries, %mask_in_bounds : i1\n'
        '%mask_interior = scalar.andi %mask_in_bounds, %mask_causal_ok : i1\n'
    ))
    for part in range(keys // 16):
        begin = source.index(f'%mk{part}_0off =')
        end = source.index('\n', source.index(f'%scores{part} =', begin)) + 1
        block = source[begin:end]
        block = block.replace(f'%scores{part} =', f'%masked_scores{part} =')
        new = (
            f'%scores{part} = scf.if %mask_interior -> (vector<8xf32>) {{\n'
            f' scf.yield %scaled{part} : vector<8xf32>\n'
            '} else {\n'
            + block
            + f' scf.yield %masked_scores{part} : vector<8xf32>\n'
            '}\n'
        )
        source = source[:begin] + new + source[end:]
    return '// Experimental interior-tile mask bypass; unchanged valid scores.\n' + source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()))
