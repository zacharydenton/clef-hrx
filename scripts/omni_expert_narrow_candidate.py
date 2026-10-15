#!/usr/bin/env python3
"""Generate M16/N32 INT4 experts with unchanged K256/K16 arithmetic."""
import argparse
from pathlib import Path
import re


def generate(source):
    source = source.replace('16 routed rows x 64 output channels', '16 routed rows x 32 output channels')
    old = ' %groups = index.mul %basegroups, %tiles : index'
    assert source.count(old) == 1
    source = source.replace(old, ''' %tile_groups = index.mul %basegroups, %tiles : index
 %two = index.constant 2 : index
 %groups = index.mul %tile_groups, %two : index''')
    source = source.replace('index.div %n, %c64', 'index.div %n, %c32')
    source = source.replace('index.mul %ntile, %c64', 'index.mul %ntile, %c32')
    start = source.index('   %nc2 =')
    end = source.index('   %l_p0_v =', start)
    source = source[:start] + source[end:]
    source = re.sub(r'^.*%(?:r[23]_p\d+(?:_v)?|v[23]_p\d+|next[23]|is[23]|pick[23]) =.*\n', '', source, flags=re.M)
    source = source.replace('%acc0, %acc1, %acc2, %acc3 =', '%acc0, %acc1 =')
    source = source.replace(', %a2 = %init : vector<8xi32>, %a3 = %init : vector<8xi32>', '')
    source = source.replace('(vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)', '(vector<8xi32>, vector<8xi32>)')
    source = source.replace('scf.yield %next0, %next1, %next2, %next3 : vector<8xi32>, vector<8xi32>, vector<8xi32>, vector<8xi32>',
                            'scf.yield %next0, %next1 : vector<8xi32>, vector<8xi32>')
    source = source.replace('scf.for %j = [%c0 to %c4 step %c1]', 'scf.for %j = [%c0 to %c2 step %c1]')
    source = source.replace('vector.fragment.store<result> %pick3,', 'vector.fragment.store<result> %pick1,')
    return '// Experimental narrower output tile; register-pressure candidate.\n' + source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('kernels/omni_expert_i4.loom'))
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()))
