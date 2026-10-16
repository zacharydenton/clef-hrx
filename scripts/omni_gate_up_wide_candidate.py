#!/usr/bin/env python3
"""Generate M16/N64 fused gate/up with K128 packets from the M16/N32 kernel.

The benchmark passes an N32 launch grid; this candidate halves it internally.
Keep ascending K16 integer accumulation and the original BF16 epilogue.
"""
import argparse
import re
from pathlib import Path
from omni_expert_k128_candidate import generate as k128


def generate(source):
    s = k128(source, rows=16)
    s = s.replace('16 routed rows x 32 gate and 32 up channels',
                  '16 routed rows x 64 gate and 64 up channels')
    s = s.replace('%ntiles = index.div %n, %c32', '%ntiles = index.div %n, %c64')
    s = s.replace('%nbase = index.mul %ntile, %c32', '%nbase = index.mul %ntile, %c64')
    old = ' %basegroups = config.get @clef.groups : index'
    assert s.count(old) == 1
    s = s.replace(old, ' %requested_groups = config.get @clef.groups : index\n'
                      ' %entry_two = index.constant 2 : index\n'
                      ' %basegroups = index.div %requested_groups, %entry_two : index')
    a = s.index('  %acc0,'); b = s.index('   %xk =', a)
    outputs = ', '.join(f'%acc{i}' for i in range(8))
    initial = ', '.join(f'%a{i} = %init : vector<8xi32>' for i in range(8))
    types = ', '.join('vector<8xi32>' for _ in range(8))
    s = s[:a] + f'  {outputs} = scf.for %kk = [%c0 to %k step %c128]({initial}) -> ({types}) {{\n' + s[b:]
    a = s.index('   %nc0 ='); b = s.index('   %nc1 =', a)
    template = s[a:b]
    names = set(re.findall(r'(%\w+) =', template))
    blocks = []
    for j in range(8):
        block = re.sub(r'%\w+', lambda m: m[0][:-1] + str(j) if m[0] in names else m[0], template)
        block = block.replace(f'%nc{j} = index.constant 0', f'%nc{j} = index.constant {(j % 4) * 16}')
        if j >= 4:
            block = block.replace('%w_len', '%u_len').replace('%wv[', '%uv[')
        blocks.append(block)
    b = s.index('   %l0_p0_v =', a)
    s = s[:a] + ''.join(blocks) + s[b:]
    a = s.index('   %l0_p0_v ='); b = s.index('\n  }', a)
    body = []
    schema = ' using {schema = %schema : encoding<schema>}'
    for part in range(8):
        body += [f'   %l_p{part}_v = vector.slice %aa[{part*2}] : vector<16xi32> -> vector<2xi32>',
                 f'   %l_p{part} = vector.fragment<lhs> %l_p{part}_v shape [%c16, %c16]{schema} : vector<2xi32>']
        for j in range(8):
            previous = f'%a{j}' if part == 0 else f'%v{j}_p{part-1}'
            dest = f'%next{j}' if part == 7 else f'%v{j}_p{part}'
            body += [f'   %r{j}_p{part}_v = vector.slice %bb{j}[{part*2}] : vector<16xi32> -> vector<2xi32>',
                     f'   %r{j}_p{part} = vector.fragment<rhs> %r{j}_p{part}_v shape [%c16, %c16]{schema} : vector<2xi32>',
                     f'   {dest} = vector.mma %l_p{part}, %r{j}_p{part}, {previous} : vector<2xi32>, vector<2xi32>, vector<8xi32>']
    body.append('   scf.yield ' + ', '.join(f'%next{i}' for i in range(8)) + ' : ' + types)
    s = s[:a] + '\n'.join(body) + s[b:]
    a = s.index('  scf.for %j = [%c0 to %c2'); b = s.index('   vector.fragment.store<result>', a)
    body = ['  scf.for %j = [%c0 to %c4 step %c1] unroll {', '   %c3 = index.constant 3 : index']
    for j in (1, 2, 3):
        body.append(f'   %is{j} = index.cmp eq, %j, %c{j} : index')
    for name, first in [('gate', 0), ('up', 4)]:
        previous = f'%acc{first}'
        for j in (1, 2, 3):
            dest = f'%{name}_pick' if j == 3 else f'%{name}_pick{j}'
            body.append(f'   {dest} = scf.select %is{j}, %acc{first+j}, {previous} : vector<8xi32>')
            previous = dest
    s = s[:a] + '\n'.join(body) + '\n' + s[b:]
    return '// Experimental N64 gate/up with K128 packets; benchmark N32 launch ABI.\n' + s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('benches/kernels/omni_gate_up_i4.loom'))
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()))
