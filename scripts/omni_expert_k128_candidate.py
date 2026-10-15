#!/usr/bin/env python3
"""Generate M16/M32 N64 INT4 experts with K128 register packets.

Keep the checkpoint's N128/K256 packing, ascending K16 integer accumulation,
and the original scaling/rounding. Smaller packets leave room for two row
fragments to share each weight load. Derived from kernels/omni_expert_i4.loom;
its SPDX and upstream attribution remain in the generated source.
"""
import argparse
import re
from pathlib import Path


def generate(source, rows=32, down_only=False):
    assert rows in (16, 32)
    s = source.replace('Prefetch K256 packed codes', 'Prefetch K128 packed codes')
    s = s.replace('step %c256]', 'step %c128]')
    s = s.replace('%xlimit = index.constant 32', '%xlimit = index.constant 16')
    # The second K128 packet starts halfway through a packed K256 row.
    s = re.sub(r'multiple_of\((%wa\d+), 32\)', r'multiple_of(\1, 16)', s)
    s = s.replace('vector<32xi32>', 'vector<16xi32>')
    if rows == 32:
        s = s.replace('16 routed rows x 64', '32 routed rows x 64')
        s = s.replace('Each 16-row partition', 'Each 32-row partition')
        s = s.replace('%c15 = index.constant 15', '%c15 = index.constant 31')
        s = s.replace('%tiles = index.div %rounded, %c16', '%tiles = index.div %rounded, %c32')
        s = s.replace(' %c16 = index.constant 16 : index\n %rounded',
                      ' %c32 = index.constant 32 : index\n %rounded', 1)
        s = s.replace('%mbase = index.mul %rowtile, %c16', '%mbase = index.mul %rowtile, %c32')
        # Gather the second row fragment, including masked final partitions.
        for begin, end, outside in [
            ('  %mr =', '  %acc0,', {'%mbase': '%mbase2'}),
            ('   %xa0 =', '   %nc0 =', {'%xbase': '%sxbase'}),
        ]:
            a = s.index(begin)
            b = s.index(end, a)
            block = s[a:b]
            names = {name: '%s' + name[1:] for name in re.findall(r'(%\w+) =', block)}
            names.update(outside)
            duplicate = re.sub(r'%\w+', lambda m: names.get(m[0], m[0]), block)
            if begin == '  %mr =':
                duplicate = '  %mbase2 = index.add %mbase, %c16 : index\n' + duplicate
            s = s[:b] + duplicate + s[b:]
    accumulators = rows // 16 * 4
    a = s.index('  %acc0,')
    b = s.index('   %xk =', a)
    outputs = ', '.join(f'%acc{i}' for i in range(accumulators))
    initial = ', '.join(f'%a{i} = %init : vector<8xi32>' for i in range(accumulators))
    types = ', '.join('vector<8xi32>' for _ in range(accumulators))
    header = f'  {outputs} = scf.for %kk = [%c0 to %k step %c128]({initial}) -> ({types}) {{\n'
    s = s[:a] + header + s[b:]
    a = s.index('   %l_p0_v =')
    b = s.index('\n  }', a)
    body = ''
    schema = ' using {schema = %schema : encoding<schema>}'
    for part in range(8):
        for m in range(rows // 16):
            array = '%aa' if m == 0 else '%saa'
            body += f'   %l{m}_p{part}_v = vector.slice {array}[{part*2}] : vector<16xi32> -> vector<2xi32>\n'
            body += f'   %l{m}_p{part} = vector.fragment<lhs> %l{m}_p{part}_v shape [%c16, %c16]{schema} : vector<2xi32>\n'
        for n in range(4):
            body += f'   %r{n}_p{part}_v = vector.slice %bb{n}[{part*2}] : vector<16xi32> -> vector<2xi32>\n'
            body += f'   %r{n}_p{part} = vector.fragment<rhs> %r{n}_p{part}_v shape [%c16, %c16]{schema} : vector<2xi32>\n'
        for j in range(accumulators):
            previous = f'%a{j}' if part == 0 else f'%v{j}_p{part-1}'
            dest = f'%next{j}' if part == 7 else f'%v{j}_p{part}'
            body += f'   {dest} = vector.mma %l{j//4}_p{part}, %r{j%4}_p{part}, {previous} : vector<2xi32>, vector<2xi32>, vector<8xi32>\n'
    body += '   scf.yield ' + ', '.join(f'%next{j}' for j in range(accumulators)) + ' : ' + types
    s = s[:a] + body + s[b:]
    if rows == 32:
        a = s.index('  scf.for %j =')
        b = s.index('   vector.fragment.store<result>', a)
        publish = '  scf.for %j = [%c0 to %c8 step %c1] unroll {\n'
        for j in range(1, 8):
            publish += f'   %outc{j} = index.constant {j} : index\n'
            publish += f'   %is{j} = index.cmp eq, %j, %outc{j} : index\n'
            previous = '%acc0' if j == 1 else f'%pick{j-1}'
            publish += f'   %pick{j} = scf.select %is{j}, %acc{j}, {previous} : vector<8xi32>\n'
        s = s[:a] + publish + s[b:]
        s = s.replace('vector.fragment.store<result> %pick3,', 'vector.fragment.store<result> %pick7,')
        s = s.replace('   %nc = index.mul %j, %c16 : index',
                      '   %nj = index.rem %j, %c4 : index\n'
                      '   %mj = index.div %j, %c4 : index\n'
                      '   %mc = index.mul %mj, %c16 : index\n'
                      '   %outmbase = index.add %mbase, %mc : index\n'
                      '   %nc = index.mul %nj, %c16 : index')
        s = s.replace('%mrout = index.add %mbase, %pr', '%mrout = index.add %outmbase, %pr')
    if down_only:
        # The projection direction is a specialization constant, so the unused
        # branch disappears during compilation. Up projections keep the source
        # kernel, including its launch geometry and original K256 packets.
        launch = re.search(r'\} launch\([^)]*\) \{\n', s)
        original_launch = re.search(r'\} launch\([^)]*\) \{\n', source)
        assert launch and original_launch
        head = s[:launch.end()]
        if rows == 32:
            head = head.replace(' %c15 = index.constant 31 : index\n %c32 = index.constant 32 : index',
                                ' %choose_route = config.get @clef.input_routed : index\n'
                                ' %zero = index.constant 0 : index\n'
                                ' %choose_down = index.cmp ne, %choose_route, %zero : index\n'
                                ' %m16 = index.constant 16 : index\n'
                                ' %m32 = index.constant 32 : index\n'
                                ' %c32 = scf.select %choose_down, %m32, %m16 : index\n'
                                ' %c15 = index.sub %c32, %one : index', 1)
        body = s[launch.end():s.rindex(' kernel.return')]
        original_body = source[original_launch.end():source.rindex(' kernel.return')]
        s = head + (' %choose_route = config.get @clef.input_routed : index\n'
                    ' %choose_zero = index.constant 0 : index\n'
                    ' %choose_down = index.cmp ne, %choose_route, %choose_zero : index\n'
                    ' scf.if %choose_down {\n') + body + ' } else {\n' + original_body + ' }\n kernel.return\n}\n'
    return '// Experimental K128 packets; checkpoint layout and arithmetic unchanged.\n' + s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rows', type=int, choices=(16, 32), default=32)
    parser.add_argument('--down-only', action='store_true', help='Retain the source kernel for up/gate projections.')
    parser.add_argument('--source', type=Path, default=Path('kernels/omni_expert_i4.loom'))
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.rows, args.down_only))
