#!/usr/bin/env python3
"""Generate register-prefetched BF16 or prepacked INT8 Omni attention.

Inspired by krea2-hrx's attention_sage_i8_fast_prefetch.loom at
0fdd677970baf5fb11cd176c625cac93d5a6eff8 (MIT; see licenses/krea2-hrx-MIT.txt). This variant retains
one LDS tile and both barriers, carrying next-tile global loads in registers.
It preserves the source's arithmetic and clamps the unused final prefetch to
the final valid tile rather than relying on additional input padding.
"""
import argparse
import re
from pathlib import Path
from omni_attention_transposed_candidate import generate as bf16
from omni_attention_int8_candidate import generate_prepacked

BARRIER = 'kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)'
LOAD = re.compile(r'^[ \t]*(%\w+) = (?:vector|view)\.load .* -> (.+)$', re.M)
TOKEN = re.compile(r'%\w+')


def generate(queries=128, keys=32, prepacked=False):
    # Full packets avoid divergent loop-carried loads for unused loader lanes.
    assert (queries, keys) in ((64,32), (64,64), (128,32), (128,64), (256,64))
    source = (generate_prepacked if prepacked else bf16)(queries, keys)
    source = '// Register-prefetched experiment; no arithmetic changes.\n' + source
    loop = re.search(r'^(.*?) = scf.for %kt = \[%c0 to %end step %c1\]\((.*?)\) -> \((.*?)\) \{\n', source, re.M)
    assert loop
    begin = source.index('\n', loop.end()) + 1  # retain %kb definition in loop
    end = source.index(BARRIER, begin)
    stage = source[begin:end]
    loads = LOAD.findall(stage)
    assert len(loads) == ((keys*8)//(queries*2)) * (3 if prepacked else 2)
    names = {name for name,_ in loads}
    # Global loads only; strip LDS stores and the scale-store predicate. Scale
    # loads become uniform, preserving the selected scalar at the writer lane.
    reads = '\n'.join(line for line in stage.splitlines()
                      if not re.match(r'\s*(?:vector|view)\.store ', line)
                      and not re.match(r'scf.if %kq\d+_first \{', line)
                      and line != '}') + '\n'
    assert 'scf.if' not in reads and 'store ' not in reads
    defined = set(re.findall(r'^[ \t]*(%\w+) =', reads, re.M))
    assert names <= defined

    def renamed(text, prefix, base):
        return TOKEN.sub(lambda m: base if m[0] == '%kb' else
                         '%' + prefix + m[0][1:] if m[0] in defined else m[0], text)

    initial = renamed(reads, 'prefetch_init_', '%c0')
    next_reads = '''%prefetch_next = index.add %kt, %c1 : index
%prefetch_last = index.sub %end, %c1 : index
%prefetch_tile = index.min %prefetch_next, %prefetch_last : index
''' + f'%prefetch_base = index.mul %prefetch_tile, %c{keys} : index\n'
    next_reads += renamed(reads, 'prefetch_next_', '%prefetch_base')
    stores = LOAD.sub('', stage)
    stores = TOKEN.sub(lambda m: '%prefetch_carry_' + m[0][1:] if m[0] in names else m[0], stores)
    lhs, params, types = loop.groups()
    extra_types = ', '.join(t for _,t in loads)
    lhs += ', ' + ', '.join('%prefetch_final_' + name[1:] for name,_ in loads)
    params += ', ' + ', '.join(f'%prefetch_carry_{name[1:]} = %prefetch_init_{name[1:]} : {t}' for name,t in loads)
    header = f'{lhs} = scf.for %kt = [%c0 to %end step %c1]({params}) -> ({types}, {extra_types}) {{\n'
    # First stage writes the values carried from the preceding iteration; next
    # global loads are placed before QK, softmax and PV, independent of that LDS.
    source = source[:loop.start()] + initial + header + source[loop.end():begin] + stores + BARRIER + '\n' + next_reads + source[end+len(BARRIER):]
    yields = re.findall(r'^scf.yield %nextmax,.*$', source, re.M)
    assert len(yields) == 1
    values, yield_types = yields[0].split(' : ', 1)
    values += ', ' + ', '.join('%prefetch_next_' + name[1:] for name,_ in loads)
    source = source.replace(yields[0], values + ' : ' + yield_types + ', ' + extra_types)
    return source


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=int, choices=(64,128,256), default=128)
    p.add_argument('--keys', type=int, choices=(32,64), default=32)
    p.add_argument('--prepacked', action='store_true')
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.write_text(generate(a.queries, a.keys, a.prepacked))
