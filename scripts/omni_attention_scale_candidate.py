#!/usr/bin/env python3
"""Transpose INT8 attention's LDS key scales for contiguous vector reads.

Preserves the scale values and QK rescaling order. The original packed Q/K
layout, quantization, attention recurrence and BF16 PV are unchanged. Derived
from omni_attention_int8_candidate.py; preserves its H3 Apache-2.0 provenance.
"""
import argparse
import re
from pathlib import Path


def generate(source, vector_multiply=False):
    declaration = re.search(r'%k_scales = buffer.view %lds\[%k_scale_offset\] : buffer -> view<(32|64)xf32>', source)
    assert declaration, 'Expected INT8 attention with per-key LDS scales'
    keys = int(declaration.group(1))
    shape = f'2x{keys // 2}'
    source = source.replace(declaration.group(0),
                            f'%k_scales = buffer.view %lds[%k_scale_offset] : buffer -> view<{shape}xf32>')
    stores = 0

    def store(match):
        nonlocal stores
        i = stores
        stores += 1
        value, row = match.groups()
        return f'''%scale_store{i}_parity = index.rem {row}, %c2 : index
%scale_store{i}_column = index.div {row}, %c2 : index
view.store {value}, %k_scales[%scale_store{i}_parity, %scale_store{i}_column] : f32, view<{shape}xf32>'''

    source = re.sub(r'view.store (%\w+), %k_scales\[(%\w+)\] : f32, view<' + str(keys) + r'xf32>', store, source)
    assert 1 <= stores <= 4, 'Unexpected scale write sites'
    for part in range(keys // 16):
        convert = f'%scaled{part} = vector.sitofp %dot{part}_7 : vector<8xi32> to vector<8xf32>'
        assert source.count(convert) == 1
        load = f'''%scale_lane{part} = index.assume %lg [range(%lg, 0, 1)] : index
%scale_column{part} = index.constant {part * 8} : index
%scale_vector{part} = vector.load %k_scales[%scale_lane{part}, %scale_column{part}] : view<{shape}xf32> -> vector<8xf32>
'''
        multiply = f'''
%scale_query_vector{part} = vector.splat %q_score_scale : vector<8xf32>
%scale_query_scores{part} = vector.mulf %scaled{part}, %scale_query_vector{part} : vector<8xf32>
%scale_scores{part} = vector.mulf %scale_query_scores{part}, %scale_vector{part} : vector<8xf32>''' if vector_multiply else ''
        source = source.replace(convert, load + convert + multiply)
        for element in range(8):
            old = f'''%ks_local{part}_{element}a = index.constant {part * 16 + element * 2} : index
%ks_local{part}_{element} = index.add %ks_local{part}_{element}a, %lg : index
%ks_value{part}_{element} = view.load %k_scales[%ks_local{part}_{element}] : view<{keys}xf32> -> f32'''
            assert source.count(old) == 1
            new = f'%ks_value{part}_{element} = vector.extract %scale_vector{part}[{element}] : vector<8xf32> -> f32'
            source = source.replace(old, '' if vector_multiply else new)
            if vector_multiply:
                old = f'''%dot_value{part}_{element} = vector.extract %scaled{part}[{element}] : vector<8xf32> -> f32
%q_scaled{part}_{element} = scalar.mulf %dot_value{part}_{element}, %q_score_scale : f32
%value{part}_{element} = scalar.mulf %q_scaled{part}_{element}, %ks_value{part}_{element} : f32'''
                assert source.count(old) == 1
                source = source.replace(old, f'%value{part}_{element} = vector.extract %scale_scores{part}[{element}] : vector<8xf32> -> f32')
    assert 'view.load %k_scales' not in source, 'Unconverted scalar scale read'
    return '// Experimental transposed LDS key scales; QK rescaling order preserved.\n' + source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--vector-multiply', action='store_true')
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.vector_multiply))
