#!/usr/bin/env python3
"""Add an FP32 key-mean input to a plain or Hadamard INT8 packing source.

The caller supplies one per-column mean over the full key sequence. Subtracting
the same vector from every key changes each query's logits by a constant, which
softmax cancels even with causal masking. Q must use the same rotation as K.
Preserves the packing source's H3 Apache-2.0 attribution and packed layout.
"""
import argparse
from pathlib import Path


def generate(source):
    launch = 'launch(%rows: index, %input: buffer, %output: buffer)'
    assert source.count(launch) == 1, 'Unsupported packing signature'
    source = source.replace(launch, 'launch(%rows: index, %input: buffer, %mean: buffer, %output: buffer)')
    names = [name for name in ['x', 'hr_input']
             if f' %{name} = vector.extf %h : vector<4xbf16> to vector<4xf32>' in source]
    assert len(names) == 1, 'Expected one plain/Hadamard BF16 input conversion'
    name = names[0]
    old = f' %{name} = vector.extf %h : vector<4xbf16> to vector<4xf32>'
    source = source.replace(old, f''' %center_raw = vector.extf %h : vector<4xbf16> to vector<4xf32>
 %center_global = buffer.assume.memory_space<global> %mean : buffer
 %center_view = buffer.view %center_global[%off] : buffer -> view<[%width]xf32>
 %center_value = vector.load %center_view[%col] : view<[%width]xf32> -> vector<4xf32>
 %{name} = vector.subf %center_raw, %center_value : vector<4xf32>''')
    return '// Experimental key centering in FP32 before quantization/rotation.\n' + source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()))
