#!/usr/bin/env python3
"""Cap expert row waves and loop over additional populated row tiles.

The ordinary grid allocates enough row partitions for every token to choose
every expert, although only eight of 128 experts are selected per token. Keep
the same independent matrix products, gather/scatter and arithmetic, but bound
the number of row waves per expert/channel tile. Each wave handles disjoint
row tiles in a strided loop. No task-list allocation or CPU readback is needed.

Derived from kernels/omni_expert_i4.loom and generated K128 variants; preserve
their SPDX and upstream attribution in the output.
"""
import argparse
import re
from pathlib import Path


def generate(source, row_waves, min_tokens=0):
    assert row_waves in (1, 2, 4, 8, 16, 32)
    assert 0 <= min_tokens <= 2048
    original = source
    # Launch and device-side wave decoding must use the same capped tile count.
    tile = r' %tiles = index.div %rounded, (%c(?:16|32)) : index'
    replacement = (
        r' %unbounded_tiles = index.div %rounded, \1 : index\n'
        f' %max_row_waves = index.constant {row_waves} : index\n'
        ' %over_limit = index.cmp ugt, %unbounded_tiles, %max_row_waves : index\n'
        ' %tiles = scf.select %over_limit, %max_row_waves, %unbounded_tiles : index'
    )
    source, grids = re.subn(tile, replacement, source)
    source, starts = re.subn(
        r' %mbase = index.mul %rowtile, (%c(?:16|32)) : index',
        r' %mbase_start = index.mul %rowtile, \1 : index\n'
        r' %row_stride = index.mul %tiles, \1 : index',
        source,
    )
    source, loops = re.subn(
        r' %active = index.cmp ult, %mbase, %count : index\n scf.if %active \{',
        ' scf.for %mbase = [%mbase_start to %count step %row_stride] {',
        source,
    )
    assert starts == loops and grids == loops + 1 and loops in (1, 2), (
        'Unrecognized source: launch decoding and row loops must agree',
        grids, starts, loops,
    )
    if min_tokens:
        # Both the grid and body branch specialize on the request's token count.
        # Small requests retain the original body, including its register use.
        source = source.replace(
            f' %max_row_waves = index.constant {row_waves} : index',
            f' %row_wave_cap = index.constant {row_waves} : index\n'
            f' %row_loop_min = index.constant {min_tokens} : index\n'
            ' %row_loop_large = index.cmp uge, %tokens, %row_loop_min : index\n'
            ' %max_row_waves = scf.select %row_loop_large, %row_wave_cap, %unbounded_tiles : index',
        )
        launch = re.search(r'\} launch\([^)]*\) \{\n', source)
        original_launch = re.search(r'\} launch\([^)]*\) \{\n', original)
        assert launch and original_launch
        body = source[launch.end():source.rindex(' kernel.return')]
        original_body = original[original_launch.end():original.rindex(' kernel.return')]
        source = source[:launch.end()] + (
            ' %row_loop_tokens = config.get @clef.tokens : index\n'
            f' %row_loop_threshold = index.constant {min_tokens} : index\n'
            ' %row_loop_enabled = index.cmp uge, %row_loop_tokens, %row_loop_threshold : index\n'
            ' scf.if %row_loop_enabled {\n'
        ) + body + ' } else {\n' + original_body + ' }\n kernel.return\n}\n'
    return (
        f'// Experimental capped row grid: at most {row_waves} waves per expert/channel tile.\n'
        + source
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--row-waves', type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument('--min-tokens', type=int, default=0,
                        help='Keep the original specialized body below this token count.')
    parser.add_argument('--source', type=Path, default=Path('kernels/omni_expert_i4.loom'))
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.row_waves, args.min_tokens))
