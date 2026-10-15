#!/usr/bin/env python3
"""Retain INT4 expert arithmetic while replacing empty-tile launch enumeration."""
import argparse
from pathlib import Path


def generate(source, row_first=False):
    launch, body = source.split('} launch(', 1)
    start = launch.index(' %basegroups =')
    launch = launch[:start] + ''' %tokens = config.get @clef.tokens : index
 %n = config.get @clef.n : index
 %c2 = index.constant 2 : index
 %c8 = index.constant 8 : index
 %c15 = index.constant 15 : index
 %c16 = index.constant 16 : index
 %c64 = index.constant 64 : index
 %c127 = index.constant 127 : index
 %routes = index.mul %tokens, %c8 : index
 %rounded = index.add %routes, %c15 : index
 %packed = index.div %rounded, %c16 : index
 %tasks = index.add %packed, %c127 : index
 %ntiles = index.div %n, %c64 : index
 %waves = index.mul %tasks, %ntiles : index
 %wave_round = index.add %waves, %one : index
 %groups = index.div %wave_round, %c2 : index
 kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%threads, %one, %one) : index
'''
    start = body.index(' %lds_bytes =')
    end = body.index(' %x_global =', start)
    lds = body[start:end]
    body = body[:start] + body[end:]
    start = body.index(' %c15 =')
    end = body.index(' %nbase =', start)
    body = body[:start] + lds + ''' %c15 = index.constant 15 : index
 %c127 = index.constant 127 : index
 %route_count = index.mul %tokens, %topk : index
 %task_round = index.add %route_count, %c15 : index
 %packed_tasks = index.div %task_round, %c16 : index
 %max_tasks = index.add %packed_tasks, %c127 : index
 %task_stride = index.add %tokens, %c1 : index
 %group_len = index.mul %task_stride, %c128 : index
 %task_header = index.add %group_len, %route_count : index
 %total_f = template.apply<@clef.load>(%routes_dtype, %routes_buffer, %routes_len, %task_header) : (index, buffer, index, index) -> (f32)
 %total_i0 = scalar.fptosi %total_f : f32 to i32
 %total_i = kernel.subgroup.broadcast.first %total_i0 : i32
 %total0 = index.cast %total_i : i32 to index
 %total = index.assume %total0 [range(%total0, 0, 1151)] : index
 %ntiles = index.div %n, %c64 : index
 %task = index.div %wave0, %ntiles : index
 %ntile = index.rem %wave0, %ntiles : index
 %live = index.cmp ult, %task, %total : index
 scf.if %live {
 %task_pair = index.mul %task, %c2 : index
 %task_at0 = index.add %task_header, %task_pair : index
 %task_at = index.add %task_at0, %c1 : index
 %row_at = index.add %task_at, %c1 : index
 %expert_f = template.apply<@clef.load>(%routes_dtype, %routes_buffer, %routes_len, %task_at) : (index, buffer, index, index) -> (f32)
 %expert_i0 = scalar.fptosi %expert_f : f32 to i32
 %expert_i = kernel.subgroup.broadcast.first %expert_i0 : i32
 %expert0 = index.cast %expert_i : i32 to index
 %expert = index.assume %expert0 [range(%expert0, 0, 127)] : index
 %row_f = template.apply<@clef.load>(%routes_dtype, %routes_buffer, %routes_len, %row_at) : (index, buffer, index, index) -> (f32)
 %row_i0 = scalar.fptosi %row_f : f32 to i32
 %row_i = kernel.subgroup.broadcast.first %row_i0 : i32
 %mbase0 = index.cast %row_i : i32 to index
 %mbase = index.assume %mbase0 [range(%mbase0, 0, 2032), multiple_of(%mbase0, 16)] : index
''' + body[end:]
    body = body.replace(' kernel.return', ' }\n kernel.return')
    if row_first:
        # Keep the original expert/channel/row order without launching empty
        # row partitions. The lookup task identifies the expert; its row offset
        # recovers that expert's prefix in the compact task list.
        body = body.replace(' %ntile = index.rem %wave0, %ntiles : index\n', '')
        body = body.replace(' %mbase = index.assume %mbase0', ' %lookup_row = index.assume %mbase0')
        body = body.replace(' %nbase = index.mul %ntile, %c64 : index\n', '')
        marker = ' %count = index.assume %count0 [range(%count0, 0, 2048)] : index\n'
        assert body.count(marker) == 1
        body = body.replace(marker, marker + ''' %tile_round = index.add %count, %c15 : index
 %expert_tiles0 = index.div %tile_round, %c16 : index
 %expert_tiles = index.assume %expert_tiles0 [range(%expert_tiles0, 1, 128)] : index
 %lookup_tile = index.div %lookup_row, %c16 : index
 %task_prefix0 = index.sub %task, %lookup_tile : index
 %task_prefix = index.assume %task_prefix0 [range(%task_prefix0, 0, 1150), le(%task_prefix0, %task)] : index
 %first_wave = index.mul %task_prefix, %ntiles : index
 %local_wave0 = index.sub %wave0, %first_wave : index
 %local_wave = index.assume %local_wave0 [range(%local_wave0, 0, 4095)] : index
 %ntile = index.div %local_wave, %expert_tiles : index
 %row_tile = index.rem %local_wave, %expert_tiles : index
 %mbase = index.mul %row_tile, %c16 : index
 %nbase = index.mul %ntile, %c64 : index
''')
    return '// Compact expert tile schedule; K order, scales and BF16 outputs are unchanged.\n' + launch + '} launch(' + body


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('kernels/omni_expert_i4.loom'))
    parser.add_argument('--row-first', action='store_true', help='preserve expert/channel/row launch order')
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.row_first))
