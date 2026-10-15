#!/usr/bin/env python3
"""Regroup independent INT4 expert waves without changing their arithmetic.

The existing kernel uses two waves per workgroup. Vary that scheduling choice,
preserving global wave IDs and each wave's private LDS slice. A final partial
workgroup masks excess waves before any tensor access. Intended for the ordinary
or generated K128 expert source, including its specialized up/down branches.
"""
import argparse
from pathlib import Path
import re


def generate(source, waves):
    assert waves in (1, 2, 4, 8)
    if waves == 2:
        return source
    launch = re.search(r'\} launch\([^)]*\) \{\n', source)
    definition = re.search(r'kernel.def[^\n]*\{\n', source)
    assert definition and launch
    head = source[definition.end():launch.start()]
    calculation, launch_config = head.split(' kernel.launch.config', 1)
    assert ' %threads = index.constant 64 : index' in calculation
    assert ' %groups = index.mul %basegroups, %tiles : index' in calculation
    assert 'scope(workgroup)' not in source, 'Independent waves must not use group-wide barriers'
    # Preserve the original number of waves, including non-divisible grids.
    new_calculation = calculation.replace(
        ' %threads = index.constant 64 : index',
        f' %threads = index.constant {waves * 32} : index',
    ).replace(
        ' %groups = index.mul %basegroups, %tiles : index',
        ' %old_groups = index.mul %basegroups, %tiles : index\n'
        ' %waves_before = index.constant 2 : index\n'
        f' %waves_after = index.constant {waves} : index\n'
        f' %wave_rounding = index.constant {waves - 1} : index\n'
        ' %wave_count = index.mul %old_groups, %waves_before : index\n'
        ' %rounded_waves = index.add %wave_count, %wave_rounding : index\n'
        ' %groups = index.div %rounded_waves, %waves_after : index',
    )
    body = source[launch.end():source.rindex(' kernel.return')]
    body, decoding = re.subn(
        r' %wgbase = index.mul %wg, %c2 : index',
        f' %waves_per_group = index.constant {waves} : index\n'
        ' %wgbase = index.mul %wg, %waves_per_group : index', body,
    )
    body, scratch = re.subn(
        r' %lds_bytes = index.constant 2048 : offset',
        f' %lds_bytes = index.constant {waves * 1024} : offset', body,
    )
    assert decoding == scratch and decoding in (1, 2), 'Unrecognized expert wave layout'
    # The guard uses the original launch arithmetic, renamed to avoid collisions
    # with the nested source body. Compilation specializes these bounds.
    guard = re.sub(r'%([A-Za-z0-9_]+)', r'%launch_guard_\1', calculation)
    guard += (
        ' %launch_guard_two = index.constant 2 : index\n'
        f' %launch_guard_waves_per_group = index.constant {waves} : index\n'
        ' %launch_guard_wg = kernel.workgroup.id<x> : index\n'
        ' %launch_guard_subgroup = kernel.subgroup.id : index\n'
        ' %launch_guard_base = index.mul %launch_guard_wg, %launch_guard_waves_per_group : index\n'
        ' %launch_guard_wave = index.add %launch_guard_base, %launch_guard_subgroup : index\n'
        ' %launch_guard_count = index.mul %launch_guard_groups, %launch_guard_two : index\n'
        ' %launch_guard_active = index.cmp ult, %launch_guard_wave, %launch_guard_count : index\n'
        ' scf.if %launch_guard_active {\n'
    )
    return (f'// Experimental expert scheduling: {waves} independent waves per workgroup.\n'
            + source[:definition.end()] + new_calculation + ' kernel.launch.config'
            + launch_config + source[launch.start():launch.end()] + guard
            + body + ' }\n kernel.return\n}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('kernels/omni_expert_i4.loom'))
    parser.add_argument('--waves', type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.waves))
