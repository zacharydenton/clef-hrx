#!/usr/bin/env python3
"""Transpose loader thread assignments while retaining the attention LDS layout.

The ordinary loader assigns consecutive 16-element column packets to adjacent
threads. The candidate assigns consecutive rows instead, with columns advancing
between groups of rows. Global reads and matching LDS writes use the same new
coordinates. Each tile element is still written once before the existing
workgroup barrier; fragment reads and all attention arithmetic are unchanged.

This trades global-read coalescing against modeled LDS store-bank conflicts.
The cacheline mapping keeps two adjacent 32-byte column packets together and
swaps row bits 1 and 2, spreading each eight-lane LDS store group over its banks.
Derived sources retain the transposed attention kernel's attribution.
"""
import argparse
from pathlib import Path
import re


def generate(source, keys, operand, mapping='row'):
    assert keys in (32, 64) and operand in ('k', 'v', 'both')
    assert mapping in ('row', 'cacheline')
    assert f'view<{keys}x136xbf16>' in source
    assert f'view<128x{keys + 8}xbf16>' in source
    assert '%kid0 =' in source
    for prefix, rows, columns in [('k', keys, 8), ('v', 128, keys // 16)]:
        if operand not in (prefix, 'both'):
            continue
        row_pattern = rf'%{prefix}r(\d+) = index.div %kid\1, %c{columns} : index'
        col_pattern = rf'%{prefix}tc(\d+) = index.rem %kid\1, %c{columns} : index'
        if mapping == 'row':
            row_replacement = rf'%{prefix}r\1 = index.rem %kid\1, %c{rows} : index'
            col_replacement = rf'%{prefix}tc\1 = index.div %kid\1, %c{rows} : index'
        else:
            def row_replacement(match):
                i = match[1]
                p = f'%{prefix}loader{i}_'
                return '\n'.join([
                    f'{p}pair = index.div %kid{i}, %c2 : index',
                    f'{p}row = index.rem {p}pair, %c{rows} : index',
                    f'{p}low = index.rem {p}row, %c2 : index',
                    f'{p}half = index.div {p}row, %c2 : index',
                    f'{p}bit1 = index.rem {p}half, %c2 : index',
                    f'{p}quarter = index.div {p}row, %c4 : index',
                    f'{p}bit2 = index.rem {p}quarter, %c2 : index',
                    f'{p}block = index.div {p}row, %c8 : index',
                    f'{p}high = index.mul {p}block, %c8 : index',
                    f'{p}mid1 = index.mul {p}bit1, %c4 : index',
                    f'{p}mid2 = index.mul {p}bit2, %c2 : index',
                    f'{p}mid = index.add {p}mid1, {p}mid2 : index',
                    f'{p}bottom = index.add {p}low, {p}mid : index',
                    f'%{prefix}r{i} = index.add {p}high, {p}bottom : index',
                ])

            def col_replacement(match):
                i = match[1]
                p = f'%{prefix}loader{i}_'
                return '\n'.join([
                    f'{p}col_block = index.div %kid{i}, %c{rows * 2} : index',
                    f'{p}col_base = index.mul {p}col_block, %c2 : index',
                    f'{p}col_low = index.rem %kid{i}, %c2 : index',
                    f'%{prefix}tc{i} = index.add {p}col_base, {p}col_low : index',
                ])
        source, row_count = re.subn(row_pattern, row_replacement, source)
        source, col_count = re.subn(col_pattern, col_replacement, source)
        assert row_count == col_count == keys // 32, 'Unrecognized loader structure'
    label = 'row-first' if mapping == 'row' else 'cacheline-paired'
    return f'// Experimental {label} {operand.upper()} loader assignment; LDS layout unchanged.\n' + source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--keys', type=int, choices=(32, 64), default=32)
    parser.add_argument('--operand', choices=('k', 'v', 'both'), required=True)
    parser.add_argument('--mapping', choices=('row', 'cacheline'), default='row')
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text(), args.keys, args.operand, args.mapping))
