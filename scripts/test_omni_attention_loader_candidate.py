"""Check generated loader addresses before running experimental GPU kernels."""
import re
import unittest

from omni_attention_loader_candidate import generate
from omni_attention_transposed_candidate import generate as attention


def coordinates(source, operand, packet, thread):
    """Interpret the emitted straight-line loader index arithmetic."""
    values = {name: int(value) for name, value in re.findall(
        r'(%\w+) = index.constant (\d+) : index', source)}
    values[f'%kid{packet}'] = thread + packet * 256
    operations = {'div': lambda a, b: a // b, 'rem': lambda a, b: a % b,
                  'mul': lambda a, b: a * b, 'add': lambda a, b: a + b}
    for dest, op, left, right in re.findall(
            r'(%\w+) = index\.(div|rem|mul|add) (%\w+), (%\w+) : index', source):
        if dest.startswith(f'%{operand}loader{packet}_') or dest in (
                f'%{operand}r{packet}', f'%{operand}tc{packet}'):
            values[dest] = operations[op](values[left], values[right])
    return values[f'%{operand}r{packet}'], values[f'%{operand}tc{packet}']


class LoaderAddresses(unittest.TestCase):
    def test_coverage_coalescing_and_store_banks(self):
        for keys in (32, 64):
            for operand, rows, columns, stride in (
                    ('k', keys, 8, 136), ('v', 128, keys // 16, keys + 8)):
                for mapping in ('row', 'cacheline'):
                    with self.subTest(keys=keys, operand=operand, mapping=mapping):
                        source = generate(attention(128, keys), keys, operand, mapping)
                        coords = [coordinates(source, operand, packet, thread)
                                  for packet in range(keys // 32) for thread in range(256)]
                        # Every original vector packet must be written exactly once,
                        # with no duplicates or addresses outside the allocated tile.
                        self.assertEqual(sorted(coords), [(r, c) for r in range(rows)
                                                          for c in range(columns)])
                        if mapping != 'cacheline':
                            continue
                        for i in range(0, len(coords), 2):
                            r, c = coords[i]
                            self.assertEqual(c % 2, 0)
                            self.assertEqual(coords[i + 1], (r, c + 1))
                        # Each 32-byte vector becomes two b128 LDS packets. In
                        # either packet each eight-lane store phase uses all banks.
                        for i in range(0, len(coords), 8):
                            for half in (0, 4):
                                banks = [(r * stride // 2 + c * 8 + half + word) % 32
                                         for r, c in coords[i:i + 8] for word in range(4)]
                                self.assertEqual(sorted(banks), list(range(32)))

    def test_preserves_unselected_operand(self):
        original = attention(128, 32)
        for selected, preserved in (('k', 'v'), ('v', 'k')):
            candidate = generate(original, 32, selected, 'cacheline')
            for thread in range(256):
                self.assertEqual(coordinates(original, preserved, 0, thread),
                                 coordinates(candidate, preserved, 0, thread))


if __name__ == '__main__':
    unittest.main()
