"""Check emitted LDS addresses independently of GPU numerical qualification."""
import re
import unittest
from omni_attention_int8_candidate import generate, generate_prepacked
from omni_attention_scale_candidate import generate as scale_candidate


class ScaleAddresses(unittest.TestCase):
    def test_scale_storage_and_fragment_reads_are_inverse(self):
        for queries in [64, 128, 256]:
            for keys in [32, 64]:
                for base in [generate, generate_prepacked]:
                    for vector in [False, True]:
                        with self.subTest(queries=queries, keys=keys, prepacked=base == generate_prepacked, vector=vector):
                            source = scale_candidate(base(queries, keys), vector)
                            shape = re.search(r'%k_scales = buffer.view .* -> view<(\d+)x(\d+)xf32>', source)
                            self.assertIsNotNone(shape)
                            height, width = map(int, shape.groups())
                            self.assertEqual((height, width), (2, keys // 2))
                            reads = re.findall(r'%scale_vector(\d+) = vector.load %k_scales\[%scale_lane(\d+), %scale_column(\d+)\] : view<\d+x\d+xf32> -> vector<8xf32>', source)
                            self.assertEqual(reads, [(str(i), str(i), str(i)) for i in range(keys // 16)])
                            for i in range(keys // 16):
                                self.assertIn(f'%scale_lane{i} = index.assume %lg [range(%lg, 0, 1)] : index', source)
                            writes = re.findall(r'(%scale_store\d+_\w+) = index\.(rem|div) (%\w+), %c2 : index', source)
                            self.assertTrue(writes)
                            # Interpret the emitted arithmetic for every key row.
                            for site in range(len(writes) // 2):
                                storage = {}
                                for row in range(keys):
                                    env = {}
                                    for name, op, _ in writes[site*2:site*2+2]:
                                        env[name] = row % 2 if op == 'rem' else row // 2
                                    address = env[f'%scale_store{site}_parity'] * width + env[f'%scale_store{site}_column']
                                    self.assertTrue(0 <= address < keys)
                                    self.assertNotIn(address, storage)
                                    storage[address] = row
                                self.assertEqual(set(storage), set(range(keys)))
                                columns = dict((int(p), int(c)) for p, c in re.findall(r'%scale_column(\d+) = index.constant (\d+) : index', source))
                                self.assertEqual(len(columns), keys // 16)
                                for lane in range(32):
                                    for part in range(keys // 16):
                                        offset = (lane // 16) * width + columns[part]
                                        self.assertEqual(offset % 8, 0)
                                        for element in range(8):
                                            self.assertEqual(storage[offset + element], part * 16 + element * 2 + lane // 16)


if __name__ == '__main__':
    unittest.main()
