"""Preserve BF16 PV/softmax and verify packed Q/K fragment coverage."""
import re
import unittest
from omni_attention_int4_candidate import generate
from omni_attention_int8_candidate import generate_prepacked


class Int4AttentionLayout(unittest.TestCase):
    def test_qk_coverage_and_unchanged_softmax_pv(self):
        for queries in [64,128,256]:
            for keys in [32,64]:
                with self.subTest(queries=queries,keys=keys):
                    old=generate_prepacked(queries,keys)
                    new=generate(queries,keys)
                    # The entire softmax/PV/output recurrence is outside the
                    # precision change; this catches accidental vector resizing.
                    marker='%partnermax, %validmax ='
                    self.assertEqual(old[old.index(marker):],new[new.index(marker):])
                    self.assertIn('element_format=i4, payload_elements=16, payload_registers=2',new)
                    qcols=[int(x) for x in re.findall(r'%qwcol\d+raw = index.add %qword_base, %c(\d+) : index',new)]
                    self.assertEqual([channel for col in qcols for channel in range(col*8,(col+2)*8)],list(range(128)))
                    for part in range(keys//16):
                        kcols=[int(x) for x in re.findall(r'%kd'+str(part)+r'_\d+ = vector.load %ks\[%krow'+str(part)+r', %c(\d+)\]',new)]
                        self.assertEqual(kcols,qcols)
                        self.assertIn(f'%dot{part}_4 = vector.mma %kf{part}_4, %qf4, %initi',new)
                        self.assertIn(f'%scaledlo{part} = vector.sitofp %dot{part}_3',new)
                        for element in range(8):
                            self.assertIn(f'%value{part}_{element} = scalar.addf %score_lo{part}_{element}, %score_hi{part}_{element}',new)
                    self.assertEqual(new.count('vector<2xi32>'),8*2+(keys//16)*8*4+((keys*8+queries*2-1)//(queries*2))*2)


if __name__ == '__main__':
    unittest.main()
