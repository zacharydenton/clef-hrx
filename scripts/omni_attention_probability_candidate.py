#!/usr/bin/env python3
"""Generate BF16 attention with probability-block-major PV scheduling.

Consume each packed probability fragment immediately, rather than retaining all
fragments while iterating over output channels. Each output still accumulates
key fragments in the same order, with the same rescaling and BF16 rounding.
The source generators retain their documented H3/krea2 provenance.
"""
import argparse
import re
from pathlib import Path
from omni_attention_transposed_candidate import generate as transposed
from omni_attention_prefetch_candidate import generate as prefetched


def generate(queries=128, keys=32, prefetch=False):
    s = prefetched(queries, keys) if prefetch else transposed(queries, keys)
    first = s.index('%delta0 =')
    last = s.index('kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)', first)
    old = s[first:last]
    probabilities = []
    for part in range(keys//16):
        begin = old.index(f'%delta{part} =')
        end = old.index('\n', old.index(f'%pf{part} =', begin)) + 1
        probabilities.append(old[begin:end])
    initializers = []
    updates = {}
    for dim in range(8):
        initializers.append(re.search(rf'^%vrow{dim} =.*\n%rescale{dim} =.*\n', old, re.M)[0])
        for part in range(keys//16):
            updates[dim,part] = re.search(rf'^%vd{dim}_{part} =.*\n%vf{dim}_{part} =.*\n%pv{dim}_{part} =.*\n', old, re.M)[0]
    # Every operation is preserved exactly once; only independent order changes.
    new = ''.join(initializers)
    for part, probability in enumerate(probabilities):
        new += probability + ''.join(updates[dim,part] for dim in range(8))
    assert sorted(old.splitlines()) == sorted(new.splitlines())
    return '// Experimental probability-block-major scheduling; unchanged arithmetic.\n' + s[:first] + new + s[last:]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries',type=int,choices=(64,128,256),default=128)
    p.add_argument('--keys',type=int,choices=(32,64),default=32)
    p.add_argument('--prefetch',action='store_true')
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.write_text(generate(a.queries,a.keys,a.prefetch))
