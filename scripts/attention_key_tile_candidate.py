#!/usr/bin/env python3
"""Generate an experimental 64-key attention tile; model evaluation is required."""
import argparse
import re
from pathlib import Path


BARRIER = 'kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n'


def _renamed(block, prefix, substitutions=None):
    """Rename definitions and uses within one straight-line Loom block."""
    substitutions = substitutions or {}
    definitions = set()
    for line in block.splitlines():
        if ' = ' in line:
            definitions.update(re.findall(r'%\w+', line.split(' = ', 1)[0]))

    def name(match):
        token = match[0]
        return substitutions.get(token, '%' + prefix + token[1:] if token in definitions else token)

    return re.sub(r'%\w+', name, block)


def generate(original):
    """Stage two K/V halves and perform one online softmax update per 64 keys."""
    if '%problow =' in original or original.count('%qkzero =') != 1:
        raise ValueError('expected the 64-query, 32-key single-component attention source')
    for marker in ('%threads = index.constant 256', '%kb = index.mul %kt, %c32',
                   '%lds_bytes = index.constant 64768', '%stagevc0 ='):
        if original.count(marker) != 1:
            raise ValueError(f'kernel structure changed: {marker}')
    s = '\n'.join(l for l in original.splitlines() if not l.startswith('//'))+'\n'
    s = '\n'.join(l for l in s.splitlines() if not l.startswith(('%ploff =','%pls =')))+'\n'
    s = s.replace('index.constant 44032 : offset','index.constant 43008 : offset')
    s = s.replace('index.constant 64512 : offset','index.constant 63488 : offset')
    s = s.replace('index.constant 64768 : offset','index.constant 63744 : offset')
    s = s.replace('view<64x32xf32>', 'view<64x64xf32>').replace('view<64x40x','view<64x72x')
    s = s.replace('%keyround = index.add %keys, %c31','%keyround = index.add %keys, %key63')
    s = s.replace('%keyround =', '%key63 = index.constant 63 : index\n%keyround =',1)
    s = s.replace('%keytiles = index.div %keyround, %c32','%keytiles = index.div %keyround, %c64')
    s = s.replace('%causaltiles = index.div %query_end, %c32','%causaltiles = index.div %query_end, %c64')
    start = s.index('%kb = index.mul %kt, %c32')
    end = s.index('\n}\nscf.if %leader {',start)
    body = s[start:end]+'\n'
    prefix = '%kb = index.mul %kt, %c64 : index\n%kb_second = index.add %kb, %c32 : index\n%lane_second = index.add %lane, %c32 : index\n%koff_second = index.add %koff, %c32 : index\n'
    k_start = body.index('%kcol0 =')
    q_start = body.index('%qkzero =')
    score_start = body.index('vector.fragment.store<result> %mask7')
    soft_start = body.index('%sv0 =')
    v_start = body.index('%stagevc0 =')
    alpha_start = body.index('%ar0 =')
    pv_start = body.index('%ph0 =')
    yield_start = body.index('scf.yield %nextmax,')
    kstage = body[k_start:q_start]
    qk = body[q_start:score_start]
    if not qk.endswith(BARRIER):
        raise ValueError('QK staging barrier changed')
    qk = qk[:-len(BARRIER)]
    score = body[score_start:soft_start]
    soft = body[soft_start:v_start]
    vstage = body[v_start:alpha_start]
    alpha = body[alpha_start:pv_start]
    pv = body[pv_start:yield_start]
    if not pv.endswith(BARRIER):
        raise ValueError('PV staging barrier changed')
    pv = pv[:-len(BARRIER)]
    soft_gather_end = soft.index('%ms1,')
    gather = soft[:soft_gather_end]
    other_gather = _renamed(gather, 'other_', {'%lane':'%lane_second'})
    soft = gather + other_gather + '%both_scores = vector.maxnumf %svec7, %other_svec7 : vector<8xf32>\n' + soft[soft_gather_end:]
    soft = soft.replace('shuffle<xor> %svec7,','shuffle<xor> %both_scores,').replace('vector.maxnumf %svec7, %ms1','vector.maxnumf %both_scores, %ms1')
    exp_line = '%prob = vector.expf<afn> %pdiff : vector<8xf32>\n'
    if soft.count(exp_line) != 1:
        raise ValueError('softmax exponent changed')
    soft = soft.replace(exp_line, exp_line + '%pdiff_second = vector.subf %other_svec7, %nextmax : vector<8xf32>\n%prob_second = vector.expf<afn> %pdiff_second : vector<8xf32>\n%both_prob = vector.addf %prob, %prob_second : vector<8xf32>\n')
    soft = soft.replace('shuffle<xor> %prob,','shuffle<xor> %both_prob,').replace('vector.addf %prob, %ss1','vector.addf %both_prob, %ss1')
    p_start = soft.index('%probhalf =')
    p_end = soft.index('scf.if %leader {', p_start)
    probability = soft[p_start:p_end]
    if probability.count(BARRIER) != 1:
        raise ValueError('probability staging barrier changed')
    probability = probability.replace(BARRIER,'')
    soft = soft[:p_start] + BARRIER + probability + _renamed(probability,'other_',{'%prob':'%prob_second','%lane':'%lane_second'}) + soft[p_end:]
    # The second PV pass continues the first pass's accumulators without a
    # second online rescaling. V occupies only 32 columns at a time.
    pv_second = '\n'.join(line for line in pv.splitlines() if not line.startswith('%rescale'))+'\n'
    pv_second = pv_second.replace('%phs[%qoff, %c0]', '%phs[%qoff, %c32]').replace('%phs[%qoff, %c16]', '%phs[%qoff, %c48]')
    pv_second = _renamed(pv_second, 'pv1_', {f'%rescale{i}':f'%pv0_next1_{i}' for i in range(8)})
    score_second = _renamed(score,'score1_',{'%mask7':'%qk1_mask7','%koff':'%koff_second'})
    # Both QK passes finish reading aliased K before either score tile is stored.
    score_first = _renamed(score.replace(BARRIER,''),'score0_',{'%mask7':'%qk0_mask7'})
    new_body = (prefix + _renamed(kstage,'k0_') + _renamed(qk,'qk0_') + BARRIER
                + _renamed(kstage,'k1_',{'%kb':'%kb_second'}) + _renamed(qk,'qk1_',{'%kb':'%kb_second'}) + BARRIER
                + score_first + score_second + soft
                + _renamed(vstage,'v0_') + alpha + _renamed(pv,'pv0_') + BARRIER
                + _renamed(vstage,'v1_',{'%kb':'%kb_second'}) + pv_second + BARRIER
                + 'scf.yield %nextmax, %nextsum, ' + ', '.join(f'%pv1_next1_{i}' for i in range(8))+' : '+', '.join(['vector<8xf32>']*10)+'\n')
    candidate = '// Experimental 64-query / 64-key single-component attention.\n// Keys and values stage in two 32-key halves; softmax reduction order changes.\n'+s[:start]+new_body+s[end:]
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(candidate)


if __name__ == '__main__':
    main()
