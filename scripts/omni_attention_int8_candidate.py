#!/usr/bin/env python3
"""Generate experimental per-row INT8 QK / BF16 PV Omni attention.

Uses the transposed BF16 generator and H3's signed INT8 WMMA operand layout
(95ed12f43bd3e596cbfad175c22de9560447e9e5, Apache-2.0). Neither mode uses
smoothing or rotation; this is a Sage-inspired experiment, not SageAttention.
The default mode quantizes inline with ordinary BF16 input/output bindings.
--prepacked consumes 36 U32 words per Q/K head row (32 packed codes, one FP32
scale, three padding words), produced by omni_prepare_attention_i8.loom.
V and output remain BF16; benchmark timings must include both packing passes.
"""
import argparse
import re
from pathlib import Path
from omni_attention_transposed_candidate import generate as bf16


def pack(name, values, inverse):
    lines = [f'%{name}_invv = vector.splat {inverse} : vector<16xf32>',
             f'%{name}_scaled = vector.mulf {values}, %{name}_invv : vector<16xf32>',
             f'%{name}_rounded = vector.roundevenf %{name}_scaled : vector<16xf32>',
             f'%{name}_lo = vector.maxnumf %{name}_rounded, %qminv : vector<16xf32>',
             f'%{name}_hi = vector.minnumf %{name}_lo, %qmaxv : vector<16xf32>',
             f'%{name}_codes = vector.fptosi %{name}_hi : vector<16xf32> to vector<16xi32>',
             f'%{name}_bytes = vector.andi %{name}_codes, %bytemaskv : vector<16xi32>']
    for word in range(4):
        for byte in range(4):
            i = word*4+byte
            lines.append(f'%{name}_b{i} = vector.extract %{name}_bytes[{i}] : vector<16xi32> -> i32')
            if byte:
                lines.append(f'%{name}_s{i} = scalar.shli %{name}_b{i}, %shift{byte*8} : i32')
        lines += [f'%{name}_a{word} = scalar.ori %{name}_b{word*4}, %{name}_s{word*4+1} : i32',
                  f'%{name}_c{word} = scalar.ori %{name}_s{word*4+2}, %{name}_s{word*4+3} : i32',
                  f'%{name}_w{word} = scalar.ori %{name}_a{word}, %{name}_c{word} : i32']
    lines.append(f'%{name} = vector.from_elements ' + ', '.join(f'%{name}_w{i}' for i in range(4)) + ' : vector<4xi32>')
    return '\n'.join(lines)


def generate(queries=128, keys=32):
    # Every K row is staged by eight adjacent lanes, so the XOR reduction below
    # cannot cross row boundaries. The same grouping holds in each packet pass.
    assert queries in (64, 128, 256) and keys in (32, 64)
    s = bf16(queries, keys)
    s = s.replace('Experimental transposed BF16 Omni attention', 'Experimental INT8 QK / BF16 PV Omni attention')
    s = s.replace(f'%voffset = index.constant {keys*136*2}', f'%voffset = index.constant {keys*36*4}')
    scale_offset = keys*36*4 + 128*(keys+8)*2
    s = s.replace(f'%lds_bytes = index.constant {keys*136*2+128*(keys+8)*2}', f'%lds_bytes = index.constant {scale_offset+keys*4}')
    s = s.replace(f'%ks = buffer.view %lds[%offset] : buffer -> view<{keys}x136xbf16>',
                  f'%ks = buffer.view %lds[%offset] : buffer -> view<{keys}x36xi32>\n%k_scale_offset = index.constant {scale_offset} : offset\n%k_scales = buffer.view %lds[%k_scale_offset] : buffer -> view<{keys}xf32>')
    constants = '''%qmax = scalar.constant 127.0 : f32
%qmin = scalar.constant -127.0 : f32
%qmaxv = vector.splat %qmax : vector<16xf32>
%qminv = vector.splat %qmin : vector<16xf32>
%tiny = scalar.constant 1e-30 : f32
%bytemaskv = vector.constant 255 : vector<16xi32>
%shift8 = scalar.constant 8 : i32
%shift16 = scalar.constant 16 : i32
%shift24 = scalar.constant 24 : i32
%xor1 = scalar.constant 1 : i32
%xor2 = scalar.constant 2 : i32
%xor4 = scalar.constant 4 : i32
%zi = vector.constant 0 : vector<8xi32>
%initi = vector.fragment<init> %zi shape [%c16, %c16] : vector<8xi32>
%i8_schema = encoding.define #encoding.operand<element_format=i8, payload_elements=16, payload_registers=4> : encoding<schema>
'''
    s = s.replace('%qcol0raw =', constants + '%qcol0raw =', 1)
    # Load Q once, quantizing each complete 128-value head row with one scale.
    for i in range(8):
        old = f'%qf{i} = vector.fragment<rhs> %qd{i} shape [%c16, %c16] : vector<16xbf16>'
        previous = '%zero' if i == 0 else f'%qmax{i-1}'
        s = s.replace(old, f'''%qx{i} = vector.extf %qd{i} : vector<16xbf16> to vector<16xf32>
%qa{i} = vector.absf %qx{i} : vector<16xf32>
%qmax{i} = vector.reduce<maxnumf> %qa{i}, {previous} : vector<16xf32>, f32''')
    quantize_q = ['%qamax = scalar.maxnumf %qmax7, %tiny : f32',
                  '%qscale = scalar.divf %qamax, %qmax : f32',
                  '%qinv = scalar.divf %qmax, %qamax : f32',
                  '%q_score_scale = scalar.mulf %qscale, %scale : f32']
    for i in range(8):
        quantize_q += [pack(f'qp{i}', f'%qx{i}', '%qinv'),
                      f'%qf{i} = vector.fragment<rhs> %qp{i} shape [%c16, %c16] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>']
    s = s.replace('%keyround =', '\n'.join(quantize_q)+'\n%keyround =', 1)
    for packet in range((keys*8+queries*2-1)//(queries*2)):
        old = f'vector.store %kv{packet}, %ks[%kr{packet}, %kcol{packet}raw] : vector<16xbf16>, view<{keys}x136xbf16>'
        assert s.count(old) == 1
        prefix=f'kq{packet}'
        lines=[f'%{prefix}_x = vector.extf %kv{packet} : vector<16xbf16> to vector<16xf32>',
               f'%{prefix}_abs = vector.absf %{prefix}_x : vector<16xf32>',
               f'%{prefix}_max0 = vector.reduce<maxnumf> %{prefix}_abs, %zero : vector<16xf32>, f32']
        previous=f'%{prefix}_max0'
        for delta in (1,2,4):
            lines += [f'%{prefix}_other{delta}, %{prefix}_valid{delta} = kernel.subgroup.shuffle<xor> {previous}, %xor{delta}, %i32 : f32, i32, i32',
                      f'%{prefix}_max{delta} = scalar.maxnumf {previous}, %{prefix}_other{delta} : f32']
            previous=f'%{prefix}_max{delta}'
        lines += [f'%{prefix}_amax = scalar.maxnumf {previous}, %tiny : f32',
                  f'%{prefix}_scale = scalar.divf %{prefix}_amax, %qmax : f32',
                  f'%{prefix}_inv = scalar.divf %qmax, %{prefix}_amax : f32',
                  pack(prefix, f'%{prefix}_x', f'%{prefix}_inv'),
                  f'%{prefix}_col = index.mul %ktc{packet}, %c4 : index',
                  f'vector.store %{prefix}, %ks[%kr{packet}, %{prefix}_col] : vector<4xi32>, view<{keys}x36xi32>',
                  f'%{prefix}_first = index.cmp eq, %ktc{packet}, %c0 : index',
                  f'scf.if %{prefix}_first {{',
                  f'view.store %{prefix}_scale, %k_scales[%kr{packet}] : f32, view<{keys}xf32>', '}']
        s=s.replace(old, '\n'.join(lines))
    for part in range(keys//16):
        for dim in range(8):
            s=s.replace(f'%kd{part}_{dim} = vector.load %ks[%krow{part}, %c{dim*16}] : view<{keys}x136xbf16> -> vector<16xbf16>',
                        f'%kd{part}_{dim} = vector.load %ks[%krow{part}, %c{dim*4}] : view<{keys}x36xi32> -> vector<4xi32>')
            s=s.replace(f'%kf{part}_{dim} = vector.fragment<lhs> %kd{part}_{dim} shape [%c16, %c16] : vector<16xbf16>',
                        f'%kf{part}_{dim} = vector.fragment<lhs> %kd{part}_{dim} shape [%c16, %c16] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>')
            prev='%init' if dim==0 else f'%dot{part}_{dim-1}'
            s=s.replace(f'%dot{part}_{dim} = vector.mma %kf{part}_{dim}, %qf{dim}, {prev} : vector<16xbf16>, vector<16xbf16>, vector<8xf32>',
                        f'%dot{part}_{dim} = vector.mma %kf{part}_{dim}, %qf{dim}, {"%initi" if dim==0 else prev} : vector<4xi32>, vector<4xi32>, vector<8xi32>')
        s=s.replace(f'%scaled{part} = vector.mulf %dot{part}_7, %scalev : vector<8xf32>',
                    f'%scaled{part} = vector.sitofp %dot{part}_7 : vector<8xi32> to vector<8xf32>')
        for i in range(8):
            old=f'%value{part}_{i} = vector.extract %scaled{part}[{i}] : vector<8xf32> -> f32'
            s=s.replace(old,f'''%ks_local{part}_{i}a = index.constant {part*16+i*2} : index
%ks_local{part}_{i} = index.add %ks_local{part}_{i}a, %lg : index
%ks_value{part}_{i} = view.load %k_scales[%ks_local{part}_{i}] : view<{keys}xf32> -> f32
%dot_value{part}_{i} = vector.extract %scaled{part}[{i}] : vector<8xf32> -> f32
%q_scaled{part}_{i} = scalar.mulf %dot_value{part}_{i}, %q_score_scale : f32
%value{part}_{i} = scalar.mulf %q_scaled{part}_{i}, %ks_value{part}_{i} : f32''')
    # Integer fragment column offsets need these constants in addition to BF16's.
    missing=[i for i in (20,24,28) if f'%c{i} =' not in s]
    s=s.replace('%offset =', '\n'.join(f'%c{i} = index.constant {i} : index' for i in missing)+'\n%offset =',1)
    assert not re.search(r'%dot\d+_\d+ =.*vector<16xbf16>',s)
    return s


def generate_prepacked(queries=128, keys=32):
    """Read 36 words/head: 32 INT8 payload words, FP32 scale, three pads."""
    s = generate(queries, keys)
    s = s.replace('Experimental INT8 QK / BF16 PV', 'Experimental prepacked INT8 QK / BF16 PV')
    s = s.replace('%qview = buffer.view %qg[%offset] : buffer -> view<[%queries]x[%qw]xbf16>', """%c36 = index.constant 36 : index
%qwords = index.mul %heads, %c36 : index
%kvheads = index.div %heads, %ratio : index
%kwords = index.mul %kvheads, %c36 : index
%qword_base = index.mul %head, %c36 : index
%kword_base = index.mul %kh, %c36 : index
%qword_limit = index.sub %qwords, %c4 : index
%kword_limit = index.sub %kwords, %c4 : index
%qview = buffer.view %qg[%offset] : buffer -> view<[%queries]x[%qwords]xi32>
%qscaleview = buffer.view %qg[%offset] : buffer -> view<[%queries]x[%qwords]xf32>""")
    s = s.replace('%kview = buffer.view %kg[%offset] : buffer -> view<[%keys]x[%kw]xbf16>',
                  '%kview = buffer.view %kg[%offset] : buffer -> view<[%keys]x[%kwords]xi32>\n%kscaleview = buffer.view %kg[%offset] : buffer -> view<[%keys]x[%kwords]xf32>')
    a=s.index('%qcol0raw ='); b=s.index('%keyround =',a)
    lines=[]
    for i in range(8):
        lines += [f'%qcol{i}raw = index.add %qc, %c{i*16} : index',
                  f'%qcol{i} = index.assume %qcol{i}raw [le(%qcol{i}raw, %qlim), multiple_of(%qcol{i}raw, 16)] : index',
                  f'%qwcol{i}raw = index.add %qword_base, %c{i*4} : index',
                  f'%qwcol{i} = index.assume %qwcol{i}raw [le(%qwcol{i}raw, %qword_limit), multiple_of(%qwcol{i}raw, 4)] : index',
                  f'%qp{i} = vector.load %qview[%qread, %qwcol{i}] : view<[%queries]x[%qwords]xi32> -> vector<4xi32>',
                  f'%qf{i} = vector.fragment<rhs> %qp{i} shape [%c16, %c16] using {{schema = %i8_schema : encoding<schema>}} : vector<4xi32>']
    lines += ['%qscolraw = index.add %qword_base, %c32 : index',
              '%qscol = index.assume %qscolraw [lt(%qscolraw, %qwords)] : index',
              '%qscale = view.load %qscaleview[%qread, %qscol] : view<[%queries]x[%qwords]xf32> -> f32',
              '%q_score_scale = scalar.mulf %qscale, %scale : f32']
    s=s[:a]+'\n'.join(lines)+'\n'+s[b:]
    for packet in range((keys*8+queries*2-1)//(queries*2)):
        a=s.index(f'%kv{packet} = vector.load'); b=s.index(f'%vr{packet} =',a)
        s=s[:a]+f"""%kq{packet}_col = index.mul %ktc{packet}, %c4 : index
%kwcol{packet}raw = index.add %kword_base, %kq{packet}_col : index
%kwcol{packet} = index.assume %kwcol{packet}raw [le(%kwcol{packet}raw, %kword_limit), multiple_of(%kwcol{packet}raw, 4)] : index
%kq{packet} = vector.load %kview[%kread{packet}, %kwcol{packet}] : view<[%keys]x[%kwords]xi32> -> vector<4xi32>
vector.store %kq{packet}, %ks[%kr{packet}, %kq{packet}_col] : vector<4xi32>, view<{keys}x36xi32>
%kq{packet}_first = index.cmp eq, %ktc{packet}, %c0 : index
scf.if %kq{packet}_first {{
 %kscol{packet}raw = index.add %kword_base, %c32 : index
 %kscol{packet} = index.assume %kscol{packet}raw [lt(%kscol{packet}raw, %kwords)] : index
 %kscale{packet} = view.load %kscaleview[%kread{packet}, %kscol{packet}] : view<[%keys]x[%kwords]xf32> -> f32
 view.store %kscale{packet}, %k_scales[%kr{packet}] : f32, view<{keys}xf32>
}}
"""+s[b:]
    return s


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=int, choices=(64,128,256), default=128)
    p.add_argument('--keys', type=int, choices=(32,64), default=32)
    p.add_argument('--prepacked', action='store_true')
    p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.write_text((generate_prepacked if a.prepacked else generate)(a.queries,a.keys))
