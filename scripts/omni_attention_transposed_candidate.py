#!/usr/bin/env python3
"""Generate an experimental BF16 Omni attention kernel with scalar softmax stats.

The transposed WMMA and packed-lane exchange follow h3-hrx's
attention_mha64t32_lds_f16_wmma.loom at 95ed12f4 (Apache-2.0).
This version uses 128-wide BF16 heads, GQA, causal masks, bounded tail loads,
and CLEF's transposed V layout. It is not a production selector.
"""
import argparse
import re
from pathlib import Path


def swizzle_lds(source, keys):
    """XOR whole 16-element packets within each row; no fragment permutation.

    Replacing padded row strides by packet swizzling saves LDS while retaining
    contiguous vector loads. The same row-dependent permutation is used by
    writers and readers. All QK, softmax and PV operations stay unchanged.
    """
    source = source.replace(f'view<{keys}x136xbf16>', f'view<{keys}x128xbf16>')
    source = source.replace(f'view<128x{keys+8}xbf16>', f'view<128x{keys}xbf16>')
    source = source.replace(
        f'%voffset = index.constant {keys*136*2} : offset',
        f'%voffset = index.constant {keys*128*2} : offset',
    )
    source = source.replace(
        f'%lds_bytes = index.constant {keys*136*2+128*(keys+8)*2} : offset',
        f'%lds_bytes = index.constant {keys*128*4} : offset',
    )
    pattern = re.compile(r'^(.*(?:vector\.load|vector\.store).*?%(ks|vs))\[(%\w+), (%\w+)\](.*)$', re.M)
    serial = 0

    def address(match):
        nonlocal serial
        prefix, matrix, row, col, suffix = match.groups()
        width = 128 if matrix == 'ks' else keys
        banks = min(4, width // 16)
        name = f'%swz{serial}'
        serial += 1
        return (
            f'{name}_row = index.rem {row}, %c{banks} : index\n'
            f'{name}_shift = index.mul {name}_row, %c16 : index\n'
            f'{name}_shift_i = index.cast {name}_shift : index to i32\n'
            f'{name}_col_i = index.cast {col} : index to i32\n'
            f'{name}_xor = scalar.xori {name}_col_i, {name}_shift_i : i32\n'
            f'{name}_raw = index.cast {name}_xor : i32 to index\n'
            f'{name} = index.assume {name}_raw [range({name}_raw, 0, {width-16}), multiple_of({name}_raw, 16)] : index\n'
            f'{prefix}[{row}, {name}]{suffix}'
        )

    source = pattern.sub(address, source)
    assert serial > 0 and not re.search(r'%(?:ks|vs)\[%\w+, %(?!swz)\w+\]', source)
    return '// Experimental packet-swizzled LDS; unchanged arithmetic.\n' + source


def generate(queries=64, keys=32, lane_controls=False, paired_accumulators=False, swizzle=False):
    assert queries in (64, 128, 256) and keys in (32, 64, 128)
    assert keys != 128 or swizzle, 'K128 requires compact LDS to fit 64 KiB'
    lines = []
    def emit(s):
        lines.extend(s.strip().splitlines())
    emit(f'''// Experimental transposed BF16 Omni attention: {queries} queries, {keys} keys.
// Adapted from h3-hrx, revision 95ed12f43bd3e596cbfad175c22de9560447e9e5, Apache-2.0.
amdgpu.target<gfx1151> @clef_target {{subgroup_size = 32}}''')
    for key in ('heads', 'ratio', 'qw', 'kw', 'causal'):
        emit(f'config.decl @clef.{key} : %value: index where [range(%value, 0, 4294967295)]')
    emit('config.decl @clef.scale : f32')
    if lane_controls:
        emit('''// Lane-dependent byte controls remove vector selections before interleaving.
low.func.def target<amdgpu.rdna3_5.core>(@clef_target) @clef_interleave_bf16(%even: reg<amdgpu.vgpr x4>, %odd: reg<amdgpu.vgpr x4>, %lo: reg<amdgpu.vgpr>, %hi: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr x8>) asm {''')
    else:
        emit('''low.func.def target<amdgpu.rdna3_5.core>(@clef_target) @clef_interleave_bf16(%even: reg<amdgpu.vgpr x4>, %odd: reg<amdgpu.vgpr x4>) -> (reg<amdgpu.vgpr x8>) asm {
 %lo = s_mov_b32 84148480
 %hi = s_mov_b32 117834498''')
    for i in range(4):
        emit(f''' %e{i} = slice %even[{i}] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>
 %o{i} = slice %odd[{i}] : reg<amdgpu.vgpr x4> -> reg<amdgpu.vgpr>
 %p{i*2} = v_perm_b32 %o{i}, %e{i}, %lo
 %p{i*2+1} = v_perm_b32 %o{i}, %e{i}, %hi''')
    emit(' %p = concat(' + ', '.join(f'%p{i}' for i in range(8)) + ') : (' + ', '.join(['reg<amdgpu.vgpr>'] * 8) + ') -> reg<amdgpu.vgpr x8>\n return %p\n}')
    if paired_accumulators:
        # H3 keeps each two adjacent PV fragments in one owned register group.
        # Only the chosen half is rescaled/updated, in the original K order.
        for slot in (0, 1):
            for scaled in (False, True):
                name = f'clef_pv_{slot}' + ('_scaled' if scaled else '')
                schedule = '' if scaled else ' schedule(locked)'
                emit(f'''low.func.def{schedule} target<amdgpu.rdna3_5.core>(@clef_target) @{name}(%v: reg<amdgpu.vgpr x8>, %p: reg<amdgpu.vgpr x8>, %acc: reg<amdgpu.vgpr x16>, %scale: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr x16>) asm {{
 %owned = copy %acc : reg<amdgpu.vgpr x16> -> reg<amdgpu.vgpr x16>
 %a0 = slice %owned[0] : reg<amdgpu.vgpr x16> -> reg<amdgpu.vgpr x8>
 %a1 = slice %owned[8] : reg<amdgpu.vgpr x16> -> reg<amdgpu.vgpr x8>''')
                operand = f'%a{slot}'
                if scaled:
                    for i in range(8):
                        emit(f''' %x{i} = slice %a{slot}[{i}] : reg<amdgpu.vgpr x8> -> reg<amdgpu.vgpr>
 %y{i} = v_mul_f32 %x{i}, %scale''')
                    emit(' %scaled = concat(' + ', '.join(f'%y{i}' for i in range(8)) + ') : (' + ', '.join(['reg<amdgpu.vgpr>'] * 8) + ') -> reg<amdgpu.vgpr x8>')
                    operand = '%scaled'
                emit(f' %r = v_wmma_f32_16x16x16_bf16 %v, %p, {operand}')
                halves = '%r, %a1' if slot == 0 else '%a0, %r'
                emit(f''' %out = concat({halves}) : (reg<amdgpu.vgpr x8>, reg<amdgpu.vgpr x8>) -> reg<amdgpu.vgpr x16>
 return %out
}}''')
        emit('''low.func.def target<amdgpu.rdna3_5.core>(@clef_target) @clef_split_pair(%acc: reg<amdgpu.vgpr x16>) -> (reg<amdgpu.vgpr x8>, reg<amdgpu.vgpr x8>) asm {
 %a = slice %acc[0] : reg<amdgpu.vgpr x16> -> reg<amdgpu.vgpr x8>
 %b = slice %acc[8] : reg<amdgpu.vgpr x16> -> reg<amdgpu.vgpr x8>
 return %a, %b
}''')
    emit(f'''kernel.def target(@clef_target) export("clef_op") @clef_op(%query_rows: index, %key_rows: index, %v_capacity: index) {{
 %one = index.constant 1 : index
 %round = index.constant {queries-1} : index
 %qt = index.constant {queries} : index
 %threads = index.constant {queries*2} : index
 %heads = config.get @clef.heads : index
 %rounded = index.add %query_rows, %round : index
 %tiles = index.div %rounded, %qt : index
 %groups = index.mul %tiles, %heads : index
 kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%query_rows: index, %key_rows: index, %v_capacity: index, %q_buffer: buffer, %k_buffer: buffer, %v_buffer: buffer, %out_buffer: buffer) {{''')
    for c in sorted(set(range(16)) | set(range(16, 129, 16)) | {31, 63, queries, keys, keys-1, queries*2}):
        emit(f'%c{c} = index.constant {c} : index')
    emit(f'''%offset = index.constant 0 : offset
%voffset = index.constant {keys*136*2} : offset
%lds_bytes = index.constant {keys*136*2+128*(keys+8)*2} : offset
%lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
%ks = buffer.view %lds[%offset] : buffer -> view<{keys}x136xbf16>
%vs = buffer.view %lds[%voffset] : buffer -> view<128x{keys+8}xbf16>
%queries = index.assume %query_rows [range(%query_rows, 1, 65536)] : index
%keys = index.assume %key_rows [range(%key_rows, 1, 65536), le(%key_rows, %v_capacity)] : index
%vstride = index.assume %v_capacity [range(%v_capacity, 64, 65536), multiple_of(%v_capacity, 64)] : index
%qw = config.get @clef.qw : index
%kw = config.get @clef.kw : index
%heads = config.get @clef.heads : index
%ratio = config.get @clef.ratio : index
%scale = config.get @clef.scale : f32
%scalev = vector.splat %scale : vector<8xf32>
%causal = config.get @clef.causal : index
%is_causal = index.cmp ne, %causal, %c0 : index
%wg = kernel.workgroup.id<x> : index
%tid = kernel.workitem.id<x> : index
%lane = index.rem %tid, %c32 : index
%wave = index.div %tid, %c32 : index
%lc = index.rem %lane, %c16 : index
%lg = index.div %lane, %c16 : index
%even = index.cmp eq, %lg, %c0 : index
%head = index.rem %wg, %heads : index
%tile = index.div %wg, %heads : index
%qbase = index.mul %tile, %c{queries} : index
%qoff = index.mul %wave, %c16 : index
%qorigin = index.add %qbase, %qoff : index
%qrow = index.add %qorigin, %lc : index
%qvalid = index.cmp ult, %qrow, %queries : index
%qsafe = scf.select %qvalid, %qrow, %c0 : index
%qread = index.assume %qsafe [lt(%qsafe, %queries)] : index
%kh = index.div %head, %ratio : index
%qc = index.mul %head, %c128 : index
%kc = index.mul %kh, %c128 : index
%qg = buffer.assume.memory_space<global> %q_buffer : buffer
%kg = buffer.assume.memory_space<global> %k_buffer : buffer
%vg = buffer.assume.memory_space<global> %v_buffer : buffer
%og = buffer.assume.memory_space<global> %out_buffer : buffer
%qview = buffer.view %qg[%offset] : buffer -> view<[%queries]x[%qw]xbf16>
%kview = buffer.view %kg[%offset] : buffer -> view<[%keys]x[%kw]xbf16>
%vview = buffer.view %vg[%offset] : buffer -> view<[%kw]x[%vstride]xbf16>
%out = buffer.view %og[%offset] : buffer -> view<[%queries]x[%qw]xbf16>
%neg = scalar.constant -3.402823466e38 : f32
%zero = scalar.constant 0.0 : f32
%zv = vector.constant 0.0 : vector<8xf32>
%init = vector.fragment<init> %zv shape [%c16, %c16] : vector<8xf32>
%i16 = scalar.constant 16 : i32
%i32 = scalar.constant 32 : i32
%qlim = index.sub %qw, %c16 : index
%klim = index.sub %kw, %c16 : index
%vlim = index.sub %vstride, %c16 : index''')
    if lane_controls:
        emit('''%pack_lo_even = scalar.constant 84148480 : i32
%pack_hi_even = scalar.constant 117834498 : i32
%pack_lo_odd = scalar.constant 16778500 : i32
%pack_hi_odd = scalar.constant 50464518 : i32
%pack_lo = scf.select %even, %pack_lo_even, %pack_lo_odd : i32
%pack_hi = scf.select %even, %pack_hi_even, %pack_hi_odd : i32''')
    for i in range(8):
        emit(f'''%qcol{i}raw = index.add %qc, %c{i*16} : index
%qcol{i} = index.assume %qcol{i}raw [le(%qcol{i}raw, %qlim), multiple_of(%qcol{i}raw, 16)] : index
%qd{i} = vector.load %qview[%qread, %qcol{i}] : view<[%queries]x[%qw]xbf16> -> vector<16xbf16>
%qf{i} = vector.fragment<rhs> %qd{i} shape [%c16, %c16] : vector<16xbf16>''')
    emit(f'''%keyround = index.add %keys, %c{keys-1} : index
%keytiles = index.div %keyround, %c{keys} : index
%qend = index.add %qbase, %c{queries} : index
%causaltiles = index.div %qend, %c{keys} : index
%causalend = index.min %causaltiles, %keytiles : index
%end = scf.select %is_causal, %causalend, %keytiles : index''')
    if paired_accumulators:
        emit('%initpair = vector.constant 0.0 : vector<16xf32>')
        results = ', '.join(['%finalmax', '%finalsum'] + [f'%finalpair{i}' for i in range(4)])
        types = ', '.join(['f32', 'f32'] + ['vector<16xf32>']*4)
        iters = ', '.join(['%oldmax = %neg : f32', '%oldsum = %zero : f32'] + [f'%accpair{i} = %initpair : vector<16xf32>' for i in range(4)])
    else:
        results = ', '.join(['%finalmax', '%finalsum'] + [f'%final{i}' for i in range(8)])
        types = ', '.join(['f32', 'f32'] + ['vector<8xf32>']*8)
        iters = ', '.join(['%oldmax = %neg : f32', '%oldsum = %zero : f32'] + [f'%acc{i} = %init : vector<8xf32>' for i in range(8)])
    emit(f'{results} = scf.for %kt = [%c0 to %end step %c1]({iters}) -> ({types}) {{\n%kb = index.mul %kt, %c{keys} : index')
    for packet in range((keys*8+queries*2-1)//(queries*2)):
        # Each thread stages one contiguous 16-element packet per pass.
        off=packet*queries*2
        partial = off + queries*2 > keys*8
        if partial:
            emit(f'''%koff{packet} = index.constant {off} : index
%kid{packet}raw = index.add %tid, %koff{packet} : index
%packet_count{packet} = index.constant {keys*8} : index
%loader{packet} = index.cmp ult, %kid{packet}raw, %packet_count{packet} : index
scf.if %loader{packet} {{
%kid{packet} = index.assume %kid{packet}raw [range(%kid{packet}raw, 0, {keys*8-1})] : index''')
        else:
            emit(f'''%koff{packet} = index.constant {off} : index
%kid{packet} = index.add %tid, %koff{packet} : index''')
        emit(f'''%kr{packet} = index.div %kid{packet}, %c8 : index
%ktc{packet} = index.rem %kid{packet}, %c8 : index
%kcol{packet}raw = index.mul %ktc{packet}, %c16 : index
%kgcol{packet}raw = index.add %kc, %kcol{packet}raw : index
%kgcol{packet} = index.assume %kgcol{packet}raw [le(%kgcol{packet}raw, %klim), multiple_of(%kgcol{packet}raw, 16)] : index
%kgrow{packet} = index.add %kb, %kr{packet} : index
%kok{packet} = index.cmp ult, %kgrow{packet}, %keys : index
%ksafe{packet} = scf.select %kok{packet}, %kgrow{packet}, %c0 : index
%kread{packet} = index.assume %ksafe{packet} [lt(%ksafe{packet}, %keys)] : index
%kv{packet} = vector.load %kview[%kread{packet}, %kgcol{packet}] : view<[%keys]x[%kw]xbf16> -> vector<16xbf16>
vector.store %kv{packet}, %ks[%kr{packet}, %kcol{packet}raw] : vector<16xbf16>, view<{keys}x136xbf16>
%vr{packet} = index.div %kid{packet}, %c{keys//16} : index
%vtc{packet} = index.rem %kid{packet}, %c{keys//16} : index
%vcol{packet} = index.mul %vtc{packet}, %c16 : index
%vgcol{packet}raw = index.add %kb, %vcol{packet} : index
%vgcol{packet} = index.assume %vgcol{packet}raw [le(%vgcol{packet}raw, %vlim), multiple_of(%vgcol{packet}raw, 16)] : index
%vgrow{packet}raw = index.add %kc, %vr{packet} : index
%vgrow{packet} = index.assume %vgrow{packet}raw [lt(%vgrow{packet}raw, %kw)] : index
%vv{packet} = vector.load %vview[%vgrow{packet}, %vgcol{packet}] : view<[%kw]x[%vstride]xbf16> -> vector<16xbf16>
vector.store %vv{packet}, %vs[%vr{packet}, %vcol{packet}] : vector<16xbf16>, view<128x{keys+8}xbf16>''')
        if partial:
            emit('}')
    emit('kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)')
    for part in range(keys//16):
        emit(f'%krow{part} = index.add %lc, %c{part*16} : index')
        for dim in range(8):
            previous = '%init' if dim == 0 else f'%dot{part}_{dim-1}'
            emit(f'''%kd{part}_{dim} = vector.load %ks[%krow{part}, %c{dim*16}] : view<{keys}x136xbf16> -> vector<16xbf16>
%kf{part}_{dim} = vector.fragment<lhs> %kd{part}_{dim} shape [%c16, %c16] : vector<16xbf16>
%dot{part}_{dim} = vector.mma %kf{part}_{dim}, %qf{dim}, {previous} : vector<16xbf16>, vector<16xbf16>, vector<8xf32>''')
        emit(f'%scaled{part} = vector.mulf %dot{part}_7, %scalev : vector<8xf32>')
        for i in range(8):
            emit(f'''%mk{part}_{i}off = index.constant {part*16+i*2} : index
%mk{part}_{i}a = index.add %kb, %mk{part}_{i}off : index
%mk{part}_{i} = index.add %mk{part}_{i}a, %lg : index
%present{part}_{i} = index.cmp ult, %mk{part}_{i}, %keys : index
%past{part}_{i} = index.cmp ule, %mk{part}_{i}, %qrow : index
%causalok{part}_{i} = scf.select %is_causal, %past{part}_{i}, %present{part}_{i} : i1
%valid{part}_{i} = scalar.andi %present{part}_{i}, %causalok{part}_{i} : i1
%value{part}_{i} = vector.extract %scaled{part}[{i}] : vector<8xf32> -> f32
%masked{part}_{i} = scf.select %valid{part}_{i}, %value{part}_{i}, %neg : f32''')
        emit(f'%scores{part} = vector.from_elements ' + ', '.join(f'%masked{part}_{i}' for i in range(8)) + ' : vector<8xf32>')
        emit(f'%max{part} = vector.reduce<maxnumf> %scores{part}, ' + ('%neg' if part == 0 else f'%max{part-1}') + ' : vector<8xf32>, f32')
    emit(f'''%partnermax, %validmax = kernel.subgroup.shuffle<xor> %max{keys//16-1}, %i16, %i32 : f32, i32, i32
%tilemax = scalar.maxnumf %max{keys//16-1}, %partnermax : f32
%nextmax = scalar.maxnumf %oldmax, %tilemax : f32
%maxv = vector.splat %nextmax : vector<8xf32>
%old_delta = scalar.subf %oldmax, %nextmax : f32
%alpha = scalar.expf<afn> %old_delta : f32
%alphav = vector.splat %alpha : vector<8xf32>
%scaled_sum = scalar.mulf %oldsum, %alpha : f32''')
    def pack(value, name):
        emit(f'''%{name}half = vector.fptrunc {value} : vector<8xf32> to vector<8xbf16>
%{name}words = vector.bitcast %{name}half : vector<8xbf16> to vector<4xi32>
%{name}partner, %{name}valid = kernel.subgroup.shuffle<xor> %{name}words, %i16, %i32 : vector<4xi32>, i32, i32''')
        if lane_controls:
            emit(f'%{name}raw = low.invoke @clef_interleave_bf16(%{name}words, %{name}partner, %pack_lo, %pack_hi) : (vector<4xi32>, vector<4xi32>, i32, i32) -> (vector<16xbf16>)')
        else:
            emit(f'''%{name}even = scf.select %even, %{name}words, %{name}partner : vector<4xi32>
%{name}odd = scf.select %even, %{name}partner, %{name}words : vector<4xi32>
%{name}raw = low.invoke @clef_interleave_bf16(%{name}even, %{name}odd) : (vector<4xi32>, vector<4xi32>) -> (vector<16xbf16>)''')
    for part in range(keys//16):
        emit(f'''%delta{part} = vector.subf %scores{part}, %maxv : vector<8xf32>
%prob{part} = vector.expf<afn> %delta{part} : vector<8xf32>
%sum{part} = vector.reduce<addf> %prob{part}, ''' + ('%scaled_sum' if part == 0 else f'%sum{part-1}') + ' : vector<8xf32>, f32')
        pack(f'%prob{part}', f'p{part}')
        emit(f'%pf{part} = vector.fragment<rhs> %p{part}raw shape [%c16, %c16] : vector<16xbf16>')
    for dim in range(8):
        if paired_accumulators:
            emit(f'%vrow{dim} = index.add %lc, %c{dim*16} : index')
            previous = f'%accpair{dim//2}' if dim % 2 == 0 else f'%pv{dim-1}_{keys//16-1}'
            for part in range(keys//16):
                name = f'clef_pv_{dim%2}' + ('_scaled' if part == 0 else '')
                emit(f'''%vd{dim}_{part} = vector.load %vs[%vrow{dim}, %c{part*16}] : view<128x{keys+8}xbf16> -> vector<16xbf16>
%pv{dim}_{part} = low.invoke @{name}(%vd{dim}_{part}, %p{part}raw, {previous}, %alpha) : (vector<16xbf16>, vector<16xbf16>, vector<16xf32>, f32) -> (vector<16xf32>)''')
                previous = f'%pv{dim}_{part}'
            continue
        emit(f'''%vrow{dim} = index.add %lc, %c{dim*16} : index
%rescale{dim} = vector.mulf %acc{dim}, %alphav : vector<8xf32>''')
        for part in range(keys//16):
            previous=f'%rescale{dim}' if part==0 else f'%pv{dim}_{part-1}'
            emit(f'''%vd{dim}_{part} = vector.load %vs[%vrow{dim}, %c{part*16}] : view<128x{keys+8}xbf16> -> vector<16xbf16>
%vf{dim}_{part} = vector.fragment<lhs> %vd{dim}_{part} shape [%c16, %c16] : vector<16xbf16>
%pv{dim}_{part} = vector.mma %vf{dim}_{part}, %pf{part}, {previous} : vector<16xbf16>, vector<16xbf16>, vector<8xf32>''')
    emit('kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)')
    outputs = range(1, 8, 2) if paired_accumulators else range(8)
    emit('scf.yield ' + ', '.join(['%nextmax', f'%sum{keys//16-1}'] + [f'%pv{i}_{keys//16-1}' for i in outputs]) + f' : {types}\n}}')
    if paired_accumulators:
        for i in range(4):
            emit(f'%final{i*2}, %final{i*2+1} = low.invoke @clef_split_pair(%finalpair{i}) : (vector<16xf32>) -> (vector<8xf32>, vector<8xf32>)')
    emit('''%partnersum, %validsum = kernel.subgroup.shuffle<xor> %finalsum, %i16, %i32 : f32, i32, i32
%sum = scalar.addf %finalsum, %partnersum : f32
%sumv = vector.splat %sum : vector<8xf32>
%writes = scalar.andi %qvalid, %even : i1''')
    for i in range(8):
        emit(f'%output{i} = vector.divf %final{i}, %sumv : vector<8xf32>')
        pack(f'%output{i}', f'o{i}')
        emit(f'''scf.if %writes {{
 %orow{i} = index.assume %qrow [lt(%qrow, %queries)] : index
 vector.store %o{i}raw, %out[%orow{i}, %qcol{i}] : vector<16xbf16>, view<[%queries]x[%qw]xbf16>
}}''')
    emit('kernel.return\n}')
    source = '\n'.join(lines)+'\n'
    if keys == 128:
        # V capacity is rounded to 64, independently of this experimental tile.
        # Clamp packets outside the last allocated row to an existing packet;
        # their probabilities are masked to zero. K loads already clamp tails.
        for packet in range((keys*8+queries*2-1)//(queries*2)):
            original = f'%vgcol{packet} = index.assume %vgcol{packet}raw [le(%vgcol{packet}raw, %vlim), multiple_of(%vgcol{packet}raw, 16)] : index'
            bounded = f'%vgcol{packet}bounded'
            replacement = (
                f'{bounded} = index.min %vgcol{packet}raw, %vlim : index\n'
                f'%vgcol{packet} = index.assume {bounded} [le({bounded}, %vlim), multiple_of({bounded}, 16)] : index'
            )
            assert original in source
            source = source.replace(original, replacement)
    return swizzle_lds(source, keys) if swizzle else source


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=int, choices=(64,128,256), default=64)
    p.add_argument('--keys', type=int, choices=(32,64,128), default=32)
    p.add_argument('--swizzle', action='store_true', help='Replace LDS row padding by packet XOR addressing.')
    p.add_argument('--lane-controls', action='store_true')
    p.add_argument('--paired-accumulators', action='store_true')
    p.add_argument('output', type=Path)
    a=p.parse_args()
    a.output.write_text(generate(a.queries,a.keys,a.lane_controls,a.paired_accumulators,a.swizzle))
