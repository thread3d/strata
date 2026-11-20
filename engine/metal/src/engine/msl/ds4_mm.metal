// metal/src/engine/msl/ds4_mm.metal - the DeepSeek prompt chunk's expert GEMMs: moe_mm.metal's mm_gu3 / mm_down3 (matmul2d on dequantized
// weight tiles) over the engine's own expert slots and its UNPADDED sort order.  Compiled after quant/units/mma/moe_mm.metal.
//   tiles[i] = (slot, first row, valid rows): a tile is TM rows of one expert; rows past `valid` (the next expert's) are computed and
//   dropped (the operand rows exist: the scratch carries TM rows of slack), the swiglu is DeepSeek's clamped one.

struct DsMmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong stride;    // bytes between slots
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
    float limit;
    uint pad0;
};

// xs[r][c] = half(x[idx[r]][c]) for r < rows
kernel void ds4_mm_gather(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]], device const int* idx [[buffer(2)]],
                          constant uint& n_embd [[buffer(3)]], device half* xs [[buffer(4)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256u + tid;
    if (c >= n_embd) return;
    xs[(ulong) tgp.y * n_embd + c] = half(x[(ulong) idx[tgp.y] * ldx + c]);
}

// The two expert GEMMs are MetalPerformancePrimitives' matmul2d where this Metal has it; below is the plain form of the same
// tiles for a Metal that does not (an Intel Mac's macOS 11).
#if STRATA_HAVE_MPP
template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_gu(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]], device half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = W + ulong(e) * a.stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            W8<F>::run(rgw, k, tab, wg, wg2);
            W8<F>::run(ruw, k, tab, wu, wu2);
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, rg, accg);
        op.run(tA, ru, accu);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = fmin(accg[i], a.limit), u = fmin(fmax(accu[i], -a.limit), a.limit);
            accg[i] = g / (1.0f + exp(-g)) * u;
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_dn(device const uchar* W [[buffer(0)]], device half* YEH [[buffer(1)]], device const half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles || a.pad0 == 3u) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = W + ulong(e) * a.stride + a.down_off + ulong(n0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    auto rw = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rw)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        const uint16_t half_cap = rw.get_capacity() / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rw.get_multidimensional_index(i);
            half4 w, w2;
            if (a.pad0 == 1u) { w = half4(0); w2 = half4(0); } else if (a.pad0 == 2u) { w = half4(1); w2 = half4(1); W8<F>::run(w0, 0, tab, w, w2); w = half4(0); } else
            W8<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab, w, w2);
            rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
            const uint16_t j = i + half_cap;
            rw[j] = w2.x; rw[j + 1] = w2.y; rw[j + 2] = w2.z; rw[j + 3] = w2.w;
        }
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, rw, acc);
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    acc.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) ((device float*) YEH)[ulong(row0 + m) * a.n_embd + n0 + f] = ot[sg][it];
    }
}


#else
// ds4_mm_gu without MetalPerformancePrimitives: the same tiles - each simdgroup owns TN weight columns (n0) and the tile's TM
// rows - decoded into threadgroup memory a TK chunk at a time (stage_w, the same helper moe_mm.metal's kernels use) and
// accumulated as plain loops over the simdgroup's 32 lanes: accg[m][f] = sum_k XS[row0 + m][k] * Wg[n0 + f][k].  The epilogue,
// the clamped swiglu and the store to HS[f][m] are the tensor version's.
template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_gu(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]], device half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg_ [[simdgroup_index_in_threadgroup]],
                      ushort lane_ [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles) return;
    const ushort sg = sg32(tid, sg_), lane = lane32(tid, lane_);   // the tile's own 32-lane group (the runtime's prologue)
    constexpr int NTH = SG * 32;
    constexpr int NOUT = TM * TN / 32;                  // outputs per thread inside its simdgroup
    threadgroup half wg[SG][TN * TK];
    threadgroup half wu[SG][TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = W + ulong(e) * a.stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    float ag[NOUT], au[NOUT];
    int mi[NOUT], fi[NOUT];
    for (int q = 0; q < NOUT; ++q) {
        const int it = int(lane) + q * 32;
        mi[q] = it / TN;
        fi[q] = it % TN;
        ag[q] = au[q] = 0.0f;
    }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        // W8<F> gives 8 values of one row: 4 at k .. k + 3 and 4 at k + 16 .. k + 19 (its cooperative tensor's layout), so a
        // step of 16 covers the whole TK chunk; the tile is stored in k order for the loops below.
        for (int kk = 0; kk < TK; kk += 16) {
            for (int f = lane; f < TN; f += 32) {
                half4 g1, g2, u1, u2;
                W8<F>::run(g0 + ulong(f) * a.gu_row, k0 + kk, tab, g1, g2);
                W8<F>::run(u0 + ulong(f) * a.gu_row, k0 + kk, tab, u1, u2);
                wg[sg][f * TK + kk + 0] = g1.x; wg[sg][f * TK + kk + 1] = g1.y;
                wg[sg][f * TK + kk + 2] = g1.z; wg[sg][f * TK + kk + 3] = g1.w;
                wu[sg][f * TK + kk + 0] = u1.x; wu[sg][f * TK + kk + 1] = u1.y;
                wu[sg][f * TK + kk + 2] = u1.z; wu[sg][f * TK + kk + 3] = u1.w;
                if (kk + 16 < TK) {
                    wg[sg][f * TK + kk + 16] = g2.x; wg[sg][f * TK + kk + 17] = g2.y;
                    wg[sg][f * TK + kk + 18] = g2.z; wg[sg][f * TK + kk + 19] = g2.w;
                    wu[sg][f * TK + kk + 16] = u2.x; wu[sg][f * TK + kk + 17] = u2.y;
                    wu[sg][f * TK + kk + 18] = u2.z; wu[sg][f * TK + kk + 19] = u2.w;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int kc = min(TK, int(a.n_embd) - k0);
        for (int k = 0; k < kc; ++k)
            for (int q = 0; q < NOUT; ++q) {
                const half xv = XS[ulong(row0 + mi[q]) * ulong(a.n_embd) + ulong(k0 + k)];
                ag[q] += float(xv) * float(wg[sg][fi[q] * TK + k]);
                au[q] += float(xv) * float(wu[sg][fi[q] * TK + k]);
            }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int q = 0; q < NOUT; ++q) {
        if (mi[q] >= valid) continue;
        const float g = fmin(ag[q], a.limit), u = fmin(fmax(au[q], -a.limit), a.limit);
        HS[ulong(row0 + mi[q]) * ulong(a.n_ff) + ulong(n0 + fi[q])] = half(g / (1.0f + exp(-g)) * u);
    }
}

// ds4_mm_dn the same way: each simdgroup owns TN down-projection columns at n0, K = n_ff over HS, and the result goes to YEH
// as fp32 (a row of n_embd per token).
template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_dn(device const uchar* W [[buffer(0)]], device half* YEH [[buffer(1)]], device const half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg_ [[simdgroup_index_in_threadgroup]],
                      ushort lane_ [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles || a.pad0 == 3u) return;
    const ushort sg = sg32(tid, sg_), lane = lane32(tid, lane_);   // the tile's own 32-lane group (the runtime's prologue)
    constexpr int NTH = SG * 32;
    constexpr int NOUT = TM * TN / 32;
    threadgroup half wd[SG][TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = W + ulong(e) * a.stride + a.down_off + ulong(n0) * a.d_row;
    float acc[NOUT];
    int mi[NOUT], fi[NOUT];
    for (int q = 0; q < NOUT; ++q) {
        const int it = int(lane) + q * 32;
        mi[q] = it / TN;
        fi[q] = it % TN;
        acc[q] = 0.0f;
    }
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        if (a.pad0 == 1u || a.pad0 == 2u) {   // the bench probes: a tile of zeros, or of ones
            const half v = a.pad0 == 2u ? half(1.0h) : half(0.0h);
            for (int it = int(lane); it < TN * TK; it += 32) wd[sg][it] = v;
        } else {
            for (int kk = 0; kk < TK; kk += 16) {
                for (int f = lane; f < TN; f += 32) {
                    half4 w1, w2;
                    W8<F>::run(w0 + ulong(f) * a.d_row, k0 + kk, tab, w1, w2);
                    wd[sg][f * TK + kk + 0] = w1.x; wd[sg][f * TK + kk + 1] = w1.y;
                    wd[sg][f * TK + kk + 2] = w1.z; wd[sg][f * TK + kk + 3] = w1.w;
                    if (kk + 16 < TK) {
                        wd[sg][f * TK + kk + 16] = w2.x; wd[sg][f * TK + kk + 17] = w2.y;
                        wd[sg][f * TK + kk + 18] = w2.z; wd[sg][f * TK + kk + 19] = w2.w;
                    }
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int kc = min(TK, int(a.n_ff) - k0);
        for (int k = 0; k < kc; ++k)
            for (int q = 0; q < NOUT; ++q) {
                const half hv = HS[ulong(row0 + mi[q]) * ulong(a.n_ff) + ulong(k0 + k)];
                acc[q] += float(hv) * float(wd[sg][fi[q] * TK + k]);
            }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int q = 0; q < NOUT; ++q)
        if (mi[q] < valid) ((device float*) YEH)[ulong(row0 + mi[q]) * ulong(a.n_embd) + ulong(n0 + fi[q])] = acc[q];
}

#endif   // (STRATA_HAVE_MPP: the tensor pair, or the plain pair above)

#define DS4_MM_INST(F, T)                                                                                                          template [[host_name("ds4_mm_gu_" #T)]] kernel decltype(ds4_mm_gu<F, 32, 32, 32, 4>) ds4_mm_gu<F, 32, 32, 32, 4>;               template [[host_name("ds4_mm_dn_" #T)]] kernel decltype(ds4_mm_dn<F, 32, 32, 32, 4>) ds4_mm_dn<F, 32, 32, 32, 4>;
DS4_MM_INST(T_IQ1_M, 29)
DS4_MM_INST(T_IQ1_S, 19)
DS4_MM_INST(T_IQ2_XXS, 16)
DS4_MM_INST(T_IQ2_XS, 17)
DS4_MM_INST(T_IQ2_S, 22)
DS4_MM_INST(T_IQ3_XXS, 18)
