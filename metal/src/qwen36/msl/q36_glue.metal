// metal/src/qwen36/msl/q36_glue.metal - the residual stream of a Qwen3.6 layer: the one step of it the port had no kernel
// for.
//
// Flash-Next's residual is a hyper-connection mix (hc_pre_norm, hc_mix_split, gr_*), and its norms live inside qsa_prep and
// the gr kernels, so nothing in metal/kernels/ is the plain thing qwen35moe does between a mixer and the router:
//
//     x += y                       (the mixer's output, or the MoE's)
//     h  = x / sqrt(mean(x^2) + eps) * w
//
// q36_engine.cu fuses exactly these two into k_add_rms (and the CUDA engine calls it twice a layer, with the first layer's
// first norm the same call without the add).  One kernel here, one threadgroup a row, so the reduction is a threadgroup
// reduction and not a grid one: the whole row is in one threadgroup's hands and there is nothing to merge.
//
// Scalars are separate arguments rather than a struct: this runtime binds each mtl::Arg to its own buffer index (mtl_rt.h
// Arg::V), so a struct would only ever see the first of them (see q36_attn.metal, where that cost a round).
//
//   x   fp32 [T][n]   the residual, updated in place when `add` is 1
//   y   fp32 [T][n]   the row being added (bound but unread when `add` is 0)
//   w   fp32 [n]      the norm's weight
//   h   fp32 [T][n]   the normed row: what the mixer or the router reads next
// The threadgroup size is this constant and not the [[threads_per_threadgroup]] built-in: a shader's grid attributes must be
// all scalar or all the same vector shape, and threadgroup_position_in_grid is a uint3 (the size is passed in the dispatch,
// d.tg[0], and the host launches exactly Q36_NTH).
constant uint Q36_NTH = 256;   // thread_index_in_threadgroup is what the port's other kernels take: a scalar beside
                                // the uint group index, which Metal accepts (thread_position_in_threadgroup is the 3-vector form)
constant uint Q36_NG = Q36_NTH / 32;   // 32-lane groups in the threadgroup

kernel void q36_add_rms(device float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                        device const float* w [[buffer(2)]], device float* h [[buffer(3)]],
                        constant uint& n [[buffer(4)]], constant uint& add [[buffer(5)]], constant float& eps [[buffer(6)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg_ [[simdgroup_index_in_threadgroup]], ushort lane_ [[thread_index_in_simdgroup]]) {
    const uint nth = Q36_NTH;
    device float* xr = x + (size_t) tg.x * n;
    if (add != 0) {
        device const float* yr = y + (size_t) tg.x * n;
        for (uint i = tid; i < n; i += nth) xr[i] += yr[i];
    }
    // mean of squares over the whole row: each lane's share, the group's sum (sum32: 32 lanes on this card's 64-wide
    // simdgroup too), then the threadgroup's groups.
    float ss = 0.0f;
    for (uint i = tid; i < n; i += nth) ss = fma(xr[i], xr[i], ss);
    ss = sum32(ss);
    const ushort sg = sg32(tid, sg_);
    const ushort lane = lane32(tid, lane_);
    threadgroup float part[Q36_NG];
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < Q36_NG; ++g) tot += part[g];
    const float inv = rsqrt(tot / (float) n + eps);
    device float* hr = h + (size_t) tg.x * n;
    for (uint i = tid; i < n; i += nth) hr[i] = xr[i] * inv * w[i];
}


// The shared expert's SwiGLU: h = silu(gate) * up, elementwise.  gemv.metal's gemv_pair fuses this into one dispatch, but it is
// only instantiated for the IQ/K-quant gates (GP_ROW lists IQ3_S, IQ4_XS, Q2_0, Q3_K..Q6_K) - not Q8_0, which is the format
// the Qwen3.6 Q8_0 pack keeps its shared expert in - so the pair would be three dispatches here anyway and there is no reason
// to make the shared expert depend on which types that kernel happens to cover.
kernel void q36_swiglu(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                       device float* h [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float g = gate[gid];
    h[gid] = g / (1.0f + exp(-g)) * up[gid];
}

// The plain residual add - x += y, no norm - which a layer needs twice without a following norm: the MoE's output goes back into
// the residual, and the norm that follows belongs to the NEXT layer (whose q36_add_rms fuses it with its own mixer).  Written
// out rather than folded into q36_add_rms because folding it meant computing a norm nobody reads.
kernel void q36_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    x[gid] += y[gid];
}

// A token's embedding row: x = dequant(token_embd row t).  The table is [n_embd, n_vocab] in the file's own order, so a row is
// n_embd contiguous values at t * row_bytes, and the format is the pack's (Q8_0 here: an fp16 scale a 32 values).  Written as
// its own kernel because the alternative - a one-hot multiply through gemv - would produce n_vocab values to throw away.
//
// Supported formats: Q8_0 (34 bytes per 32 values: fp16 scale + 32 int8), Q6_K (210 bytes per 256 values: 6-bit codes).
kernel void q36_embed(device const uchar* table [[buffer(0)]], device float* x [[buffer(1)]],
                      constant uint& n_embd [[buffer(2)]], constant uint& token [[buffer(3)]],
                      constant uint& etype [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n_embd) return;
    if (etype == 8) {   // Q8_0
        device const uchar* row = table + (size_t) token * ((size_t) n_embd / 32u) * 34u;
        device const uchar* blk = row + (size_t) (gid / 32u) * 34u;
        const float d = float(as_type<half>(ushort(blk[0] | (blk[1] << 8))));
        x[gid] = d * float((int) (signed char) blk[2 + (gid & 31u)]);
    } else if (etype == 14) {   // Q6_K: 210 bytes per 256 values; matches dequantize_q6_K
        const size_t row_bytes = ((size_t) n_embd / 256u) * 210u;
        device const uchar* row = table + (size_t) token * row_bytes;
        int bid = (int) (gid / 256);          // super-block index within row
        int off = (int) (gid % 256);          // offset within super-block (0-255)
        device const uchar* b = row + (size_t) bid * 210u;
        int hf = off / 128;                 // 0 = first 128, 1 = second 128
        int pos_in_half = off % 128;          // 0-127
        int l = pos_in_half % 32;             // 0-31
        int group = pos_in_half / 32;         // 0-3 → q1,q2,q3,q4
        int is = l / 16;                      // 0 or 1
        device const uchar* ql = b + (size_t) hf * 64u;   // ql advances by 64 per half
        device const uchar* qh = b + 128u + (size_t) hf * 32u; // qh advances by 32 per half
        uchar ql_byte = (group == 1 || group == 3) ? ql[l + 32] : ql[l];
        uint8_t nibble = (group == 0 || group == 2) ? (ql_byte & 0x0Fu) : (ql_byte >> 4);
        int qh_shift = group * 2;
        int qh_bits = ((int)qh[l] >> qh_shift) & 3;
        int code = (int)(nibble | (uint8_t)(qh_bits << 4)) - 32;
        device const signed char* scs = (device const signed char*)(b + 192);
        int sc_index = hf * 8 + is + group * 2;
        const float d = float(as_type<half>(ushort(b[208] | (b[209] << 8))));
        x[gid] = d * (float)scs[sc_index] * (float)code;
    } else if (etype == 12) {   // Q4_K: 144 bytes per 256 values; d*scale*nibble - dmin*min_scale
        const size_t row_bytes = ((size_t) n_embd / 256u) * 144u;
        device const uchar* row = table + (size_t) token * row_bytes;
        int bid = (int) (gid / 256);          // super-block index within row
        int off = (int) (gid % 256);          // offset within super-block (0-255)
        device const uchar* b = row + (size_t) bid * 144u;
        const float d  = float(as_type<half>(ushort(b[0] | (b[1] << 8))));
        const float mn = float(as_type<half>(ushort(b[2] | (b[3] << 8))));
        device const uchar* scales = b + 4;   // 12 bytes of packed 6-bit scale/min
        device const uchar* qs = b + 16;      // 128 bytes of 4-bit quantized values
        int j4 = off / 64;                    // which 64-value group (0-3)
        int pos = off % 64;                   // offset within group (0-63)
        int li = pos / 32;                    // 0 = low nibbles, 1 = high nibbles
        int lj = pos % 32;                    // position within 32 (0-31)
        int is = j4 * 2 + li;                 // scale index (0-7)
        uchar sc, mm;
        if (is < 4) {
            sc = scales[is] & 63;
            mm = scales[is + 4] & 63;
        } else {
            sc = (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
            mm = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
        }
        const float d1 = d * (float)sc;
        const float m1 = mn * (float)mm;
        device const uchar* qb = qs + (size_t) j4 * 32u + (size_t) lj;
        uint8_t nibble = (li == 0) ? (*qb & 0x0Fu) : (*qb >> 4);
        x[gid] = d1 * (float)nibble - m1;
    }
}

// Debug: copy first n values of src to dst, to verify GPU can read src.
kernel void q36_debug_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    dst[gid] = src[gid];
}
