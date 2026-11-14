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

