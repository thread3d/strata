// metal/kernels/dense_mm.metal - dense projections for a PROMPT chunk (after quant/units/mma.metal):
//   act_f16    X[t][k] -> half (the matrix units' operand)
//   dq_f16<F>  a quantized matrix -> half rows (ref_quant's values rounded to fp16), once per chunk
//   gemm_f16   Y[t][n] = Xh[t] . Wh[n] (fp32) by MetalPerformancePrimitives matmul2d, TM x TN tiles, K dynamic;
//              rows of Y past n_out / t past n_tok are not written (the tensors carry the extents)
// The GEMM below is written on MetalPerformancePrimitives (Metal 4's matmul2d, the M5's neural accelerators).  A Metal
// without it - an Intel Mac's macOS 11 - is told so by the runtime (mtl_rt.mm, probe_mpp), and takes the plain tiled GEMM at
// the bottom of this file instead: same name, same buffers, same argument, so no host code changes.
#if STRATA_HAVE_MPP
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif

kernel void act_f16(device const float* x [[buffer(0)]], device half* xh [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid < n) xh[gid] = half(x[gid]);
}

struct DqArgs {
    uint n_in;       // values per row
    uint n_out;      // rows
    uint row_bytes;
    uint perm;       // 0: row r -> r; p > 0: row r = c * (n_out / p) + d -> d * p + c (the HC up matrix with the p
                     // streams of an output interleaved, for gemm_hc_up's epilogue)
};

// one thread per unit: grid (units per row / 64, n_out), 64 threads
template <int F>
kernel void dq_f16(device const uchar* W [[buffer(0)]], device half* Wh [[buffer(1)]], constant DqArgs& a [[buffer(2)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, H = UNIT / 2;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u = int(tg.x) * 64 + tid, r = int(tg.y);
    if (u * UNIT >= int(a.n_in)) return;
    // the unit's chunk origin: a 128-wide chunk holds whole units of every format
    const int k0 = (u * UNIT) / 128 * 128;
    float w[UNIT];
    int p0, p1;
    DW<F>::run(W + ulong(r) * a.row_bytes, u, k0, w, p0, p1, tab);
    const int q = a.perm ? int(a.n_out / a.perm) : 1;
    const int ro = a.perm ? (r % q) * int(a.perm) + r / q : r;
    device half* o = Wh + ulong(ro) * a.n_in + k0;
    for (int j = 0; j < H; ++j) {
        o[p0 + j] = half(w[j]);
        o[p1 + j] = half(w[H + j]);
    }
}

struct GemmArgs { int M, N, K; int pad; };   // Y is M rows of N floats

#if STRATA_HAVE_MPP
template <int TM, int TN, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mY);
}
#else
// The same GEMM on a Metal without MetalPerformancePrimitives: a TM x TN tile of Y per threadgroup, K walked in steps of KC
// with both operand tiles staged in threadgroup memory.  The 128 threads the host launches are laid out 8 across the rows
// and 16 across the columns, so every thread accumulates (TM/8) x (TN/16) outputs; rows and columns past M and N are staged
// as zero and not written back (the tensors carry the extents, as with the tensor API).  It is the arithmetic of the
// tensor API's GEMM - Y[t][n] = sum_k A[t][k] * B[n][k], half operands, float accumulator - not its speed.
template <int TM, int TN, int SG>
kernel void gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* Y [[buffer(2)]],
                     constant GemmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NT = 128;          // threads the host launches for this kernel
    constexpr int KC = 16;           // K values staged per step
    constexpr int TR = NT / 16;      // threads across the rows (8)
    constexpr int TC = 16;           // threads across the columns (16)
    constexpr int AR = TM / TR;      // outputs per thread down a column
    constexpr int AC = TN / TC;      // ... and across a row
    threadgroup half sA[TM * KC];
    threadgroup half sB[TN * KC];
    const int t0 = int(tg.y) * TM, n0 = int(tg.x) * TN;
    const int r = int(tid) % TR, c = int(tid) / TR;
    float acc[AR][AC];
    for (int i = 0; i < AR; ++i)
        for (int j = 0; j < AC; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < a.K; k0 += KC) {
        for (int i = int(tid); i < TM * KC; i += NT) {
            const int rr = i / KC, cc = i % KC, t = t0 + rr, k = k0 + cc;
            sA[i] = (t < a.M && k < a.K) ? A[ulong(t) * ulong(a.K) + ulong(k)] : half(0.0h);
        }
        for (int i = int(tid); i < TN * KC; i += NT) {
            const int rr = i / KC, cc = i % KC, n = n0 + rr, k = k0 + cc;
            sB[i] = (n < a.N && k < a.K) ? B[ulong(n) * ulong(a.K) + ulong(k)] : half(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int kc = min(KC, a.K - k0);
        for (int k = 0; k < kc; ++k) {
            half av[AR];
            for (int i = 0; i < AR; ++i) av[i] = sA[(r + i * TR) * KC + k];
            for (int j = 0; j < AC; ++j) {
                const half bv = sB[(c + j * TC) * KC + k];
                for (int i = 0; i < AR; ++i) acc[i][j] += float(av[i]) * float(bv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < AR; ++i) {
        const int t = t0 + r + i * TR;
        if (t >= a.M) break;
        for (int j = 0; j < AC; ++j) {
            const int n = n0 + c + j * TC;
            if (n < a.N) Y[ulong(t) * ulong(a.N) + ulong(n)] = acc[i][j];
        }
    }
}
#endif

#define DQ(F) template [[host_name("dq_f16_" #F)]] kernel decltype(dq_f16<F>) dq_f16<F>;
#define DQ_(F) DQ(F)
DQ_(T_F32) DQ_(T_BF16) DQ_(T_Q8_0) DQ_(T_Q4_0) DQ_(T_Q5_0) DQ_(T_IQ4_NL) DQ_(T_IQ4_XS) DQ_(T_Q2_0) DQ_(T_IQ3_S) DQ_(T_Q3_K)
DQ_(T_Q4_K) DQ_(T_Q5_K) DQ_(T_Q6_K)
template [[host_name("gemm_f16_64_64_4")]] kernel decltype(gemm_f16<64, 64, 4>) gemm_f16<64, 64, 4>;
template [[host_name("gemm_f16_128_64_4")]] kernel decltype(gemm_f16<128, 64, 4>) gemm_f16<128, 64, 4>;
template [[host_name("gemm_f16_64_128_4")]] kernel decltype(gemm_f16<64, 128, 4>) gemm_f16<64, 128, 4>;
template [[host_name("gemm_f16_64_32_4")]] kernel decltype(gemm_f16<64, 32, 4>) gemm_f16<64, 32, 4>;
template [[host_name("gemm_f16_128_128_4")]] kernel decltype(gemm_f16<128, 128, 4>) gemm_f16<128, 128, 4>;
template [[host_name("gemm_f16_128_128_8")]] kernel decltype(gemm_f16<128, 128, 8>) gemm_f16<128, 128, 8>;
template [[host_name("gemm_f16_128_64_8")]] kernel decltype(gemm_f16<128, 64, 8>) gemm_f16<128, 64, 8>;
template [[host_name("gemm_f16_64_64_8")]] kernel decltype(gemm_f16<64, 64, 8>) gemm_f16<64, 64, 8>;
template [[host_name("gemm_f16_256_64_8")]] kernel decltype(gemm_f16<256, 64, 8>) gemm_f16<256, 64, 8>;
template [[host_name("gemm_f16_64_256_8")]] kernel decltype(gemm_f16<64, 256, 8>) gemm_f16<64, 256, 8>;
template [[host_name("gemm_f16_128_32_4")]] kernel decltype(gemm_f16<128, 32, 4>) gemm_f16<128, 32, 4>;
template [[host_name("gemm_f16_32_128_4")]] kernel decltype(gemm_f16<32, 128, 4>) gemm_f16<32, 128, 4>;
