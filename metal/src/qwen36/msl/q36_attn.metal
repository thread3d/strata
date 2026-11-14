// metal/src/qwen36/msl/q36_attn.metal - Qwen3.6-35B-A3B's full-attention layer, decode (one query token a head).
//
// The family's attention is the plain kind: 16 query heads over 2 KV heads, width 256, no indexer and no compressor, so
// there is nothing to select and nothing to merge - every query reads every row of its window.  That is why this file is
// small next to qsa.metal, which is Flash-Next's indexed attention over a pooled cell graph; only qsa's idea of an online
// softmax is shared, and it is written here directly rather than through a split-K partial.
//
// One 32-lane group a head: the group's lane l holds query components l, l + 32, l + 64, ... (QD / 32 of them), a row's dot
// product is that group's sum, and the softmax state (m, l) is replicated across the group because the sum is.  The
// reduction is sum32 from the runtime prologue, so the kernel is right on a 64-lane AMD simdgroup as well as on a 32-lane
// Apple one (mtl_rt.mm's STRATA_SIMD_WIDTH).
//
//   q      fp32 [QH][QD]   the query after its qk-norm and RoPE (the host's business, not this kernel's)
//   kb     bf16 [n_kv][KVH][QD]  the window's keys, token-major, oldest row first
//   vb     bf16 [n_kv][KVH][QD]  the same rows' values
//   gate   fp32 [QH][QD] or nothing: attention's output gate, sigmoid(gate) * out (qwen35moe's attn_q second half)
//   out    fp32 [QH][QD]
//
// The KV ring itself is the host's: this kernel is handed the window in logical order, oldest first, which is what
// qsa.metal's callers also do (ring_to_ext).

// A bf16 half pair read as fp32 without needing a bfloat type (macOS 11's Metal has none): the top 16 bits of the float.
static inline float q36_bf16(uint16_t h) { return as_type<float>((uint) h << 16); }

// The scalars are separate arguments rather than one struct: a dispatch binds each mtl::Arg to its own buffer index
// (mtl_rt.h Arg::V), so three values passed for a struct would land in buffers 5, 6 and 7 with only the first inside the
// struct - which reads scale as 0, and a 0 scale is invisible at one row (a uniform softmax is a one-row softmax) and
// wrong at every other length.
template <int QH, int KVH, int QD>
kernel void q36_attn(device const float* q [[buffer(0)]], device const uint16_t* kb [[buffer(1)]],
                     device const uint16_t* vb [[buffer(2)]], device const float* gate [[buffer(3)]],
                     device float* out [[buffer(4)]], constant uint& n_kv [[buffer(5)]],
                     constant uint& has_gate [[buffer(6)]], constant float& scale [[buffer(7)]],
                     ushort tid [[thread_index_in_threadgroup]], ushort sg_ [[simdgroup_index_in_threadgroup]],
                     ushort lane_ [[thread_index_in_simdgroup]]) {
    const ushort h = sg32(tid, sg_);   // one 32-lane group a query head (the runtime's prologue)
    if (h >= QH) return;
    const ushort lane = lane32(tid, lane_);
    constexpr int LPR = QD / 32;   // query components a lane holds
    const int kvh = (int) h * KVH / QH;   // the group a query head reads: qwen35moe's 16 over 2 is 8 queries a KV head

    device const float* qh = q + (size_t) h * QD;
    float qv[LPR];
    for (int i = 0; i < LPR; ++i) qv[i] = qh[lane + 32 * i];

    float acc[LPR];
    for (int i = 0; i < LPR; ++i) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    if (n_kv > 0) {
        // TOKEN-MAJOR rows: row r of the window holds every KV head's 256 values side by side, ((r * KVH) + kvh) * QD.
        // That is the layout an append wants - one q36_head_norm_rope_bf16 / q36_store_bf16 dispatch writes a whole
        // token's KV heads contiguously at the row the token belongs to, because those kernels write head h at
        // h * QD from the base they are handed.  The head-major alternative costs n_kv_head dispatches a token, and a
        // dispatch is ~0.7 ms on this card whatever it computes (NEXT.md rounds 92-97).
        for (uint r = 0; r < n_kv; ++r) {
            device const uint16_t* krow = kb + ((size_t) r * KVH + kvh) * QD;
            float dot = 0.0f;
            for (int i = 0; i < LPR; ++i) dot = fma(qv[i], q36_bf16(krow[lane + 32 * i]), dot);
            const float s = sum32(dot) * scale;   // the group's dot product, then the 1/sqrt(QD)
            // online softmax: the running max, and the rescale of both the sum and the accumulator
            const float mn = max(m, s);
            const float corr = exp(m - mn);
            const float p = exp(s - mn);
            l = l * corr + p;
            m = mn;
            device const uint16_t* vrow = vb + ((size_t) r * KVH + kvh) * QD;
            for (int i = 0; i < LPR; ++i) acc[i] = fma(p, q36_bf16(vrow[lane + 32 * i]), acc[i] * corr);
        }
    }

    const float inv = n_kv > 0 ? 1.0f / l : 0.0f;
    device float* oh = out + (size_t) h * QD;
    for (int i = 0; i < LPR; ++i) {
        const int j = lane + 32 * i;
        float o = acc[i] * inv;
        // the gate is the second QD of this head's row in the q projection: (h * 2 + 1) * QD
        if (has_gate) o *= 1.0f / (1.0f + exp(-gate[(size_t) h * 2 * QD + QD + j]));
        oh[j] = o;
    }
}

// Qwen3.6's values need no norm and no rotation - they go to the ring as they come from their projection, in bf16.  This is
// that store: the same round-to-nearest-even on the top half the other bf16 writers use, one thread a value.
kernel void q36_store_bf16(device const float* src [[buffer(0)]], device uint16_t* dst [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint u = as_type<uint>(src[gid]);
    u += 0x7fffu + ((u >> 16) & 1u);
    dst[gid] = uint16_t(u >> 16);
}

// The shapes the port compiles: the family's own (16 query heads over 2 KV, width 256) and the small ones the parity
// program drives it at, which are what makes a lane-mapping bug visible before a model is loaded.
#define Q36_ATTN(QH, KVH, QD) template [[host_name("q36_attn_" #QH "_" #KVH "_" #QD)]] kernel decltype(q36_attn<QH, KVH, QD>) q36_attn<QH, KVH, QD>;
Q36_ATTN(16, 2, 256)
Q36_ATTN(4, 2, 32)
Q36_ATTN(8, 4, 64)
Q36_ATTN(2, 1, 32)
Q36_ATTN(2, 1, 256)   // the small parity model's head count at the family's width (q36_bind_check, q36_forward_check)
