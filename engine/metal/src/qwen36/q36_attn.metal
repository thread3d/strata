// metal/src/qwen36/msl/q36_attn.metal - dense causal attention for Qwen3.6-35B-A3B on Metal:
// 16 query heads over 2 KV heads, head dim 256, no indexer, online softmax over KV cells.
// One threadgroup per query head, 256 threads = head_dim.  Each thread handles one output
// dimension j.  KV ring is token-major: KV[(t * n_kv_head + kvh) * head_dim + j].
// Bit-identical to q36_attn on the CUDA side.
#include <metal_stdlib>
using namespace metal;

inline __attribute__((always_inline)) float silu(float v) { return v / (1.0f + expf(-v)); }
inline __attribute__((always_inline)) float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }

kernel void q36_attn(
    device const float* q [[buffer(0)]],      // [n_head, head_dim] per token (queries)
    device const float* gate_q [[buffer(1)]], // [n_head, head_dim] per token (output gate)
    device const float* k_ring [[buffer(2)]], // token-major KV ring: [(t * n_kv + kvh) * head_dim + j]
    device const float* v_ring [[buffer(3)]],
    device float* o [[buffer(4)]],            // [n_head, head_dim] per token
    constant int& n_head [[buffer(5)]],
    constant int& n_kv [[buffer(6)]],         // number of tokens in the ring
    constant int& n_kv_head [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& chunk_cells [[buffer(9)]],  // unused (kept for ABI compatibility)
    constant float& scale [[buffer(10)]],     // 1/sqrt(head_dim)
    constant uint& is_decode [[buffer(11)]],  // 1: one query; 0: batched prompt
    uint32_t tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
    const int head = int(tg_pos.x);
    const int tok = int(tg_pos.y);   // prompt token index (batch dim); decode: tok == 0
    const int lane = int(tid);       // output dimension j this thread owns (0..head_dim-1)

    const int n_head_i = int(n_head);
    const int n_kv_i = int(n_kv);
    const int n_kvh_i = int(n_kv_head);
    const int hd_i = int(head_dim);
    const float scale_f = float(scale);

    device const float* qh = q + (size_t) tok * n_head_i * hd_i + (size_t) head * hd_i;
    device const float* gh = gate_q + (size_t) tok * n_head_i * hd_i + (size_t) head * hd_i;
    device float* oh = o + (size_t) tok * n_head_i * hd_i + (size_t) head * hd_i;

    const int kvh = head % n_kvh_i;
    const int j = lane;

    // Online softmax state (per output dimension j, shared across all threads via reduction)
    float max_v = -INFINITY;
    float lse = 0.0f;
    float acc = 0.0f;

    // Threadgroup memory for dot-product reduction (256 floats, one per thread)
    threadgroup float tg_score[256];

    for (int c = 0; c < n_kv_i; ++c) {
        bool causal_ok = is_decode ? true : (c <= tok);
        if (!causal_ok) continue;

        device const float* kv = k_ring + (size_t) c * n_kvh_i * hd_i + (size_t) kvh * hd_i;
        device const float* vv = v_ring + (size_t) c * n_kvh_i * hd_i + (size_t) kvh * hd_i;

        // Partial dot product: q[j] * k[c][j]
        tg_score[lane] = qh[j] * kv[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce sum across all 256 threads
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) tg_score[lane] += tg_score[lane + offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float score = tg_score[0] * scale_f;  // all threads read the full dot product

        // Online softmax update (all threads compute identical values)
        float old_max = max_v;
        max_v = fmaxf(max_v, score);
        lse = lse * expf(old_max - max_v) + expf(score - max_v);
        acc = acc * expf(old_max - max_v) + expf(score - max_v) * vv[j];
    }

    // Finalize: o[j] = acc / lse, then apply output gate sigmoid(gate_q[j])
    if (lse > 0.0f) {
        float ov = acc / lse;
        ov = ov * sigm(gh[j]);
        oh[j] = ov;
    } else {
        oh[j] = 0.0f;
    }
}
