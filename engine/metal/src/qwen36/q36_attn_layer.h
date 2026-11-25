// metal/src/qwen36/q36_attn_layer.h - one Qwen3.6 attention layer, wired: three projections, the head norm and RoPE, the
// attention kernel, and the output projection.
//
// This is the first place the port puts measured parts together.  Its arithmetic is q36_engine.cu's, in its order:
//
//     qg = W_q . h                      [2 * n_head * head_dim]: the queries, then the output gate
//     k  = W_k . h,  v = W_v . h       [n_kv_head * head_dim] each
//     q  = rope(rms(q_head) * q_norm)  per head            -> q36_head_norm_rope_f32
//     k  = rope(rms(k_head) * k_norm)  per KV head         -> q36_head_norm_rope_bf16, into the ring
//     v  -> bf16                                           -> q36_store_bf16, into the ring
//     a  = attn(q, k[...], v[...])     sigmoid(gate) folded in -> q36_attn
//     y  = W_o . a
//
// The KV ring is TOKEN-MAJOR - row r holds every KV head's head_dim values side by side, (r * n_kv_head + kvh) * head_dim -
// which is what lets one dispatch write a whole token's KV heads (q36_attn.metal's comment: the head-major alternative costs
// a dispatch a head, and a dispatch is ~0.7 ms on this card whatever it computes).
#pragma once

#include "q36_dense.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

struct AttnShape {
    int n_embd = 2048;
    int n_head = 16;
    int n_kv_head = 2;
    int head_dim = 256;
    float eps = 1e-6f;   // the layer norm's epsilon, from the model file (never a literal)
};

struct AttnWeights {
    DenseMatrix q, k, v, o;
    const mtl::Buffer* q_norm = nullptr;   // [head_dim]
    size_t q_norm_off = 0;
    const mtl::Buffer* k_norm = nullptr;   // [head_dim]
    size_t k_norm_off = 0;
};

// The scratch: sized once for the model's shape and the run's KV window, then reused by every token and layer.
struct AttnScratch {
    mtl::Buffer qg, k_raw, v_raw, q, kring, vring, a, y;
    int n_embd = 0, n_head = 0, n_kv_head = 0, head_dim = 0, cap = 0;
};

struct AttnKernels {
    int lib = -1;
    mtl::Pipeline rope_f32, rope_bf16, store, attn;
};

bool attn_init(mtl::Context& cx, AttnKernels& k, const AttnShape& sh, std::string& err);
bool attn_scratch(mtl::Context& cx, AttnScratch& s, const AttnShape& sh, int kv_cap, std::string& err);

// One token at position `pos` over `n_kv` rows of the window (rows 0 .. n_kv-1, oldest first); this call appends the token's
// own row at `n_kv - 1` before attending.  `h` is the normed residual row (fp32, n_embd); `cs` the engine's RoPE table
// (float2 [pos][head_dim/2]).  The layer's output is left in `s.y` (fp32, n_embd).  The dense scratch is the caller's, so a
// token's projections all share one prepared activation.
bool attn_forward(mtl::Context& cx, AttnKernels& k, DenseKernels& dk, DenseScratch& ds, const AttnShape& sh,
                  const AttnWeights& w, AttnScratch& s, const mtl::Buffer& h, const mtl::Buffer& cs, int pos, int n_kv,
                  std::string& err);

}  // namespace strata::qwen36
