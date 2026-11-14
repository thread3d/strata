// metal/src/qwen36/q36_gdn_layer.h - one Qwen3.6 GDN layer (30 of the 40), wired: four projections, the step kernel, and the
// output projection.
//
//     qkv   = W_qkv  . h     [channels = (2 h_k + h_v) * 128] = 8192 for this family
//     z     = W_gate . h     [h_v * 128] = 4096
//     alpha = W_alpha. h     [h_v]      RAW: gdn_step_t applies softplus(alpha + dt) * ssm_a itself (gdn.metal:65-69)
//     beta  = W_beta . h     [h_v]      RAW for the same reason
//     y     = gdn_step_t(state, hist, qkv, alpha, beta, z, conv_w, dt, ssm_a, gamma)   [h_v * 128]
//     out   = W_out . y
//
// The state and the conv history are the caller's: they live across tokens and, on this card, across layers - one state block
// and one history block per layer, carried by the engine (a per-layer `state` is 32 x 128 x 128 floats = 2 MiB, so 30 GDN
// layers are 60 MiB of the card's budget and belong in the same accounting as the dense weights).
#pragma once

#include "q36_dense.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

struct GdnShape {
    int n_embd = 2048;
    int h_k = 16;      // ssm.group_count
    int h_v = 32;      // ssm.time_step_rank
    int state = 128;   // ssm.state_size
    float eps = 1e-6f;
    int channels() const { return (2 * h_k + h_v) * state; }
    int value_dim() const { return h_v * state; }
};

struct GdnWeights {
    DenseMatrix qkv, gate, alpha, beta, out;
    const mtl::Buffer* conv_w = nullptr;   // ssm_conv1d.weight [channels][4]
    size_t conv_w_off = 0;
    const mtl::Buffer* dt = nullptr;       // ssm_dt.bias [h_v]
    size_t dt_off = 0;
    const mtl::Buffer* ssm_a = nullptr;    // ssm_a [h_v]
    size_t ssm_a_off = 0;
    const mtl::Buffer* gamma = nullptr;    // ssm_norm.weight [state]
    size_t gamma_off = 0;
};

/// The per-token scratch (not the state: that is the caller's, per layer).
struct GdnScratch {
    mtl::Buffer qkv, z, alpha, beta, y, out, hist_out;   // hist_out: the kernel's write side (never the buffer it reads)
    int n_embd = 0, h_k = 0, h_v = 0, state = 0;
};

struct GdnKernels {
    int lib = -1;
    mtl::Pipeline step;
};

bool gdn_init(mtl::Context& cx, GdnKernels& k, std::string& err);
bool gdn_scratch(mtl::Context& cx, GdnScratch& s, const GdnShape& sh, std::string& err);

/// One token through the layer.  `h` is the normed residual row (fp32, n_embd); `state` is this layer's block
/// (h_v * state * state floats) and `hist` its conv history (channels * 3 floats), both updated in place.  The layer's output
/// is left in `s.out` (fp32, n_embd).
bool gdn_forward(mtl::Context& cx, GdnKernels& k, DenseKernels& dk, DenseScratch& ds, const GdnShape& sh,
                 const GdnWeights& w, GdnScratch& s, mtl::Buffer& state, mtl::Buffer& hist, const mtl::Buffer& h,
                 std::string& err);

/// The state and history a layer needs: allocate both for a run (the caller keeps one pair a layer).
bool gdn_state(mtl::Context& cx, const GdnShape& sh, mtl::Buffer& state, mtl::Buffer& hist, std::string& err);

}  // namespace strata::qwen36
