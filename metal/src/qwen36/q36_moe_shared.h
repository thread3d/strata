// metal/src/qwen36/q36_moe_shared.h - the shared expert of a Qwen3.6 MoE block.
//
//     h  = silu(W_gate . x) * (W_up . x)        [n_ff]      gemv_pair, one dispatch, the SwiGLU fused
//     y  = W_down . h                           [n_embd]
//
// and the gate it is scaled by - sigmoid(W_gate_inp_shexp . x) - is applied by the combine (`moe_combine_sh`), not here, so this
// file is the two projections and nothing else.  The pair kernel is already parity-tested against double (pair_parity.cpp);
// what this adds is the wiring, from the engine's own call in metal/tests/token_chain.cpp:564-580: the pipeline is named for
// the two types, the two weights may have different formats (each reads its own prepared plane), and rb_up is the up matrix's
// own row stride.
#pragma once

#include "q36_dense.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

struct SharedShape {
    int n_embd = 2048;
    int n_ff = 512;
};

struct SharedWeights {
    DenseMatrix gate, up, down;   // ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp
};

struct SharedScratch {
    mtl::Buffer h;    // n_ff, silu(gate) * up - raw, as the down projection reads it
    mtl::Buffer g, u; // the two projections before the SwiGLU
    mtl::Buffer y;    // n_embd
    int n_embd = 0, n_ff = 0;
    int lib = -1;   // the q36 glue kernels (q36_glue.metal), compiled once
};

/// y = W_down . (silu(W_gate . x) * (W_up . x)); `x` is the normed residual row (fp32, n_embd) and the dense scratch is the
/// caller's, so the prepared activation this needs is the same one every other projection of the token uses.
bool shared_forward(mtl::Context& cx, DenseKernels& dk, DenseScratch& ds, const SharedShape& sh, const SharedWeights& w,
                    SharedScratch& s, const mtl::Buffer& x, std::string& err);

}  // namespace strata::qwen36
