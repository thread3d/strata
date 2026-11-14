// metal/src/qwen36/q36_layer.h - one Qwen3.6 decoder layer: the norms, the mixer, the norms, and the MoE block.
//
// q36_engine.cu's order, with q36_add_rms fusing each join (x += y, then h = rms(x) * w):
//
//     h = rms(x) * attn_norm            the first layer's x IS the embedding, so its first join takes no add
//     y = mixer(h)                      GDN (30 layers) or attention (10)
//     x += y;  h = rms(x) * post_attention_norm
//     y = moe(h)                        the gated shared expert, the routed experts, the combine
//     x += y                            a plain add: the norm that follows belongs to the NEXT layer
//
// Everything here is a part that has its own parity test; what this file adds is the order, the three buffers the parts do not
// own (the normed row, the shared gate, the MoE's output), and the two norm weights, which the device half uploads but no part's
// weight struct takes.
#pragma once

#include "q36_attn_layer.h"
#include "q36_dense.h"
#include "q36_gdn_layer.h"
#include "q36_moe_routed.h"
#include "q36_moe_shared.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

struct LayerKernels {
    int glue_lib = -1;
    mtl::Pipeline add_rms, add;
};

struct LayerWeights {
    bool attention = false;
    AttnWeights attn;             // attention layers
    GdnWeights gdn;               // GDN layers
    const mtl::Buffer* attn_norm = nullptr;   // attn_norm.weight
    size_t attn_norm_off = 0;
    const mtl::Buffer* post_norm = nullptr;   // post_attention_norm.weight
    size_t post_norm_off = 0;
    DenseMatrix gate_inp;         // ffn_gate_inp.weight (the router)
    DenseMatrix gate_inp_shexp;   // ffn_gate_inp_shexp.weight (the shared expert's gate, n_embd -> 1)
    SharedWeights shared;
};

/// The scratch the parts do not own, plus the parts' own.
struct LayerScratch {
    mtl::Buffer h;         // n_embd: the normed row the mixer and then the MoE read
    mtl::Buffer sg;        // 1: the shared expert's pre-sigmoid gate
    mtl::Buffer moe_out;   // n_embd: the combine's output
    int n_embd = 0;
};

struct LayerIo {
    AttnShape attn;
    GdnShape gdn;
    SharedShape shared;
    RoutedShape routed;
    AttnScratch* attn_scratch = nullptr;
    GdnScratch* gdn_scratch = nullptr;
    SharedScratch* shared_scratch = nullptr;
    RoutedScratch* routed_scratch = nullptr;
    AttnKernels* attn_kernels = nullptr;
    GdnKernels* gdn_kernels = nullptr;
    RoutedKernels* routed_kernels = nullptr;
    mtl::Buffer* state = nullptr;        // this layer's GDN state
    mtl::Buffer* hist = nullptr;         // this layer's conv history
    const mtl::Buffer* cs = nullptr;     // the RoPE table (attention layers)
    const mtl::Buffer* slots = nullptr;  // the packed expert slots (resident)
    size_t slots_off = 0;
    StreamFill fill{};                   // streaming: the model, layer and pool to fill instead (pool null = resident)
    int pos = 0, n_kv = 0;
    size_t layer_index = 0;   // which layer this is (for the dump hooks only)
};

bool layer_init(mtl::Context& cx, LayerKernels& k, std::string& err);
bool layer_scratch(mtl::Context& cx, LayerScratch& s, int n_embd, std::string& err);

/// One token through one layer.  `x` is the residual (fp32, n_embd) and is updated in place; `from_embedding` marks the first
/// layer, whose first join takes no add.  `s.h` is left holding the MoE's normed row and `s.moe_out` the block's output.
bool layer_forward(mtl::Context& cx, LayerKernels& lk, DenseKernels& dk, DenseScratch& ds, const LayerWeights& w,
                   LayerIo& io, LayerScratch& s, mtl::Buffer& x, bool from_embedding, std::string& err);

}  // namespace strata::qwen36
