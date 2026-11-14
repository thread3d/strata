// metal/src/qwen36/q36_moe_routed.h - the routed half of a Qwen3.6 MoE block: the router, the plan, the grouped experts and the
// combine.
//
//     logits = W_router . x                      [n_expert], written at a 512-float row stride (moe_route's window)
//     route  = moe_route                          top-k ids and weights, and the plan (moe_route writes it itself)
//     ye     = moe_gu / moe_down                  the selected experts' gate|up|down, over the plan's groups
//     bo     = moe_combine_sh                     sum_k ye[plan[t][k]] + sigmoid(sg) * ysh
//
// The experts are read from **blobs that are slots** (q36_expert_blob.h): expert e at blobs + e * blob, gate at +0, up at
// up_off, down at down_off.  A resident model packs every expert once; a streamed one fills a slot per token - the kernel cannot
// tell the difference, which is the point of that layout.
//
// The shared expert (q36_moe_shared.h) and the shared gate `sg` are the caller's: the combine is what joins them, so this file
// takes `ysh` and `sg` already computed.
#pragma once

#include "q36_dense.h"
#include "q36_expert_blob.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

struct RoutedShape {
    ExpertBlob blob;      // n_embd, n_ff, the two types, and the offsets they imply
    int n_expert = 256;
    int top_k = 8;
    int n_tok = 1;
};

struct RoutedWeights {
    const mtl::Buffer* blobs = nullptr;   // n_expert slots of blob.blob bytes
    size_t blobs_off = 0;
    DenseMatrix router;                   // ffn_gate_inp.weight [n_embd, n_expert]
};

struct RoutedScratch {
    mtl::Buffer logits, ids, wts, plan, h, ye;
    int n_embd = 0, n_ff = 0, n_expert = 0, top_k = 0, n_tok = 0;
};

struct RoutedKernels {
    int lib = -1;
    mtl::Pipeline route, gu, dn, combine;
};

bool routed_init(mtl::Context& cx, RoutedKernels& k, const RoutedShape& sh, std::string& err);
bool routed_scratch(mtl::Context& cx, RoutedScratch& s, const RoutedShape& sh, std::string& err);

/// One token through the routed path.  `x` is the normed residual row (fp32, n_embd); `ysh` and `sg` are the shared expert's
/// output and its pre-sigmoid gate (both optional: pass a zeroed buffer and 0 to see the routed sum alone).  The result is left
/// in `bo` (fp32, n_embd).  The plan buffer is scratch and is rewritten on every call.
bool routed_forward(mtl::Context& cx, RoutedKernels& k, DenseKernels& dk, DenseScratch& ds, const RoutedShape& sh,
                    const RoutedWeights& w, RoutedScratch& s, const mtl::Buffer& x, const mtl::Buffer& ysh,
                    const mtl::Buffer& sg, mtl::Buffer& bo, std::string& err);

}  // namespace strata::qwen36
