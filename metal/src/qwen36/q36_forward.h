// metal/src/qwen36/q36_forward.h - the model: one token through every layer, then the final norm and the logits.
//
//     x = embed(token)
//     for every layer (trunk, then the MTP block): layer_forward(x)     each layer adds its own mixer and MoE output
//     h = rms(x) * output_norm;  logits = W_output . h
//
// What makes this a model rather than a layer is the per-layer state: 30 GDN layers carry a state (h_v * 128 * 128 floats - 2 MiB,
// so 60 MiB for the trunk) and a conv history, and the 10 attention layers carry a KV ring each.  Those, plus the scratch, are
// what Forward::resident_bytes() counts before the first token.
//
// The experts are the one place this file has a policy rather than a shape: `slot_budget_bytes` says how much of the card the
// caller will give to the resident expert slots, and a model whose experts do not fit is REFUSED here with the number - because
// the alternative on this card is a 34 GiB allocation against 8 GiB, and the path that makes a real Qwen3.6 pack run is streaming
// one expert a token into these same slots, which is the next piece of work rather than something to fake.
#pragma once

#include "q36_attn_layer.h"
#include "q36_bind.h"
#include "q36_dense.h"
#include "q36_device.h"
#include "q36_gdn_layer.h"
#include "q36_layer.h"
#include "q36_model.h"
#include "q36_moe_routed.h"
#include "q36_moe_shared.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen36 {

struct Forward {
    int kv_cap = 0;
    std::vector<mtl::Buffer> state, hist;    // per GDN layer
    std::vector<mtl::Buffer> kring, vring;   // per attention layer
    std::vector<mtl::Buffer> slots;          // per layer: every expert's blob, RESIDENT (unused when streaming)
    ExpertStream stream;                     // or one pool of top_k slots, filled per layer (the path that fits this card)
    bool streaming = false;
    mtl::Buffer x, xh, logits, cs;

    AttnScratch attn;
    GdnScratch gdn;
    SharedScratch shared;
    RoutedScratch routed;
    DenseScratch dense;
    DenseKernels dk;
    LayerScratch layer;
    AttnKernels ak;
    GdnKernels gk;
    RoutedKernels rk;
    LayerKernels lk;

    uint64_t resident_bytes() const;
};

bool forward_init(mtl::Context& cx, const Model& m, const DeviceModel& d, int kv_cap, uint64_t slot_budget_bytes, Forward& f,
                  std::string& err);

/// One token at `pos` over a window of `n_kv` rows (every attention layer shares the position).  The logits are left in
/// `f.logits` (n_vocab floats).
/// A token's embedding row into `f.x` - the first step of forward_token, on its own so a caller can check it against the file's
/// own bytes (a real pack's row 3 of 248,320 has no fixture to vouch for it).
bool embed_token(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, uint32_t token, std::string& err);

bool forward_token(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, uint32_t token, int pos, int n_kv,
                   std::string& err);

}  // namespace strata::qwen36
