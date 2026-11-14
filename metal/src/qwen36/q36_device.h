// metal/src/qwen36/q36_device.h - the Qwen3.6 engine's device side: what stays in graphics memory, and what does not.
//
// The number that decides this port is 37.8 GB of file against 8 GB of card.  q36_model.h already separated the three
// kinds of bytes (embedding, dense, experts) from the file's own header; this is the half that acts on it: every dense
// tensor - the norms, the four mixer projections of the GDN layers, the six of the attention layers, the router, the gated
// shared expert, the nextn block - is uploaded once and stays, because every token reads it.  The routed expert tensors
// are deliberately NOT uploaded: on this card they cannot be, and how they stream is the next question rather than this
// one.  A 37.8 GB Q8_0 file is ~2 GB dense and ~34 GB experts (measured on a synthetic file of the family's shape by
// q36_model_check's describe()), so the split is not a tuning choice.
//
// Uploads are byte for byte: a Q8_0 matrix goes to the device as the blocks the file holds, which is what the port's
// gemv / moe kernels read, so nothing is repacked and nothing can drift.
#pragma once

#include "mtl_rt.h"
#include "q36_model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen36 {

/// One layer's dense tensors on the device.  The GDN fields are empty on the attention layers and the other way round,
/// so a buffer that is not there is a buffer the layer does not read.
struct DeviceLayer {
    strata::core::LayerKind kind = strata::core::LayerKind::Gdn;

    mtl::Buffer gate_inp, gate_inp_shexp, gate_shexp, up_shexp, down_shexp;
    mtl::Buffer attn_norm, post_norm;

    mtl::Buffer qkv, sgate, ssm_out, conv1d, alpha, beta, ssm_a, dt_bias, ssm_norm;   // GDN
    mtl::Buffer q, k, v, o, q_norm, k_norm;                                          // attention
    mtl::Buffer eh_proj, enorm, hnorm, shared_head_norm;                            // MTP

    uint64_t bytes = 0;
};

struct DeviceModel {
    mtl::Buffer embed, output, output_norm;
    std::vector<DeviceLayer> layers;
    uint64_t dense_bytes = 0;   ///< everything uploaded, embedding included
};

/// Uploads every dense tensor of `m`.  The expert tensors are left in the file.  False with `err` set when a tensor the
/// plan named is absent from the mapping or the device refuses the allocation.
bool upload_dense(mtl::Context& cx, const Model& m, DeviceModel& d, std::string& err);

/// Frees what `upload_dense` allocated (the expert tensors were never allocated) and empties `d`.
void release(mtl::Context& cx, DeviceModel& d);

}  // namespace strata::qwen36
