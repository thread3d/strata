// metal/src/qwen36/q36_model.h - Qwen3.6-35B-A3B (GGUF architecture qwen35moe) on this Metal runtime: everything the
// engine can decide from the file's own header, before a single device buffer exists.
//
// The CUDA/HIP engine (src/qwen35/q36_engine.cu) does the same reading inline, spread through its load(); this file is
// that half on its own, because on this card the reading is not the hard part - the split between what stays in graphics
// memory and what streams from the SSD is, and that split is decided here, from the file's own numbers.
//
// What it does not do: no device allocation, no kernel, no upload.  `open()` is pure host work over an mmapped header, so
// it can run (and be tested) with no GPU at all.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen36 {

/// One tensor as the file holds it: its header entry, its shard, and (for reading) the pointer into that shard's mapping.
struct TensorRef {
    const strata::TensorInfo* t = nullptr;
    size_t shard = 0;

    bool ok() const { return t != nullptr; }
    uint64_t bytes() const { return t ? strata::tensor_payload_bytes(*t) : 0; }
    int type() const { return t ? (int) t->type : -1; }
    const char* name() const { return t ? t->name.c_str() : "(absent)"; }
    int64_t width() const { return (t && !t->shape.empty()) ? (int64_t) t->shape[0] : 0; }   ///< the row width (GGUF dim 0)
    int64_t rows() const { return (t && t->shape.size() > 1) ? (int64_t) t->shape[1] : 1; }
    const uint8_t* data(const strata::GgufModel& m) const { return t ? m.shard(shard).tensor_data(*t) : nullptr; }
    /// Bytes of one row at this tensor's own type: the unit an expert slab and a matrix-vector read are measured in.
    uint64_t row_bytes() const { return rows() > 0 ? bytes() / (uint64_t) rows() : 0; }
};

/// One layer of the trunk (or the in-file MTP block, which is a layer of the same shape plus four tensors).
struct Layer {
    strata::core::LayerKind kind = strata::core::LayerKind::Gdn;

    // every layer
    TensorRef gate_inp, gate_inp_shexp, gate_shexp, up_shexp, down_shexp;
    TensorRef gate_exps, up_exps, down_exps;
    TensorRef attn_norm, post_norm;

    // GDN layers (30 of 40 in the published file)
    TensorRef qkv, sgate, ssm_out, conv1d, alpha, beta, ssm_a, dt_bias, ssm_norm;

    // full-attention layers (10 of 40), and the MTP block
    TensorRef q, k, v, o, q_norm, k_norm;

    // the MTP block only
    TensorRef eh_proj, enorm, hnorm, shared_head_norm;

    uint64_t expert_bytes = 0;   ///< the three stacked expert tensors (the 90% that must stream on this card)
    uint64_t dense_bytes = 0;    ///< everything else in the layer (norms, the shared expert, the mixers, the router)
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    strata::core::ModelGeometry g;
    int64_t top_k = 0;               ///< experts per token, from expert_used_count
    float eps = 1e-6f;              ///< attention.layer_norm_rms_epsilon
    float rope_base = 10000000.0f; ///< rope.freq_base
    int64_t n_rot = 0;             ///< rope.dimension_count (the rotated slice of the head)

    TensorRef embed, output, output_norm;
    std::vector<Layer> layers;       ///< n_layers, then the n_mtp_layers blocks

    // the split this file implies, for a plan against a card's budget
    uint64_t embed_bytes = 0;
    uint64_t dense_bytes = 0;        ///< every layer's dense_bytes, plus output and output_norm
    uint64_t expert_bytes = 0;       ///< every layer's expert_bytes
    uint64_t total_bytes = 0;        ///< the tensors above (not the files' padding)
    uint64_t gu_row = 0;             ///< row bytes of the stacked expert gate/up tensors
    uint64_t down_row = 0;           ///< row bytes of the stacked expert down tensors

    int64_t n_trunk() const { return g.n_layers; }
    int64_t n_mtp() const { return g.n_mtp_layers; }

    /// Bytes one expert's [gate | up | down] slab takes: the unit a per-token stream reads, top_k times a layer.  The expert tensors
    /// are STACKED (`[n_embd, n_ff, n_expert]`), so their `row_bytes()` is the width of all 256 experts - the row of ONE expert is
    /// the format's row for `n_embd` (gate/up) and for `n_ff` (down), and this is that arithmetic, the same as ExpertBlob::shape().
    uint64_t expert_slab_bytes() const;
    /// Bytes a decoded token must read from the expert tensors alone (n_layers x top_k slabs): the number that decides
    /// whether streaming can be fast enough on this card.
    uint64_t expert_bytes_per_token() const { return expert_slab_bytes() * (uint64_t) top_k * (uint64_t) g.n_layers; }
};

/// Opens `path` (any shard of a split model), reads its header, resolves every tensor the layout table names, and
/// refuses what this port's kernels cannot take.  No device is touched.  False with `err` set: the file's own words.
bool open(const std::string& path, Model& m, std::string& err);

/// The model as text: the geometry, the dense/stream split, the kernel kinds each weight class resolves to, and the
/// per-layer kinds.  This is the plan a run is started from.
std::string describe(const Model& m);

/// Whether this port has a kernel of that class for a weight of `type` (a GGUF ggml_type id).  `Expert` asks about the
/// decode window's grouped expert kernels (moe.metal), `Dense` about the dense matvec (gemv.metal).  The lists are the
/// types those two files instantiate; a type outside them is refused at open rather than at the first dispatch.
enum class Role { Dense, ExpertGateUp, ExpertDown };
bool type_supported(Role role, int type);

}  // namespace strata::qwen36
