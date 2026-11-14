// metal/src/qwen36/q36_expert_blob.h - the layout of ONE expert's weights, which moe_gu / moe_down read and which the streaming
// path fills.  The decision this file records: **a blob is a slot**.  An expert's three matrices live in one contiguous region
//   [ gate (n_ff rows of gu_row bytes) | up (n_ff rows) | down (n_embd rows of d_row bytes) ]
// so that moe_gu finds gate at +0 and up at +n_ff*gu_row from the blob's start, and moe_down finds the down at
// +2*n_ff*gu_row (moe.metal's MoeArgs: gate_off / up_off / down_off, gu_stride = d_stride = one blob).  Whatever fills a slot -
// a resident pack at load, or a streamed expert a token - writes exactly these bytes, in this order, and nothing else has to
// know how the file stores them.
//
// The file does NOT store them this way: its three stacked tensors are expert-major but separate, so expert e's slab is
// contiguous *within* each tensor at e * rows * row_bytes.  pack() is that gather, and it is the only place that knows both
// layouts.
#pragma once

#include "ref_quant.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace strata::qwen36 {

struct ExpertBlob {
    int n_embd = 2048;
    int n_ff = 512;
    int gu_type = 8;    // Q8_0
    int down_type = 8;
    uint64_t gu_row = 0;   // set by shape()
    uint64_t d_row = 0;
    uint64_t up_off = 0;
    uint64_t down_off = 0;
    uint64_t blob = 0;

    void shape() {
        gu_row = (uint64_t) ref::row_bytes(gu_type, n_embd);
        d_row = (uint64_t) ref::row_bytes(down_type, n_ff);
        up_off = (uint64_t) n_ff * gu_row;
        down_off = 2 * up_off;
        blob = down_off + (uint64_t) n_embd * d_row;
    }
    /// Bytes of a slot holding `n_expert` experts (the resident case: every expert packed once, slot e = e).
    uint64_t slots(int n_expert) const { return blob * (uint64_t) n_expert; }
};

/// Gathers expert `expert` out of the file's three stacked tensors into `dst` (one blob), from the mappings `gate`, `up` and
/// `down` at their tensors' payload bytes.  Returns false when a size does not fit what shape() says.
inline bool pack(const ExpertBlob& b, int expert, const uint8_t* gate, const uint8_t* up, const uint8_t* down,
                 uint8_t* dst, std::string& err) {
    if (gate == nullptr || up == nullptr || down == nullptr || dst == nullptr) {
        err = "a packed expert needs all four pointers";
        return false;
    }
    if (b.blob == 0 || b.gu_row == 0 || b.d_row == 0) {
        err = "the blob's shape was never computed";
        return false;
    }
    const uint64_t gu_slab = (uint64_t) b.n_ff * b.gu_row;     // expert e's gate occupies e * gu_slab .. within its tensor
    const uint64_t d_slab = (uint64_t) b.n_embd * b.d_row;
    const uint64_t e = (uint64_t) expert;
    std::memcpy(dst, gate + e * gu_slab, (size_t) gu_slab);
    std::memcpy(dst + b.up_off, up + e * gu_slab, (size_t) gu_slab);
    std::memcpy(dst + b.down_off, down + e * d_slab, (size_t) d_slab);
    return true;
}

}  // namespace strata::qwen36
