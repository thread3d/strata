// metal/src/qwen36/q36_bind.h - the bridge between what the file declares and what the parts read.
//
// Every part takes a DenseMatrix (buffer, offset, type, n_in, n_out) or a buffer+offset; the model knows a tensor's shard, its
// GGUF type and its shape.  Getting one of those wrong - n_in and n_out swapped, a transposed matrix read the wrong way, a 1-D
// tensor bound where a matrix was meant - produces a wrong answer with no crash and no error message, and the layer tests cannot
// see it because they build their own matrices.
//
// So this is its own function, checked against the layout table rather than against a formula (q36_bind_check): for every dense
// weight the table names, the binding's type is the tensor's type and its widths are the tensor's own, in the order the kernels
// multiply in (a GGUF matrix is [n_in, n_out]: dim 0 is the row width, which is what a gemv's K is).
#pragma once

#include "q36_attn_layer.h"
#include "q36_device.h"
#include "q36_gdn_layer.h"
#include "q36_layer.h"
#include "q36_model.h"
#include "q36_moe_shared.h"

#include <string>

namespace strata::qwen36 {

/// A DenseMatrix pointing at a device buffer.  The device half uploads each tensor into its own buffer, so the offset is 0 - but
/// it is carried, because nothing here may assume a tensor starts at a buffer's beginning once slots exist.
inline DenseMatrix bind(const TensorRef& r, const mtl::Buffer& b) {
    DenseMatrix m;
    if (r.ok() && b.ptr != nullptr) {
        m.buf = &b;
        m.off = 0;
        m.type = r.type();
        m.n_in = (int) ((r.t->shape.size() > 0) ? r.t->shape[0] : 0);
        m.n_out = (int) ((r.t->shape.size() > 1) ? r.t->shape[1] : 1);
    }
    return m;
}

/// Fills `w` for layer `i` of `m`, reading the buffers `d`.  Returns false with `err` when a tensor the layer needs is absent from
/// either side - which is the difference between "the file is not what the plan said" and a dispatch that reads zeros.
bool bind_layer(const Model& m, const DeviceModel& d, size_t i, LayerWeights& w, std::string& err);

}  // namespace strata::qwen36
