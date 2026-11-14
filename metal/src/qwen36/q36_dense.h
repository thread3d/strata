// metal/src/qwen36/q36_dense.h - one dense matrix-vector through this port's kernels, in one place.
//
// Every projection of a Qwen3.6 layer is the same call: a Q8_0 (or K-quant / IQ / F32 / F16 / BF16) matrix against one
// token's activation, through the prepared-activation form the port's gemv tiles read (units.metal's xprep writes four
// kinds of the same row plus a 16-wide sum table, and each format reads the one it needs).  The calling convention is
// fiddly - the tile name, the row stride, the plane offsets into the prepared buffer, the grid's rows per threadgroup -
// and getting one of them wrong costs a silent wrong answer, so it lives here once, tested against a CPU dequantisation of
// the same bytes (q36_dense_parity) rather than open-coded per caller.
#pragma once

#include "mtl_rt.h"

#include <cstdint>
#include <string>

namespace strata::qwen36 {

/// Where a dense weight is and what it is.  `data` may be a device buffer (resident dense weights) or the file's mapping
/// (an expert slab read straight from the mapping, or a weight streamed in) - a gemv only needs a binding.
struct DenseMatrix {
    const mtl::Buffer* buf = nullptr;   // the buffer the weight lives in
    size_t off = 0;                     // bytes from that buffer's start
    int type = 0;                       // the GGUF ggml_type id
    int n_in = 0;                       // the row width
    int n_out = 0;                      // rows
};

/// The per-token scratch the prepared activations need: 4 planes of n floats plus the 16-wide sums.
struct DenseScratch {
    mtl::Buffer xp;
    int n_in = 0;
};

/// The kernels, resolved once (the library is compiled once per process).
struct DenseKernels {
    int lib = -1;
    mtl::Pipeline prep;
};

/// Compiles the two kernel files this port's dense path needs and resolves xprep.  Idempotent; false with `err` when the
/// device refuses (a Metal 2 card missing a feature, say).
bool dense_init(mtl::Context& cx, DenseKernels& k, std::string& err);

/// Allocates the scratch for a width (once per model; every projection of a token shares it).
bool dense_scratch(mtl::Context& cx, DenseScratch& s, int n_in, std::string& err);

/// Fills `s` with the prepared forms of `x` (the four kinds and the 16-wide sums) for a width of `n_in`.  A caller that runs a
/// kernel of its own on the prepared activation (gemv_pair for the shared expert) calls this once, then binds `s.xp`.
bool dense_prep(mtl::Context& cx, DenseKernels& k, const mtl::Buffer& x, DenseScratch& s, int n_in, std::string& err);

/// y[0..n_out) = W . x, one token.  `x` holds n_in fp32 values.  The pipeline for the weight's format and shape is resolved
/// on each call (Metal caches compiled functions; the host map is the only cost) and refused by name when the kernel does
/// not exist - which is the failure a caller can act on, unlike a silently empty dispatch.
bool dense_gemv(mtl::Context& cx, DenseKernels& k, const DenseMatrix& W, const mtl::Buffer& x, mtl::Buffer& y,
                DenseScratch& s, std::string& err);

}  // namespace strata::qwen36
