// metal/src/qwen36/q36_moe_shared.cpp - the shared expert's two dispatches: gemv_pair (silu fused) and the down projection.
#include "q36_moe_shared.h"

#include "ref_quant.h"

#include <cstdio>

namespace strata::qwen36 {
namespace {

// which prepared plane a format reads (q36_dense.cpp's xkind, repeated here because the pair kernel takes the same planes)
int xkind(int t) {
    switch (t) {
    case 42: return 1;            // Q2_0
    case 12: case 13: return 2;   // Q4_K, Q5_K
    case 14: case 11: return 3;   // Q6_K, Q3_K
    default: return 0;
    }
}

}  // namespace

bool shared_forward(mtl::Context& cx, DenseKernels& dk, DenseScratch& ds, const SharedShape& sh, const SharedWeights& w,
                    SharedScratch& s, const mtl::Buffer& x, std::string& err) {
    if (s.h.ptr == nullptr || s.n_embd != sh.n_embd || s.n_ff != sh.n_ff) {
        s.h = cx.alloc(4 * (size_t) sh.n_ff);
        s.y = cx.alloc(4 * (size_t) sh.n_embd);
        s.g = cx.alloc(4 * (size_t) sh.n_ff);
        s.u = cx.alloc(4 * (size_t) sh.n_ff);
        if (s.h.ptr == nullptr || s.y.ptr == nullptr || s.g.ptr == nullptr || s.u.ptr == nullptr) {
            err = "the device refused the shared expert's scratch";
            return false;
        }
        s.n_embd = sh.n_embd;
        s.n_ff = sh.n_ff;
    }
    if (w.gate.buf == nullptr || w.up.buf == nullptr || w.down.buf == nullptr) {
        err = "the shared expert is missing a weight";
        return false;
    }
    // the two projections, then the SwiGLU, then the down.  Three dispatches rather than gemv_pair's one: that kernel is
    // instantiated only for the IQ/K-quant gates, and this port's pack keeps the shared expert in Q8_0 (see q36_glue.metal).
    if (!dense_gemv(cx, dk, w.gate, x, s.g, ds, err)) return false;
    if (!dense_gemv(cx, dk, w.up, x, s.u, ds, err)) return false;
    if (s.lib < 0) {
        try {
            s.lib = cx.compile({"../src/qwen36/msl/q36_glue.metal"}, "", false);
        } catch (const std::exception& e) {
            err = std::string("the q36 glue kernels did not compile: ") + e.what();
            return false;
        }
    }
    mtl::Pipeline sw;
    try {
        sw = cx.pipeline(s.lib, "q36_swiglu");
    } catch (const std::exception& e) {
        err = std::string("no q36_swiglu: ") + e.what();
        return false;
    }
    mtl::Dispatch d;
    d.p = &sw;
    d.args = {mtl::Arg::B(s.g), mtl::Arg::B(s.u), mtl::Arg::B(s.h), mtl::Arg::V((uint32_t) sh.n_ff)};
    d.tg[0] = 64;
    d.tg[1] = d.tg[2] = 1;
    d.grid[0] = (uint32_t) ((sh.n_ff + 63) / 64);
    d.grid[1] = d.grid[2] = 1;
    cx.run({d});
    return dense_gemv(cx, dk, w.down, s.h, s.y, ds, err);
}

}  // namespace strata::qwen36
