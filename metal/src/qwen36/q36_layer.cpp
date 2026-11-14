// metal/src/qwen36/q36_layer.cpp - one decoder layer: the order, and nothing new computed.
#include "q36_layer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::qwen36 {
namespace {

// x += y (when add), then h = rms(x) * norm_w  (q36_glue.metal's q36_add_rms)
bool join(mtl::Context& cx, LayerKernels& lk, mtl::Buffer& x, const mtl::Buffer* y, const mtl::Buffer& norm_w,
          size_t norm_off, mtl::Buffer& h, int n, float eps, bool add, std::string& err) {
    mtl::Dispatch d;
    d.p = &lk.add_rms;
    // the add's source is bound even when it is not added: the kernel reads it only when `add` is 1
    d.args = {mtl::Arg::B(x), y != nullptr ? mtl::Arg::B(*y) : mtl::Arg::B(x), mtl::Arg::B(norm_w, norm_off),
              mtl::Arg::B(h), mtl::Arg::V((uint32_t) n), mtl::Arg::V(add ? 1u : 0u), mtl::Arg::V(eps)};
    d.grid[0] = 1;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256;
    d.tg[1] = d.tg[2] = 1;
    cx.run({d});
    (void) err;
    return true;
}

}  // namespace

bool layer_init(mtl::Context& cx, LayerKernels& lk, std::string& err) {
    if (lk.glue_lib >= 0) return true;
    try {
        lk.glue_lib = cx.compile({"../src/qwen36/msl/q36_glue.metal"}, "", false);
        lk.add_rms = cx.pipeline(lk.glue_lib, "q36_add_rms");
        lk.add = cx.pipeline(lk.glue_lib, "q36_add");
    } catch (const std::exception& e) {
        err = std::string("the q36 glue kernels did not compile: ") + e.what();
        return false;
    }
    return true;
}

bool layer_scratch(mtl::Context& cx, LayerScratch& s, int n_embd, std::string& err) {
    if (s.h.ptr != nullptr && s.n_embd == n_embd) return true;
    s.h = cx.alloc(4 * (size_t) n_embd);
    s.sg = cx.alloc(4);
    s.moe_out = cx.alloc(4 * (size_t) n_embd);
    if (s.h.ptr == nullptr || s.sg.ptr == nullptr || s.moe_out.ptr == nullptr) {
        err = "the device refused the layer's scratch";
        return false;
    }
    s.n_embd = n_embd;
    return true;
}

bool layer_forward(mtl::Context& cx, LayerKernels& lk, DenseKernels& dk, DenseScratch& ds, const LayerWeights& w,
                   LayerIo& io, LayerScratch& s, mtl::Buffer& x, bool from_embedding, std::string& err) {
    if (!layer_init(cx, lk, err)) return false;
    const int n = w.attention ? io.attn.n_embd : io.gdn.n_embd;
    const float eps = w.attention ? io.attn.eps : io.gdn.eps;
    if (w.attn_norm == nullptr || w.post_norm == nullptr) {
        err = "a layer needs both of its norms";
        return false;
    }
    if (!layer_scratch(cx, s, n, err)) return false;

    // 1. h = rms(x) * attn_norm, with NO add - always.  A layer is a closed unit: its last step adds the MoE's output to x, so
    // the residual already holds everything up to this point and the next layer has nothing to add.  (This was wrong first time:
    // the join was given a null source and added x to itself on every layer after the first, which the second-layer case below
    // caught - the residual doubled a layer, rms went to inf and everything downstream was NaN.)
    (void) from_embedding;
    if (!join(cx, lk, x, nullptr, *w.attn_norm, w.attn_norm_off, s.h, n, eps, false, err)) return false;
    {   // Q36_DUMP_H2: the normed row this layer's mixer read, for Q36_DUMP_LAYER - the calibration input a host reference needs
        const char* lp = std::getenv("Q36_DUMP_LAYER");
        const size_t want = lp != nullptr ? (size_t) std::atoi(lp) : 0;
        const char* hp = std::getenv("Q36_DUMP_H2");
        if (hp != nullptr && io.layer_index == want) {
            if (FILE* fh = std::fopen(hp, "wb")) {
                std::fwrite(s.h.ptr, 4, (size_t) n, fh);
                std::fclose(fh);
            }
        }
    }   // BEFORE the mixer: attn_forward takes h by non-const reference and clobbers it

    // 2. the mixer, its output read from the part's own scratch
    if (w.attention) {
        if (!attn_forward(cx, *io.attn_kernels, dk, ds, io.attn, w.attn, *io.attn_scratch, s.h, *io.cs, io.pos, io.n_kv,
                          err))
            return false;
    } else {
        if (!gdn_forward(cx, *io.gdn_kernels, dk, ds, io.gdn, w.gdn, *io.gdn_scratch, *io.state, *io.hist, s.h, err,
                         io.layer_index))
            return false;
    }
    const mtl::Buffer& mix = w.attention ? io.attn_scratch->y : io.gdn_scratch->out;
    if (std::getenv("Q36_TRACE")) {
        auto scan = [](const mtl::Buffer& b, int n, const char* what) {
            const float* v = (const float*) b.ptr;
            size_t bad = 0;
            for (int i = 0; i < n; ++i) if (!std::isfinite(v[i])) ++bad;
            std::printf("Q36_TRACE   %s: %g %g %g, non-finite %zu of %d\n", what, (double) v[0], (double) v[1], (double) v[2], bad, n);
            std::fflush(stdout);
        };
        scan(s.h, n, "h after the first norm");
        scan(mix, n, "the mixer's output");
        scan(x, n, "x after the mixer join");
    }

    // 3. x += mixer; h = rms(x) * post_attention_norm
    if (!join(cx, lk, x, &mix, *w.post_norm, w.post_norm_off, s.h, n, eps, true, err)) return false;

    // 4. the MoE: the shared expert, its gate, then the routed experts and the combine
    if (!shared_forward(cx, dk, ds, io.shared, w.shared, *io.shared_scratch, s.h, err)) return false;
    if (!dense_gemv(cx, dk, w.gate_inp_shexp, s.h, s.sg, ds, err)) return false;   // n_embd -> 1
    RoutedWeights rw{io.slots, io.slots_off, nullptr, w.gate_inp};
    const StreamFill* fill = io.fill.pool != nullptr ? &io.fill : nullptr;
    if (!routed_forward(cx, *io.routed_kernels, dk, ds, io.routed, rw, *io.routed_scratch, s.h, io.shared_scratch->y, s.sg,
                        s.moe_out, err, fill))
        return false;

    // 5. x += the MoE's output (a plain add: the next layer's first join carries its own norm)
    mtl::Dispatch d;
    d.p = &lk.add;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(s.moe_out), mtl::Arg::V((uint32_t) n)};
    d.tg[0] = 64;
    d.tg[1] = d.tg[2] = 1;
    d.grid[0] = (uint32_t) ((n + 63) / 64);
    d.grid[1] = d.grid[2] = 1;
    cx.run({d});
    return true;
}

}  // namespace strata::qwen36
