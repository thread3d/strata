// metal/src/qwen36/q36_gdn_layer.cpp - the GDN layer's wiring.
#include "q36_gdn_layer.h"

#include <cstdio>
#include <cstring>

namespace strata::qwen36 {

bool gdn_init(mtl::Context& cx, GdnKernels& k, std::string& err) {
    if (k.lib < 0) {
        try {
            k.lib = cx.compile({"quant.metal", "units.metal", "gdn.metal"}, "", false);
        } catch (const std::exception& e) {
            err = std::string("the GDN kernels did not compile: ") + e.what();
            return false;
        }
    }
    if (k.step.handle == nullptr) {
        try {
            // function constant 2 is the token count the specialization is cut for (gdn_parity's convention), 1 here
            k.step = cx.pipeline(k.lib, "gdn_step", mtl::Constants().i(2, 1));
        } catch (const std::exception& e) {
            err = std::string("no gdn_step kernel: ") + e.what();
            return false;
        }
    }
    return true;
}

bool gdn_state(mtl::Context& cx, const GdnShape& sh, mtl::Buffer& state, mtl::Buffer& hist, std::string& err) {
    state = cx.alloc(4 * (size_t) sh.h_v * sh.state * sh.state);
    hist = cx.alloc(4 * (size_t) sh.channels() * 3);
    if (state.ptr == nullptr || hist.ptr == nullptr) {
        err = "the device refused the GDN state";
        return false;
    }
    std::memset(state.ptr, 0, 4 * (size_t) sh.h_v * sh.state * sh.state);
    std::memset(hist.ptr, 0, 4 * (size_t) sh.channels() * 3);
    return true;
}

bool gdn_scratch(mtl::Context& cx, GdnScratch& s, const GdnShape& sh, std::string& err) {
    if (s.qkv.ptr != nullptr && s.n_embd == sh.n_embd && s.h_k == sh.h_k && s.h_v == sh.h_v && s.state == sh.state) return true;
    s.qkv = cx.alloc(4 * (size_t) sh.channels());
    s.z = cx.alloc(4 * (size_t) sh.value_dim());
    s.alpha = cx.alloc(4 * (size_t) sh.h_v);
    s.beta = cx.alloc(4 * (size_t) sh.h_v);
    s.y = cx.alloc(4 * (size_t) sh.value_dim());
    s.out = cx.alloc(4 * (size_t) sh.n_embd);
    s.hist_out = cx.alloc(4 * (size_t) sh.channels() * 3);
    for (const mtl::Buffer* b : {&s.qkv, &s.z, &s.alpha, &s.beta, &s.y, &s.out, &s.hist_out}) {
        if (b->ptr == nullptr) {
            err = "the device refused the GDN scratch";
            return false;
        }
    }
    s.n_embd = sh.n_embd;
    s.h_k = sh.h_k;
    s.h_v = sh.h_v;
    s.state = sh.state;
    return true;
}

bool gdn_forward(mtl::Context& cx, GdnKernels& k, DenseKernels& dk, DenseScratch& ds, const GdnShape& sh,
                 const GdnWeights& w, GdnScratch& s, mtl::Buffer& state, mtl::Buffer& hist, const mtl::Buffer& h,
                 std::string& err) {
    if (!gdn_init(cx, k, err)) return false;
    if (!gdn_scratch(cx, s, sh, err)) return false;

    // 1. the four projections against the same prepared activation (alpha and beta stay RAW: gdn_step_t transforms them)
    if (!dense_gemv(cx, dk, w.qkv, h, s.qkv, ds, err)) return false;
    if (!dense_gemv(cx, dk, w.gate, h, s.z, ds, err)) return false;
    if (!dense_gemv(cx, dk, w.alpha, h, s.alpha, ds, err)) return false;
    if (!dense_gemv(cx, dk, w.beta, h, s.beta, ds, err)) return false;

    // 2. the step: conv + L2 + the delta rule + the SiLU-gated norm, state and history committed
    struct GdnArgs {
        uint32_t h_k, h_v, channels, commit;
        float eps;
        uint32_t plane;
    } a{(uint32_t) sh.h_k, (uint32_t) sh.h_v, (uint32_t) sh.channels(), 1u, sh.eps, 0u};
    mtl::Dispatch d;
    d.p = &k.step;
    d.args = {mtl::Arg::B(state), mtl::Arg::B(hist), mtl::Arg::B(s.qkv), mtl::Arg::B(s.alpha), mtl::Arg::B(s.beta),
              mtl::Arg::B(s.z), mtl::Arg::B(*w.conv_w, w.conv_w_off), mtl::Arg::B(*w.dt, w.dt_off),
              mtl::Arg::B(*w.ssm_a, w.ssm_a_off), mtl::Arg::B(*w.gamma, w.gamma_off), mtl::Arg::B(s.y), mtl::Arg::V(a),
              mtl::Arg::B(s.hist_out)};
    d.grid[0] = (uint32_t) sh.h_v;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 512;
    d.tg[1] = d.tg[2] = 1;
    // The history the kernel WRITES goes to its own buffer, and only then back into the caller's: the same buffer for both is a
    // race, because a threadgroup covers head hk = h % h_k for the queries and keys, so heads h and h + h_k write the same
    // channels while other threadgroups are still reading them (gdn_parity passes two buffers for exactly this).
    cx.run({d});
    std::memcpy(hist.ptr, s.hist_out.ptr, 4 * (size_t) sh.channels() * 3);

    // 3. the output projection
    if (!dense_gemv(cx, dk, w.out, s.y, s.out, ds, err)) return false;
    return true;
}

}  // namespace strata::qwen36
