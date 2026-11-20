// metal/tests/ds4_pnorm_parity.cpp - ds4_hc_pre_norm (the DeepSeek engine's activation producer) against a double reference.
//
// The engine's matvecs consumed zeros on an Intel Mac Pro because this kernel wrote zeros: a = sum_s mix[s]*h[s][e] over the hc
// residual streams, x = a * rsqrt(mean(a^2) + eps) (times w when given).  Tested here on its own, away from the engine: the same
// shapes (dim 5120, hc 4, 1 and 3 tokens), a 1024-thread threadgroup, values that make the norm's scale matter.
#include "mtl_rt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int fails = 0;
void report(const char* what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-56s err %.3e (tol %.1e)  %s\n", what, err, tol, ok ? "PASS" : "FAIL");
    fails += !ok;
}
}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const int lib = cx.compile({"../src/engine/msl/ds4_init.metal"}, "", false);
    mtl::Pipeline p;
    try {
        p = cx.pipeline(lib, "ds4_hc_pre_norm");
    } catch (const std::exception& e) {
        std::printf("no ds4_hc_pre_norm: %s\n", e.what());
        return 1;
    }
    for (const int T : {1, 3}) {
        const int dim = 5120, hc = 4, n_mix = 8;
        const float eps = 1e-20f;
        std::vector<float> h((size_t) T * hc * dim), mix((size_t) T * n_mix), w(dim, 0.5f), ref((size_t) T * dim, 0.f);
        // values spread over the range the engine sees: the embedding (about +-0.5) and mixes of order one
        for (size_t i = 0; i < h.size(); ++i) h[i] = 0.5f * std::sin(0.00037f * (float) i) + 0.06f * ((int) (i % 7) - 3);
        for (size_t i = 0; i < mix.size(); ++i) mix[i] = (i % 2 ? 1.0f : -1.0f) * (1.0f + 0.1f * (float) (i % 3));
        for (int t = 0; t < T; ++t)
            for (int e = 0; e < dim; ++e) {
                double a = 0.0;
                for (int s = 0; s < hc; ++s) a += (double) mix[(size_t) t * n_mix + s] * h[((size_t) t * hc + s) * dim + e];
                ref[(size_t) t * dim + e] = (float) a;
            }
        // the reference's mean of squares, then the normalisation (with w)
        std::vector<float> want(ref);
        for (int t = 0; t < T; ++t) {
            double ss = 0.0;
            for (int e = 0; e < dim; ++e) ss += (double) ref[(size_t) t * dim + e] * ref[(size_t) t * dim + e];
            const double inv = 1.0 / std::sqrt(ss / dim + (double) eps);
            for (int e = 0; e < dim; ++e)
                want[(size_t) t * dim + e] = (float) ((double) ref[(size_t) t * dim + e] * inv * (double) w[e]);
        }
        mtl::Buffer bh = cx.alloc(4 * h.size()), bm = cx.alloc(4 * mix.size()), bw = cx.alloc(4 * w.size());
        mtl::Buffer bx = cx.alloc(4 * (size_t) T * dim);
        std::memcpy(bh.ptr, h.data(), 4 * h.size());
        std::memcpy(bm.ptr, mix.data(), 4 * mix.size());
        std::memcpy(bw.ptr, w.data(), 4 * w.size());
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(bh), mtl::Arg::B(bm), mtl::Arg::B(bw), mtl::Arg::V((unsigned) dim),
                  mtl::Arg::V((unsigned) hc), mtl::Arg::V(eps), mtl::Arg::B(bx), mtl::Arg::V(1u)};
        d.grid[0] = (uint32_t) T;
        d.grid[1] = d.grid[2] = 1;
        d.tg[0] = 1024;
        d.tg[1] = d.tg[2] = 1;
        cx.run({d});   // (the test's runtime commits and waits inside run)
        const float* got = (const float*) bx.ptr;
        double num = 0.0, den = 0.0, worst = 0.0;
        size_t zeros = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            const double e = (double) got[i] - (double) want[i];
            num += e * e;
            den += (double) want[i] * (double) want[i];
            worst = std::fmax(worst, std::fabs(e));
            if (got[i] == 0.0f) ++zeros;
        }
        char nm[96];
        std::snprintf(nm, sizeof nm, "hc_pre_norm T=%d dim=%d hc=%d (relative L2)", T, dim, hc);
        report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-6);
        std::printf("    zeros %zu of %zu, worst |err| %.3e\n", zeros, want.size(), worst);
    }
    std::printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
