// metal/tests/q36_glue_parity.cpp - q36_add_rms (metal/src/qwen36/msl/q36_glue.metal) against a double reference.
//
// The residual stream is the one place a small error is invisible until it is fatal: this kernel runs twice a layer, 40
// layers, and a wrong mean of squares is a wrong scale on everything downstream.  What can go wrong is the threadgroup
// reduction (a group's partial not summed, or summed twice), the stride pattern when the width is not a multiple of the
// threadgroup (2048 / 256 is 8 values a thread, 100 is not), and the in-place add.
//
//   q36_glue_parity
#include "mtl_rt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-78s rel L2 %.3e (tol %.1e)  %s\n", what.c_str(), err, tol, ok ? "PASS" : "FAIL");
    g_fail += !ok;
}

struct Case {
    int n, T;
    bool add;
};

void run_case(mtl::Context& cx, const Case& c) {
    const int N = c.n, T = c.T;
    const float eps = 1e-6f;
    std::vector<float> x((size_t) T * N), y((size_t) T * N), w((size_t) N), want((size_t) T * N), wantx((size_t) T * N);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.9f * std::sin(0.0013f * (float) i + 0.2f) + 0.05f * (float) ((int) (i % 5) - 2);
    for (size_t i = 0; i < y.size(); ++i) y[i] = 0.4f * std::cos(0.0007f * (float) i + 1.7f);
    for (int i = 0; i < N; ++i) w[i] = 0.5f + 0.5f * std::sin(0.01f * (float) i);

    // the reference: the same order the kernel does it (add, then the row's mean of squares in double, then scale)
    for (int t = 0; t < T; ++t) {
        double ss = 0.0;
        for (int e = 0; e < N; ++e) {
            const double xo = (double) x[(size_t) t * N + e];
            const double v = c.add ? xo + (double) y[(size_t) t * N + e] : xo;
            wantx[(size_t) t * N + e] = (float) v;
            ss += v * v;
        }
        const double inv = 1.0 / std::sqrt(ss / (double) N + (double) eps);
        for (int e = 0; e < N; ++e)
            want[(size_t) t * N + e] = (float) ((double) wantx[(size_t) t * N + e] * inv * (double) w[(size_t) e]);
    }

    mtl::Buffer bx = cx.alloc(4 * x.size()), by = cx.alloc(4 * y.size()), bw = cx.alloc(4 * w.size()),
                bh = cx.alloc(4 * want.size());
    std::memcpy(bx.ptr, x.data(), 4 * x.size());
    std::memcpy(by.ptr, y.data(), 4 * y.size());
    std::memcpy(bw.ptr, w.data(), 4 * w.size());

    mtl::Pipeline p = cx.pipeline(cx.compile({"../src/qwen36/msl/q36_glue.metal"}, "", false), "q36_add_rms");
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(bx), mtl::Arg::B(by), mtl::Arg::B(bw), mtl::Arg::B(bh), mtl::Arg::V((unsigned) N),
              mtl::Arg::V(c.add ? 1u : 0u), mtl::Arg::V(eps)};
    d.grid[0] = (uint32_t) T;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = 256;
    d.tg[1] = d.tg[2] = 1;
    cx.run({d});

    const float* got = (const float*) bh.ptr;
    const float* gotx = (const float*) bx.ptr;
    double num = 0.0, den = 0.0, xnum = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = (double) got[i] - (double) want[i];
        num += e * e;
        den += (double) want[i] * (double) want[i];
        const double ex = (double) gotx[i] - (double) wantx[i];
        xnum += ex * ex;
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "q36_add_rms n=%d T=%d%s", N, T, c.add ? ", add" : ", no add");
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);
    // the residual itself has to be exactly what was added when `add` is set (it is the whole point of the in-place form)
    if (c.add) {
        char nmx[128];
        std::snprintf(nmx, sizeof nmx, "    the residual x after the add (n=%d T=%d)", N, T);
        report(nmx, std::sqrt(xnum), 1e-6);
    }
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    // the family's hidden width, a width that is not a whole number of values a thread, and a single row
    for (const Case& c : {Case{2048, 1, true}, Case{2048, 3, true}, Case{100, 2, true}, Case{2048, 1, false},
                          Case{64, 4, false}})
        run_case(cx, c);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
