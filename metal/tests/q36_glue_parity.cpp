// metal/tests/q36_glue_parity.cpp - q36_add_rms (metal/src/qwen36/msl/q36_glue.metal) against a double reference.
//
// The residual stream is the one place a small error is invisible until it is fatal: this kernel runs twice a layer, 40
// layers, and a wrong mean of squares is a wrong scale on everything downstream.  What can go wrong is the threadgroup
// reduction (a group's partial not summed, or summed twice), the stride pattern when the width is not a multiple of the
// threadgroup (2048 / 256 is 8 values a thread, 100 is not), and the in-place add.
//
//   q36_glue_parity
#include "mtl_rt.h"
#include "ref_quant.h"

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

// q36_add: the plain residual add (no norm), bit for bit - it is the step that carries the MoE's output back into the residual.
void add_case(mtl::Context& cx, int n) {
    std::vector<float> x((size_t) n), y((size_t) n);
    for (int i = 0; i < n; ++i) {
        x[(size_t) i] = 0.8f * std::sin(0.021f * (float) i);
        y[(size_t) i] = 0.6f * std::cos(0.013f * (float) i + 0.4f);
    }
    mtl::Buffer bx = cx.alloc(4 * (size_t) n), by = cx.alloc(4 * (size_t) n);
    std::memcpy(bx.ptr, x.data(), 4 * (size_t) n);
    std::memcpy(by.ptr, y.data(), 4 * (size_t) n);
    mtl::Pipeline p = cx.pipeline(cx.compile({"../src/qwen36/msl/q36_glue.metal"}, "", false), "q36_add");
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(bx), mtl::Arg::B(by), mtl::Arg::V((unsigned) n)};
    d.tg[0] = 64;
    d.grid[0] = (uint32_t) ((n + 63) / 64);
    cx.run({d});
    const float* got = (const float*) bx.ptr;
    size_t bad = 0;
    for (int i = 0; i < n; ++i)
        if (got[(size_t) i] != x[(size_t) i] + y[(size_t) i]) ++bad;
    char nm[96];
    std::snprintf(nm, sizeof nm, "q36_add %d values (bit for bit)", n);
    report(nm, bad == 0 ? 0.0 : 1.0, 0.0);
}

// q36_embed: one token's row out of a Q8_0 table, against ref::dequant of the same bytes.
void embed_case(mtl::Context& cx, int n_embd, uint32_t token, int vocab) {
    const size_t rb = ref::row_bytes(ref::Q8_0, n_embd);
    std::vector<uint8_t> table(rb * (size_t) vocab);
    ref::random_blocks(ref::Q8_0, table.data(), vocab, n_embd, 77 + token, 0.02f);
    std::vector<float> want((size_t) n_embd);
    ref::dequant(ref::Q8_0, table.data() + (size_t) token * rb, want.data(), want.size());
    mtl::Buffer bt = cx.alloc(table.size()), bx = cx.alloc(4 * (size_t) n_embd);
    std::memcpy(bt.ptr, table.data(), table.size());
    mtl::Pipeline p = cx.pipeline(cx.compile({"../src/qwen36/msl/q36_glue.metal"}, "", false), "q36_embed");
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(bt), mtl::Arg::B(bx), mtl::Arg::V((unsigned) n_embd), mtl::Arg::V(token)};
    d.tg[0] = 64;
    d.grid[0] = (uint32_t) ((n_embd + 63) / 64);
    cx.run({d});
    const float* got = (const float*) bx.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n_embd; ++i) {
        const double e = (double) got[i] - (double) want[(size_t) i];
        num += e * e;
        den += (double) want[(size_t) i] * (double) want[(size_t) i];
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "q36_embed token %u of %d, n_embd %d", token, vocab, n_embd);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-6);
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    // the family's hidden width, a width that is not a whole number of values a thread, and a single row
    for (const Case& c : {Case{2048, 1, true}, Case{2048, 3, true}, Case{100, 2, true}, Case{2048, 1, false},
                          Case{64, 4, false}})
        run_case(cx, c);
    add_case(cx, 2048);
    add_case(cx, 1000);
    // the family's hidden width, the first and last tokens of a table, and a narrower table whose rows are still whole Q8_0
    // blocks (96 = three of them - a width that is not a multiple of 32 is not a valid Q8_0 row at all, which the first
    // version of this case tried and failed on)
    embed_case(cx, 2048, 0, 64);
    embed_case(cx, 2048, 63, 64);
    embed_case(cx, 96, 7, 16);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
