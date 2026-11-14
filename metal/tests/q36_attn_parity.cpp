// metal/tests/q36_attn_parity.cpp - metal/src/qwen36/msl/q36_attn.metal against a double reference.
//
// The kernel is the one piece of Qwen3.6's engine that has no counterpart in the Flash-Next set: 16 query heads over 2 KV
// heads, width 256, every row of the window read, online softmax.  What can go wrong in it is a lane mapping (which group
// owns which head, which lane holds which component), the GQA group a query head reads, the online rescale, and the output
// gate - so all four are exercised here at the family's own width and at small widths where a wrong lane shows up as a
// wrong answer rather than as a small difference.
//
//   q36_attn_parity [--ntok N]      (default: every case below)
#include "mtl_rt.h"

#include <algorithm>
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

// The kernel's row values are bf16; the reference reads exactly what the kernel reads, so the same rounding is applied
// here (round to nearest even on the 16 high bits) and then widened back, exactly as q36_bf16 does in the shader.
uint16_t to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);   // (a NaN stays a NaN)
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Case {
    int qh, kvh, qd, n_kv;
    bool gate;
};

void run_case(mtl::Context& cx, const Case& c) {
    const int QH = c.qh, KVH = c.kvh, QD = c.qd, NKV = c.n_kv;
    const float scale = 1.0f / std::sqrt((float) QD);
    const std::string fn = "q36_attn_" + std::to_string(QH) + "_" + std::to_string(KVH) + "_" + std::to_string(QD);
    mtl::Pipeline p = cx.pipeline(cx.compile({"../src/qwen36/msl/q36_attn.metal"}, "", false), fn);

    // deterministic values with a spread that makes the softmax's max matter (logits of order +-3)
    std::vector<float> q((size_t) QH * QD), gate((size_t) QH * QD);
    std::vector<uint16_t> kb((size_t) KVH * NKV * QD), vb((size_t) KVH * NKV * QD);
    std::vector<float> kf((size_t) KVH * NKV * QD), vf((size_t) KVH * NKV * QD);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.7f * std::sin(0.013f * (float) i + 0.3f);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = 3.0f * std::sin(0.007f * (float) i + 1.1f);
    for (size_t i = 0; i < kb.size(); ++i) {
        kf[i] = 0.8f * std::sin(0.011f * (float) i + 2.0f);
        vf[i] = 0.9f * std::cos(0.005f * (float) i + 0.7f);
        kb[i] = to_bf16(kf[i]);
        vb[i] = to_bf16(vf[i]);
        kf[i] = from_bf16(kb[i]);
        vf[i] = from_bf16(vb[i]);
    }

    // the reference: every query head over its KV group's rows, in double
    std::vector<float> want((size_t) QH * QD, 0.0f);
    for (int h = 0; h < QH; ++h) {
        const int kvh = h * KVH / QH;   // the group the kernel gives this query head
        std::vector<double> s((size_t) NKV);
        double mx = -INFINITY;
        for (int r = 0; r < NKV; ++r) {
            double d = 0.0;
            for (int e = 0; e < QD; ++e) d += (double) q[(size_t) h * QD + e] * (double) kf[(size_t) (kvh * NKV + r) * QD + e];
            s[(size_t) r] = d * (double) scale;
            mx = std::fmax(mx, s[(size_t) r]);
        }
        double sum = 0.0;
        for (int r = 0; r < NKV; ++r) {
            s[(size_t) r] = std::exp(s[(size_t) r] - mx);
            sum += s[(size_t) r];
        }
        if (NKV == 0) sum = 1.0;
        for (int e = 0; e < QD; ++e) {
            double a = 0.0;
            for (int r = 0; r < NKV; ++r) a += s[(size_t) r] * (double) vf[(size_t) (kvh * NKV + r) * QD + e];
            a /= sum;
            if (c.gate) a *= 1.0 / (1.0 + std::exp(-(double) gate[(size_t) h * QD + e]));
            want[(size_t) h * QD + e] = (float) a;
        }
    }

    // (an empty window still needs a buffer that exists: alloc(0) is refused by this runtime, and says so)
    mtl::Buffer bq = cx.alloc(4 * q.size()), bg = cx.alloc(4 * gate.size()), bk = cx.alloc(2 * std::max<size_t>(1, kb.size())),
                bv = cx.alloc(2 * std::max<size_t>(1, vb.size())), bo = cx.alloc(4 * want.size());
    std::memcpy(bq.ptr, q.data(), 4 * q.size());
    std::memcpy(bg.ptr, gate.data(), 4 * gate.size());
    if (!kb.empty()) std::memcpy(bk.ptr, kb.data(), 2 * kb.size());
    if (!vb.empty()) std::memcpy(bv.ptr, vb.data(), 2 * vb.size());

    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(bq), mtl::Arg::B(bk), mtl::Arg::B(bv), mtl::Arg::B(bg), mtl::Arg::B(bo),
              mtl::Arg::V((unsigned) NKV), mtl::Arg::V(c.gate ? 1u : 0u), mtl::Arg::V(scale)};
    d.grid[0] = 1;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = (uint32_t) (QH * 32);
    d.tg[1] = d.tg[2] = 1;
    cx.run({d});

    const float* got = (const float*) bo.ptr;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double e = (double) got[i] - (double) want[i];
        num += e * e;
        den += (double) want[i] * (double) want[i];
    }
    char nm[160];
    std::snprintf(nm, sizeof nm, "q36_attn %d q / %d kv x %d, %d rows%s", QH, KVH, QD, NKV, c.gate ? ", gated" : "");
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);
}

}  // namespace

int main(int argc, char** argv) {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    (void) argc;
    (void) argv;
    // the family's own shape first, then widths where a lane mapping error cannot hide in a 256-wide vector, then the
    // window lengths that exercise the online softmax (empty, one row, a row past any chunk boundary, long)
    for (const Case& c : {Case{16, 2, 256, 1, false}, Case{16, 2, 256, 37, false}, Case{16, 2, 256, 300, true},
                          Case{16, 2, 256, 2048, false}, Case{4, 2, 32, 10, false}, Case{8, 4, 64, 64, true},
                          Case{2, 1, 32, 0, false}})
        run_case(cx, c);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
