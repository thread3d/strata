// metal/tests/q36_rope_parity.cpp - q36_head_norm_rope_{f32,bf16} (metal/src/qwen36/msl/q36_rope.metal) against a double
// reference, with the RoPE table built exactly as the engine builds it (setup_host.cpp build_rope_table: double angles,
// base from the file, NEOX pairs (i, i + 32) over the first 64 of a 256-wide head).
//
// A head-wide norm is where a wrong reduction is a wrong scale on everything attention then reads, and the rotation is
// where a wrong pairing silently permutes the model (a dump then "looks permuted", which is a whole afternoon to chase).
// So: the query path (fp32), the KV path (bf16), positions from 0 to a long one, and the whole head rather than a slice.
#include "mtl_rt.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int QD = 256, HALF = 32, QROT = 64;

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-78s rel L2 %.3e (tol %.1e)  %s\n", what.c_str(), err, tol, ok ? "PASS" : "FAIL");
    g_fail += !ok;
}

uint16_t to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
}

void run(mtl::Context& cx, int heads, bool bf16, int pos, const char* name) {
    const float eps = 1e-6f;
    const double base = 1e7;
    std::vector<float> cs((size_t) (pos + 1) * HALF * 2);
    for (int p = 0; p <= pos; ++p)
        for (int i = 0; i < HALF; ++i) {
            const double th = (double) p * std::pow(base, -2.0 * (double) i / (double) QD);
            cs[(size_t) (p * HALF + i) * 2] = (float) std::cos(th);
            cs[(size_t) (p * HALF + i) * 2 + 1] = (float) std::sin(th);
        }
    // the projection is per head [q(QD) | gate(QD)]: heads * 2 * QD values, of which only every other QD is a query
    std::vector<float> raw((size_t) heads * 2 * QD), w((size_t) QD);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = 1.3f * std::sin(0.0031f * (float) i + 0.4f);
    for (int i = 0; i < QD; ++i) w[i] = 0.6f + 0.4f * std::cos(0.017f * (float) i);

    // the reference, in double: the head's mean of squares, then the NEOX rotation on the first 64
    std::vector<float> want((size_t) heads * QD);
    std::vector<uint16_t> wantb((size_t) heads * QD);
    for (int h = 0; h < heads; ++h) {
        double ss = 0.0;
        for (int e = 0; e < QD; ++e) ss += (double) raw[(size_t) h * 2 * QD + e] * (double) raw[(size_t) h * 2 * QD + e];
        const double rs = 1.0 / std::sqrt(ss / QD + (double) eps);
        std::vector<double> y((size_t) QD);
        for (int e = 0; e < QD; ++e) y[(size_t) e] = (double) raw[(size_t) h * 2 * QD + e] * rs * (double) w[e];
        for (int e = 0; e < QD; ++e) {
            double o = y[(size_t) e];
            if (e < QROT) {
                const int i = e & (HALF - 1);
                const double c = (double) cs[(size_t) (pos * HALF + i) * 2], s = (double) cs[(size_t) (pos * HALF + i) * 2 + 1];
                const double a = y[(size_t) i], b = y[(size_t) (i + HALF)];
                o = e < HALF ? a * c - b * s : a * s + b * c;
            }
            want[(size_t) h * QD + e] = (float) o;
            wantb[(size_t) h * QD + e] = to_bf16((float) o);
        }
    }

    mtl::Buffer br = cx.alloc(4 * raw.size()), bw = cx.alloc(4 * w.size()), bc = cx.alloc(4 * cs.size());
    std::memcpy(br.ptr, raw.data(), 4 * raw.size());
    std::memcpy(bw.ptr, w.data(), 4 * w.size());
    std::memcpy(bc.ptr, cs.data(), 4 * cs.size());

    mtl::Pipeline p = cx.pipeline(cx.compile({"../src/qwen36/msl/q36_rope.metal"}, "", false),
                                  bf16 ? "q36_head_norm_rope_bf16" : "q36_head_norm_rope_f32");
    mtl::Buffer bo = cx.alloc((bf16 ? 2 : 4) * (size_t) want.size());
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(br), mtl::Arg::B(bw), mtl::Arg::B(bc), mtl::Arg::B(bo), mtl::Arg::V((unsigned) pos),
              mtl::Arg::V(eps)};
    d.grid[0] = (uint32_t) heads;
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = QD;
    d.tg[1] = d.tg[2] = 1;
    cx.run({d});

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double got = bf16 ? (double) from_bf16(((const uint16_t*) bo.ptr)[i]) : (double) ((const float*) bo.ptr)[i];
        const double ref = bf16 ? (double) from_bf16(wantb[i]) : (double) want[i];
        const double e = got - ref;
        num += e * e;
        den += ref * ref;
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "q36_head_norm_rope %s %d heads, pos %d", name, heads, pos);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), bf16 ? 1e-6 : 2e-6);
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    // the family's shapes: 16 query heads (fp32 out) and 2 KV heads (bf16 out, the ring), at the first position and at
    // ones far enough that the angle's own magnitude matters
    for (int pos : {0, 7, 1000, 32768}) {
        run(cx, 16, false, pos, "q  fp32");
        run(cx, 2, true, pos, "kv bf16");
    }
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
