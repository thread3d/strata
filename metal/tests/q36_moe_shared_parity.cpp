// metal/tests/q36_moe_shared_parity.cpp - q36_moe_shared (the shared expert's gate+up via gemv_pair, then its down) against a
// double reference over the same Q8_0 bytes.
//
// gemv_pair is parity-tested on its own (pair_parity.cpp); what is checked here is the WIRING - the pair's pipeline name for
// the two types, the two prepared planes each weight reads, rb_up as the up matrix's own row stride, h read RAW by the down
// projection, and the two dispatches in the right order against the same prepared activation.
#include "qwen36/q36_moe_shared.h"

#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int N = 512, NFF = 128;   // (the down projection's rows are 128 wide: a Q8_0 gemv tile needs 4 units)

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-78s rel L2 %.3e (tol %.1e)  %s\n", what.c_str(), err, tol, ok ? "PASS" : "FAIL");
    g_fail += !ok;
}

struct Mat {
    std::vector<uint8_t> bytes;
    std::vector<float> f;
    mtl::Buffer buf;
    strata::qwen36::DenseMatrix dm;
};
void make_mat(mtl::Context& cx, Mat& m, int n_in, int n_out, uint64_t seed) {
    const size_t rb = ref::row_bytes(ref::Q8_0, n_in);
    m.bytes.resize(rb * (size_t) n_out);
    ref::random_blocks(ref::Q8_0, m.bytes.data(), n_out, n_in, seed, 0.02f);
    m.f.resize((size_t) n_in * (size_t) n_out);
    ref::dequant(ref::Q8_0, m.bytes.data(), m.f.data(), m.f.size());
    m.buf = cx.alloc(m.bytes.size());
    std::memcpy(m.buf.ptr, m.bytes.data(), m.bytes.size());
    m.dm = strata::qwen36::DenseMatrix{&m.buf, 0, ref::Q8_0, n_in, n_out};
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    strata::qwen36::DenseKernels dk;
    strata::qwen36::DenseScratch ds;
    std::string err;
    if (!strata::qwen36::dense_init(cx, dk, err)) { std::printf("dense_init: %s\n", err.c_str()); return 1; }

    Mat wg, wu, wd;
    make_mat(cx, wg, N, NFF, 31);
    make_mat(cx, wu, N, NFF, 32);
    make_mat(cx, wd, NFF, N, 33);
    std::vector<float> x((size_t) N);
    for (int i = 0; i < N; ++i) x[(size_t) i] = 1.15f * std::sin(0.019f * (float) i + 0.6f);
    mtl::Buffer bx = cx.alloc(4 * (size_t) N);
    std::memcpy(bx.ptr, x.data(), 4 * (size_t) N);

    const strata::qwen36::SharedShape sh{N, NFF};
    strata::qwen36::SharedWeights w{wg.dm, wu.dm, wd.dm};
    strata::qwen36::SharedScratch s;
    if (!strata::qwen36::shared_forward(cx, dk, ds, sh, w, s, bx, err)) {
        std::printf("shared_forward: %s\n", err.c_str());
        return 1;
    }

    // the reference: silu(gate . x) * (up . x), then down . h - all in double over the dequantised bytes
    auto dot = [](const std::vector<float>& W, int n_in, int row, const std::vector<float>& v) {
        double a = 0.0;
        for (int i = 0; i < n_in; ++i) a += (double) W[(size_t) row * n_in + i] * (double) v[(size_t) i];
        return a;
    };
    std::vector<float> h((size_t) NFF);
    for (int r = 0; r < NFF; ++r) {
        const double g = dot(wg.f, N, r, x), u = dot(wu.f, N, r, x);
        h[(size_t) r] = (float) (g / (1.0 + std::exp(-g)) * u);
    }
    std::vector<double> want((size_t) N);
    for (int o = 0; o < N; ++o) want[(size_t) o] = dot(wd.f, NFF, o, h);

    const float* got = (const float*) s.y.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = (double) got[i] - want[(size_t) i];
        num += e * e;
        den += want[(size_t) i] * want[(size_t) i];
    }
    char nm[96];
    std::snprintf(nm, sizeof nm, "shared expert: %d -> %d -> %d (Q8_0 pair + down)", N, NFF, N);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
