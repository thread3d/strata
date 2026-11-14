// metal/tests/q36_dense_parity.cpp - q36_dense_gemv (metal/src/qwen36/q36_dense.{h,cpp}) against a double reference.
//
// This is the first function of the driver proper: one token's activation through one dense matrix, with the
// prepared-activation form, the tile, the grid and the argument order that every projection of every layer will use.  The
// reference dequantises the same Q8_0 bytes on the CPU (ref::dequant, the port's transcription of ggml) and dots them with
// the same activation in double, so a wrong row stride, a wrong plane offset or a wrong tile shows as a wrong answer.
//
//   q36_dense_parity
#include "qwen36/q36_dense.h"

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

// q36's own widths: the hidden (2048) for q/k/v/o and the mixers, the value dim (512 for the tiny shape), and an expert FF
// width; each at Q8_0, the format the pack the user asked for carries.
struct Case {
    int n_in, n_out;
    const char* what;
};

void run(mtl::Context& cx, strata::qwen36::DenseKernels& k, const Case& c) {
    const int n_in = c.n_in, n_out = c.n_out;
    const uint64_t rb = (uint64_t) ref::row_bytes(ref::Q8_0, n_in);
    std::vector<uint8_t> w((size_t) rb * n_out);
    ref::random_blocks(ref::Q8_0, w.data(), n_out, n_in, 1234u + (uint64_t) n_in, 0.02f);
    std::vector<float> x((size_t) n_in);
    for (int i = 0; i < n_in; ++i) x[i] = 1.1f * std::sin(0.013f * (float) i + 0.9f);

    // the reference: the same blocks, dequantised on the CPU, dotted in double
    std::vector<float> wf((size_t) n_in * n_out);
    ref::dequant(ref::Q8_0, w.data(), wf.data(), wf.size());
    std::vector<double> want((size_t) n_out);
    for (int o = 0; o < n_out; ++o) {
        double s = 0.0;
        for (int i = 0; i < n_in; ++i) s += (double) wf[(size_t) o * n_in + i] * (double) x[i];
        want[(size_t) o] = s;
    }

    mtl::Buffer bw = cx.alloc(w.size()), bx = cx.alloc(4 * (size_t) n_in), by = cx.alloc(4 * (size_t) n_out);
    std::memcpy(bw.ptr, w.data(), w.size());
    std::memcpy(bx.ptr, x.data(), 4 * (size_t) n_in);

    strata::qwen36::DenseMatrix W{&bw, 0, ref::Q8_0, n_in, n_out};
    strata::qwen36::DenseScratch s;
    std::string err;
    const bool ok = strata::qwen36::dense_gemv(cx, k, W, bx, by, s, err);
    if (!ok) {
        std::printf("%-78s FAIL (%s)\n", c.what, err.c_str());
        ++g_fail;
        return;
    }
    const float* got = (const float*) by.ptr;
    double num = 0.0, den = 0.0;
    for (int o = 0; o < n_out; ++o) {
        const double e = (double) got[o] - want[(size_t) o];
        num += e * e;
        den += want[(size_t) o] * want[(size_t) o];
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "%s (%d x %d, Q8_0)", c.what, n_in, n_out);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);
    if (s.xp.ptr != nullptr) cx.free(s.xp);
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    strata::qwen36::DenseKernels k;
    std::string err;
    if (!strata::qwen36::dense_init(cx, k, err)) {
        std::printf("dense_init failed: %s\n", err.c_str());
        return 1;
    }
    // the family's hidden, the GDN value dim, the shared expert's FF, and a width below the 640 tile threshold
    for (const Case& c : {Case{2048, 64, "hidden -> rows"}, Case{2048, 512, "hidden -> value dim"},
                          Case{512, 32, "value dim -> hidden"}, Case{320, 16, "a narrow row width"}})
        run(cx, k, c);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
