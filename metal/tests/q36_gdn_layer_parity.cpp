// metal/tests/q36_gdn_layer_parity.cpp - q36_gdn_forward (one whole GDN layer, wired) against a double reference.
//
// The step kernel is parity-checked on its own (gdn_parity: y 1.6e-07, state 7.8e-08, history exact); this catches the
// WIRING - the four projections into the right buffers, alpha and beta left RAW (gdn_step_t applies softplus + dt + ssm_a
// itself), the state and history carried across tokens, and ssm_out fed the step's output and not its input. The reference
// below is gdn_parity's own, which the kernel is already known to agree with, so a mismatch here is the layer and not the maths.
#include "qwen36/q36_gdn_layer.h"

#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int N = 512;    // the hidden width (small: the GDN's own numbers below are the family's)
constexpr int HK = 16, HV = 32, S = 128, C = (2 * HK + HV) * S, ZV = HV * S;

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

double dot(const std::vector<float>& W, int n_in, int row, const std::vector<float>& x) {
    double acc = 0.0;
    for (int i = 0; i < n_in; ++i) acc += (double) W[(size_t) row * n_in + i] * (double) x[(size_t) i];
    return acc;
}

// one layer, `T` tokens in sequence (the state and history must carry from one to the next)
void run_case(mtl::Context& cx, strata::qwen36::GdnKernels& k, int T, int pos_unused) {
    (void) pos_unused;
    strata::qwen36::DenseKernels dk;
    strata::qwen36::DenseScratch ds;
    std::string err;
    if (!strata::qwen36::dense_init(cx, dk, err)) { report("dense_init", 1.0, 0.0); return; }

    Mat wqkv, wgate, wal, wbe, wout;
    make_mat(cx, wqkv, N, C, 21);
    make_mat(cx, wgate, N, ZV, 22);
    make_mat(cx, wal, N, HV, 23);
    make_mat(cx, wbe, N, HV, 24);
    make_mat(cx, wout, ZV, N, 25);

    // the 1-D GDN tensors, as the file stores them (F32 here)
    std::vector<float> cw((size_t) C * 4), dt((size_t) HV), sa((size_t) HV), gm((size_t) S);
    for (size_t i = 0; i < cw.size(); ++i) cw[i] = 0.3f * std::sin(0.001f * (float) i + 0.7f);
    for (int i = 0; i < HV; ++i) { dt[(size_t) i] = 0.2f * std::cos(0.11f * (float) i); sa[(size_t) i] = -0.15f - 0.05f * (float) (i % 3); }
    for (int i = 0; i < S; ++i) gm[(size_t) i] = 0.8f + 0.2f * std::sin(0.03f * (float) i);
    mtl::Buffer bcw = cx.alloc(4 * cw.size()), bdt = cx.alloc(4 * dt.size()), bsa = cx.alloc(4 * sa.size()), bgm = cx.alloc(4 * gm.size());
    std::memcpy(bcw.ptr, cw.data(), 4 * cw.size());
    std::memcpy(bdt.ptr, dt.data(), 4 * dt.size());
    std::memcpy(bsa.ptr, sa.data(), 4 * sa.size());
    std::memcpy(bgm.ptr, gm.data(), 4 * gm.size());

    const strata::qwen36::GdnShape sh{N, HK, HV, S, 1e-6f};
    strata::qwen36::GdnWeights w{wqkv.dm, wgate.dm, wal.dm, wbe.dm, wout.dm, &bcw, 0, &bdt, 0, &bsa, 0, &bgm, 0};
    strata::qwen36::GdnScratch sc;
    mtl::Buffer state, hist;
    if (!strata::qwen36::gdn_state(cx, sh, state, hist, err) || !strata::qwen36::gdn_scratch(cx, sc, sh, err)) {
        report("gdn scratch/state: " + err, 1.0, 0.0);
        return;
    }
    // a non-zero starting state and history, so a wrong offset cannot hide behind zeros
    std::vector<float> st0((size_t) HV * S * S), h0((size_t) C * 3);
    for (size_t i = 0; i < st0.size(); ++i) st0[i] = 0.02f * std::sin(0.0007f * (float) i + 0.3f);
    for (size_t i = 0; i < h0.size(); ++i) h0[i] = 0.1f * std::cos(0.0011f * (float) i + 0.9f);
    std::memcpy(state.ptr, st0.data(), 4 * st0.size());
    std::memcpy(hist.ptr, h0.data(), 4 * h0.size());

    std::vector<std::vector<float>> hs((size_t) T, std::vector<float>(N));
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < N; ++i) hs[(size_t) t][(size_t) i] = 1.1f * std::sin(0.017f * (float) (i + 37 * t) + 0.4f);
    std::vector<mtl::Buffer> bh((size_t) T);
    std::vector<float> want((size_t) T * N);
    std::vector<double> Sd(st0.begin(), st0.end()), Hd(h0.begin(), h0.end());

    for (int t = 0; t < T; ++t) {
        bh[(size_t) t] = cx.alloc(4 * (size_t) N);
        std::memcpy(bh[(size_t) t].ptr, hs[(size_t) t].data(), 4 * (size_t) N);
        if (!strata::qwen36::gdn_forward(cx, k, dk, ds, sh, w, sc, state, hist, bh[(size_t) t], err)) {
            report("gdn_forward: " + err, 1.0, 0.0);
            return;
        }
        // ---- the reference: gdn_parity's, which this kernel is known to agree with ----
        std::vector<double> qkv((size_t) C), z((size_t) ZV), al((size_t) HV), be((size_t) HV);
        for (int o = 0; o < C; ++o) qkv[(size_t) o] = dot(wqkv.f, N, o, hs[(size_t) t]);
        for (int o = 0; o < ZV; ++o) z[(size_t) o] = dot(wgate.f, N, o, hs[(size_t) t]);
        for (int o = 0; o < HV; ++o) {
            al[(size_t) o] = dot(wal.f, N, o, hs[(size_t) t]);
            be[(size_t) o] = dot(wbe.f, N, o, hs[(size_t) t]);
        }
        if (t == 0) {   // the four projections first: they are pure gemvs, so a mismatch here is the wiring and not the step
            auto cmp = [&](const mtl::Buffer& b, const std::vector<double>& ref, int n, const char* what) {
                const float* g = (const float*) b.ptr;
                double en = 0.0, ed = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double e = (double) g[i] - ref[(size_t) i];
                    en += e * e;
                    ed += ref[(size_t) i] * ref[(size_t) i];
                }
                std::printf("    %-12s rel L2 %.3e\n", what, ed > 0 ? std::sqrt(en / ed) : std::sqrt(en));
            };
            std::vector<double> rq((size_t) C), rz((size_t) ZV), ra((size_t) HV), rb((size_t) HV);
            for (int o = 0; o < C; ++o) rq[(size_t) o] = dot(wqkv.f, N, o, hs[0]);
            for (int o = 0; o < ZV; ++o) rz[(size_t) o] = dot(wgate.f, N, o, hs[0]);
            for (int o = 0; o < HV; ++o) {
                ra[(size_t) o] = dot(wal.f, N, o, hs[0]);
                rb[(size_t) o] = dot(wbe.f, N, o, hs[0]);
            }
            cmp(sc.qkv, rq, C, "qkv");
            cmp(sc.z, rz, ZV, "z");
            cmp(sc.alpha, ra, HV, "alpha");
            cmp(sc.beta, rb, HV, "beta");
        }
        std::vector<double> cv((size_t) C);
        for (int c = 0; c < C; ++c) {
            const double in = qkv[(size_t) c];
            double s = Hd[(size_t) (c * 3)] * (double) cw[(size_t) (c * 4)] + Hd[(size_t) (c * 3 + 1)] * (double) cw[(size_t) (c * 4 + 1)] +
                       Hd[(size_t) (c * 3 + 2)] * (double) cw[(size_t) (c * 4 + 2)] + in * (double) cw[(size_t) (c * 4 + 3)];
            cv[(size_t) c] = s / (1.0 + std::exp(-s));
            Hd[(size_t) (c * 3)] = Hd[(size_t) (c * 3 + 1)];
            Hd[(size_t) (c * 3 + 1)] = Hd[(size_t) (c * 3 + 2)];
            Hd[(size_t) (c * 3 + 2)] = in;
        }
        for (int hh = 0; hh < 2 * HK; ++hh) {
            double ss = 0;
            for (int i = 0; i < S; ++i) ss += cv[(size_t) (hh * S + i)] * cv[(size_t) (hh * S + i)];
            const double rs = 1.0 / std::sqrt(ss / S + sh.eps / S);
            for (int i = 0; i < S; ++i) cv[(size_t) (hh * S + i)] = cv[(size_t) (hh * S + i)] * rs / std::sqrt((double) S);
        }
        std::vector<double> yv((size_t) ZV);
        for (int h = 0; h < HV; ++h) {
            const int hk = h % HK;
            const double av = al[(size_t) h] + (double) dt[(size_t) h];
            const double sp = av > 20 ? av : std::log1p(std::exp(av));
            const double g = std::exp(sp * (double) sa[(size_t) h]);
            const double b = 1.0 / (1.0 + std::exp(-be[(size_t) h]));
            std::vector<double> o((size_t) S);
            for (int jj = 0; jj < S; ++jj) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += Sd[(size_t) ((i * HV + h) * S + jj)] * cv[(size_t) (HK * S + hk * S + i)];
                const double delta = (cv[(size_t) (2 * HK * S + h * S + jj)] - g * kv) * b;
                double acc = 0;
                for (int i = 0; i < S; ++i) {
                    double& sv = Sd[(size_t) ((i * HV + h) * S + jj)];
                    sv = g * sv + cv[(size_t) (HK * S + hk * S + i)] * delta;
                    acc += sv * cv[(size_t) (hk * S + i)];
                }
                o[(size_t) jj] = acc / std::sqrt((double) S);
            }
            double ss = 0;
            for (int jj = 0; jj < S; ++jj) ss += o[(size_t) jj] * o[(size_t) jj];
            const double rs = 1.0 / std::sqrt(ss / S + (double) sh.eps);
            for (int jj = 0; jj < S; ++jj) {
                const double zz = z[(size_t) (h * S + jj)];
                yv[(size_t) (h * S + jj)] = o[(size_t) jj] * rs * (double) gm[(size_t) jj] / (1.0 + std::exp(-zz));
            }
        }
        std::vector<float> yf((size_t) ZV);
        for (size_t i = 0; i < yf.size(); ++i) yf[i] = (float) yv[i];
        for (int o = 0; o < N; ++o) want[(size_t) (t * N + o)] = dot(wout.f, ZV, o, yf);
    }

    // the last token's output against the reference, and the state the layer left behind
    const float* got = (const float*) sc.out.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = (double) got[i] - (double) want[(size_t) ((T - 1) * N + i)];
        num += e * e;
        den += want[(size_t) ((T - 1) * N + i)] * want[(size_t) ((T - 1) * N + i)];
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "gdn layer: hidden %d, %d K / %d V heads, %d state, %d token(s)", N, HK, HV, S, T);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);

    const float* sgot = (const float*) state.ptr;
    double sn = 0.0, sd = 0.0;
    for (size_t i = 0; i < Sd.size(); ++i) {
        const double e = (double) sgot[i] - Sd[i];
        sn += e * e;
        sd += Sd[i] * Sd[i];
    }
    char ns[96];
    std::snprintf(ns, sizeof ns, "    the state after %d token(s)", T);
    report(ns, sd > 0 ? std::sqrt(sn / sd) : std::sqrt(sn), 2e-5);
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    strata::qwen36::GdnKernels k;
    std::string err;
    if (!strata::qwen36::gdn_init(cx, k, err)) {
        std::printf("gdn_init failed: %s\n", err.c_str());
        return 1;
    }
    for (const auto& t : {1, 3}) run_case(cx, k, t, 0);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
