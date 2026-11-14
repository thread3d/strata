// metal/tests/q36_attn_layer_parity.cpp - q36_attn_forward (one whole attention layer, wired) against a double reference
// over the same bytes: no model, no checkpoint, tiny Q8_0 matrices at the family's head shape.
//
// The five measured parts are each parity-checked on their own; this catches the WIRING - a projection pointed at the wrong
// buffer, the gate read from the queries' half, a KV row written one head short, the ring read before the append, the output
// projection fed the pre-gate attention.  Every one of those is invisible in the parts and fatal in the layer.
#include "qwen36/q36_attn_layer.h"

#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int N = 512;      // the hidden width (a small one: the head shape below is the family's own)
constexpr int QH = 16, KVH = 2, QD = 256;

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

/// A Q8_0 matrix in a device buffer plus its CPU dequantisation for the reference.  n_out rows of n_in values.
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

void run_case(mtl::Context& cx, strata::qwen36::AttnKernels& k, int pos, int n_kv) {
    const int cap = n_kv + 4;
    strata::qwen36::DenseKernels dk;
    strata::qwen36::DenseScratch ds;
    std::string err;
    if (!strata::qwen36::dense_init(cx, dk, err)) {
        report("dense_init", 1.0, 0.0);
        return;
    }

    // the layer's weights and norms
    Mat wq, wk, wv, wo;
    make_mat(cx, wq, N, 2 * QH * QD, 11);
    make_mat(cx, wk, N, KVH * QD, 12);
    make_mat(cx, wv, N, KVH * QD, 13);
    make_mat(cx, wo, QH * QD, N, 14);
    std::vector<float> qn((size_t) QD), kn((size_t) QD);
    for (int i = 0; i < QD; ++i) {
        qn[(size_t) i] = 0.7f + 0.3f * std::sin(0.011f * (float) i);
        kn[(size_t) i] = 0.7f + 0.3f * std::cos(0.009f * (float) i);
    }
    mtl::Buffer bqn = cx.alloc(4 * qn.size()), bkn = cx.alloc(4 * kn.size());
    std::memcpy(bqn.ptr, qn.data(), 4 * qn.size());
    std::memcpy(bkn.ptr, kn.data(), 4 * kn.size());

    // the RoPE table, built as setup_host.cpp does
    // n_rot is 64 of the head's 256 - the family's rope.dimension_count, which the port's kernel is cut for - so the
    // rotation covers the first 64 dims in 32 NEOX pairs, and the table is 32 (cos, sin) pairs a position.  (The first
    // version of this test used QD / 2 = 128 and rotated half the head: with one KV row a uniform softmax hides both the
    // query and the key path entirely, so it passed.)
    const double base = 1e7;
    constexpr int HALF = 32, QROT = 64;
    std::vector<float> cs((size_t) (pos + 1) * HALF * 2);
    for (int p = 0; p <= pos; ++p)
        for (int i = 0; i < HALF; ++i) {
            const double th = (double) p * std::pow(base, -2.0 * (double) i / (double) QROT);
            cs[(size_t) (p * HALF + i) * 2] = (float) std::cos(th);
            cs[(size_t) (p * HALF + i) * 2 + 1] = (float) std::sin(th);
        }
    mtl::Buffer bcs = cx.alloc(4 * cs.size());
    std::memcpy(bcs.ptr, cs.data(), 4 * cs.size());

    const strata::qwen36::AttnShape sh{N, QH, KVH, QD, 1e-6f};
    strata::qwen36::AttnWeights w{wq.dm, wk.dm, wv.dm, wo.dm, &bqn, 0, &bkn, 0};
    strata::qwen36::AttnScratch s;
    if (!strata::qwen36::attn_scratch(cx, s, sh, cap, err)) {
        report("attn_scratch: " + err, 1.0, 0.0);
        return;
    }
    // the window's earlier rows: whatever the ring held, which the reference must read from the same place
    std::vector<uint16_t> pre_k((size_t) (n_kv - 1) * KVH * QD), pre_v((size_t) (n_kv - 1) * KVH * QD);
    for (size_t i = 0; i < pre_k.size(); ++i) {
        pre_k[i] = to_bf16(0.4f * std::sin(0.01f * (float) i + 0.2f));
        pre_v[i] = to_bf16(0.5f * std::cos(0.013f * (float) i + 0.6f));
    }
    if (!pre_k.empty()) std::memcpy(s.kring.ptr, pre_k.data(), 2 * pre_k.size());
    if (!pre_v.empty()) std::memcpy(s.vring.ptr, pre_v.data(), 2 * pre_v.size());

    std::vector<float> h((size_t) N);
    for (int i = 0; i < N; ++i) h[(size_t) i] = 1.2f * std::sin(0.021f * (float) i + 0.5f);
    mtl::Buffer bh = cx.alloc(4 * h.size());
    std::memcpy(bh.ptr, h.data(), 4 * h.size());

    if (!strata::qwen36::attn_forward(cx, k, dk, ds, sh, w, s, bh, bcs, pos, n_kv, err)) {
        report("attn_forward: " + err, 1.0, 0.0);
        return;
    }

    // ---- the reference, in double, over the same dequantised bytes and the same bf16 rows ----
    auto dot = [](const std::vector<float>& W, int n_in, int row, const std::vector<float>& x) {
        double acc = 0.0;
        for (int i = 0; i < n_in; ++i) acc += (double) W[(size_t) row * n_in + i] * (double) x[(size_t) i];
        return acc;
    };
    std::vector<double> qg((size_t) 2 * QH * QD), kr((size_t) KVH * QD), vr((size_t) KVH * QD);
    for (int o = 0; o < 2 * QH * QD; ++o) qg[(size_t) o] = dot(wq.f, N, o, h);
    for (int o = 0; o < KVH * QD; ++o) {
        kr[(size_t) o] = dot(wk.f, N, o, h);
        vr[(size_t) o] = dot(wv.f, N, o, h);
    }
    // norm + rope, per head (the ring row for this token, token-major)
    auto norm_rope = [&](double* y, int head_off, const std::vector<float>& wn) {
        double ss = 0.0;
        for (int e = 0; e < QD; ++e) ss += y[e] * y[e];
        const double rs = 1.0 / std::sqrt(ss / QD + (double) sh.eps);
        std::vector<double> r((size_t) QD);
        for (int e = 0; e < QD; ++e) r[(size_t) e] = y[e] * rs * (double) wn[(size_t) e];
        for (int e = 0; e < QD; ++e) {
            double o = r[(size_t) e];
            if (e < QROT) {   // NEOX pairs (i, i + 32) over the first 64 dims; everything past them is untouched
                const int i = e < HALF ? e : e - HALF;
                const double c = (double) cs[(size_t) (pos * HALF + i) * 2];
                const double s_ = (double) cs[(size_t) (pos * HALF + i) * 2 + 1];
                o = e < HALF ? r[(size_t) i] * c - r[(size_t) (i + HALF)] * s_
                             : r[(size_t) i] * s_ + r[(size_t) e] * c;
            }
            y[e] = o;
        }
        (void) head_off;
    };
    std::vector<double> q((size_t) QH * QD), kk((size_t) KVH * QD);
    for (int hd = 0; hd < KVH; ++hd)   // (this line was missing on the first run: with one KV row a uniform softmax ignores
        for (int e = 0; e < QD; ++e)   //  the keys entirely, so a reference with zeroed keys still matched - the same shape of
            kk[(size_t) hd * QD + e] = kr[(size_t) hd * QD + e];   // trap as the zero `scale` in q36_attn_parity)
    for (int hd = 0; hd < QH; ++hd) {   // (the queries come from the projection's FIRST half, as the kernel reads them)
        for (int e = 0; e < QD; ++e) q[(size_t) hd * QD + e] = qg[(size_t) hd * 2 * QD + e];   // per head [q | gate]
        norm_rope(&q[(size_t) hd * QD], pos, qn);   // the ropes are per POSITION: a hardcoded 0 matched the layer only at pos 0
    }
    for (int hd = 0; hd < KVH; ++hd) norm_rope(&kk[(size_t) hd * QD], pos, kn);   // the key of THIS token ropes at its position too

    // the window as the kernel reads it: earlier rows from the ring, this token's appended here
    std::vector<float> kwin((size_t) n_kv * KVH * QD), vwin((size_t) n_kv * KVH * QD);
    for (int r = 0; r < n_kv - 1; ++r) {
        std::memcpy(&kwin[(size_t) r * KVH * QD], &pre_k[(size_t) r * KVH * QD], 2 * (size_t) KVH * QD);
        for (int i = 0; i < KVH * QD; ++i) kwin[(size_t) (r * KVH * QD + i)] = from_bf16(pre_k[(size_t) (r * KVH * QD + i)]);
        for (int i = 0; i < KVH * QD; ++i) vwin[(size_t) (r * KVH * QD + i)] = from_bf16(pre_v[(size_t) (r * KVH * QD + i)]);
    }
    for (int hd = 0; hd < KVH; ++hd) {
        for (int e = 0; e < QD; ++e) kwin[(size_t) ((n_kv - 1) * KVH * QD + hd * QD + e)] = from_bf16(to_bf16((float) kk[(size_t) hd * QD + e]));
        for (int e = 0; e < QD; ++e) vwin[(size_t) ((n_kv - 1) * KVH * QD + hd * QD + e)] = from_bf16(to_bf16((float) vr[(size_t) hd * QD + e]));
    }

    const double scale = 1.0 / std::sqrt((double) QD);
    std::vector<double> a((size_t) QH * QD);
    for (int hd = 0; hd < QH; ++hd) {
        const int kvh = hd * KVH / QH;
        std::vector<double> sc((size_t) n_kv);
        double mx = -INFINITY;
        for (int r = 0; r < n_kv; ++r) {
            double d = 0.0;
            for (int e = 0; e < QD; ++e)
                d += q[(size_t) hd * QD + e] * (double) kwin[(size_t) (r * KVH * QD + kvh * QD + e)];
            sc[(size_t) r] = d * scale;
            mx = std::fmax(mx, sc[(size_t) r]);
        }
        double sum = 0.0;
        for (int r = 0; r < n_kv; ++r) {
            sc[(size_t) r] = std::exp(sc[(size_t) r] - mx);
            sum += sc[(size_t) r];
        }
        for (int e = 0; e < QD; ++e) {
            double acc = 0.0;
            for (int r = 0; r < n_kv; ++r) acc += sc[(size_t) r] * (double) vwin[(size_t) (r * KVH * QD + kvh * QD + e)];
            acc /= sum;
            acc *= 1.0 / (1.0 + std::exp(-qg[(size_t) (hd * 2 * QD + QD + e)]));   // the gate is the second QD of this head's row
            a[(size_t) hd * QD + e] = acc;
        }
    }
    std::vector<float> af((size_t) QH * QD);
    for (size_t i = 0; i < af.size(); ++i) af[i] = (float) a[i];
    std::vector<double> yref((size_t) N);
    for (int o = 0; o < N; ++o) yref[(size_t) o] = dot(wo.f, QH * QD, o, af);

    // WHERE it diverges: the ring's earlier rows as the kernel left them, then the attention output before the o projection
    if (n_kv > 1) {
        const uint16_t* ring = (const uint16_t*) s.kring.ptr;
        size_t pre_bad = 0;
        for (size_t i = 0; i < pre_k.size(); ++i)
            if (ring[i] != pre_k[i]) ++pre_bad;
        std::printf("    ring rows 0..%d intact: %s\n", n_kv - 2, pre_bad == 0 ? "yes" : "NO");
        const uint16_t* last = ring + (size_t) (n_kv - 1) * KVH * QD;
        size_t ap_bad = 0;
        for (int hd = 0; hd < KVH; ++hd)
            for (int e = 0; e < QD; ++e)
                if (last[(size_t) (hd * QD + e)] != to_bf16((float) kk[(size_t) hd * QD + e])) ++ap_bad;
        char b[96];
        std::snprintf(b, sizeof b, "    the appended row matches the reference's keys: %s (%zu of %d differ)", ap_bad == 0 ? "yes" : "NO",
                      ap_bad, KVH * QD);
        std::printf("%s\n", b);
        const float* gq = (const float*) s.q.ptr;
        double qn_ = 0.0, qd_ = 0.0;
        for (int i = 0; i < QH * QD; ++i) {
            const double e = (double) gq[i] - (double) q[(size_t) i];
            qn_ += e * e;
            qd_ += (double) q[(size_t) i] * (double) q[(size_t) i];
        }
        std::printf("    the queries after norm+rope: rel L2 %.3e\n", qd_ > 0 ? std::sqrt(qn_ / qd_) : std::sqrt(qn_));
        const float* ga = (const float*) s.a.ptr;
        double an = 0.0, ad = 0.0;
        for (size_t i = 0; i < af.size(); ++i) {
            const double e = (double) ga[i] - (double) af[i];
            an += e * e;
            ad += (double) af[i] * (double) af[i];
        }
        std::printf("    attention before the o projection: rel L2 %.3e\n", ad > 0 ? std::sqrt(an / ad) : std::sqrt(an));
    }
    const float* got = (const float*) s.y.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = (double) got[i] - yref[(size_t) i];
        num += e * e;
        den += yref[(size_t) i] * yref[(size_t) i];
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "attn layer: %d q / %d kv x %d, hidden %d, pos %d, %d KV rows", QH, KVH, QD, N, pos, n_kv);
    // the layer is measured at ~1.4e-07 against a double reference; the tolerance says so rather than being roomy
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 1e-5);
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    const strata::qwen36::AttnShape sh{N, QH, KVH, QD, 1e-6f};
    strata::qwen36::AttnKernels k;
    std::string err;
    if (!strata::qwen36::attn_init(cx, k, sh, err)) {
        std::printf("attn_init failed: %s\n", err.c_str());
        return 1;
    }
    // the first token of a conversation, and one with a window behind it
    for (const auto& c : {std::pair<int, int>{0, 1}, std::pair<int, int>{3, 4}, std::pair<int, int>{37, 6}})
        run_case(cx, k, c.first, c.second);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
