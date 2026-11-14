// metal/tests/q36_layer_check.cpp - one whole decoder layer, with every weight zeroed except the norms.
//
// With a zero mixer and a zero MoE the layer's answer is known exactly: the mixer adds nothing, the shared expert and the
// router are zero so the combine is zero, and the residual comes out equal to what went in.  What that pins is the ORDER and the
// norms - the two norm weights reach q36_add_rms, the first join adds nothing, the second adds the mixer's (zero) output, and
// the MoE's (zero) output lands back in the residual through the plain add.  An order error, a missing norm, a buffer bound to
// the wrong place, or a norm applied twice all move the answer away from x.
//
// The parts themselves are checked elsewhere against double references (attention 1.4e-07, GDN 2.3e-07, shared 1.58e-07,
// routed 1.75e-07); this is the wiring between them.
#include "qwen36/q36_layer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int N = 512, HK = 2, HV = 4, S = 128, NFF = 128, NE = 256, TOPK = 8;

void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

mtl::Buffer zeros(mtl::Context& cx, size_t bytes) {
    mtl::Buffer b = cx.alloc(bytes);
    if (b.ptr != nullptr) std::memset(b.ptr, 0, bytes);
    return b;
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    strata::qwen36::DenseKernels dk;
    strata::qwen36::DenseScratch ds;
    std::string err;
    if (!strata::qwen36::dense_init(cx, dk, err)) { std::printf("dense_init: %s\n", err.c_str()); return 1; }

    strata::qwen36::GdnShape gsh{N, HK, HV, S, 1e-6f};
    strata::qwen36::ExpertBlob blob;
    blob.n_embd = N;
    blob.n_ff = NFF;
    blob.gu_type = 8;
    blob.down_type = 8;
    blob.shape();

    // every matrix zero (Q8_0 blocks of zero are all-zero bytes: d = 0), the norms and the 1-D tensors real
    auto q8zero = [&](mtl::Context& c, int n_in, int n_out) { return zeros(c, (size_t) (n_in / 32 * 34) * (size_t) n_out); };
    mtl::Buffer wqkv = q8zero(cx, N, gsh.channels()), wgate = q8zero(cx, N, gsh.value_dim()), wa = q8zero(cx, N, HV),
                wb = q8zero(cx, N, HV), wout = q8zero(cx, gsh.value_dim(), N), router = q8zero(cx, N, NE),
                shg = q8zero(cx, N, NFF), shu = q8zero(cx, N, NFF), shd = q8zero(cx, NFF, N),
                shexp = q8zero(cx, N, 1), slots = zeros(cx, (size_t) blob.slots(NE));
    std::vector<float> cw((size_t) gsh.channels() * 4, 0.0f), dtv((size_t) HV, 0.0f), sav((size_t) HV, 0.0f),
        gmv((size_t) S), an((size_t) N), pn((size_t) N);
    for (int i = 0; i < S; ++i) gmv[(size_t) i] = 1.0f;
    for (int i = 0; i < N; ++i) {
        an[(size_t) i] = 0.9f + 0.1f * std::sin(0.01f * (float) i);
        pn[(size_t) i] = 0.8f + 0.2f * std::cos(0.013f * (float) i);
    }
    mtl::Buffer bcw = cx.alloc(4 * cw.size()), bdt = cx.alloc(4 * dtv.size()), bsa = cx.alloc(4 * sav.size()),
                bgm = cx.alloc(4 * gmv.size()), ban = cx.alloc(4 * an.size()), bpn = cx.alloc(4 * pn.size());
    std::memcpy(bcw.ptr, cw.data(), 4 * cw.size());
    std::memcpy(bdt.ptr, dtv.data(), 4 * dtv.size());
    std::memcpy(bsa.ptr, sav.data(), 4 * sav.size());
    std::memcpy(bgm.ptr, gmv.data(), 4 * gmv.size());
    std::memcpy(ban.ptr, an.data(), 4 * an.size());
    std::memcpy(bpn.ptr, pn.data(), 4 * pn.size());

    strata::qwen36::LayerWeights w;
    w.attention = false;
    w.gdn = strata::qwen36::GdnWeights{{&wqkv, 0, 8, N, gsh.channels()}, {&wgate, 0, 8, N, gsh.value_dim()},
                                       {&wa, 0, 8, N, HV},          {&wb, 0, 8, N, HV},
                                       {&wout, 0, 8, gsh.value_dim(), N}, &bcw, 0, &bdt, 0, &bsa, 0, &bgm, 0};
    w.attn_norm = &ban;
    w.post_norm = &bpn;
    w.gate_inp = strata::qwen36::DenseMatrix{&router, 0, 8, N, NE};
    w.gate_inp_shexp = strata::qwen36::DenseMatrix{&shexp, 0, 8, N, 1};
    w.shared = strata::qwen36::SharedWeights{{&shg, 0, 8, N, NFF}, {&shu, 0, 8, N, NFF}, {&shd, 0, 8, NFF, N}};

    strata::qwen36::GdnScratch gsc;
    strata::qwen36::SharedScratch ssc;
    strata::qwen36::RoutedScratch rsc;
    strata::qwen36::GdnKernels gk;
    strata::qwen36::RoutedKernels rk;
    strata::qwen36::LayerKernels lk;
    strata::qwen36::LayerScratch ls;
    mtl::Buffer state, hist;
    const strata::qwen36::RoutedShape rsh{blob, NE, TOPK, 1};
    strata::qwen36::LayerIo io;
    io.gdn = gsh;
    io.shared = strata::qwen36::SharedShape{N, NFF};
    io.routed = rsh;
    io.gdn_scratch = &gsc;
    io.shared_scratch = &ssc;
    io.routed_scratch = &rsc;
    io.gdn_kernels = &gk;
    io.routed_kernels = &rk;
    io.state = &state;
    io.hist = &hist;
    io.slots = &slots;
    if (!strata::qwen36::gdn_state(cx, gsh, state, hist, err) || !strata::qwen36::gdn_scratch(cx, gsc, gsh, err) ||
        !strata::qwen36::routed_scratch(cx, rsc, rsh, err) || !strata::qwen36::layer_init(cx, lk, err) ||
        !strata::qwen36::layer_scratch(cx, ls, N, err)) {
        std::printf("setup: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> x((size_t) N), x0;
    for (int i = 0; i < N; ++i) x[(size_t) i] = 1.3f * std::sin(0.017f * (float) i + 0.3f);
    x0 = x;
    mtl::Buffer bx = cx.alloc(4 * (size_t) N);
    std::memcpy(bx.ptr, x.data(), 4 * (size_t) N);

    const bool first = true;
    if (!strata::qwen36::layer_forward(cx, lk, dk, ds, w, io, ls, bx, first, err)) {
        std::printf("layer_forward: %s\n", err.c_str());
        return 1;
    }

    // the residual must be exactly what went in: the mixer is zero, the shared expert is zero, the router is zero, so the combine
    // is sigmoid(0) * 0 = 0 and there is nothing to add
    const float* got = (const float*) bx.ptr;
    size_t diff = 0;
    for (int i = 0; i < N; ++i)
        if (got[i] != x0[(size_t) i]) ++diff;
    char nm[128];
    std::snprintf(nm, sizeof nm, "with a zero mixer and a zero MoE, the residual comes back unchanged (%zu of %d differ)", diff, N);
    check(diff == 0, nm);

    // and the MoE's normed row is rms(x) * post_norm, in double
    double ss = 0.0;
    for (int i = 0; i < N; ++i) ss += (double) x0[(size_t) i] * (double) x0[(size_t) i];
    const double rs = 1.0 / std::sqrt(ss / N + 1e-6);
    const float* h = (const float*) ls.h.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const double want = (double) x0[(size_t) i] * rs * (double) pn[(size_t) i];
        num += ((double) h[i] - want) * ((double) h[i] - want);
        den += want * want;
    }
    char nm2[128];
    std::snprintf(nm2, sizeof nm2, "the MoE's normed row is rms(x) * post_attention_norm (rel L2 %.2e)", std::sqrt(num / den));
    check(std::sqrt(num / den) <= 2e-5, nm2);

    // A SECOND layer over the same scratch, with the first layer's output as the residual: the parts share one AttnScratch,
    // GdnScratch, SharedScratch, RoutedScratch and LayerScratch, so a buffer one part still reads while the next writes it would
    // show here and nowhere else (each part is checked alone elsewhere).  The weights are still zero, so the answer is still x.
    std::vector<float> x1((size_t) N);
    std::memcpy(x1.data(), bx.ptr, 4 * (size_t) N);
    if (!strata::qwen36::layer_forward(cx, lk, dk, ds, w, io, ls, bx, false, err)) {
        std::printf("layer_forward (second): %s\n", err.c_str());
        return 1;
    }
    const float* got2 = (const float*) bx.ptr;
    size_t diff2 = 0;
    for (int i = 0; i < N; ++i)
        if (got2[i] != x1[(size_t) i]) ++diff2;
    char nm3[128];
    std::snprintf(nm3, sizeof nm3, "a second layer over the same scratch leaves the residual alone too (%zu of %d differ)", diff2, N);
    check(diff2 == 0, nm3);

    std::printf("%s\n", g_fail ? "q36_layer_check: FAILURES" : "q36_layer_check: all ok");
    return g_fail ? 1 : 0;
}
