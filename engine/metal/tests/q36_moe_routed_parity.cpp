// metal/tests/q36_moe_routed_parity.cpp - q36_routed_forward (router → plan → grouped experts → combine) against a double
// reference over the same blobs: one token, 256 experts, top-8, Q8_0.
//
// moe_route is checked at this shape already (q36_router_parity), and moe_gu / moe_down are checked against double in
// expert_parity; what only the whole path can catch is the plumbing between them - the 512-wide logits stride, the plan the
// route kernel writes and the indirect grids read, the blob offsets, the prepared activation the gate/up reads, and the
// combine's plan lookup.
#include "qwen36/q36_moe_routed.h"

#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
// NFF is the PACK's FF (512), not the fixture's 128: the expert kernels template on the row counts, and a partial reduction that is
// right at 128 was measured wrong at 512 on the real Q8_0 pack (round 60: the engine's expert output differed from a double reference
// by factors of 3 to -111, which is a different sum rather than a scaled one).  A parity test at the real width is the only thing that
// can catch that class.
constexpr int N = 2048, NFF = 512, NE = 256, TOPK = 8;   // the PACK's own numbers: n_embd 2048, FF 512, 256 experts top-8

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-78s rel L2 %.3e (tol %.1e)  %s\n", what.c_str(), err, tol, ok ? "PASS" : "FAIL");
    g_fail += !ok;
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    strata::qwen36::DenseKernels dk;
    strata::qwen36::DenseScratch ds;
    std::string err;
    if (!strata::qwen36::dense_init(cx, dk, err)) { std::printf("dense_init: %s\n", err.c_str()); return 1; }

    strata::qwen36::ExpertBlob b;
    b.n_embd = N;
    b.n_ff = NFF;
    b.gu_type = 8;
    b.down_type = 8;
    b.shape();

    // the three stacked tensors as the file holds them, and the packed slots the kernel reads
    std::vector<uint8_t> gate((size_t) (b.n_ff * b.gu_row * NE)), up(gate.size()), down((size_t) (b.n_embd * b.d_row * NE));
    ref::random_blocks(8, gate.data(), NFF * NE, N, 41, 0.02f);
    ref::random_blocks(8, up.data(), NFF * NE, N, 42, 0.02f);
    ref::random_blocks(8, down.data(), N * NE, NFF, 43, 0.02f);
    mtl::Buffer slots = cx.alloc((size_t) b.slots(NE));
    for (int e = 0; e < NE; ++e)
        if (!strata::qwen36::pack(b, e, gate.data(), up.data(), down.data(), slots.as<uint8_t>() + (size_t) e * b.blob, err)) {
            std::printf("pack: %s\n", err.c_str());
            return 1;
        }

    // the router
    std::vector<uint8_t> rbytes((size_t) ref::row_bytes(8, N) * NE);
    ref::random_blocks(8, rbytes.data(), NE, N, 44, 0.02f);
    std::vector<float> rf((size_t) N * NE);
    ref::dequant(8, rbytes.data(), rf.data(), rf.size());
    mtl::Buffer br = cx.alloc(rbytes.size());
    std::memcpy(br.ptr, rbytes.data(), rbytes.size());

    std::vector<float> x((size_t) N), ysh((size_t) N, 0.0f), sgv(1, 0.0f);
    for (int i = 0; i < N; ++i) x[(size_t) i] = 1.2f * std::sin(0.017f * (float) i + 0.5f);
    mtl::Buffer bx = cx.alloc(4 * (size_t) N), bysh = cx.alloc(4 * (size_t) N), bsg = cx.alloc(4), bo = cx.alloc(4 * (size_t) N);
    std::memcpy(bx.ptr, x.data(), 4 * (size_t) N);
    std::memset(bysh.ptr, 0, 4 * (size_t) N);   // the shared expert silent: the routed sum alone
    std::memset(bsg.ptr, 0, 4);

    strata::qwen36::RoutedShape sh{b, NE, TOPK, 1};
    strata::qwen36::RoutedWeights w{&slots, 0, nullptr, strata::qwen36::DenseMatrix{&br, 0, 8, N, NE}};
    strata::qwen36::RoutedScratch s;
    strata::qwen36::RoutedKernels k;
    if (!strata::qwen36::routed_forward(cx, k, dk, ds, sh, w, s, bx, bysh, bsg, bo, err)) {
        std::printf("routed_forward: %s\n", err.c_str());
        return 1;
    }

    // the reference: the same routing (softmax, top-8, renormalised), then the selected experts' arithmetic in double
    std::vector<double> logits((size_t) NE);
    double mx = -INFINITY;
    for (int e = 0; e < NE; ++e) {
        double d = 0.0;
        for (int i = 0; i < N; ++i) d += (double) rf[(size_t) e * N + i] * (double) x[(size_t) i];
        logits[(size_t) e] = d;
        mx = std::fmax(mx, d);
    }
    double sum = 0.0;
    for (int e = 0; e < NE; ++e) { logits[(size_t) e] = std::exp(logits[(size_t) e] - mx); sum += logits[(size_t) e]; }
    std::vector<int> pick;
    std::vector<char> used((size_t) NE, 0);
    for (int r = 0; r < TOPK; ++r) {
        int best = -1;
        for (int e = 0; e < NE; ++e)
            if (!used[(size_t) e] && (best < 0 || logits[(size_t) e] > logits[(size_t) best])) best = e;
        used[(size_t) best] = 1;
        pick.push_back(best);
    }
    double ssum = 0.0;
    for (int r = 0; r < TOPK; ++r) ssum += logits[(size_t) pick[(size_t) r]];
    std::vector<double> want((size_t) N, 0.0);
    for (int r = 0; r < TOPK; ++r) {
        const int e = pick[(size_t) r];
        const double wt = logits[(size_t) e] / ssum;
        std::vector<float> gf((size_t) NFF * N), uf((size_t) NFF * N), df((size_t) N * NFF);
        ref::dequant(8, gate.data() + (size_t) e * NFF * b.gu_row, gf.data(), gf.size());
        ref::dequant(8, up.data() + (size_t) e * NFF * b.gu_row, uf.data(), uf.size());
        ref::dequant(8, down.data() + (size_t) e * N * b.d_row, df.data(), df.size());
        std::vector<double> h((size_t) NFF);
        for (int o = 0; o < NFF; ++o) {
            double g = 0.0, u = 0.0;
            for (int i = 0; i < N; ++i) {
                g += (double) gf[(size_t) o * N + i] * (double) x[(size_t) i];
                u += (double) uf[(size_t) o * N + i] * (double) x[(size_t) i];
            }
            h[(size_t) o] = g / (1.0 + std::exp(-g)) * u;
        }
        for (int o = 0; o < N; ++o) {
            double acc = 0.0;
            for (int i = 0; i < NFF; ++i) acc += (double) df[(size_t) o * NFF + i] * h[(size_t) i];
            want[(size_t) o] += wt * acc;
        }
    }

    const float* got = (const float*) bo.ptr;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; ++i) {
        const double e = (double) got[i] - want[(size_t) i];
        num += e * e;
        den += want[(size_t) i] * want[(size_t) i];
    }
    char nm[128];
    std::snprintf(nm, sizeof nm, "routed MoE: %d experts top-%d, blob %.1f KiB, 1 token", NE, TOPK, (double) b.blob / 1024.0);
    report(nm, den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-4);

    // THE SAME EXPERTS THROUGH THE ARENA, against the resident answer above: this is what the streaming path rests on - a slot
    // loaded a moment ago must give the arithmetic the resident pack gives, to the last bit that float arithmetic allows.
    {
        // the experts the resident run routed to: read them off its ids, which the router wrote
        std::vector<int> picked;
        strata::qwen36::stream_ids(s.ids, 1, TOPK, picked);
        strata::qwen36::ExpertStream stream;
        if (!strata::qwen36::stream_init(cx, stream, b, (int) picked.size(), NE, err)) {
            std::printf("stream_init: %s\n", err.c_str());
            return 1;
        }
        const int loaded = strata::qwen36::stream_load_raw(b, stream, picked, gate.data(), up.data(), down.data(), err);
        if (loaded != (int) picked.size()) {
            std::printf("stream_load: %s\n", err.c_str());
            return 1;
        }
        strata::qwen36::RoutedWeights w2{nullptr, 0, &stream, strata::qwen36::DenseMatrix{&br, 0, 8, N, NE}};
        strata::qwen36::RoutedScratch s2;
        strata::qwen36::RoutedKernels k2;
        mtl::Buffer bo2 = cx.alloc(4 * (size_t) N);
        if (!strata::qwen36::routed_forward(cx, k2, dk, ds, sh, w2, s2, bx, bysh, bsg, bo2, err)) {
            std::printf("routed_forward (arena): %s\n", err.c_str());
            return 1;
        }
        const float* g2 = (const float*) bo2.ptr;
        double n2 = 0.0, d2 = 0.0;
        for (int i = 0; i < N; ++i) {
            const double e = (double) g2[i] - (double) got[i];
            n2 += e * e;
            d2 += (double) got[i] * (double) got[i];
        }
        report("the same token through the ARENA gives the resident answer", d2 > 0 ? std::sqrt(n2 / d2) : std::sqrt(n2), 2e-6);
    }
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
