// metal/tests/q36_router_parity.cpp - moe.metal's route kernel at Qwen3.6's routing shape (256 experts, top-8), against the
// rule the engine's own kernels use, with the row stride the kernel actually reads.
//
// Two things here are not obvious from q36's metadata and would mis-route silently:
//
//   * the kernel reads the logits at a row stride of 512 whatever the expert count (moe.metal: `logits + sg * 512`), and
//     route_tok<8> reads only lane + 32*i for i < 8, i.e. the first 256 values.  So a 256-expert model must place its router
//     output at 512 floats a token and leave 256..511 alone - this test fills that half with garbage large enough to win any
//     top-k if it were read, and checks that it is not;
//   * ROUTE_K comes from GEO_TOPK (the host prelude), so the library is compiled with "#define GEO_TOPK 8" - without it the
//     default 10 would read the weights of a 10-expert router out of an 8-wide window.
//
// The rule: softmax over all experts, take the top k (ties to the lower index), renormalise over the selected with a 2^-14
// floor - the same rule route_tok gives Flash-Next at 512/10.
#include "mtl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
constexpr int T = 4, NEXP = 256, ROW = 512, TOPK = 8;

void report(const std::string& what, double err, double tol) {
    const bool ok = err <= tol;
    std::printf("%-78s rel L2 %.3e (tol %.1e)  %s\n", what.c_str(), err, tol, ok ? "PASS" : "FAIL");
    g_fail += !ok;
}

struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    std::vector<float> logits((size_t) T * ROW), ids_ref((size_t) T * TOPK), w_ref((size_t) T * TOPK);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < NEXP; ++e) logits[(size_t) (t * ROW + e)] = 3.0f * std::sin(0.37f * (float) (e + 11 * t) + 0.2f);
        // the half the kernel must not read: large enough that any top-k would pick from it
        for (int e = NEXP; e < ROW; ++e) logits[(size_t) (t * ROW + e)] = 50.0f;
        // the reference: softmax over the 256, top-8, renormalised
        std::vector<double> p((size_t) NEXP);
        double mx = -INFINITY;
        for (int e = 0; e < NEXP; ++e) mx = std::fmax(mx, (double) logits[(size_t) (t * ROW + e)]);
        double sum = 0.0;
        for (int e = 0; e < NEXP; ++e) {
            p[(size_t) e] = std::exp((double) logits[(size_t) (t * ROW + e)] - mx);
            sum += p[(size_t) e];
        }
        std::vector<double> top(p);
        std::sort(top.begin(), top.end(), std::greater<double>());
        double ssum = 0.0;
        for (int r = 0; r < TOPK; ++r) ssum += top[(size_t) r];
        // the ids the reference picks: the r-largest p, ties to the lower index
        std::vector<int> pick;
        std::vector<char> used((size_t) NEXP, 0);
        for (int r = 0; r < TOPK; ++r) {
            int best = -1;
            for (int e = 0; e < NEXP; ++e)
                if (!used[(size_t) e] && (best < 0 || p[(size_t) e] > p[(size_t) best])) best = e;
            used[(size_t) best] = 1;
            pick.push_back(best);
        }
        for (int r = 0; r < TOPK; ++r) {
            ids_ref[(size_t) (t * TOPK + r)] = (float) pick[(size_t) r];
            w_ref[(size_t) (t * TOPK + r)] = p[(size_t) pick[(size_t) r]] / ssum;
        }
    }

    mtl::Buffer bl = cx.alloc(4 * logits.size()), bids = cx.alloc(4 * (size_t) T * TOPK), bw = cx.alloc(4 * (size_t) T * TOPK),
                bplan = cx.alloc(4 * 4096);
    std::memcpy(bl.ptr, logits.data(), 4 * logits.size());
    MoeArgs a{};
    a.n_embd = 2048;
    a.n_ff = 512;
    a.n_tok = T;
    a.topk = TOPK;
    a.n_expert = NEXP;
    a.gx_gu = 1;
    a.gx_dn = 1;
    mtl::Pipeline p;
    try {
        p = cx.pipeline(cx.compile({"quant.metal", "units.metal", "moe.metal"}, "#define GEO_TOPK 8\n"), "moe_route");
    } catch (const std::exception& e) {
        std::printf("no moe_route: %s\n", e.what());
        return 1;
    }
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(bl), mtl::Arg::B(bids), mtl::Arg::B(bw), mtl::Arg::B(bplan), mtl::Arg::V(a)};
    d.tg[0] = 512;
    d.grid[0] = 1;
    cx.run({d});

    const int* ids = (const int*) bids.ptr;
    const float* w = (const float*) bw.ptr;
    size_t id_bad = 0, id_max = 0;
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < TOPK; ++r) {
            const int want = (int) ids_ref[(size_t) (t * TOPK + r)];
            const int got = ids[(size_t) (t * TOPK + r)];
            if (got != want) ++id_bad;
            if (got > id_max) id_max = (size_t) got;
        }
    report("moe_route 256 experts top-8: every id matches the reference (ties to the lower index)",
           id_bad == 0 ? 0.0 : 1.0, 0.0);
    std::printf("    %zu of %d ids differ; the largest id chosen is %zu (a choice from 256..511 would prove the far half was read)\n",
                id_bad, T * TOPK, id_max);
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < w_ref.size(); ++i) {
        const double e = (double) w[i] - w_ref[i];
        num += e * e;
        den += w_ref[i] * w_ref[i];
    }
    report("moe_route 256 experts top-8: the weights", den > 0 ? std::sqrt(num / den) : std::sqrt(num), 2e-5);
    std::printf("%s\n", g_fail ? "FAILURES" : "ALL PASS");
    return g_fail ? 1 : 0;
}
