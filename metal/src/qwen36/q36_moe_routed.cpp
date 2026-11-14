// metal/src/qwen36/q36_moe_routed.cpp - the routed half's five dispatches.
#include "q36_moe_routed.h"

#include <cstdio>
#include <vector>

namespace strata::qwen36 {
namespace {

// moe.metal's MoeArgs, field for field (expert_parity's copy is the same shape)
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
// The plan's layout, moe.metal's constants: the two indirect grids sit at these words.
constexpr int P_IGU = 4, P_IDN = 8;

// The tiles expert_parity runs by default, and the sizes they imply.
struct Cfg {
    int L, R, S;
};
constexpr Cfg kGu{32, 2, 4}, kDn{8, 2, 4};
int rows_per_tg(Cfg c) { return c.S * (32 / c.L) * c.R; }
std::string tile(const char* kernel, int type, Cfg c) {
    return std::string(kernel) + "_" + std::to_string(type) + "_" + std::to_string(c.L) + "_" + std::to_string(c.R) + "_" +
           std::to_string(c.S);
}

}  // namespace

bool routed_init(mtl::Context& cx, RoutedKernels& k, const RoutedShape& sh, std::string& err, bool arena) {
    if (k.lib < 0) {
        try {
            // GEO_TOPK is the router's window (ROUTE_K): 8 for this family, and without it the default 10 reads past it
            k.lib = cx.compile({"quant.metal", "units.metal", "moe.metal", "glue.metal"}, "#define GEO_TOPK 8\n", false);
        } catch (const std::exception& e) {
            err = std::string("the MoE kernels did not compile: ") + e.what();
            return false;
        }
    }
    const int want = arena ? 1 : 0;
    if (k.arena == want) return true;   // (the two modes are different pipelines: function constant 20 changes the addressing)
    try {
        const mtl::Constants fc = mtl::Constants().b(20, arena);
        k.route = cx.pipeline(k.lib, "moe_route");
        k.gu = cx.pipeline(k.lib, tile("moe_gu", sh.blob.gu_type, kGu), fc);
        k.dn = cx.pipeline(k.lib, tile("moe_down", sh.blob.down_type, kDn), fc);
        k.combine = cx.pipeline(k.lib, "moe_combine_sh");
        k.arena = want;
    } catch (const std::exception& e) {
        err = std::string("a MoE kernel is missing for this model: ") + e.what();
        return false;
    }
    return true;
}

bool routed_scratch(mtl::Context& cx, RoutedScratch& s, const RoutedShape& sh, std::string& err) {
    if (s.logits.ptr != nullptr && s.n_embd == sh.blob.n_embd && s.n_ff == sh.blob.n_ff && s.n_expert == sh.n_expert &&
        s.top_k == sh.top_k && s.n_tok == sh.n_tok)
        return true;
    const size_t T = (size_t) sh.n_tok, K = (size_t) sh.top_k, E = T * K;
    s.logits = cx.alloc(4 * 512 * T);   // the router's row is 512 wide whatever the expert count (moe_route)
    s.ids = cx.alloc(4 * K * T);
    s.wts = cx.alloc(4 * K * T);
    s.plan = cx.alloc(4 * 4096);
    s.h = cx.alloc(4 * (size_t) sh.blob.n_ff * E);
    s.ye = cx.alloc(4 * (size_t) sh.blob.n_embd * E);
    for (const mtl::Buffer* b : {&s.logits, &s.ids, &s.wts, &s.plan, &s.h, &s.ye}) {
        if (b->ptr == nullptr) {
            err = "the device refused the MoE scratch";
            return false;
        }
    }
    s.n_embd = sh.blob.n_embd;
    s.n_ff = sh.blob.n_ff;
    s.n_expert = sh.n_expert;
    s.top_k = sh.top_k;
    s.n_tok = sh.n_tok;
    return true;
}

bool routed_forward(mtl::Context& cx, RoutedKernels& k, DenseKernels& dk, DenseScratch& ds, const RoutedShape& sh,
                    const RoutedWeights& w, RoutedScratch& s, const mtl::Buffer& x, const mtl::Buffer& ysh,
                    const mtl::Buffer& sg, mtl::Buffer& bo, std::string& err, const StreamFill* fill) {
    const bool arena = w.stream != nullptr || (fill != nullptr && fill->pool != nullptr);
    if (!routed_init(cx, k, sh, err, arena)) return false;
    if (!routed_scratch(cx, s, sh, err)) return false;
    const ExpertBlob& b = sh.blob;
    if (w.blobs == nullptr && w.stream == nullptr && (fill == nullptr || fill->pool == nullptr)) {
        err = "the routed experts have no buffers";
        return false;
    }
    const uint32_t n_ff = (uint32_t) b.n_ff, n_embd = (uint32_t) b.n_embd;
    if ((int) n_ff % rows_per_tg(kGu) != 0 || (int) n_embd % rows_per_tg(kDn) != 0) {
        err = "the expert widths are not a whole number of threadgroups for the tiles this port runs";
        return false;
    }

    // 1. the router's logits, at the 512-wide row stride moe_route reads
    if (!dense_gemv(cx, dk, w.router, x, s.logits, ds, err, 512)) return false;

    MoeArgs a{};
    a.gate_off = 0;
    a.up_off = b.up_off;
    a.down_off = b.down_off;
    a.gu_stride = b.blob;
    a.d_stride = b.blob;
    a.gu_row = (uint32_t) b.gu_row;
    a.d_row = (uint32_t) b.d_row;
    a.n_embd = n_embd;
    a.n_ff = n_ff;
    a.n_tok = (uint32_t) sh.n_tok;
    a.topk = (uint32_t) sh.top_k;
    a.n_expert = (uint32_t) sh.n_expert;
    a.gx_gu = n_ff / (uint32_t) rows_per_tg(kGu);
    a.gx_dn = n_embd / (uint32_t) rows_per_tg(kDn);

    // 2. route + plan (moe_route writes the plan itself, so there is no separate plan dispatch here)
    mtl::Dispatch dr;
    dr.p = &k.route;
    dr.args = {mtl::Arg::B(s.logits), mtl::Arg::B(s.ids), mtl::Arg::B(s.wts), mtl::Arg::B(s.plan), mtl::Arg::V(a)};
    dr.tg[0] = 512;
    dr.tg[1] = dr.tg[2] = 1;
    dr.grid[0] = 1;
    dr.grid[1] = dr.grid[2] = 1;
    cx.run({dr});

    // 2b. STREAMING: the router's picks go back to the host, the pool is filled with those experts, and the experts then run
    // through the arena.  cx.run() has already waited, so the ids are the ones this token routed to.
    RoutedWeights we = w;
    if (fill != nullptr && fill->pool != nullptr) {
        if (fill->m == nullptr || fill->L == nullptr) {
            err = "a streamed block needs the model and the layer to fill from";
            return false;
        }
        std::vector<int> picked;
        stream_ids(s.ids, sh.n_tok, sh.top_k, picked);
        if ((int) picked.size() > fill->pool->n_slot) {
            err = "a token routed to more experts than the pool holds";
            return false;
        }
        if (!picked.empty()) {
            err.clear();
            const int loaded = stream_load(*fill->m, *fill->L, *fill->pool, picked, err);
            if (loaded <= 0) {   // (0 with no message is a bug in the caller, so say which)
                if (err.empty()) err = "the pool filled nothing from the router's picks";
                return false;
            }
        }
        we.stream = fill->pool;
    }

    // 3. the grouped experts, their grids read out of the plan
    const size_t plane = (size_t) n_embd * 4;
    // In streaming mode the pool is buffer 0, the slot map is buffer 6, and W1..W3 are the pool again (a pool that spans more
    // than one buffer would use them; this one does not, so they are bound to the same buffer and never addressed).
    const mtl::Buffer& pool = we.stream != nullptr ? we.stream->slots : *we.blobs;
    const size_t pool_off = we.stream != nullptr ? 0 : we.blobs_off;
    const mtl::Buffer& smap = we.stream != nullptr ? we.stream->smap : s.plan;   // (unread when the arena is off)
    mtl::Dispatch dg;
    dg.p = &k.gu;
    dg.args = {mtl::Arg::B(pool, pool_off), mtl::Arg::B(ds.xp, plane * 0), mtl::Arg::B(s.h), mtl::Arg::B(s.plan),
               mtl::Arg::V(a), mtl::Arg::B(ds.xp, plane * 4), mtl::Arg::B(smap), mtl::Arg::B(pool), mtl::Arg::B(pool),
               mtl::Arg::B(pool)};
    dg.tg[0] = (uint32_t) (kGu.S * 32);
    dg.tg[1] = dg.tg[2] = 1;
    dg.indirect = &s.plan;
    dg.ind_off = (size_t) P_IGU * 4;

    mtl::Dispatch dd;
    dd.p = &k.dn;
    dd.args = {mtl::Arg::B(pool, pool_off), mtl::Arg::B(s.h), mtl::Arg::B(s.ye), mtl::Arg::B(s.plan), mtl::Arg::V(a),
               mtl::Arg::B(ds.xp), mtl::Arg::B(smap), mtl::Arg::B(pool), mtl::Arg::B(pool), mtl::Arg::B(pool)};
    dd.tg[0] = (uint32_t) (kDn.S * 32);
    dd.tg[1] = dd.tg[2] = 1;
    dd.indirect = &s.plan;
    dd.ind_off = (size_t) P_IDN * 4;
    cx.run({dg, dd});

    // 4. the combine: the routed picks, plus sigmoid(sg) times the shared expert
    struct GlueArgs {
        uint32_t n, n_tok, n_embd, topk;
    } g{n_embd, (uint32_t) sh.n_tok, n_embd, (uint32_t) sh.top_k};
    mtl::Dispatch dc;
    dc.p = &k.combine;
    dc.args = {mtl::Arg::B(s.ye), mtl::Arg::B(s.plan), mtl::Arg::B(ysh), mtl::Arg::B(sg), mtl::Arg::B(bo), mtl::Arg::V(g)};
    dc.tg[0] = 64;
    dc.tg[1] = dc.tg[2] = 1;
    dc.grid[0] = (uint32_t) ((n_embd + 63) / 64);
    dc.grid[1] = dc.grid[2] = 1;
    cx.run({dc});
    return true;
}

}  // namespace strata::qwen36
