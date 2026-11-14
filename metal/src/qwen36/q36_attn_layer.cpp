// metal/src/qwen36/q36_attn_layer.cpp - the attention layer's wiring: the measured parts, in q36_engine.cu's order.
#include "q36_attn_layer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::qwen36 {
namespace {

std::string attn_name(const AttnShape& s) {
    return "q36_attn_" + std::to_string(s.n_head) + "_" + std::to_string(s.n_kv_head) + "_" + std::to_string(s.head_dim);
}

}  // namespace

bool attn_init(mtl::Context& cx, AttnKernels& k, const AttnShape& sh, std::string& err) {
    if (k.lib < 0) {
        try {
            k.lib = cx.compile({"../src/qwen36/msl/q36_rope.metal", "../src/qwen36/msl/q36_attn.metal"}, "", false);
        } catch (const std::exception& e) {
            err = std::string("the q36 attention kernels did not compile: ") + e.what();
            return false;
        }
    }
    if (k.rope_f32.handle == nullptr || k.rope_bf16.handle == nullptr || k.store.handle == nullptr ||
        k.attn.handle == nullptr) {
        try {
            k.rope_f32 = cx.pipeline(k.lib, "q36_head_norm_rope_f32");
            k.rope_bf16 = cx.pipeline(k.lib, "q36_head_norm_rope_bf16");
            k.store = cx.pipeline(k.lib, "q36_store_bf16");
            k.attn = cx.pipeline(k.lib, attn_name(sh));
        } catch (const std::exception& e) {
            // the shape's attention kernel is instantiated per (heads, kv heads, width): name what is missing
            err = "this port has no kernel for " + attn_name(sh) + " (" + k.rope_f32.name + " " + e.what() + ")";
            return false;
        }
    }
    // the rope kernel stages a whole head row in threadgroup memory and dispatches head_dim threads
    if (k.rope_f32.max_threads < sh.head_dim) {
        err = "q36_head_norm_rope_f32 takes " + std::to_string(k.rope_f32.max_threads) + " threads a threadgroup, a head is " +
              std::to_string(sh.head_dim);
        return false;
    }
    return true;
}

bool attn_scratch(mtl::Context& cx, AttnScratch& s, const AttnShape& sh, int kv_cap, std::string& err) {
    if (s.qg.ptr != nullptr && s.n_embd == sh.n_embd && s.n_head == sh.n_head && s.n_kv_head == sh.n_kv_head &&
        s.head_dim == sh.head_dim && s.cap == kv_cap)
        return true;
    const size_t hq = (size_t) sh.n_head * sh.head_dim, hkv = (size_t) sh.n_kv_head * sh.head_dim;
    s.kring = cx.alloc(2 * (size_t) kv_cap * hkv);
    s.vring = cx.alloc(2 * (size_t) kv_cap * hkv);
    s.qg = cx.alloc(4 * 2 * hq);
    s.k_raw = cx.alloc(4 * hkv);
    s.v_raw = cx.alloc(4 * hkv);
    s.q = cx.alloc(4 * hq);
    s.a = cx.alloc(4 * hq);
    s.y = cx.alloc(4 * (size_t) sh.n_embd);
    for (const mtl::Buffer* b : {&s.qg, &s.k_raw, &s.v_raw, &s.q, &s.kring, &s.vring, &s.a, &s.y}) {
        if (b->ptr == nullptr) {
            err = "the device refused the attention scratch";
            return false;
        }
    }
    s.n_embd = sh.n_embd;
    s.n_head = sh.n_head;
    s.n_kv_head = sh.n_kv_head;
    s.head_dim = sh.head_dim;
    s.cap = kv_cap;
    return true;
}

bool attn_forward(mtl::Context& cx, AttnKernels& k, DenseKernels& dk, DenseScratch& ds, const AttnShape& sh,
                  const AttnWeights& w, AttnScratch& s, const mtl::Buffer& h, const mtl::Buffer& cs, int pos, int n_kv,
                  std::string& err) {
    if (!attn_init(cx, k, sh, err)) return false;
    if (n_kv < 1) {
        err = "attn_forward needs at least one KV row (the token's own)";
        return false;
    }
    if (s.qg.ptr == nullptr || s.cap < n_kv) {
        if (!attn_scratch(cx, s, sh, n_kv > s.cap ? n_kv : s.cap, err)) return false;
    }
    const uint32_t QD = (uint32_t) sh.head_dim, QH = (uint32_t) sh.n_head, KVH = (uint32_t) sh.n_kv_head;
    const size_t hq = (size_t) QH * QD, hkv = (size_t) KVH * QD;
    const size_t row = (size_t) (n_kv - 1) * hkv;   // this token's row, token-major

    // 1. the three input projections, all against the same prepared activation
    if (std::getenv("Q36_TRACE_H")) {   // THE ROW THIS LAYER ACTUALLY PROJECTS: the one number that settles the wiring question
        const float* hh = (const float*) h.ptr;
        std::printf("attn_forward h[0..2] = %g %g %g   (dumped s.h was -0.3494 -0.1587 1.5184)\n", (double) hh[0], (double) hh[1],
                    (double) hh[2]);
        std::fflush(stdout);
    }
    if (!dense_gemv(cx, dk, w.q, h, s.qg, ds, err)) return false;
    if (std::getenv("Q36_TRACE_QG")) {   // qg[0] the moment it is produced: 0.331396 means someone overwrites s.qg later, 5.04 means two
        std::printf("attn_forward qg[0] = %.6g after the gemv\n", (double) ((const float*) s.qg.ptr)[0]);   // identical gemvs disagree
        std::fflush(stdout);
    }
    if (!dense_gemv(cx, dk, w.k, h, s.k_raw, ds, err)) return false;
    if (!dense_gemv(cx, dk, w.v, h, s.v_raw, ds, err)) return false;

    // 2. the queries: normed per head, roped, fp32 (the gate stays where the projection put it, second in s.qg)
    mtl::Dispatch dq;
    dq.p = &k.rope_f32;
    dq.args = {mtl::Arg::B(s.qg), mtl::Arg::B(*w.q_norm, w.q_norm_off), mtl::Arg::B(cs), mtl::Arg::B(s.q),
               mtl::Arg::V((uint32_t) pos), mtl::Arg::V(sh.eps)};
    dq.grid[0] = QH;
    dq.grid[1] = dq.grid[2] = 1;
    dq.tg[0] = QD;
    dq.tg[1] = dq.tg[2] = 1;

    // 3. the keys and values into this token's ring row: one dispatch each, because the row is token-major
    mtl::Dispatch dk_;
    dk_.p = &k.rope_bf16;
    dk_.args = {mtl::Arg::B(s.k_raw), mtl::Arg::B(*w.k_norm, w.k_norm_off), mtl::Arg::B(cs), mtl::Arg::B(s.kring, row * 2),
                mtl::Arg::V((uint32_t) pos), mtl::Arg::V(sh.eps)};
    dk_.grid[0] = KVH;
    dk_.grid[1] = dk_.grid[2] = 1;
    dk_.tg[0] = QD;
    dk_.tg[1] = dk_.tg[2] = 1;

    mtl::Dispatch dv;
    dv.p = &k.store;
    dv.args = {mtl::Arg::B(s.v_raw), mtl::Arg::B(s.vring, row * 2), mtl::Arg::V((uint32_t) hkv)};
    dv.tg[0] = 64;
    dv.tg[1] = dv.tg[2] = 1;
    dv.grid[0] = (uint32_t) ((hkv + 63) / 64);
    dv.grid[1] = dv.grid[2] = 1;

    // 4. attention over the window, the output gate folded in (sigmoid(gate) * out).  The gate is the SECOND QD of each head's
    // row in the q projection - per head [q | gate], not all-q-then-all-gate - so the kernel is handed the projection itself and
    // indexes (h * 2 + 1) * QD.
    const float scale = 1.0f / std::sqrt((float) QD);
    mtl::Dispatch da;
    da.p = &k.attn;
    da.args = {mtl::Arg::B(s.q), mtl::Arg::B(s.kring), mtl::Arg::B(s.vring), mtl::Arg::B(s.qg, 0),
               mtl::Arg::B(s.a), mtl::Arg::V((uint32_t) n_kv), mtl::Arg::V(1u), mtl::Arg::V(scale)};
    da.grid[0] = 1;
    da.grid[1] = da.grid[2] = 1;
    da.tg[0] = QH * 32;   // one 32-lane group a query head
    da.tg[1] = da.tg[2] = 1;

    cx.run({dq, dk_, dv, da});
    if (std::getenv("Q36_TRACE_QG")) { std::printf("  qg[0] after the group = %.6g\n", (double) ((const float*) s.qg.ptr)[0]); std::fflush(stdout); }
    if (const char* qp = std::getenv("Q36_DUMP_QG")) {   // the query projection: with v, the pair says whether h or one dispatch is wrong
        if (FILE* fq = std::fopen(qp, "wb")) {
            std::fwrite(s.qg.ptr, 4, 2 * hq, fq);
            std::fclose(fq);
        }
    }
    if (const char* vp = std::getenv("Q36_DUMP_V")) {   // the value projection before the ring store
        if (FILE* fv = std::fopen(vp, "wb")) {
            std::fwrite(s.v_raw.ptr, 4, hkv, fv);
            std::fclose(fv);
        }
    }
    if (std::getenv("Q36_TRACE_QG")) { std::printf("  qg[0] after attention = %.6g\n", (double) ((const float*) s.qg.ptr)[0]); std::fflush(stdout); }
    if (const char* ap = std::getenv("Q36_DUMP_A")) {   // the attention's own output, before the o projection: the discriminator
        if (FILE* fa = std::fopen(ap, "wb")) {
            std::fwrite(s.a.ptr, 4, hq, fa);
            std::fclose(fa);
        }
    }

    // 5. the output projection
    if (!dense_gemv(cx, dk, w.o, s.a, s.y, ds, err)) return false;
    return true;
}

}  // namespace strata::qwen36
