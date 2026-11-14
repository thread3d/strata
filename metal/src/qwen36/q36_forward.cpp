// metal/src/qwen36/q36_forward.cpp - every layer, once per token.
#include "q36_forward.h"

#include <cmath>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::qwen36 {
namespace {

ExpertBlob blob_of(const Model& m, const Layer& L) {
    ExpertBlob b;
    b.n_embd = (int) m.g.n_embd;
    b.n_ff = (int) m.g.n_ff;
    if (L.gate_exps.ok()) b.gu_type = L.gate_exps.type();
    if (L.down_exps.ok()) b.down_type = L.down_exps.type();
    b.shape();
    return b;
}

}  // namespace

uint64_t Forward::resident_bytes() const {
    uint64_t n = 0;
    for (const auto* v : {&state, &hist, &kring, &vring, &slots})
        for (const mtl::Buffer& b : *v) n += b.bytes;
    for (const mtl::Buffer& b : {x, xh, logits, cs, layer.h, layer.sg, layer.moe_out, dense.xp, shared.h, shared.y, shared.g,
                                 shared.u, routed.logits, routed.ids, routed.wts, routed.plan, routed.h, routed.ye})
        n += b.bytes;
    return n;
}

bool forward_init(mtl::Context& cx, const Model& m, const DeviceModel& d, int kv_cap, uint64_t slot_budget_bytes, Forward& f,
                  std::string& err) {
    (void) d;
    f.kv_cap = kv_cap;
    const int N = (int) m.g.n_embd;
    const GdnShape gsh{N, (int) m.g.ssm_k_heads, (int) m.g.ssm_v_heads, (int) m.g.ssm_state_size, m.eps};
    const AttnShape ash{N, (int) m.g.n_head, (int) m.g.n_head_kv, (int) m.g.head_dim, m.eps};
    const SharedShape ssh{N, (int) m.g.n_ff_shexp};
    if (!attn_init(cx, f.ak, ash, err) || !gdn_init(cx, f.gk, err) || !layer_init(cx, f.lk, err) ||
        !dense_init(cx, f.dk, err))
        return false;
    if (!attn_scratch(cx, f.attn, ash, kv_cap, err) || !gdn_scratch(cx, f.gdn, gsh, err) ||
        !layer_scratch(cx, f.layer, N, err))
        return false;

    // the expert slots: only if the caller's budget holds them all (the streaming path is what makes a real pack fit)
    // EVERY layer's slots count towards the budget, not the first one's: a `break` here made the sum a fortieth of the truth
    // (816 MiB instead of 33.5 GiB for the Q8_0 pack), which is why the resident plan was chosen against a 2 GiB budget and the run
    // only survived because the slots are host-backed on a 96 GB machine.  The kernels are resolved once, from the first layer that
    // has experts, because their instantiation depends on the types and every layer of a pack shares them.
    uint64_t resident = 0;
    bool resolved = false;
    for (const Layer& L : m.layers) {
        if (!L.gate_exps.ok()) continue;
        const ExpertBlob b = blob_of(m, L);
        resident += b.slots((int) m.g.n_expert);
        if (!resolved) {
            const RoutedShape rsh{b, (int) m.g.n_expert, (int) m.top_k, 1};
            if (!routed_init(cx, f.rk, rsh, err, false)) return false;
            resolved = true;
        }
    }
    if (resident > slot_budget_bytes && !m.layers.empty()) {
        // STREAMING: top_k slots, filled per layer.  Every layer's experts share one pool because the layers run in sequence.
        const ExpertBlob b = blob_of(m, m.layers[0]);
        if (!stream_init(cx, f.stream, b, (int) m.top_k, (int) m.g.n_expert, err)) return false;
        f.streaming = true;
        const RoutedShape rsh{b, (int) m.g.n_expert, (int) m.top_k, 1};
        if (!routed_init(cx, f.rk, rsh, err, true) || !routed_scratch(cx, f.routed, rsh, err)) return false;
        err.clear();
    }
    if (!f.streaming && resident > slot_budget_bytes) {
        char buf[320];
        std::snprintf(buf, sizeof buf,
                      "the resident expert slots need %.1f GiB and the budget is %.1f GiB: this card cannot hold them, and the "
                      "path that runs a real pack is streaming one expert a token into these slots",
                      (double) resident / 1073741824.0, (double) slot_budget_bytes / 1073741824.0);
        err = buf;
        return false;
    }
    if (resident > 0) {
        const ExpertBlob b = blob_of(m, m.layers[0]);
        RoutedShape rsh{b, (int) m.g.n_expert, (int) m.top_k, 1};
        if (!routed_scratch(cx, f.routed, rsh, err)) return false;
    }

    f.x = cx.alloc(4 * (size_t) N);
    f.xh = cx.alloc(4 * (size_t) N);
    f.logits = cx.alloc(4 * (size_t) m.g.n_vocab);
    const int pos_rows = kv_cap + 16;
    f.cs = cx.alloc(4 * (size_t) pos_rows * 32 * 2);
    if (f.x.ptr == nullptr || f.xh.ptr == nullptr || f.logits.ptr == nullptr || f.cs.ptr == nullptr) {
        err = "the device refused the forward's buffers";
        return false;
    }
    float* tab = f.cs.as<float>();
    for (int p = 0; p < pos_rows; ++p)
        for (int i = 0; i < 32; ++i) {
            const double th = (double) p * std::pow((double) m.rope_base, -2.0 * (double) i / 64.0);
            tab[(p * 32 + i) * 2] = (float) std::cos(th);
            tab[(p * 32 + i) * 2 + 1] = (float) std::sin(th);
        }

    f.slots.assign(m.layers.size(), mtl::Buffer{});
    f.state.assign(m.layers.size(), mtl::Buffer{});
    f.hist.assign(m.layers.size(), mtl::Buffer{});
    f.kring.assign(m.layers.size(), mtl::Buffer{});
    f.vring.assign(m.layers.size(), mtl::Buffer{});
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& L = m.layers[i];
        if (L.kind == strata::core::LayerKind::Gdn) {
            if (!gdn_state(cx, gsh, f.state[i], f.hist[i], err)) return false;
        } else {
            const size_t bytes = 2 * (size_t) kv_cap * m.g.n_head_kv * m.g.head_dim;
            f.kring[i] = cx.alloc(bytes);
            f.vring[i] = cx.alloc(bytes);
            if (f.kring[i].ptr == nullptr || f.vring[i].ptr == nullptr) {
                err = "the device refused a KV ring";
                return false;
            }
        }
        if (L.gate_exps.ok() && !f.streaming) {
            const ExpertBlob b = blob_of(m, L);
            f.slots[i] = cx.alloc((size_t) b.slots((int) m.g.n_expert));
            if (f.slots[i].ptr == nullptr) {
                err = "the device refused layer " + std::to_string(i) + "'s expert slots";
                return false;
            }
            for (int e = 0; e < (int) m.g.n_expert; ++e)
                if (!pack(b, e, L.gate_exps.data(*m.file), L.up_exps.data(*m.file), L.down_exps.data(*m.file),
                          f.slots[i].as<uint8_t>() + (size_t) e * b.blob, err))
                    return false;
        }
    }
    return true;
}

bool embed_token(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, uint32_t token, std::string& err) {
    const int N = (int) m.g.n_embd;
    mtl::Pipeline emb;
    try {
        emb = cx.pipeline(f.lk.glue_lib, "q36_embed");
    } catch (const std::exception& e) {
        err = std::string("no q36_embed: ") + e.what();
        return false;
    }
    mtl::Dispatch de;
    de.p = &emb;
    de.args = {mtl::Arg::B(d.embed), mtl::Arg::B(f.x), mtl::Arg::V((uint32_t) N), mtl::Arg::V(token)};
    de.tg[0] = 64;
    de.tg[1] = de.tg[2] = 1;
    de.grid[0] = (uint32_t) ((N + 63) / 64);
    de.grid[1] = de.grid[2] = 1;
    cx.run({de});
    return true;
}

bool forward_token(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, uint32_t token, int pos, int n_kv,
                   std::string& err) {
    const int N = (int) m.g.n_embd;
    const GdnShape gsh{N, (int) m.g.ssm_k_heads, (int) m.g.ssm_v_heads, (int) m.g.ssm_state_size, m.eps};
    const AttnShape ash{N, (int) m.g.n_head, (int) m.g.n_head_kv, (int) m.g.head_dim, m.eps};
    const SharedShape ssh{N, (int) m.g.n_ff_shexp};

    if (!embed_token(cx, m, d, f, token, err)) return false;
    if (std::getenv("Q36_TRACE")) {
        const float* xv = (const float*) f.x.ptr;
        size_t bad = 0;
        for (int e = 0; e < N; ++e) if (!std::isfinite(xv[e])) ++bad;
        std::printf("Q36_TRACE embed: x[0..2] = %g %g %g, non-finite %zu\n", (double) xv[0], (double) xv[1], (double) xv[2], bad);
        std::fflush(stdout);
    }

    for (size_t i = 0; i < m.layers.size(); ++i) {
        // The in-file MTP block is a DRAFT HEAD, not a trunk layer: the engine runs the trunk, and the MTP block is what a
        // speculative decode evaluates against the trunk's own last state.  Running it in the trunk would feed a draft layer's
        // output into the final norm - which is one way to get fluent-looking nonsense out of a correct model.
        if (m.layers[i].kind == strata::core::LayerKind::Mtp) continue;
        LayerWeights w;
        if (!bind_layer(m, d, i, w, err)) return false;
        LayerIo io;
        io.attn = ash;
        io.gdn = gsh;
        io.shared = ssh;
        io.routed = RoutedShape{blob_of(m, m.layers[i]), (int) m.g.n_expert, (int) m.top_k, 1};
        io.attn_scratch = &f.attn;
        io.gdn_scratch = &f.gdn;
        io.shared_scratch = &f.shared;
        io.routed_scratch = &f.routed;
        io.attn_kernels = &f.ak;
        io.gdn_kernels = &f.gk;
        io.routed_kernels = &f.rk;
        io.state = &f.state[i];
        io.hist = &f.hist[i];
        io.cs = &f.cs;
        io.slots = f.streaming ? nullptr : &f.slots[i];
        io.slots_off = 0;
        if (f.streaming) {
            io.fill.m = &m;
            io.fill.L = &m.layers[i];
            io.fill.pool = &f.stream;
        }
        io.pos = pos;
        io.n_kv = n_kv;
        if (w.attention) {   // this layer's own ring: the parts read the one they are handed
            f.attn.kring = f.kring[i];
            f.attn.vring = f.vring[i];
        }
        if (!layer_forward(cx, f.lk, f.dk, f.dense, w, io, f.layer, f.x, i == 0, err)) return false;
        // Q36_TRACE=1: the residual after each layer, with the count of non-finite values - the trace that found the
        // round-27 defect (a norm weight read from a fixture byte pattern), and cheap enough to keep
        if (std::getenv("Q36_TRACE")) {
            const float* xv = (const float*) f.x.ptr;
            size_t bad = 0, first = 0;
            for (int e = 0; e < N; ++e)
                if (!std::isfinite(xv[e])) {
                    if (!bad) first = (size_t) e;
                    ++bad;
                }
            std::printf("Q36_TRACE layer %zu (%s): x[0..2] = %g %g %g, non-finite %zu (first %zu)\n", i,
                        w.attention ? "attention" : "gdn", (double) xv[0], (double) xv[1], (double) xv[2], bad, first);
            std::fflush(stdout);
        }
    }

    // the final norm (a layer's last step already added its own output, so there is nothing to add here) and the projection
    mtl::Dispatch dn;
    dn.p = &f.lk.add_rms;
    dn.args = {mtl::Arg::B(f.x), mtl::Arg::B(f.x), mtl::Arg::B(d.output_norm), mtl::Arg::B(f.xh), mtl::Arg::V((uint32_t) N),
               mtl::Arg::V(0u), mtl::Arg::V(m.eps)};
    dn.grid[0] = 1;
    dn.grid[1] = dn.grid[2] = 1;
    dn.tg[0] = 256;
    dn.tg[1] = dn.tg[2] = 1;
    cx.run({dn});
    return dense_gemv(cx, f.dk, bind(m.output, d.output), f.xh, f.logits, f.dense, err);
}

}  // namespace strata::qwen36
