// metal/src/qwen36/q36_engine.cpp - strata-q36 on Metal: the Qwen3.6-35B-A3B engine port.
// Uses the port's layered API: q36_model (open), q36_device (upload_dense), q36_dense (dense_init/gemv),
// q36_gdn_layer (gdn_*), q36_attn_layer (attn_*), q36_layer (layer_*), q36_moe_* (shared/routed).
// Driver: GGUF open, device upload, per-layer forward (GDN or attention + MoE), output projection.
// Modes: --tokens IDS [--n N], --serve (stub).
#include "q36_model.h"
#include "q36_device.h"
#include "q36_dense.h"
#include "q36_gdn_layer.h"
#include "q36_attn_layer.h"
#include "q36_layer.h"
#include "q36_moe_routed.h"
#include "q36_moe_shared.h"

#include "../mtl_rt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::qwen36 {

namespace {

DenseMatrix dm_matrix(const mtl::Buffer* b, const TensorRef& t) {
    DenseMatrix m;
    m.buf = b;
    m.off = 0;
    m.type = t.type();
    m.n_in = (int) t.width();
    m.n_out = (int) t.rows();
    return m;
}

GdnWeights gdn_weights(const DeviceLayer& dl, const Layer& L) {
    GdnWeights w;
    w.qkv = dm_matrix(&dl.qkv, L.qkv);
    w.gate = dm_matrix(&dl.sgate, L.sgate);
    w.alpha = dm_matrix(&dl.alpha, L.alpha);
    w.beta = dm_matrix(&dl.beta, L.beta);
    w.out = dm_matrix(&dl.ssm_out, L.ssm_out);
    w.conv_w = &dl.conv1d;
    w.dt = &dl.dt_bias;
    w.ssm_a = &dl.ssm_a;
    w.gamma = &dl.ssm_norm;
    return w;
}

AttnWeights attn_weights(const DeviceLayer& dl, const Layer& L) {
    AttnWeights w;
    w.q = dm_matrix(&dl.q, L.q);
    w.k = dm_matrix(&dl.k, L.k);
    w.v = dm_matrix(&dl.v, L.v);
    w.o = dm_matrix(&dl.o, L.o);
    w.q_norm = &dl.q_norm;
    w.k_norm = &dl.k_norm;
    return w;
}

}  // namespace

int run(int argc, char** argv) {
    std::string model_path;
    std::string ids_s;
    int n = 0;
    bool serve_mode = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" || a == "--native") {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); return 2; }
            model_path = argv[++i];
        } else if (a == "--tokens" || a == "--ids") {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); return 2; }
            ids_s = argv[++i];
        } else if (a == "--n") {
            if (i + 1 >= argc) { std::fprintf(stderr, "--n needs a value\n"); return 2; }
            n = std::atoi(argv[++i]);
        } else if (a == "--serve") {
            serve_mode = true;
        } else if (a == "--help" || a == "-h") {
            std::printf("strata-q36-metal: Qwen3.6-35B-A3B on Metal\n");
            std::printf("  --model PATH    GGUF model path\n");
            std::printf("  --tokens IDS    comma-separated token ids\n");
            std::printf("  --n N           number of tokens to generate\n");
            std::printf("  --serve         serve mode\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "strata-q36-metal: --model required\n");
        return 2;
    }

    // Open the model (header parse only, no GPU).
    Model model;
    std::string err;
    if (!open(model_path, model, err)) {
        std::fprintf(stderr, "qwen36: open failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "qwen36: model opened: %lld layers, embd=%lld, vocab=%lld, n_expert=%lld, top_k=%lld\n",
                 (long long) model.n_trunk(), (long long) model.g.n_embd,
                 (long long) model.g.n_vocab, (long long) model.g.n_expert,
                 (long long) model.g.n_expert_used);

    // Metal context.
    mtl::Context ctx;
    std::fprintf(stderr, "qwen36: device %s\n", ctx.device_name().c_str());

    // Compile kernel libraries.  quant.metal (fmt<F>) and units.metal (Dec<F>) must come first;
    // ggml-common.h is auto-prepended by compile_try when the prelude lacks NO_GGML_COMMON.
    std::vector<std::string> kernel_files = {
        (ctx.kernel_dir + "/quant.metal"),
        (ctx.kernel_dir + "/units.metal"),
        (ctx.kernel_dir + "/gemv.metal"),
        (ctx.kernel_dir + "/gemvu.metal"),
        (ctx.kernel_dir + "/gdn.metal"),
        (ctx.kernel_dir + "/moe.metal"),
        (ctx.kernel_dir + "/glue.metal"),
    };
    int lib = ctx.compile(kernel_files, "", false);
    if (lib < 0) {
        std::fprintf(stderr, "qwen36: kernel compilation failed\n");
        return 1;
    }

    // Init layer (glue) kernels.
    LayerKernels lk;
    if (!layer_init(ctx, lk, err)) {
        std::fprintf(stderr, "qwen36: layer_init failed: %s\n", err.c_str());
        return 1;
    }

    // Init dense kernels.
    DenseKernels dk;
    if (!dense_init(ctx, dk, err)) {
        std::fprintf(stderr, "qwen36: dense_init failed: %s\n", err.c_str());
        return 1;
    }

    // Upload all dense tensors.
    DeviceModel dm;
    if (!upload_dense(ctx, model, dm, err)) {
        std::fprintf(stderr, "qwen36: upload_dense failed: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "qwen36: dense upload complete, %llu bytes\n",
                 (unsigned long long) dm.dense_bytes);

    // Allocate layer scratch.
    LayerScratch ls;
    if (!layer_scratch(ctx, ls, (int) model.g.n_embd, err)) {
        std::fprintf(stderr, "qwen36: layer_scratch failed: %s\n", err.c_str());
        return 1;
    }

    // Allocate dense scratch.
    DenseScratch ds;
    if (!dense_scratch(ctx, ds, (int) model.g.n_embd, err)) {
        std::fprintf(stderr, "qwen36: dense_scratch failed: %s\n", err.c_str());
        return 1;
    }

    // Parse token ids.
    std::vector<int> ids;
    if (!ids_s.empty()) {
        size_t pos = 0;
        while (pos < ids_s.size()) {
            while (pos < ids_s.size() && (ids_s[pos] == ',' || ids_s[pos] == ' ')) ++pos;
            if (pos >= ids_s.size()) break;
            size_t end = ids_s.find(',', pos);
            if (end == std::string::npos) end = ids_s.size();
            ids.push_back(std::atoi(ids_s.substr(pos, end - pos).c_str()));
            pos = end;
        }
    }

    if (serve_mode) {
        std::fprintf(stderr, "qwen36: serve mode not yet implemented\n");
        return 1;
    }

    if (ids.empty()) {
        std::fprintf(stderr, "qwen36: no tokens provided\n");
        return 2;
    }

    int64_t n_embd = model.g.n_embd;
    int64_t n_vocab = model.g.n_vocab;

    // Embedding lookup: x = embed[ids[0]].  On AMD RX 580 (macpro51), CPU writes to Shared
    // memory are invisible to GPU compute kernels.  ctx.alloc(bytes, src) now stages via a
    // Shared buffer (newBufferWithBytes copies) and blits to a Private (GPU-coherent)
    // destination via MTLBlitCommandEncoder, following llama.cpp's macpro51 pattern.
    std::vector<float> x_host((size_t) n_embd);
    {
        int64_t tok0 = ids[0];
        if (tok0 < 0 || tok0 >= n_vocab) {
            std::fprintf(stderr, "qwen36: token %lld out of range [0, %lld)\n", (long long) tok0, (long long) n_vocab);
            return 2;
        }
        const uint8_t* emb = model.embed.data(*model.file);
        if (emb == nullptr) {
            std::fprintf(stderr, "qwen36: embedding tensor not available\n");
            return 1;
        }
        int emb_type = model.embed.type();
        size_t emb_row = ref::row_bytes(emb_type, (int) n_embd);
        ref::dequant(emb_type, emb + (size_t) tok0 * emb_row, x_host.data(), (size_t) n_embd);
    }
    std::fprintf(stderr, "q36_engine: about to call ctx.alloc for x, bytes=%zu\n", 4 * (size_t) n_embd);
    std::fflush(stderr);
    mtl::Buffer x = ctx.alloc(4 * (size_t) n_embd, x_host.data());

    // Per-layer state.
    int max_ctx = 4096;
    size_t rope_bytes = (size_t) max_ctx * (size_t) (model.g.head_dim / 2) * 2 * sizeof(float);
    mtl::Buffer rope_tab = ctx.alloc(rope_bytes);
    std::memset(rope_tab.ptr, 0, rope_bytes);

    // Expert streaming pool: one slot per routed expert a token can pick (top_k = 8 for this family)
    ExpertBlob stream_blob;
    stream_blob.n_embd = (int) model.g.n_embd;
    stream_blob.n_ff = (int) model.g.n_ff;
    stream_blob.shape();
    ExpertStream pool;
    if (!stream_init(ctx, pool, stream_blob, (int) model.g.n_expert_used, (int) model.g.n_expert, err)) {
        std::fprintf(stderr, "qwen36: stream_init failed: %s\n", err.c_str());
        return 1;
    }

    std::vector<mtl::Buffer> gdn_state_bufs(model.n_trunk());
    std::vector<mtl::Buffer> gdn_hist_bufs(model.n_trunk());
    std::vector<AttnScratch> attn_scratches(model.n_trunk());
    std::vector<GdnScratch> gdn_scratches(model.n_trunk());
    std::vector<AttnKernels> attn_kernels(model.n_trunk());
    std::vector<GdnKernels> gdn_kernels(model.n_trunk());
    std::vector<SharedScratch> shared_scratches(model.n_trunk());
    std::vector<RoutedScratch> routed_scratches(model.n_trunk());
    std::vector<RoutedKernels> routed_kernels(model.n_trunk());
    std::vector<bool> layer_ready(model.n_trunk(), false);

    int pos = 0;
    for (int64_t l = 0; l < model.n_trunk(); ++l) {
        bool is_attn = (l % model.g.qsa_interval == model.g.qsa_interval - 1);
        const Layer& L = model.layers[(size_t) l];
        DeviceLayer& dl = dm.layers[(size_t) l];

        LayerWeights w;
        w.attention = is_attn;
        if (is_attn) {
            w.attn = attn_weights(dl, L);
        } else {
            w.gdn = gdn_weights(dl, L);
        }
        w.attn_norm = &dl.attn_norm;
        w.attn_norm_off = 0;
        w.post_norm = &dl.post_norm;
        w.post_norm_off = 0;
        w.gate_inp = dm_matrix(&dl.gate_inp, L.gate_inp);
        w.gate_inp_shexp = dm_matrix(&dl.gate_inp_shexp, L.gate_inp_shexp);
        w.shared.gate = dm_matrix(&dl.gate_shexp, L.gate_shexp);
        w.shared.up = dm_matrix(&dl.up_shexp, L.up_shexp);
        w.shared.down = dm_matrix(&dl.down_shexp, L.down_shexp);

        LayerIo io;
        io.layer_index = (size_t) l;
        if (is_attn) {
            io.attn.n_embd = (int) n_embd;
            io.attn.n_head = (int) model.g.n_head;
            io.attn.n_kv_head = (int) model.g.n_head_kv;
            io.attn.head_dim = (int) model.g.head_dim;
            io.attn.eps = model.eps;
            if (!layer_ready[(size_t) l]) {
                if (!attn_init(ctx, attn_kernels[(size_t) l], io.attn, err)) {
                    std::fprintf(stderr, "qwen36: attn_init layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
                if (!attn_scratch(ctx, attn_scratches[(size_t) l], io.attn, 16, err)) {
                    std::fprintf(stderr, "qwen36: attn_scratch layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
            }
            io.attn_kernels = &attn_kernels[(size_t) l];
            io.attn_scratch = &attn_scratches[(size_t) l];
            io.cs = &rope_tab;
            io.pos = pos;
            io.n_kv = 1;
        } else {
            io.gdn.n_embd = (int) n_embd;
            io.gdn.h_k = (int) model.g.ssm_k_heads;
            io.gdn.h_v = (int) model.g.ssm_v_heads;
            io.gdn.state = (int) model.g.ssm_state_size;
            io.gdn.eps = model.eps;
            if (!layer_ready[(size_t) l]) {
                if (!gdn_init(ctx, gdn_kernels[(size_t) l], err)) {
                    std::fprintf(stderr, "qwen36: gdn_init layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
                if (!gdn_scratch(ctx, gdn_scratches[(size_t) l], io.gdn, err)) {
                    std::fprintf(stderr, "qwen36: gdn_scratch layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
                if (!gdn_state(ctx, io.gdn, gdn_state_bufs[(size_t) l], gdn_hist_bufs[(size_t) l], err)) {
                    std::fprintf(stderr, "qwen36: gdn_state layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
                SharedShape ssh;
                ssh.n_embd = (int) n_embd;
                ssh.n_ff = (int) model.g.ssm_value_dim;
                shared_scratches[(size_t) l].lib = lk.glue_lib;
                shared_scratches[(size_t) l].n_embd = (int) n_embd;
                shared_scratches[(size_t) l].n_ff = ssh.n_ff;
                shared_scratches[(size_t) l].h = ctx.alloc(4 * (size_t) ssh.n_ff);
                shared_scratches[(size_t) l].g = ctx.alloc(4 * (size_t) ssh.n_ff);
                shared_scratches[(size_t) l].u = ctx.alloc(4 * (size_t) ssh.n_ff);
                shared_scratches[(size_t) l].y = ctx.alloc(4 * (size_t) n_embd);

                RoutedShape rsh;
                rsh.n_expert = (int) model.g.n_expert;
                rsh.top_k = (int) model.g.n_expert_used;
                rsh.n_tok = 1;
                if (!routed_scratches[(size_t) l].logits.ptr) {
                    if (!routed_scratch(ctx, routed_scratches[(size_t) l], rsh, err)) {
                        std::fprintf(stderr, "qwen36: routed_scratch layer %lld failed: %s\n", (long long) l, err.c_str());
                        return 1;
                    }
                }
                if (!routed_init(ctx, routed_kernels[(size_t) l], rsh, err, false)) {
                    std::fprintf(stderr, "qwen36: routed_init layer %lld failed: %s\n", (long long) l, err.c_str());
                    return 1;
                }
            }
            io.gdn_kernels = &gdn_kernels[(size_t) l];
            io.gdn_scratch = &gdn_scratches[(size_t) l];
            io.state = &gdn_state_bufs[(size_t) l];
            io.hist = &gdn_hist_bufs[(size_t) l];
            io.shared_scratch = &shared_scratches[(size_t) l];
            io.routed_scratch = &routed_scratches[(size_t) l];
            io.routed_kernels = &routed_kernels[(size_t) l];
            io.fill.m = &model;
            io.fill.L = &L;
            io.fill.pool = &pool;
        }
        layer_ready[(size_t) l] = true;

        std::string layer_err;
        if (!layer_forward(ctx, lk, dk, ds, w, io, ls, x, l == 0, layer_err)) {
            std::fprintf(stderr, "qwen36: layer %lld failed: %s\n", (long long) l, layer_err.c_str());
            return 1;
        }
        pos++;
    }

    // Output projection: logits = output . (rms(x) * output_norm).
    mtl::Buffer h_norm = ctx.alloc(4 * (size_t) n_embd);
    {
        static mtl::Buffer zero_buf;
        if (!zero_buf.ptr) {
            zero_buf = ctx.alloc(4 * (size_t) n_embd);
            std::memset(zero_buf.ptr, 0, 4 * (size_t) n_embd);
        }
        mtl::Constants c;
        c.u(0, (uint32_t) n_embd);
        uint32_t eps_bits;
        std::memcpy(&eps_bits, &model.eps, 4);
        c.u(1, eps_bits);
        mtl::Pipeline p = ctx.pipeline(lk.glue_lib, "q36_add_rms", c);
        if (!p.handle) {
            std::fprintf(stderr, "qwen36: q36_add_rms not available\n");
            return 1;
        }
        mtl::Dispatch d;
        d.p = &p;
        d.args = {mtl::Arg::B(x), mtl::Arg::B(zero_buf), mtl::Arg::B(dm.output_norm),
                  mtl::Arg::B(h_norm), mtl::Arg::V((uint32_t) n_embd), mtl::Arg::V(0u), mtl::Arg::V(eps_bits)};
        d.tg[0] = 256; d.tg[1] = 1; d.tg[2] = 1;
        d.grid[0] = 1; d.grid[1] = 1; d.grid[2] = 1;
        ctx.run({d});
    }

    mtl::Buffer logits_buf = ctx.alloc(4 * (size_t) n_vocab);
    DenseMatrix out_w = dm_matrix(&dm.output, model.output);
    if (!dense_gemv(ctx, dk, out_w, h_norm, logits_buf, ds, err)) {
        std::fprintf(stderr, "qwen36: output projection failed: %s\n", err.c_str());
        return 1;
    }

    std::vector<float> logits((size_t) n_vocab);
    std::memcpy(logits.data(), logits_buf.ptr, (size_t) n_vocab * 4);
    std::vector<int> idx((size_t) n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) idx[(size_t) i] = (int) i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    std::fprintf(stderr, "qwen36: top-5 logits:");
    for (int k = 0; k < 5 && k < n_vocab; ++k)
        std::fprintf(stderr, " %d=%.4f", idx[(size_t) k], logits[(size_t) idx[(size_t) k]]);
    std::fprintf(stderr, "\n");

    std::fprintf(stderr, "qwen36: forward pass complete for %zu tokens\n", ids.size());
    return 0;
}

}  // namespace strata::qwen36

namespace strata::dsv41 {
int run(int, char**) {
    std::fprintf(stderr, "dsv41::run: stub (Qwen3.6 build; DeepSeek not linked)\n");
    return 0;
}
}  // namespace strata::dsv41
