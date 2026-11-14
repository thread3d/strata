// metal/src/qwen36/q36_model.cpp - the host half of the Qwen3.6 Metal engine: read the file, decide what it is, and refuse
// what this port cannot do.  No device work of any kind.
//
// Why this is its own unit rather than part of a driver: the CUDA engine (src/qwen35/q36_engine.cu) learns a model by
// loading it, so "does this file fit" and "can the kernels take it" are answered by trying.  On the RX 580 neither is
// true - the card is 8 GB against a 37.8 GB file, and several of the port's kernels are compiled only for the types the
// Flash-Next packs use - so both questions have to be answered from the header, before an allocation, with a message that
// says which one failed.
#include "q36_model.h"

#include "ref_quant.h"

#include <cstdio>
#include <cstring>

namespace strata::qwen36 {

uint64_t Model::expert_slab_bytes() const {
    const Layer& l0 = layers.empty() ? Layer() : layers.front();
    if (!l0.gate_exps.ok() || !l0.down_exps.ok()) return 0;
    const uint64_t gu = (uint64_t) ref::row_bytes(l0.gate_exps.type(), (int) g.n_embd);
    const uint64_t dn = (uint64_t) ref::row_bytes(l0.down_exps.type(), (int) g.n_ff);
    return 2 * (uint64_t) g.n_ff * gu + (uint64_t) g.n_embd * dn;
}

namespace {

using strata::core::ModelGeometry;

// The types gemv.metal instantiates: gemv_parity walks exactly this list (metal/tests/gemv_parity.cpp), so a type here is a
// type with a parity test behind it, not merely a type a kernel names.
bool dense_ok(int t) {
    switch (t) {
    case ref::F32: case ref::BF16: case ref::F16:
    case ref::Q2_0: case ref::Q4_0: case ref::Q5_0: case ref::Q5_1: case ref::Q8_0:
    case ref::Q3_K: case ref::Q4_K: case ref::Q5_K: case ref::Q6_K:
    case ref::IQ4_NL: case ref::IQ4_XS: case ref::IQ1_S: case ref::IQ1_M:
    case ref::IQ2_XXS: case ref::IQ2_XS: case ref::IQ2_S: case ref::IQ3_XXS: case ref::IQ3_S:
        return true;
    default:
        return false;
    }
}

// moe.metal's GU_SET / DN_SET instantiation lists.  Q8_0 is in both - which is the one the Qwen3.6 Q8_0 pack needs, and
// which expert_parity was run at on this card (h rel L2 1.52e-07 at one token, 1.47e-07 at eight).
bool expert_gu_ok(int t) {
    switch (t) {
    case ref::Q2_0: case ref::Q4_K: case ref::Q5_K: case ref::Q6_K: case ref::Q8_0:
    case ref::IQ2_XXS: case ref::IQ2_XS: case ref::IQ2_S: case ref::IQ3_XXS: case ref::IQ3_S: case ref::IQ4_XS:
        return true;
    default:
        return false;
    }
}
bool expert_dn_ok(int t) {
    switch (t) {
    case ref::Q2_0: case ref::Q5_0: case ref::Q5_1: case ref::Q8_0: case ref::IQ4_NL: case ref::IQ4_XS: case ref::IQ3_S:
        return true;
    default:
        return false;
    }
}

struct Resolver {
    const strata::GgufModel& m;
    std::string err;

    TensorRef get(const std::string& name, bool required = true) {
        TensorRef r;
        size_t shard = 0;
        const strata::TensorInfo* t = m.find(name, &shard);
        if (t == nullptr) {
            if (required && err.empty()) err = "the file has no " + name;
            return r;
        }
        if (!m.in_bounds(*t, shard)) {
            if (err.empty()) err = name + " is not inside " + m.shard(shard).path();
            return r;
        }
        r.t = t;
        r.shard = shard;
        return r;
    }
};

// A matrix whose type the port's kernels do not take is a refusal with the tensor's own name - not a dispatch that throws
// later, which on this runtime means a process that stops inside Metal.
bool check_type(std::string& err, const TensorRef& r, Role role, const char* what) {
    if (!r.ok()) return true;   // (absence is the resolver's error, already reported)
    const int t = r.type();
    if (type_supported(role, t)) return true;
    char buf[512];
    std::snprintf(buf, sizeof buf, "%s is %s, and this port has no %s kernel for that type (it takes: %s)", r.name(),
                  r.t->type_name(), what,
                  role == Role::Dense ? "dense Q8_0, the K-quants, the IQ types, F16/BF16/F32"
                                      : (role == Role::ExpertGateUp ? "Q8_0, the K-quants, the IQ types and Q2_0"
                                                                    : "Q8_0, Q5_0, Q5_1, IQ4_NL, IQ4_XS, IQ3_S and Q2_0"));
    if (err.empty()) err = buf;
    return false;
}

}  // namespace

bool type_supported(Role role, int type) {
    switch (role) {
    case Role::Dense:
        return dense_ok(type);
    case Role::ExpertGateUp:
        return expert_gu_ok(type);
    case Role::ExpertDown:
        return expert_dn_ok(type);
    }
    return false;
}

bool open(const std::string& path, Model& m, std::string& err) {
    try {
        m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    } catch (const std::exception& e) {
        err = std::string("cannot open the model: ") + e.what();
        return false;
    }
    const strata::GgufModel& gm = *m.file;

    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = std::string("this engine runs ") + strata::model_family_title(strata::ModelFamily::Qwen35Moe) +
              " (qwen35moe) files only; " + gm.meta().path() + " is " + strata::model_family_title(m.g.family);
        return false;
    }

    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;

    // The gate the CUDA driver applies, kept here in the same words: a geometry these kernels do not take is refused by
    // name, at open, rather than by a dispatch that returns garbage three layers in.
    const ModelGeometry& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "a geometry this port's kernels do not take: head %lld, ssm state %lld, conv %lld, %lld experts, "
                      "top-%lld (the kernels are cut for head 256, state 128, conv 4, <= 512 experts, top-k <= 15)",
                      (long long) g.head_dim, (long long) g.ssm_state_size, (long long) g.ssm_d_conv, (long long) g.n_expert,
                      (long long) m.top_k);
        err = buf;
        return false;
    }
    if (g.has_hyper_connections()) {
        err = "this is a hyper-connection model; the Qwen3.6 engine path is the plain pre-norm one";
        return false;
    }

    const std::string k = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(k + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(k + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(k + "rope.dimension_count")) m.n_rot = (int64_t) v->u;

    Resolver r{gm, {}};
    m.embed = r.get("token_embd.weight");
    m.output = r.get("output.weight");
    m.output_norm = r.get("output_norm.weight");
    if (!r.err.empty()) {
        err = r.err;
        return false;
    }

    m.layers.clear();
    m.layers.reserve((size_t) (g.n_layers + g.n_mtp_layers));
    for (int64_t il = 0; il < g.n_layers + g.n_mtp_layers; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        Layer L;
        L.kind = strata::core::layer_kind(g, il);
        L.gate_inp = r.get(p + "ffn_gate_inp.weight");
        L.gate_inp_shexp = r.get(p + "ffn_gate_inp_shexp.weight");
        L.gate_shexp = r.get(p + "ffn_gate_shexp.weight");
        L.up_shexp = r.get(p + "ffn_up_shexp.weight");
        L.down_shexp = r.get(p + "ffn_down_shexp.weight");
        L.gate_exps = r.get(p + "ffn_gate_exps.weight");
        L.up_exps = r.get(p + "ffn_up_exps.weight");
        L.down_exps = r.get(p + "ffn_down_exps.weight");
        L.attn_norm = r.get(p + "attn_norm.weight");
        L.post_norm = r.get(p + "post_attention_norm.weight");
        if (L.kind == strata::core::LayerKind::Gdn) {
            L.qkv = r.get(p + "attn_qkv.weight");
            L.sgate = r.get(p + "attn_gate.weight");
            L.ssm_out = r.get(p + "ssm_out.weight");
            L.conv1d = r.get(p + "ssm_conv1d.weight");
            L.alpha = r.get(p + "ssm_alpha.weight");
            L.beta = r.get(p + "ssm_beta.weight");
            L.ssm_a = r.get(p + "ssm_a");
            L.dt_bias = r.get(p + "ssm_dt.bias");
            L.ssm_norm = r.get(p + "ssm_norm.weight");
        } else {
            L.q = r.get(p + "attn_q.weight");
            L.k = r.get(p + "attn_k.weight");
            L.v = r.get(p + "attn_v.weight");
            L.o = r.get(p + "attn_output.weight");
            L.q_norm = r.get(p + "attn_q_norm.weight");
            L.k_norm = r.get(p + "attn_k_norm.weight");
        }
        if (L.kind == strata::core::LayerKind::Mtp) {
            L.eh_proj = r.get(p + "nextn.eh_proj.weight");
            L.enorm = r.get(p + "nextn.enorm.weight");
            L.hnorm = r.get(p + "nextn.hnorm.weight");
            L.shared_head_norm = r.get(p + "nextn.shared_head_norm.weight");
        }
        if (!r.err.empty()) {
            err = r.err;
            return false;
        }

        L.expert_bytes = L.gate_exps.bytes() + L.up_exps.bytes() + L.down_exps.bytes();
        L.dense_bytes = L.gate_inp.bytes() + L.gate_inp_shexp.bytes() + L.gate_shexp.bytes() + L.up_shexp.bytes() +
                        L.down_shexp.bytes() + L.attn_norm.bytes() + L.post_norm.bytes() + L.qkv.bytes() + L.sgate.bytes() +
                        L.ssm_out.bytes() + L.conv1d.bytes() + L.alpha.bytes() + L.beta.bytes() + L.ssm_a.bytes() +
                        L.dt_bias.bytes() + L.ssm_norm.bytes() + L.q.bytes() + L.k.bytes() + L.v.bytes() + L.o.bytes() +
                        L.q_norm.bytes() + L.k_norm.bytes() + L.eh_proj.bytes() + L.enorm.bytes() + L.hnorm.bytes() +
                        L.shared_head_norm.bytes();
        m.layers.push_back(L);
    }

    // The kernels this port would call on each weight class, checked here so a file the port cannot run says so at open.
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const Layer& L = m.layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        if (!check_type(err, L.gate_exps, Role::ExpertGateUp, "grouped expert gate/up")) return false;
        if (!check_type(err, L.up_exps, Role::ExpertGateUp, "grouped expert gate/up")) return false;
        if (!check_type(err, L.down_exps, Role::ExpertDown, "grouped expert down")) return false;
        for (const TensorRef* dense : {&L.qkv, &L.sgate, &L.ssm_out, &L.q, &L.k, &L.v, &L.o, &L.gate_shexp, &L.up_shexp,
                                       &L.down_shexp, &L.eh_proj})
            if (!check_type(err, *dense, Role::Dense, "dense matvec")) return false;
        (void) p;
    }
    if (!check_type(err, m.embed, Role::Dense, "dense matvec")) return false;
    if (!check_type(err, m.output, Role::Dense, "dense matvec")) return false;

    m.embed_bytes = m.embed.bytes();
    m.dense_bytes = m.output.bytes() + m.output_norm.bytes();
    m.expert_bytes = 0;
    for (const Layer& L : m.layers) {
        m.dense_bytes += L.dense_bytes;
        m.expert_bytes += L.expert_bytes;
    }
    m.total_bytes = m.embed_bytes + m.dense_bytes + m.expert_bytes;
    const Layer& l0 = m.layers.front();
    m.gu_row = l0.gate_exps.ok() ? l0.gate_exps.row_bytes() : 0;
    m.down_row = l0.down_exps.ok() ? l0.down_exps.row_bytes() : 0;
    return true;
}

std::string describe(const Model& m) {
    const ModelGeometry& g = m.g;
    char buf[1024];
    std::string s;
    auto line = [&](const char* fmt, auto... args) {
        std::snprintf(buf, sizeof buf, fmt, args...);
        s += buf;
        s += '\n';
    };
    line("%s (%s), %lld trunk layers + %lld MTP block(s)", strata::model_family_title(g.family),
         strata::model_family_arch(g.family), (long long) g.n_layers, (long long) g.n_mtp_layers);
    line("  hidden %lld, vocab %lld, %lld experts top-%lld, FF %lld (shared %lld)", (long long) g.n_embd,
         (long long) g.n_vocab, (long long) g.n_expert, (long long) m.top_k, (long long) g.n_ff,
         (long long) g.n_ff_shexp);
    line("  layers: %lld GDN, %lld full attention (every %lldth), + %lld MTP", (long long) g.n_gdn_layers(),
         (long long) g.n_qsa_layers(), (long long) g.qsa_interval, (long long) g.n_mtp_layers);
    line("  attention: %lld query heads over %lld KV x %lld, %lld rotated; no indexer", (long long) g.n_head,
         (long long) g.n_head_kv, (long long) g.head_dim, (long long) m.n_rot);
    line("  GDN: state %lld, %lld K groups, %lld V heads, conv %lld, inner %lld", (long long) g.ssm_state_size,
         (long long) g.ssm_k_heads, (long long) g.ssm_v_heads, (long long) g.ssm_d_conv, (long long) g.ssm_value_dim);
    line("  eps %g, rope base %g", (double) m.eps, (double) m.rope_base);
    line("  bytes: embedding %.2f GiB, dense %.2f GiB, experts %.2f GiB, of %.2f GiB of tensors",
         (double) m.embed_bytes / 1073741824.0, (double) m.dense_bytes / 1073741824.0,
         (double) m.expert_bytes / 1073741824.0, (double) m.total_bytes / 1073741824.0);
    const Layer& l0 = m.layers.front();
    line("  one expert's slab: %.2f MiB (%s gate/up, %s down), %.2f GiB a token at top-%lld over %lld layers",
         (double) m.expert_slab_bytes() / 1048576.0, l0.gate_exps.ok() ? l0.gate_exps.t->type_name() : "?",
         l0.down_exps.ok() ? l0.down_exps.t->type_name() : "?", (double) m.expert_bytes_per_token() / 1073741824.0,
         (long long) m.top_k, (long long) g.n_layers);
    return s;
}

}  // namespace strata::qwen36
