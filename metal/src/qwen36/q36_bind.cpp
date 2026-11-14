// metal/src/qwen36/q36_bind.cpp - one layer's weights, bound from the model's tensors to the parts' structs.
#include "q36_bind.h"

namespace strata::qwen36 {
namespace {

bool need(bool ok, const char* what, std::string& err) {
    if (!ok && err.empty()) err = std::string("layer weights: missing ") + what;
    return ok;
}

bool need_dense(const DenseMatrix& m, const char* what, std::string& err) { return need(m.buf != nullptr, what, err); }

}  // namespace

bool bind_layer(const Model& m, const DeviceModel& d, size_t i, LayerWeights& w, std::string& err) {
    if (i >= m.layers.size() || i >= d.layers.size()) {
        err = "bind_layer: no layer " + std::to_string(i);
        return false;
    }
    const Layer& L = m.layers[i];
    const DeviceLayer& D = d.layers[i];
    err.clear();
    w = LayerWeights{};
    // the in-file MTP block is an attention layer with four extra tensors (the layout table gives it attn_q/k/v/output),
    // not a GDN one
    w.attention = L.kind != strata::core::LayerKind::Gdn;

    w.attn_norm = &D.attn_norm;
    w.attn_norm_off = 0;
    w.post_norm = &D.post_norm;
    w.post_norm_off = 0;
    if (!need(w.attn_norm->ptr != nullptr && w.post_norm->ptr != nullptr, "a norm", err)) return false;

    w.gate_inp = bind(L.gate_inp, D.gate_inp);
    w.gate_inp_shexp = bind(L.gate_inp_shexp, D.gate_inp_shexp);
    if (!need_dense(w.gate_inp, "ffn_gate_inp", err) || !need_dense(w.gate_inp_shexp, "ffn_gate_inp_shexp", err)) return false;

    w.shared.gate = bind(L.gate_shexp, D.gate_shexp);
    w.shared.up = bind(L.up_shexp, D.up_shexp);
    w.shared.down = bind(L.down_shexp, D.down_shexp);
    if (!need_dense(w.shared.gate, "ffn_gate_shexp", err) || !need_dense(w.shared.up, "ffn_up_shexp", err) ||
        !need_dense(w.shared.down, "ffn_down_shexp", err))
        return false;

    if (w.attention) {
        w.attn.q = bind(L.q, D.q);
        w.attn.k = bind(L.k, D.k);
        w.attn.v = bind(L.v, D.v);
        w.attn.o = bind(L.o, D.o);
        w.attn.q_norm = &D.q_norm;
        w.attn.k_norm = &D.k_norm;
        if (!need_dense(w.attn.q, "attn_q", err) || !need_dense(w.attn.k, "attn_k", err) || !need_dense(w.attn.v, "attn_v", err) ||
            !need_dense(w.attn.o, "attn_output", err) || !need(w.attn.q_norm->ptr != nullptr && w.attn.k_norm->ptr != nullptr,
                                                                  "a head norm", err))
            return false;
    } else {
        w.gdn.qkv = bind(L.qkv, D.qkv);
        w.gdn.gate = bind(L.sgate, D.sgate);
        w.gdn.out = bind(L.ssm_out, D.ssm_out);
        w.gdn.alpha = bind(L.alpha, D.alpha);
        w.gdn.beta = bind(L.beta, D.beta);
        w.gdn.conv_w = &D.conv1d;
        w.gdn.dt = &D.dt_bias;
        w.gdn.ssm_a = &D.ssm_a;
        w.gdn.gamma = &D.ssm_norm;
        if (!need_dense(w.gdn.qkv, "attn_qkv", err) || !need_dense(w.gdn.gate, "attn_gate", err) ||
            !need_dense(w.gdn.out, "ssm_out", err) || !need_dense(w.gdn.alpha, "ssm_alpha", err) ||
            !need_dense(w.gdn.beta, "ssm_beta", err))
            return false;
        if (!need(w.gdn.conv_w->ptr != nullptr && w.gdn.dt->ptr != nullptr && w.gdn.ssm_a->ptr != nullptr &&
                      w.gdn.gamma->ptr != nullptr,
                  "a GDN 1-D tensor", err))
            return false;
    }
    return true;
}

}  // namespace strata::qwen36
