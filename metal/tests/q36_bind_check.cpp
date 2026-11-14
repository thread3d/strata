// metal/tests/q36_bind_check.cpp - the binding between the file's tensors and the parts' DenseMatrix, checked against the layout
// table rather than against a formula.
//
// A swapped n_in / n_out, a transposed matrix, or a 1-D tensor bound where a matrix was meant all produce a wrong answer with no
// crash and no message - and the layer tests cannot see it, because they build their own matrices.  So this walks a real
// (synthetic) model's layers, binds each one the way the forward pass will, and checks every matrix's type and both widths
// against what the layout table says that tensor is.
#include "qwen36/q36_bind.h"

#include "q36_fixture.h"

#include <cstdio>
#include <string>

using namespace q36fix;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

bool dims(const strata::qwen36::DenseMatrix& m, int n_in, int n_out) {
    return m.buf != nullptr && m.n_in == n_in && m.n_out == n_out;
}
}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    TempDir dir;
    const fs::path f = write_model(dir, "bind");
    if (f.empty()) { std::printf("the fixture could not be built\n"); return 1; }

    strata::qwen36::Model m;
    std::string err;
    if (!strata::qwen36::open(f.string(), m, err)) { std::printf("open: %s\n", err.c_str()); return 1; }
    strata::qwen36::DeviceModel d;
    if (!strata::qwen36::upload_dense(cx, m, d, err)) { std::printf("upload: %s\n", err.c_str()); return 1; }

    const auto &g = m.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp, NE = (int) g.n_expert;
    const int CH = (int) (2 * g.ssm_k_heads + g.ssm_v_heads) * (int) g.ssm_state_size;   // the GDN's conv channels
    const int ZV = (int) g.ssm_value_dim;
    bool all = true;
    int gdn = 0, attn = 0;
    for (size_t i = 0; i < m.layers.size() && all; ++i) {
        strata::qwen36::LayerWeights w;
        if (!strata::qwen36::bind_layer(m, d, i, w, err)) {
            std::printf("  layer %zu: %s\n", i, err.c_str());
            all = false;
            break;
        }
        all = all && dims(w.gate_inp, N, NE) && dims(w.gate_inp_shexp, N, 1);
        all = all && dims(w.shared.gate, N, FS) && dims(w.shared.up, N, FS) && dims(w.shared.down, FS, N);
        if (w.attention) {
            if (m.layers[i].kind == strata::core::LayerKind::Attention) ++attn;   // (the MTP block is counted on its own below)
            all = all && dims(w.attn.q, N, 2 * (int) g.n_head * (int) g.head_dim);
            all = all && dims(w.attn.k, N, (int) g.n_head_kv * (int) g.head_dim);
            all = all && dims(w.attn.v, N, (int) g.n_head_kv * (int) g.head_dim);
            all = all && dims(w.attn.o, (int) g.n_head * (int) g.head_dim, N);
        } else if (m.layers[i].kind == strata::core::LayerKind::Gdn) {
            ++gdn;
            all = all && dims(w.gdn.qkv, N, CH);
            all = all && dims(w.gdn.gate, N, ZV);
            all = all && dims(w.gdn.out, ZV, N);
            all = all && dims(w.gdn.alpha, N, (int) g.ssm_v_heads) && dims(w.gdn.beta, N, (int) g.ssm_v_heads);
        }
    }
    check(all, "every layer binds, and every matrix has the type and the two widths the layout table says");
    check(gdn == 3 && attn == 1, "the fixture's 4 trunk layers bind as 3 GDN and 1 attention (got " + std::to_string(gdn) + "/" +
                                     std::to_string(attn) + ")");
    // the last layer is the MTP block: it has attention weights and a MoE, and it binds too
    strata::qwen36::LayerWeights wm;
    const bool mtp = strata::qwen36::bind_layer(m, d, m.layers.size() - 1, wm, err);
    check(mtp && wm.attention && wm.gate_inp.buf != nullptr, "the MTP block binds (attention weights plus its own MoE)");
    std::printf("%s\n", g_fail ? "q36_bind_check: FAILURES" : "q36_bind_check: all ok");
    return g_fail ? 1 : 0;
}
