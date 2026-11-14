// metal/tests/q36_model_check.cpp - the Qwen3.6 host half on synthetic files: no model, no GPU, no Metal device.
//
//   * a conforming tiny qwen35moe file opens, and describe() reports its geometry, the dense/expert split and the
//     per-token expert traffic;
//   * a dense weight at a type the port's matvec does not read is refused at open, naming the tensor;
//   * a missing tensor is named;
//   * an expert tensor at a type the port's grouped-expert kernels do not take is refused at open, naming the tensor,
//     instead of at the first dispatch (which on this runtime stops the process inside Metal);
//   * a geometry outside what the kernels are cut for (head 256, state 128, conv 4) is refused by name.
//
// The fixture (q36_fixture.h) builds the same tiny model q36_upload_check drives onto the device.
#include "qwen36/q36_model.h"

#include "q36_fixture.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace q36fix;
using fixture::Kv;
using fixture::Tensor;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// A type outside gemv.metal's set (Q4_1 is a ggml type Strata's kernels do not read): the dense path has to refuse it.
void retype_embedding_q4_1(std::vector<Tensor>& ts) {
    for (Tensor& t : ts)
        if (t.name == "token_embd.weight") t.type = 3;   // Q4_1
}

void retype_experts_q4_0(std::vector<Tensor>& ts) {
    for (Tensor& t : ts)
        if (t.name.find("ffn_gate_exps.weight") != std::string::npos) t.type = 2;   // Q4_0: not in moe.metal's gate/up set
}

void drop_a_tensor(std::vector<Tensor>& ts) {
    for (size_t i = 0; i < ts.size(); ++i)
        if (ts[i].name == "blk.2.ffn_down_exps.weight") {
            ts.erase(ts.begin() + (long) i);
            return;
        }
}

}  // namespace

int main() {
    std::printf("q36_model_check: the Qwen3.6 host half on synthetic files (no model, no GPU)\n");
    TempDir dir;

    // 1. a conforming file opens and describes itself
    {
        const fs::path f = write_model(dir, "ok", "qwen35moe", kTiny);
        check(!f.empty(), "the fixture file was written");
        strata::qwen36::Model m;
        std::string err;
        const bool ok = !f.empty() && strata::qwen36::open(f.string(), m, err);
        check(ok, "a conforming qwen35moe file opens" + (ok ? std::string() : ": " + err));
        if (ok) {
            std::printf("%s", strata::qwen36::describe(m).c_str());
            // block_count 5 with one nextn_predict_layers: four trunk layers and the MTP block the file also carries
            check(m.g.n_layers == (int64_t) kTiny.block_count - 1, "the trunk is the file's block count less its MTP blocks");
            check(m.n_mtp() == 1, "the in-file MTP block is counted separately from the trunk");
            check((int64_t) m.layers.size() == m.g.n_layers + m.n_mtp(), "every trunk layer and the MTP block are resolved");
            check(m.top_k == (int64_t) kTiny.used, "the experts per token are the file's expert_used_count");
            check(m.expert_bytes > 0 && m.dense_bytes > 0 && m.embed_bytes > 0, "the dense / expert / embedding split is filled");
            check(m.total_bytes == m.embed_bytes + m.dense_bytes + m.expert_bytes, "the three totals add up to the whole");
            check(m.expert_bytes_per_token() > 0, "the per-token expert traffic is computed");
            check(m.g.n_gdn_layers() == 3 && m.g.n_qsa_layers() == 1, "3 GDN layers and 1 full-attention layer of 4");
            check(m.layers[3].kind == strata::core::LayerKind::Attention, "layer 3 is the full-attention one");
            check(m.layers[4].kind == strata::core::LayerKind::Mtp, "the block after the trunk is the MTP one");
            check(m.layers[4].eh_proj.ok(), "the MTP block's eh_proj is resolved");
        }
    }

    // 2. a dense weight at a type the port's matvec does not read is refused at open, naming the tensor
    {
        const fs::path f = write_model(dir, "q41", "qwen35moe", kTiny, retype_embedding_q4_1);
        strata::qwen36::Model m;
        std::string err;
        const bool opened = !f.empty() && strata::qwen36::open(f.string(), m, err);
        check(!opened && err.find("token_embd.weight") != std::string::npos && err.find("Q4_1") != std::string::npos,
              std::string("an unsupported dense type is refused naming the tensor: ") + (opened ? "(it opened)" : err));
    }

    // 3. a missing tensor is named
    {
        const fs::path f = write_model(dir, "missing", "qwen35moe", kTiny, drop_a_tensor);
        strata::qwen36::Model m;
        std::string err;
        const bool opened = !f.empty() && strata::qwen36::open(f.string(), m, err);
        check(!opened && err.find("blk.2.ffn_down_exps.weight") != std::string::npos,
              std::string("a missing tensor is named: ") + (opened ? "(it opened)" : err));
    }

    // 4. an expert type the port's kernels do not take is refused at open, naming the tensor
    {
        const fs::path f = write_model(dir, "q40", "qwen35moe", kTiny, retype_experts_q4_0);
        strata::qwen36::Model m;
        std::string err;
        const bool opened = !f.empty() && strata::qwen36::open(f.string(), m, err);
        check(!opened && err.find("ffn_gate_exps.weight") != std::string::npos && err.find("Q4_0") != std::string::npos,
              std::string("an unsupported expert type is refused naming the tensor: ") + (opened ? "(it opened)" : err));
    }

    // 5. the kernel gate: a head width the port's attention kernels are not cut for
    {
        Dims d = kTiny;
        d.key_len = 128;
        const fs::path f = write_model(dir, "head128", "qwen35moe", d);
        strata::qwen36::Model m;
        std::string err;
        const bool opened = !f.empty() && strata::qwen36::open(f.string(), m, err);
        check(!opened && err.find("head 256") != std::string::npos,
              std::string("a head width the kernels do not take is refused by name: ") + (opened ? "(it opened)" : err));
    }

    std::printf("%s\n", g_fail == 0 ? "q36_model_check: all ok" : "q36_model_check: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
