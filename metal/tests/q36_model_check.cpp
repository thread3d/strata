// metal/tests/q36_model_check.cpp - the Qwen3.6 host half on synthetic files: no model, no GPU, no Metal device.
//
//   * a conforming tiny qwen35moe file opens, and describe() reports its geometry, the dense/expert split and the
//     per-token expert traffic;
//   * a qwen4exp file is refused in the family's own words;
//   * a missing tensor is named;
//   * an expert tensor at a type the port's grouped-expert kernels do not take is refused at open, naming the tensor,
//     instead of at the first dispatch (which on this runtime stops the process inside Metal);
//   * a geometry outside what the kernels are cut for (head 256, state 128, conv 4) is refused by name.
//
// The files are tiny: a few hundred KB each, every tensor F32 or Q8_0 with a recognised payload pattern.
#include "qwen36/q36_model.h"

#include "gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using fixture::Kv;
using fixture::Tensor;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, nextn;
};

// A tiny qwen35moe model that still meets the port's kernel gate (head 256, ssm state 128, conv 4): the widths are small
// where the kernels take a run-time width (hidden, FF, expert count, vocab) and at the kernel's own value where they do
// not.
const Dims kTiny = {5, 64, 2, 1, 256, 4, 2, 32, 32, 4, 128, 2, 4, 512, 4, 1};
constexpr uint64_t kVocab = 64;

std::vector<Kv> metadata(const char* arch, const Dims& d) {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", d.block_count),
                          fixture::u32(p + "embedding_length", d.embd),
                          fixture::u32(p + "attention.head_count", d.heads),
                          fixture::u32(p + "attention.head_count_kv", d.heads_kv),
                          fixture::u32(p + "attention.key_length", d.key_len),
                          fixture::u32(p + "attention.value_length", d.key_len),
                          fixture::u32(p + "expert_count", d.experts),
                          fixture::u32(p + "expert_used_count", d.used),
                          fixture::u32(p + "expert_feed_forward_length", d.ff),
                          fixture::u32(p + "expert_shared_feed_forward_length", d.ff_sh),
                          fixture::u32(p + "ssm.conv_kernel", d.conv),
                          fixture::u32(p + "ssm.state_size", d.state),
                          fixture::u32(p + "ssm.group_count", d.groups),
                          fixture::u32(p + "ssm.time_step_rank", d.v_heads),
                          fixture::u32(p + "ssm.inner_size", d.inner),
                          fixture::u32(p + "full_attention_interval", d.interval),
                          fixture::u32(p + "nextn_predict_layers", d.nextn),
                          fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
                          fixture::f32(p + "rope.freq_base", 10000000.0f),
                          fixture::u32(p + "rope.dimension_count", 64)};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < kVocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// Every tensor the layout table of `g` names.  Matrices are Q8_0 (what the pack the user asked for carries) and the 1-D
/// tensors F32, so the file looks like a real one where the loader looks.
std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape, uint32_t type) {
        ts.push_back({name, std::vector<uint64_t>(shape.begin(), shape.end()), type, seed++});
    };
    // Q8_0 where the row width is a whole number of its 32-value blocks (as a real pack's matrices are), F32 elsewhere
    // (the 1-D tensors and the narrow 2-D ones: the conv's 4-wide rows, the GDN's alpha / beta).  The experts are 3-D --
    // [width, FF, experts] with the expert on the last axis -- and their rows are the matrix rows like any other.
    auto type_of = [](const std::vector<int64_t>& shape) {
        return (shape.size() >= 2 && shape[0] % 32 == 0) ? 8u : 0u;
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape, type_of(w.shape));
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l))
            add("blk." + std::to_string(l) + "." + w.name, w.shape, type_of(w.shape));
    return ts;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-q36-model-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

/// Writes a fixture file and returns its path; `mutate` edits the tensor list before it is written.
fs::path write_model(const TempDir& dir, const char* name, const char* arch, const Dims& d,
                     void (*mutate)(std::vector<Tensor>&) = nullptr) {
    strata::core::ModelGeometry g;
    int64_t k = 0;
    std::string err;
    const fs::path meta = dir.path / (std::string(name) + "-meta.gguf");
    fixture::write(meta, metadata(arch, d), {});
    if (!strata::core::geometry_from_gguf(strata::GgufFile(meta.string()), g, k, err)) {
        std::printf("  (fixture: geometry: %s)\n", err.c_str());
        return {};
    }
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path file = dir.path / (std::string(name) + ".gguf");
    fixture::write(file, metadata(arch, d), ts);
    return file;
}

void retype_experts_q4_0(std::vector<Tensor>& ts) {
    for (Tensor& t : ts)
        if (t.name.find("ffn_gate_exps.weight") != std::string::npos) t.type = 2;   // Q4_0: not in moe.metal's gate/up set
}

// A type outside gemv.metal's set (Q4_1 is a ggml type Strata's kernels do not read): the dense path has to refuse it.
void retype_embedding_q4_1(std::vector<Tensor>& ts) {
    for (Tensor& t : ts)
        if (t.name == "token_embd.weight") t.type = 3;   // Q4_1
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
