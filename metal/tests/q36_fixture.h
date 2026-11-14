// metal/tests/q36_fixture.h - a tiny synthetic Qwen3.6-35B-A3B (qwen35moe) GGUF for the host-half and device-half checks.
//
// The file carries the metadata a real pack carries and every tensor the layout table names, at widths small enough to
// build in a fraction of a second: small where the kernels take a run-time width (hidden, FF, expert count, vocab) and at
// the kernel's own value where they do not (head 256, ssm state 128, conv 4) - so a model built here passes the port's
// kernel gate, which is the point: everything downstream is the real code path at a small size.
#pragma once

#include "gguf_fixture.hpp"
#include "strata/core/family_layout.hpp"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace q36fix {

namespace fs = std::filesystem;
using fixture::Kv;
using fixture::Tensor;

struct Dims {
    uint64_t block_count, embd, heads, heads_kv, key_len, experts, used, ff, ff_sh, conv, state, groups, v_heads,
        inner, interval, nextn;
};

/// 5 blocks, of which the last is the MTP block: 4 trunk layers, layer 3 the full-attention one.
inline constexpr Dims kTiny = {5, 64, 2, 1, 256, 4, 2, 32, 32, 4, 128, 2, 4, 512, 4, 1};
inline constexpr uint64_t kVocab = 64;
/// A second model, wide enough to run the ARITHMETIC on: every matrix row is a whole number of the gemv tiles' units (n_embd 512,
/// n_ff 128), which kTiny's 64/32 are not - kTiny is for the loader tests, where nothing is multiplied.
inline constexpr Dims kSmall = {5, 512, 2, 1, 256, 8, 2, 128, 128, 4, 128, 2, 4, 512, 4, 1};
//                                              ^ 8 experts: the router's top-8 needs eight to pick from, or four of its picks
//                                                fall outside a smaller model and the pool's range check refuses them

inline std::vector<Kv> metadata(const char* arch, const Dims& d) {
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
    kv.push_back(fixture::str_array("tokenizer.ggml_tokens", tokens));
    return kv;
}

/// Q8_0 where the row width is a whole number of its 32-value blocks (as a real pack's matrices are), F32 elsewhere (the
/// 1-D tensors and the narrow 2-D ones: the conv's 4-wide rows, the GDN's alpha / beta).  The experts are 3-D -
/// [width, FF, experts] with the expert on the last axis - and their rows are matrix rows like any other.
/// NOTE for arithmetic tests: the payload of every tensor here is a BYTE PATTERN (fixture::Tensor's seed), which is meaningless
/// as an F32 - denormals, 1e38, NaN.  That is fine for the loader checks, which only read shapes, types and bytes; a test that
/// MULTIPLIES with these tensors must overwrite the ones it uses (q36_forward_check fills the norms with 1.0 for exactly this
/// reason - a norm weight of 1e24 turns a well-formed row into garbage and then into NaN).
inline uint32_t type_of(const std::vector<int64_t>& shape) {
    return (shape.size() >= 2 && shape[0] % 32 == 0) ? 8u : 0u;
}

inline std::vector<Tensor> tensors_of(const strata::core::ModelGeometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        ts.push_back({name, std::vector<uint64_t>(shape.begin(), shape.end()), type_of(shape), seed++});
    };
    for (const auto& w : strata::core::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const auto& w : strata::core::layer_tensors(g, l))
            add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-q36-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

/// Writes `<name>.gguf` with the tiny model's metadata and tensors (`mutate` edits the tensor list first).  Empty path
/// when the fixture itself cannot be built (the metadata was refused).
inline fs::path write_model(const TempDir& dir, const char* name, const char* arch = "qwen35moe",
                            const Dims& d = kTiny, void (*mutate)(std::vector<Tensor>&) = nullptr) {
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

}  // namespace q36fix
