// bench/macpro51/llama_ref.cpp - the llama.cpp side of the macpro51 comparison, from source.
//
// Takes the context as TOKEN IDS and prints the greedy continuation as token ids, which is the
// same contract as `strata-q36-metal --tokens ... --n N`.  That lets the two engines be compared
// id for id instead of by eye over decoded text, and it removes the dependency on the
// undocumented `~/q36_llama_dump` binary the first round of measurements used.
//
//   clang++ -O2 -std=c++17 \
//     -I~/projects/llama.cpp/include -I~/projects/llama.cpp/ggml/include \
//     bench/macpro51/llama_ref.cpp \
//     -L~/projects/llama.cpp/build/bin -lllama -lggml -lggml-base \
//     -o ~/llama_ref
//
//   DYLD_LIBRARY_PATH=~/projects/llama.cpp/build/bin ~/llama_ref \
//     --model ~/models/qwen36/Qwen3.6-35B-A3B-Q8_0.gguf --tokens 3,1,42498,456,11 --n 8
//
// Greedy means argmax of the last position's logits: no sampler chain, no temperature, no seed, so
// the output depends only on the weights and the ids.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> parse_ids(const std::string& s) {
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back((llama_token) std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model, toks;
    int n_gen = 16;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) toks = argv[++i];
        else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n_gen = std::atoi(argv[++i]);
    }
    if (model.empty() || toks.empty()) {
        std::fprintf(stderr, "usage: %s --model PATH --tokens a,b,c --n N\n", argv[0]);
        return 2;
    }

    std::vector<llama_token> ids = parse_ids(toks);
    if (ids.empty()) {
        std::fprintf(stderr, "no token ids\n");
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model* m = llama_model_load_from_file(model.c_str(), mp);
    if (m == nullptr) {
        std::fprintf(stderr, "cannot load %s\n", model.c_str());
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(m);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (ids.size() + (size_t) n_gen + 8);
    cp.n_batch = cp.n_ctx;
    cp.n_threads = 24;
    llama_context* ctx = llama_init_from_model(m, cp);
    if (ctx == nullptr) {
        std::fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    std::printf("prompt:");
    for (llama_token t : ids) std::printf(" %d", (int) t);
    std::printf("\n");

    // The prompt goes through in one batch; llama_decode tracks the positions itself.
    if (llama_decode(ctx, llama_batch_get_one(ids.data(), (int32_t) ids.size())) != 0) {
        std::fprintf(stderr, "prompt decode failed\n");
        return 1;
    }

    std::vector<llama_token> gen;
    for (int i = 0; i < n_gen; ++i) {
        const float* logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            std::fprintf(stderr, "no logits at step %d\n", i);
            return 1;
        }
        const int nv = llama_vocab_n_tokens(vocab);
        int best = 0;
        for (int t = 1; t < nv; ++t)
            if (logits[t] > logits[best]) best = t;
        gen.push_back((llama_token) best);
        if (llama_vocab_is_eog(vocab, (llama_token) best)) break;
        llama_token one = (llama_token) best;
        if (llama_decode(ctx, llama_batch_get_one(&one, 1)) != 0) {
            std::fprintf(stderr, "decode failed at step %d\n", i);
            return 1;
        }
    }

    std::printf("gen:");
    for (llama_token t : gen) std::printf(" %d", (int) t);
    std::printf("\n");

    llama_free(ctx);
    llama_model_free(m);
    llama_backend_free();
    return 0;
}
