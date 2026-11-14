// metal/src/qwen36/q36_generate.cpp - the loop, and nothing else.
#include "q36_generate.h"

#include <cstdio>

namespace strata::qwen36 {

int32_t argmax(const float* logits, int n) {
    if (logits == nullptr || n <= 0) return -1;
    int32_t best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i)
        if (logits[i] > bv) {   // strictly greater: ties keep the lower id
            bv = logits[i];
            best = i;
        }
    return best;
}

bool generate(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, const std::vector<uint32_t>& prompt,
              int n_new, std::vector<uint32_t>& out, std::string& err) {
    out.clear();
    if (prompt.empty()) {
        err = "generate needs at least one prompt token";
        return false;
    }
    if (n_new <= 0) return true;
    const int vocab = (int) m.g.n_vocab;
    int pos = 0;
    uint32_t token = prompt[0];
    // The prompt goes through the model one token at a time (chunked prefill is the next piece of work, not this one), and the
    // LAST prompt token's logits are what produces the first new token.
    for (size_t i = 0; i < prompt.size(); ++i) {
        const int n_kv = f.kv_cap < pos + 1 ? f.kv_cap : pos + 1;
        token = prompt[i];
        if (!forward_token(cx, m, d, f, token, pos, n_kv, err)) return false;
        ++pos;
    }
    for (int s = 0; s < n_new; ++s) {
        const int32_t next = argmax((const float*) f.logits.ptr, vocab);
        if (next < 0) {
            err = "the sampler had no logits to read";
            return false;
        }
        out.push_back((uint32_t) next);
        if (s + 1 == n_new) break;
        const int n_kv = f.kv_cap < pos + 1 ? f.kv_cap : pos + 1;
        if (!forward_token(cx, m, d, f, (uint32_t) next, pos, n_kv, err)) return false;
        ++pos;
    }
    return true;
}

}  // namespace strata::qwen36
