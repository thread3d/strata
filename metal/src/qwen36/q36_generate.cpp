// metal/src/qwen36/q36_generate.cpp - the loop, and nothing else.
#include "q36_generate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

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

uint32_t sample(const float* logits, int n, const Sampling& s, uint64_t& rng) {
    if (s.temperature <= 0.0f) return (uint32_t) argmax(logits, n);
    // temperature, then a nucleus over the sorted probabilities; the RNG is splitmix64 so a seed reproduces a run on any platform.
    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>((size_t) n, 256), idx.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    const size_t k = std::min<size_t>((size_t) n, 256);   // the nucleus lives at the top of most distributions
    std::vector<double> w(k);
    double mx = (double) logits[idx[0]];
    for (size_t i = 0; i < k; ++i) w[i] = std::exp(((double) logits[idx[i]] - mx) / (double) s.temperature);
    double sum = 0.0;
    for (double v : w) sum += v;
    for (double& v : w) v /= sum;
    size_t keep = k;
    if (s.top_p < 1.0f) {
        double acc = 0.0;
        for (size_t i = 0; i < k; ++i) {
            acc += w[i];
            if (acc >= (double) s.top_p) { keep = i + 1; break; }
        }
    }
    double r = 0.0, acc = 0.0;
    for (size_t i = 0; i < keep; ++i) acc += w[i];
    rng += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    r = (double) (z >> 11) / 9007199254740992.0;   // [0, 1)
    r *= acc;
    double c = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        c += w[i];
        if (r < c) return (uint32_t) idx[i];
    }
    return (uint32_t) idx[keep - 1];
}

bool generate(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, const std::vector<uint32_t>& prompt,
              int n_new, std::vector<uint32_t>& out, const Sampling& s, std::string& err) {
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
    uint64_t rng = s.seed;
    for (int step = 0; step < n_new; ++step) {
        const uint32_t next = sample((const float*) f.logits.ptr, vocab, s, rng);
        if ((int32_t) next < 0) {
            err = "the sampler had no logits to read";
            return false;
        }
        if (std::find(s.stop.begin(), s.stop.end(), next) != s.stop.end()) return true;   // the model ended its turn: not a token to keep
        out.push_back(next);
        if (step + 1 == n_new) break;
        const int n_kv = f.kv_cap < pos + 1 ? f.kv_cap : pos + 1;
        if (!forward_token(cx, m, d, f, next, pos, n_kv, err)) return false;
        ++pos;
    }
    return true;
}

}  // namespace strata::qwen36
