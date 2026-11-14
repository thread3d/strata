// metal/src/qwen36/q36_generate.h - the decode loop: one token in, one token out, the KV window growing behind it.
//
//     for each step:  forward_token(token, pos, n_kv)  ->  logits  ->  pick  ->  that token is the next input
//
// Sampling here is GREEDY by default (the argmax of the logit row, on the host: the logits are in a host-visible buffer and 248,320 of them are
// not worth a dispatch) - ties to the lower id, the rule the port's routers use.  Temperature and nucleus sampling are opt-in (§Sampling), and
// a `stop` set ends the loop the way a real engine does: the model's own end-of-turn token is not something to generate past.
#pragma once

#include "q36_forward.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen36 {

/// How the next token is chosen.  `temperature == 0` is greedy and byte-identical to the port's first version; anything else samples.
struct Sampling {
    float temperature = 0.0f;   // 0 = greedy (argmax); > 0 divides the logits
    float top_p = 1.0f;         // nucleus: keep the smallest set whose probability sums past this (1 = off)
    uint64_t seed = 0;          // the PRNG's seed, so a sampled run can be repeated
    std::vector<uint32_t> stop; // tokens that END the answer; they are not appended to `out`
};

/// The argmax of a logit row (ties to the lower id, as the port's routers do).  `n` is the vocabulary.
int32_t argmax(const float* logits, int n);

/// A token from the row under `s`, deterministic for the same seed.
uint32_t sample(const float* logits, int n, const Sampling& s, uint64_t& rng);

/// Runs `n_new` steps after `prompt`, stopping early at a token in `s.stop`.  The window is `min(kv_cap, pos + 1)` rows, so `kv_cap` bounds the
/// context.  The generated ids are appended to `out` (which starts empty); `f.logits` holds the last step's row.
bool generate(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, const std::vector<uint32_t>& prompt,
              int n_new, std::vector<uint32_t>& out, const Sampling& s, std::string& err);

inline bool generate(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, const std::vector<uint32_t>& prompt,
                     int n_new, std::vector<uint32_t>& out, std::string& err) {
    return generate(cx, m, d, f, prompt, n_new, out, Sampling{}, err);
}

}  // namespace strata::qwen36
