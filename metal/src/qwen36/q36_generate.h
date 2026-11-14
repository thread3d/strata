// metal/src/qwen36/q36_generate.h - the decode loop: one token in, one token out, the KV window growing behind it.
//
//     for each step:  forward_token(token, pos, n_kv)  ->  logits  ->  pick  ->  that token is the next input
//
// Sampling here is GREEDY (the argmax of the logit row, on the host: the logits are in a host-visible buffer and 248,320 of them
// are not worth a dispatch).  The engine's temperature/top-p/penalty sampler is `strata::qwen35::sampling` on the CUDA side and
// lands here next; what this file has to get right is the loop itself, which is where the KV window, the position and the step
// count can each go wrong by one.
#pragma once

#include "q36_forward.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen36 {

/// The argmax of a logit row (ties to the lower id, as the port's routers do).  `n` is the vocabulary.
int32_t argmax(const float* logits, int n);

/// Runs `n_new` steps after `prompt`.  The window is `min(kv_cap, pos + 1)` rows, so `kv_cap` bounds the context.  The generated
/// ids are appended to `out` (which starts empty) and every step's logits stay in `f.logits`.
bool generate(mtl::Context& cx, const Model& m, const DeviceModel& d, Forward& f, const std::vector<uint32_t>& prompt,
              int n_new, std::vector<uint32_t>& out, std::string& err);

}  // namespace strata::qwen36
