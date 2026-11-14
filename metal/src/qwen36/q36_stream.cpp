// metal/src/qwen36/q36_stream.cpp - filling the slot pool a token's routing asks for.
#include "q36_stream.h"

#include <cstring>

namespace strata::qwen36 {

bool stream_init(mtl::Context& cx, ExpertStream& s, const ExpertBlob& b, int n_slot, int n_expert, std::string& err) {
    if (n_slot <= 0 || n_expert <= 0 || b.blob == 0) {
        err = "a stream needs slots, experts and a blob shape";
        return false;
    }
    s.slots = cx.alloc((size_t) b.slots(n_slot));
    s.smap = cx.alloc(4 * (size_t) n_expert);
    if (s.slots.ptr == nullptr || s.smap.ptr == nullptr) {
        err = "the device refused the expert pool";
        return false;
    }
    s.blob = b;
    s.n_slot = n_slot;
    s.n_expert = n_expert;
    uint32_t* map = s.smap.as<uint32_t>();
    for (int e = 0; e < n_expert; ++e) map[e] = kNotLoaded;
    return true;
}

void stream_ids(const mtl::Buffer& ids, int n_tok, int top_k, std::vector<int>& out) {
    out.clear();
    if (ids.ptr == nullptr) return;
    const int* p = ids.as<int>();
    for (int i = 0; i < n_tok * top_k; ++i) {
        const int e = p[i];
        bool seen = false;
        for (int q : out)
            if (q == e) {
                seen = true;
                break;
            }
        if (!seen && e >= 0) out.push_back(e);
    }
}

int stream_load(const Model& m, const Layer& L, ExpertStream& s, const std::vector<int>& ids, std::string& err) {
    uint32_t* map = s.smap.as<uint32_t>();
    int slot = 0, loaded = 0;
    for (int e : ids) {
        if (e < 0 || e >= s.n_expert) {
            err = "expert " + std::to_string(e) + " is outside the model's " + std::to_string(s.n_expert);
            return 0;
        }
        if (map[e] != kNotLoaded) continue;   // (a verify window can route the same expert twice)
        if (slot >= s.n_slot) {
            err = "a token routed to more experts than the pool holds (" + std::to_string(s.n_slot) + ")";
            return 0;
        }
        if (!pack(s.blob, e, L.gate_exps.data(*m.file), L.up_exps.data(*m.file), L.down_exps.data(*m.file),
                  slot_ptr(s, slot), err))
            return 0;
        map[e] = (uint32_t) slot;
        ++slot;
        ++loaded;
    }
    return loaded;
}

}  // namespace strata::qwen36
