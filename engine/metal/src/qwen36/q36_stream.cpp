// metal/src/qwen36/q36_stream.cpp - filling the slot pool a token's routing asks for.
#include "q36_stream.h"

#include <cstring>

namespace strata::qwen36 {

bool stream_init(mtl::Context& cx, ExpertStream& s, const ExpertBlob& b, int n_slot, int n_expert, std::string& err) {
    if (n_slot <= 0 || n_expert <= 0 || b.blob == 0) {
        err = "a stream needs slots, experts and a blob shape";
        return false;
    }
    s.stride = slot_stride(b);
    s.slots = cx.alloc((size_t) (s.stride * (uint64_t) n_slot));
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
    // The blob this LAYER has, not the one the pool was sized with: a pack may mix formats per layer (Q4_K_M keeps a Q6_K
    // down in layers 0-4 and a Q4_K one from layer 5), and packing a layer's downs at another format's row stride puts
    // the wrong bytes - or bytes past the tensor - into the slot.  The pool's stride only has to be big enough.
    ExpertBlob b;
    b.n_embd = (int) m.g.n_embd;
    b.n_ff = (int) m.g.n_ff;
    if (L.gate_exps.ok()) b.gu_type = L.gate_exps.type();
    if (L.down_exps.ok()) b.down_type = L.down_exps.type();
    b.shape();
    return stream_load_raw(b, s, ids, L.gate_exps.data(*m.file), L.up_exps.data(*m.file), L.down_exps.data(*m.file), err);
}

int stream_load_raw(const ExpertBlob& b, ExpertStream& s, const std::vector<int>& ids, const uint8_t* gate, const uint8_t* up,
                    const uint8_t* down, std::string& err) {
    if (b.blob == 0 || b.blob > s.stride) {   // the layout must fit a slot; the pool is sized by the largest layer blob
        err = "an expert blob of " + std::to_string(b.blob) + " bytes does not fit the pool's " + std::to_string(s.stride);
        return 0;
    }
    uint32_t* map = s.smap.as<uint32_t>();
    // every slot now belongs to THIS layer: forget which expert the previous one left in which slot
    for (int e = 0; e < s.n_expert; ++e) map[e] = kNotLoaded;
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
        if (!pack(b, e, gate, up, down, slot_ptr(s, slot), err)) return 0;
        map[e] = slot_word(s, slot);
        ++slot;
        ++loaded;
    }
    return loaded;
}

}  // namespace strata::qwen36
