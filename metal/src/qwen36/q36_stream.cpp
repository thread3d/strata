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

int stream_load(const Model& m, const Layer& L, ExpertStream& s, const std::vector<int>& ids,
                       ExpertBlob& out_blob, std::string& err) {
    ExpertBlob b = s.blob;
    b.n_embd = (int) m.g.n_embd;
    b.n_ff = (int) m.g.n_ff;
    int gate_t = L.gate_exps.ok() ? L.gate_exps.type() : b.gu_type;
    int up_t = L.up_exps.ok() ? L.up_exps.type() : gate_t;
    int down_t = L.down_exps.ok() ? L.down_exps.type() : b.down_type;
    if (gate_t != up_t) {
        std::fprintf(stderr, "stream_load: layer gate_type=%d up_type=%d MISMATCH\n", gate_t, up_t);
    }
    b.gu_type = gate_t;
    b.down_type = down_t;
    b.shape();
    std::fprintf(stderr, "stream_load: gu_type=%d down_type=%d gu_row=%llu d_row=%llu blob=%llu stride=%llu\n",
                 b.gu_type, b.down_type, (unsigned long long)b.gu_row, (unsigned long long)b.d_row, (unsigned long long)b.blob, (unsigned long long)s.stride);
    if (b.blob > s.stride) {
        err = "an expert blob does not fit the stream slot";
        return 0;
    }
    out_blob = b;
    return stream_load_raw(b, s, ids, L.gate_exps.data(*m.file), L.up_exps.data(*m.file), L.down_exps.data(*m.file), err);
}

int stream_load_raw(const ExpertBlob& b, ExpertStream& s, const std::vector<int>& ids, const uint8_t* gate, const uint8_t* up,
                    const uint8_t* down, std::string& err) {
    if (b.blob > s.stride) {
        err = "an expert blob does not fit the stream slot";
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
