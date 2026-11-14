// metal/src/qwen36/q36_stream.h - the experts, streamed instead of resident.
//
// A real Qwen3.6 Q8_0 pack keeps ~34 GiB of routed experts against this card's 8, so they cannot be resident: the streaming path
// gives a layer a POOL of slots and fills them, per token, with only the experts that token routes to.  The blob layout is the
// one q36_expert_blob fixes (gate | up | down), which is exactly what a slot is - so the kernels do not know whether an expert was
// loaded at model load or a millisecond ago, and the resident path (q36_forward) and this one share every dispatch after the fill.
//
// What a fill costs: `top_k` blobs of a layer, copied from the file's mapping into the pool.  For this family that is 8 x ~3.3 MiB
// = ~26 MiB a layer, 30 GDN and 10 attention, so ~1 GiB of copying a token - which is why the design keeps the copy sequential and
// the pool as small as the routing needs rather than caching whole layers.
//
// The slot map the kernels read: `smap[e]` is the slot holding expert e, or 0xFFFFFFFF for an expert that is not loaded.  A caller
// fills it as it fills slots; the kernels' arena path reads it in place of `W + e * gu_stride`.
#pragma once

#include "q36_expert_blob.h"
#include "q36_model.h"
#include "mtl_rt.h"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::qwen36 {

struct ExpertStream {
    mtl::Buffer slots;    // n_slot blobs, back to back at `stride` bytes each
    mtl::Buffer smap;     // n_expert uint32: the arena word for the slot an expert is in, or kNotLoaded
    ExpertBlob blob{};
    uint64_t stride = 0;  // one slot's stride: blob.blob rounded UP to the arena's 16 KiB granule
    int n_slot = 0;
    int n_expert = 0;
};

inline constexpr uint32_t kNotLoaded = 0xFFFFFFFFu;

/// A pool of `n_slot` slots and the map that goes with it (every entry starts as kNotLoaded).
bool stream_init(mtl::Context& cx, ExpertStream& s, const ExpertBlob& b, int n_slot, int n_expert, std::string& err);

/// Fills slots with `ids` (a token's routed experts for ONE LAYER) and points `smap` at them, so that exactly those experts are
/// addressable.  A repeated id is loaded once.
///
/// **The map is cleared first, and that is the whole contract**: a slot holds one layer's expert, and one pool serves every layer of
/// a token (the layers run in sequence), so an entry left from the previous layer would point at THAT layer's weights - the numbers
/// stay finite and the answer is quietly wrong, expert by expert.  `loaded` is the number of blobs copied, which can be fewer than
/// `ids.size()` when a token routes the same expert twice.
int stream_load(const Model& m, const Layer& L, ExpertStream& s, const std::vector<int>& ids, std::string& err);

/// The same fill from three raw tensor mappings (the file's gate / up / down payloads), for a caller that already holds them - a
/// test with a packed blob rather than a model, or an engine that streamed the slabs itself.
int stream_load_raw(const ExpertBlob& b, ExpertStream& s, const std::vector<int>& ids, const uint8_t* gate, const uint8_t* up,
                    const uint8_t* down, std::string& err);

/// The routed ids, off the device and into a host list: the one synchronisation a token needs.  `ids` holds n_tok * top_k ints as
/// moe_route wrote them; the list comes back deduplicated (a verify window routes the same expert for two tokens) and in the order
/// they were found, which is the order stream_load fills slots in.
void stream_ids(const mtl::Buffer& ids, int n_tok, int top_k, std::vector<int>& out);

/// One slot's stride: the blob rounded up to the arena's granule.  moe.metal's arena_slot() encodes a slot as
/// (buffer << 28) | (byte offset / 16384), so a slot that is not a whole number of 16 KiB granules cannot be addressed - the
/// real pack's blob (3,342,336 bytes for n_embd 2048 / n_ff 512 at Q8_0) is exactly 204 granules, and anything else is padded here.
inline constexpr uint64_t kArenaGranule = 16384;
inline uint64_t slot_stride(const ExpertBlob& b) { return (b.blob + kArenaGranule - 1) / kArenaGranule * kArenaGranule; }

/// The arena word for slot `i` in a pool that lives in one buffer: chunk 0, offset i * stride / 16384.
inline uint32_t slot_word(const ExpertStream& s, int i) { return (uint32_t) ((uint64_t) i * s.stride / kArenaGranule); }

/// The host address of slot `i`.
inline uint8_t* slot_ptr(ExpertStream& s, int i) { return s.slots.as<uint8_t>() + (size_t) i * s.stride; }

}  // namespace strata::qwen36
