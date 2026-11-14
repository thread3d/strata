// metal/tests/q36_stream_check.cpp - the expert pool: what a fill puts in a slot, and what the map then says.
//
// The claim the streaming path rests on is that a slot is identical whether it was filled at model load or a millisecond ago - so
// a slot's bytes are compared against the same gather the resident path uses (q36_expert_blob::pack) and against the file's own
// slabs, expert by expert and slot by slot.  A wrong offset here hands a token the wrong expert's weights, which no downstream test
// can see: the arithmetic would be right and the answer wrong.
#include "qwen36/q36_stream.h"

#include "q36_fixture.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace q36fix;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // a crash must not take the trace with it
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    TempDir dir;
    const fs::path f = write_model(dir, "stream", "qwen35moe", kSmall);
    if (f.empty()) { std::printf("the fixture could not be built\n"); return 1; }
    strata::qwen36::Model m;
    std::string err;
    if (!strata::qwen36::open(f.string(), m, err)) { std::printf("open: %s\n", err.c_str()); return 1; }

    const strata::qwen36::Layer& L = m.layers[0];
    strata::qwen36::ExpertBlob b;
    b.n_embd = (int) m.g.n_embd;
    b.n_ff = (int) m.g.n_ff;
    b.gu_type = L.gate_exps.type();
    b.down_type = L.down_exps.type();
    b.shape();

    strata::qwen36::ExpertStream st;
    const int kSlots = 4;
    if (!strata::qwen36::stream_init(cx, st, b, kSlots, (int) m.g.n_expert, err)) {
        std::printf("stream_init: %s\n", err.c_str());
        return 1;
    }
    // everything starts unloaded
    {
        const uint32_t* map = st.smap.as<uint32_t>();
        size_t loaded = 0;
        for (int e = 0; e < (int) m.g.n_expert; ++e)
            if (map[e] != strata::qwen36::kNotLoaded) ++loaded;
        check(loaded == 0, "every expert starts unloaded");
    }

    // fill with four experts, then compare each slot against the same gather the resident path packs
    const std::vector<int> ids{3, 1, 2, 3};   // (the repeated 3 is what a verify window does; the fixture has 4 experts)
    const int n = strata::qwen36::stream_load(m, L, st, ids, err);
    check(n == 3, "three distinct experts load from four routed ids (got " + std::to_string(n) + (err.empty() ? "" : ": " + err) + ")");
    {
        const uint32_t* map = st.smap.as<uint32_t>();
        std::vector<uint8_t> want((size_t) b.blob);
        bool all = map[3] == 0 && map[1] == 1 && map[2] == 2;
        for (int r = 0; r < 3 && all; ++r) {
            const int e = ids[(size_t) r];
            if (!strata::qwen36::pack(b, e, L.gate_exps.data(*m.file), L.up_exps.data(*m.file), L.down_exps.data(*m.file),
                                      want.data(), err)) {
                all = false;
                break;
            }
            all = std::memcmp(strata::qwen36::slot_ptr(st, r), want.data(), (size_t) b.blob) == 0;
        }
        check(all, "each slot holds exactly its expert's gate | up | down (same bytes as the resident gather)");
        check(map[0] == strata::qwen36::kNotLoaded, "an expert that was not routed stays unloaded");
    }

    // a second fill reuses the pool from slot 0 and re-points the map
    const std::vector<int> ids2{2, 0, 1};
    const int n2 = strata::qwen36::stream_load(m, L, st, ids2, err);
    check(n2 == 1, "a second fill loads only what is not already in a slot (only 0 is new)");
    {
        const uint32_t* map = st.smap.as<uint32_t>();
        check(map[2] == 2 && map[0] == 0 && map[1] == 1, "the map points each loaded expert at its own slot");
        std::vector<uint8_t> want((size_t) b.blob);
        bool all = strata::qwen36::pack(b, 0, L.gate_exps.data(*m.file), L.up_exps.data(*m.file),
                                        L.down_exps.data(*m.file), want.data(), err) &&
                   std::memcmp(strata::qwen36::slot_ptr(st, 0), want.data(), (size_t) b.blob) == 0;
        check(all, "slot 0 now holds expert 0, not the expert it held before");
    }
    // and a pool smaller than the routing is refused, not silently truncated
    strata::qwen36::ExpertStream small;
    if (!strata::qwen36::stream_init(cx, small, b, 2, (int) m.g.n_expert, err)) { std::printf("init: %s\n", err.c_str()); return 1; }
    const int n3 = strata::qwen36::stream_load(m, L, small, {1, 2, 3}, err);
    check(n3 == 0 && !err.empty(), "a routing wider than the pool is refused with a message (" + err + ")");

    std::printf("%s\n", g_fail ? "q36_stream_check: FAILURES" : "q36_stream_check: all ok");
    return g_fail ? 1 : 0;
}
