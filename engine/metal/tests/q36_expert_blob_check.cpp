// metal/tests/q36_expert_blob_check.cpp - the expert blob layout, and the gather that fills it.  No GPU: this is the agreement
// between the file's three stacked tensors and the single region moe_gu / moe_down read, and a mismatch there is a silently
// wrong expert rather than a crash.
#include "qwen36/q36_expert_blob.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-92s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    strata::qwen36::ExpertBlob b;
    b.n_embd = 2048;
    b.n_ff = 512;
    b.gu_type = 8;    // Q8_0: 34 bytes a 32 values
    b.down_type = 8;
    b.shape();
    check(b.gu_row == 2048 / 32 * 34, "a Q8_0 gate row is 2176 bytes at n_embd 2048");
    check(b.up_off == 512ull * b.gu_row, "up starts one gate matrix in");
    check(b.down_off == 2 * b.up_off, "down starts two matrices in");
    check(b.blob == b.down_off + 2048ull * (512 / 32 * 34), "the blob is gate + up + down");
    check(b.slots(256) == b.blob * 256, "a resident layer's slots are one blob each");

    // the gather, on small tensors with recognisable bytes
    strata::qwen36::ExpertBlob s;
    s.n_embd = 64;
    s.n_ff = 32;
    s.shape();
    const int NE = 3;
    std::vector<uint8_t> gate((size_t) (s.n_ff * s.gu_row * NE)), up(gate.size()), down((size_t) (s.n_embd * s.d_row * NE));
    for (size_t i = 0; i < gate.size(); ++i) {
        gate[i] = (uint8_t) (i * 7 + 1);
        up[i] = (uint8_t) (i * 11 + 3);
    }
    for (size_t i = 0; i < down.size(); ++i) down[i] = (uint8_t) (i * 13 + 5);
    std::vector<uint8_t> dst((size_t) s.blob);
    std::string err;
    bool all = true;
    for (int e = 0; e < NE; ++e) {
        std::memset(dst.data(), 0, dst.size());
        all = all && strata::qwen36::pack(s, e, gate.data(), up.data(), down.data(), dst.data(), err);
        // gate at 0, up at up_off, down at down_off, each exactly its own slab, and nothing else touched
        const uint64_t gu_slab = (uint64_t) s.n_ff * s.gu_row, d_slab = (uint64_t) s.n_embd * s.d_row;
        all = all && std::memcmp(dst.data(), gate.data() + (size_t) e * gu_slab, (size_t) gu_slab) == 0;
        all = all && std::memcmp(dst.data() + s.up_off, up.data() + (size_t) e * gu_slab, (size_t) gu_slab) == 0;
        all = all && std::memcmp(dst.data() + s.down_off, down.data() + (size_t) e * d_slab, (size_t) d_slab) == 0;
        const size_t used = (size_t) s.down_off + (size_t) d_slab;
        for (size_t i = used; i < dst.size(); ++i) all = all && dst[i] == 0;
    }
    check(all, "every expert's gate, up and down land at their offsets and nowhere else");
    check(!strata::qwen36::pack(s, 0, nullptr, up.data(), down.data(), dst.data(), err), "a missing tensor is refused");
    std::printf("%s\n", g_fail ? "q36_expert_blob_check: FAILURES" : "q36_expert_blob_check: all ok");
    return g_fail ? 1 : 0;
}
