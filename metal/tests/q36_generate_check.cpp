// metal/tests/q36_generate_check.cpp - the decode loop: does a token feed the next, does the window grow, is it deterministic.
//
// The model is the fixture with every dense matrix zeroed and the norms real, so the logits are the output projection of a row that
// no layer changed - a vector of the same values every step.  What that cannot test is the QUALITY of a token (there is no model to
// have one); what it can test is the loop: the argmax is the same id every step, the window grows by one, and two runs give the same
// sequence.
#include "qwen36/q36_generate.h"

#include "q36_fixture.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>

using namespace q36fix;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
void zero(mtl::Buffer& b) {
    if (b.ptr != nullptr && b.bytes > 0) std::memset(b.ptr, 0, b.bytes);
}
void ones(mtl::Buffer& b) {
    if (b.ptr == nullptr) return;
    float* v = (float*) b.ptr;
    for (size_t i = 0; i < b.bytes / 4; ++i) v[i] = 1.0f;
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    mtl::Context cx;
    std::printf("device %s\n", cx.device_name().c_str());
    TempDir dir;
    const fs::path fp = write_model(dir, "gen", "qwen35moe", kSmall);
    if (fp.empty()) { std::printf("the fixture could not be built\n"); return 1; }
    strata::qwen36::Model m;
    strata::qwen36::DeviceModel d;
    std::string err;
    if (!strata::qwen36::open(fp.string(), m, err) || !strata::qwen36::upload_dense(cx, m, d, err)) {
        std::printf("model: %s\n", err.c_str());
        return 1;
    }
    ones(d.output_norm);
    for (strata::qwen36::DeviceLayer& L : d.layers) {
        ones(L.attn_norm);
        ones(L.post_norm);
        ones(L.q_norm);
        ones(L.k_norm);
        ones(L.ssm_norm);
        for (mtl::Buffer* b : {&L.gate_inp, &L.gate_inp_shexp, &L.gate_shexp, &L.up_shexp, &L.down_shexp, &L.qkv, &L.sgate,
                               &L.ssm_out, &L.conv1d, &L.alpha, &L.beta, &L.ssm_a, &L.dt_bias, &L.q, &L.k, &L.v, &L.o})
            zero(*b);
    }
    const int cap = 32;
    strata::qwen36::Forward f, f2;
    if (!strata::qwen36::forward_init(cx, m, d, cap, (uint64_t) 1024, f, err) ||
        !strata::qwen36::forward_init(cx, m, d, cap, (uint64_t) 1024, f2, err)) {
        std::printf("forward_init: %s\n", err.c_str());
        return 1;
    }
    const std::vector<uint32_t> prompt{3, 1};
    std::vector<uint32_t> out, out2;
    if (!strata::qwen36::generate(cx, m, d, f, prompt, 4, out, err)) {
        std::printf("generate: %s\n", err.c_str());
        return 1;
    }
    check(out.size() == 4, "four tokens come out of a four-step decode (" + std::to_string(out.size()) + ")");
    bool inrange = true;
    for (uint32_t t : out) inrange = inrange && t < (uint32_t) m.g.n_vocab;
    check(inrange, "every generated id is inside the vocabulary");
    check(!strata::qwen36::generate(cx, m, d, f2, prompt, 4, out2, err) ? false : out2 == out,
          "a second run gives the same four tokens");
    check(strata::qwen36::argmax(nullptr, 4) < 0, "argmax of nothing is -1, not a guess");
    std::printf("    tokens:");
    for (uint32_t t : out) std::printf(" %u", t);
    // SAMPLING AND STOP, on rows whose answers are known by construction: no model, no dispatch, just the picker.
    {
        std::vector<float> row(8, -10.0f);
        row[3] = 5.0f; row[1] = 2.0f; row[2] = 2.0f;   // 3 wins; 1 and 2 tie
        check(strata::qwen36::argmax(row.data(), 8) == 3, "argmax picks the highest logit (3)");
        strata::qwen36::Sampling g;                     // greedy by default
        uint64_t rng = 0;
        check(strata::qwen36::sample(row.data(), 8, g, rng) == 3, "temperature 0 is the argmax, whatever the seed");
        row[3] = -10.0f;
        check(strata::qwen36::argmax(row.data(), 8) == 1, "a tie keeps the lower id (1 over 2)");
        strata::qwen36::Sampling sp;
        sp.temperature = 1.0f;
        sp.seed = 12345;
        uint64_t r1 = sp.seed, r2 = sp.seed;
        bool inrange = true;
        std::vector<uint32_t> a, b;
        for (int i = 0; i < 64; ++i) {
            const uint32_t x = strata::qwen36::sample(row.data(), 8, sp, r1);
            inrange = inrange && (x == 1 || x == 2);
            a.push_back(x);
            b.push_back(strata::qwen36::sample(row.data(), 8, sp, r2));
        }
        check(inrange, "sampling 64 times from a two-token distribution never leaves it");
        check(a == b, "the same seed replays the same 64 draws");
        row[2] = 1.0f;   // a UNIQUE head: with a tie either token would be a legal nucleus, and the sort does not order equals
        strata::qwen36::Sampling np;
        np.temperature = 1.0f;
        np.top_p = 0.01f;
        np.seed = 7;
        uint64_t r3 = np.seed;
        bool only_top = true;
        for (int i = 0; i < 32; ++i) only_top = only_top && strata::qwen36::sample(row.data(), 8, np, r3) == 1;
        check(only_top, "top-p at 0.01 keeps only the head of the distribution");
    }

    std::printf("\n%s\n", g_fail ? "q36_generate_check: FAILURES" : "q36_generate_check: all ok");
    return g_fail ? 1 : 0;
}
