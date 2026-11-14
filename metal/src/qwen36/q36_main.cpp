// metal/src/qwen36/q36_main.cpp - strata-q36-metal: open a Qwen3.6-35B-A3B Q8_0 pack and generate tokens on this Metal runtime.
//
//   strata-q36-metal --model PATH [--tokens a,b,c] [--n N] [--kv-cap N] [--resident-byte-budget GiB] [--describe] [--no-stream]
//
// What it is for: the engine's own entry point on this card, and the program the first real run is made with.  It prints the model it
// read, the plan it chose (resident or streamed experts), then the tokens and the time - so a run says what it did rather than only
// what came out.
#include "q36_forward.h"
#include "q36_generate.h"
#include "q36_model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::vector<uint32_t> parse_ids(const char* s) {
    std::vector<uint32_t> out;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) break;
        out.push_back((uint32_t) v);
        p = end;
        if (*p == ',') ++p;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model, tokens_s = "3,1";
    int n_new = 8, kv_cap = 64;
    double budget_gib = 2.0;
    bool describe = false, no_stream = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : (std::fprintf(stderr, "%s needs a value\n", a.c_str()), std::exit(2), std::string()); };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--n") n_new = std::atoi(next().c_str());
        else if (a == "--kv-cap") kv_cap = std::atoi(next().c_str());
        else if (a == "--resident-byte-budget") budget_gib = std::atof(next().c_str());
        else if (a == "--describe") describe = true;
        else if (a == "--no-stream") no_stream = true;
        else { std::fprintf(stderr, "strata-q36-metal: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-q36-metal --model PATH [--tokens a,b,c] [--n N] [--kv-cap N] [--describe]\n");
        return 2;
    }

    mtl::Context cx;
    std::printf("device: %s (%llu-byte buffers, %d-lane simdgroups)\n", cx.device_name().c_str(),
                (unsigned long long) cx.max_buffer_length(), cx.simd_width());

    strata::qwen36::Model m;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!strata::qwen36::open(model, m, err)) {
        std::printf("cannot read the model: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s", strata::qwen36::describe(m).c_str());
    strata::qwen36::DeviceModel d;
    if (!strata::qwen36::upload_dense(cx, m, d, err)) {
        std::printf("cannot upload the dense half: %s\n", err.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("loaded %.2f GiB of dense weights in %.1f s\n", (double) d.dense_bytes / 1073741824.0, load_s);
    if (describe) return 0;

    const uint64_t budget = no_stream ? ~uint64_t{0} : (uint64_t) (budget_gib * 1073741824.0);
    strata::qwen36::Forward f;
    t0 = std::chrono::steady_clock::now();
    if (!strata::qwen36::forward_init(cx, m, d, kv_cap, budget, f, err)) {
        std::printf("cannot start the engine: %s\n", err.c_str());
        return 1;
    }
    const double init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("engine: %s experts, %d-slot pool, %.1f MiB of state and scratch, %.1f s\n",
                f.streaming ? "STREAMED (a pool filled per layer)" : "resident", f.streaming ? (int) m.top_k : 0,
                (double) f.resident_bytes() / 1048576.0, init_s);

    const std::vector<uint32_t> prompt = parse_ids(tokens_s.c_str());
    std::vector<uint32_t> out;
    t0 = std::chrono::steady_clock::now();
    if (!strata::qwen36::generate(cx, m, d, f, prompt, n_new, out, err)) {
        std::printf("generation failed: %s\n", err.c_str());
        return 1;
    }
    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("prompt %zu token(s) + %zu generated in %.2f s (%.3f s a step)\n", prompt.size(), out.size(), gen_s,
                gen_s / (double) (prompt.size() + out.size()));
    std::printf("tokens:");
    for (uint32_t t : out) std::printf(" %u", t);
    std::printf("\n");
    return 0;
}
