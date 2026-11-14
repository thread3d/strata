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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// The byte-level BPE alphabet: a token's string holds its bytes remapped to printable codepoints (the GPT-2 table every Qwen
// tokenizer ships), so decoding is that map undone and the bytes written out.  Enough to READ a continuation; merges are not needed
// for that, because the vocabulary already holds whole pieces.
std::string detok(const std::vector<std::string>& vocab, uint32_t id, const std::vector<int>& byte_of) {
    if ((size_t) id >= vocab.size()) return "<id ?>";
    std::string out;
    for (size_t i = 0; i < vocab[id].size();) {
        const unsigned char c = (unsigned char) vocab[id][i];
        int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : 4;
        uint32_t cp = len == 1 ? c : len == 2 ? ((c & 0x1f) << 6) : len == 3 ? ((c & 0x0f) << 12) : ((c & 0x07) << 18);
        for (int k = 1; k < len && i + (size_t) k < vocab[id].size(); ++k) cp = (cp << 6) | ((unsigned char) vocab[id][i + (size_t) k] & 0x3f);
        i += (size_t) len;
        if ((size_t) cp < byte_of.size() && byte_of[(size_t) cp] >= 0) out.push_back((char) byte_of[(size_t) cp]);
        else return out + "<?>";
    }
    return out;
}

std::vector<int> byte_table() {
    std::vector<int> byte_of(512, -1), cp_of(256, 0);
    int n = 0;
    for (int b = 0; b < 256; ++b) cp_of[(size_t) b] = b;
    for (int b = 0; b < 256; ++b)
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) cp_of[(size_t) b] = 256 + n++;
    for (int b = 0; b < 256; ++b) byte_of[(size_t) cp_of[(size_t) b]] = b;
    return byte_of;
}

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
    bool describe = false, no_stream = false, text = false;
    std::string prompt_text;
    int check_embed = -1, check_output = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : (std::fprintf(stderr, "%s needs a value\n", a.c_str()), std::exit(2), std::string()); };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens_s = next();
        else if (a == "--n") n_new = std::atoi(next().c_str());
        else if (a == "--kv-cap") kv_cap = std::atoi(next().c_str());
        else if (a == "--resident-byte-budget") budget_gib = std::atof(next().c_str());
        else if (a == "--describe") describe = true;
    else if (a == "--text") text = true;
    else if (a == "--prompt") prompt_text = next();
    else if (a == "--check-embed") check_embed = (int) std::atoi(next().c_str());
    else if (a == "--check-output") check_output = (int) std::atoi(next().c_str());
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

    if (text || !prompt_text.empty()) strata::keep_all_meta_arrays() = true;   // a detokeniser needs the whole vocabulary
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

    if (check_embed >= 0) {
        // The embedding gather against the file's own bytes: the row decoded on the host and compared with what the kernel produced.
        // A real pack's row of 248,320 has no fixture to vouch for it, and everything downstream starts here.
        if (!strata::qwen36::embed_token(cx, m, d, f, (uint32_t) check_embed, err)) {
            std::printf("embed: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> want((size_t) m.g.n_embd);
        ref::dequant(m.embed.type(),
                     m.embed.data(*m.file) + (size_t) check_embed * ref::row_bytes(m.embed.type(), (int) m.g.n_embd),
                     want.data(), want.size());
        const float* got = (const float*) f.x.ptr;
        double num = 0.0, den = 0.0;
        for (int i = 0; i < (int) m.g.n_embd; ++i) {
            const double e = (double) got[i] - (double) want[(size_t) i];
            num += e * e;
            den += (double) want[(size_t) i] * (double) want[(size_t) i];
        }
        std::printf("embed row %d: got %g %g %g | file %g %g %g | rel L2 %.3e\n", check_embed, (double) got[0], (double) got[1],
                    (double) got[2], (double) want[0], (double) want[1], (double) want[2],
                    den > 0 ? std::sqrt(num / den) : std::sqrt(num));
        return 0;
    }

    if (check_output > 0) {
        // The OUTPUT projection, checked against the file: logits[k] must be the dot of the final normed row with the file's row k of
        // output.weight.  That verifies the matrix's orientation and its vocabulary-row indexing on a real 248,320-row matrix - the
        // one thing no synthetic test has ever done - given the h this engine produced.
        if (!strata::qwen36::forward_token(cx, m, d, f, 760, 0, 1, err)) {
            std::printf("forward: %s\n", err.c_str());
            return 1;
        }
        const float* h = (const float*) f.xh.ptr;
        const float* lg = (const float*) f.logits.ptr;
        const size_t rb = ref::row_bytes(m.output.type(), (int) m.g.n_embd);
        std::vector<float> row((size_t) m.g.n_embd);
        int worst = 0;
        double worst_rel = 0.0;
        for (int k = 0; k < check_output; ++k) {
            ref::dequant(m.output.type(), m.output.data(*m.file) + (size_t) k * rb, row.data(), row.size());
            double d = 0.0, nn = 0.0;
            for (int i = 0; i < (int) m.g.n_embd; ++i) {
                d += (double) row[(size_t) i] * (double) h[i];
                nn += (double) row[(size_t) i] * (double) row[(size_t) i];
            }
            const double rel = std::fabs(d - (double) lg[k]) / (std::fabs(d) + 1e-9);
            if (rel > worst_rel) { worst_rel = rel; worst = k; }
        }
        // and where the engine's own best logit sits, to say whether the head is at least self-consistent
        int best = 0;
        for (int k = 1; k < (int) m.g.n_vocab; ++k)
            if (lg[k] > lg[best]) best = k;
        std::printf("output projection: %d rows checked, worst rel %.3e at row %d; engine argmax %d (logit %g, h[0] %g)\n", check_output,
                    worst_rel, worst, best, (double) lg[best], (double) h[0]);
        return worst_rel < 2e-5 ? 0 : 1;
    }

    std::vector<uint32_t> prompt = parse_ids(tokens_s.c_str());
    if (!prompt_text.empty()) {
        // Encode by greedy longest match over the pack's own vocabulary: each token's decoded string, the longest piece that fits.
        // A crude encoder, but it needs no merge table - and what it is FOR is finding the ids of a phrase so the continuation can be
        // read, not reproducing a tokenizer exactly.
        const strata::MetaValue* tv = m.file->meta().get("tokenizer.ggml.tokens");
        if (tv == nullptr || tv->items.empty()) {
            std::printf("--prompt needs the vocabulary, which this pack did not give (%zu entries)\n", tv ? tv->items.size() : 0);
            return 1;
        }
        const std::vector<int> bt = byte_table();
        std::vector<std::string> piece;
        piece.reserve(tv->items.size());
        for (const auto& it : tv->items) piece.push_back(detok(([&] {
            static std::vector<std::string> v;
            v.clear();
            for (const auto& x : tv->items) v.push_back(x.s);
            return v;
        })(), (uint32_t) piece.size(), bt));
        size_t at = 0;
        while (at < prompt_text.size()) {
            size_t best = 0, bid = 0;
            for (size_t i = 0; i < piece.size(); ++i) {
                const std::string& p = piece[i];
                if (p.size() <= best || p.size() > prompt_text.size() - at) continue;
                if (prompt_text.compare(at, p.size(), p) == 0) { best = p.size(); bid = i; }
            }
            if (best == 0) { std::printf("--prompt: no token covers \"%s\"\n", prompt_text.c_str() + at); return 1; }
            prompt.push_back((uint32_t) bid);
            at += best;
        }
        std::printf("prompt \"%s\" is %zu token(s):", prompt_text.c_str(), prompt.size());
        for (uint32_t t : prompt) std::printf(" %u", t);
        std::printf("\n");
    }
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
    if (text) {   // the same ids as text, through the pack's own vocabulary
        const strata::MetaValue* tv = m.file->meta().get("tokenizer.ggml.tokens");
        if (tv == nullptr) {
            std::printf("text: this pack has no tokenizer.ggml.tokens\n");
            return 0;
        }
        std::printf("text: the vocabulary holds %llu entries, the model %lld\n", (unsigned long long) tv->items.size(),
                    (long long) m.g.n_vocab);
        std::vector<std::string> vocab;
        vocab.reserve(tv->items.size());
        for (const auto& it : tv->items) vocab.push_back(it.s);
        const std::vector<int> bt = byte_table();
        std::printf("text:");
        for (uint32_t t : out) std::printf("%s", detok(vocab, t, bt).c_str());
        std::printf("\n");
    }
    return 0;
}
