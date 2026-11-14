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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    int check_embed = -1, check_output = 0, check_bind = -1, check_expert = -1;
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
    else if (a == "--check-bind") check_bind = (int) std::atoi(next().c_str());
    else if (a == "--check-expert") check_expert = (int) std::atoi(next().c_str());
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

    if (check_bind >= 0) {
        // Is the binding right for a REAL layer?  Every DeviceLayer buffer must hold exactly its tensor's file bytes - upload_dense
        // copies them, so a byte comparison convicts a wrong tensor, a wrong offset or a wrong order, none of which the fixture's
        // uniform shapes can reveal.
        if (check_bind >= (int) m.layers.size() || check_bind >= (int) d.layers.size()) {
            std::printf("no layer %d\n", check_bind);
            return 1;
        }
        const strata::qwen36::Layer& L = m.layers[(size_t) check_bind];
        const strata::qwen36::DeviceLayer& D = d.layers[(size_t) check_bind];
        struct Pair { const char* name; const strata::qwen36::TensorRef* t; const mtl::Buffer* b; };
        std::vector<Pair> pairs{{"attn_norm", &L.attn_norm, &D.attn_norm}, {"post_norm", &L.post_norm, &D.post_norm},
                                {"ffn_gate_inp", &L.gate_inp, &D.gate_inp},
                                {"ffn_gate_inp_shexp", &L.gate_inp_shexp, &D.gate_inp_shexp},
                                {"ffn_gate_shexp", &L.gate_shexp, &D.gate_shexp},
                                {"ffn_up_shexp", &L.up_shexp, &D.up_shexp},
                                {"ffn_down_shexp", &L.down_shexp, &D.down_shexp},
                                {"attn_qkv", &L.qkv, &D.qkv}, {"ssm_out", &L.ssm_out, &D.ssm_out},
                                {"ssm_alpha", &L.alpha, &D.alpha}, {"ssm_beta", &L.beta, &D.beta},
                                {"ssm_conv1d", &L.conv1d, &D.conv1d}, {"ssm_norm", &L.ssm_norm, &D.ssm_norm},
                                {"attn_q", &L.q, &D.q}, {"attn_k", &L.k, &D.k}, {"attn_v", &L.v, &D.v},
                                {"attn_output", &L.o, &D.o}, {"q_norm", &L.q_norm, &D.q_norm}, {"k_norm", &L.k_norm, &D.k_norm}};
        int bad = 0, checked = 0;
        for (const Pair& pr : pairs) {
            if (pr.b->ptr == nullptr || pr.b->bytes == 0) continue;
            if (pr.t != nullptr) {
                if (!pr.t->ok()) { std::printf("  %-20s DEVICE BUFFER BUT NO TENSOR\n", pr.name); ++bad; continue; }
                const uint8_t* f = pr.t->data(*m.file);
                const size_t n = (size_t) std::min<uint64_t>(pr.b->bytes, pr.t->bytes());
                if (std::memcmp(pr.b->ptr, f, n) != 0) { std::printf("  %-20s MISMATCH (%zu bytes)\n", pr.name, n); ++bad; }
                ++checked;
            }
        }
        std::printf("layer %d bindings: %d checked, %d wrong (%s, %zu tensors)\n", check_bind, checked, bad,
                    L.kind == strata::core::LayerKind::Gdn ? "GDN" : (L.kind == strata::core::LayerKind::Attention ? "attention" : "MTP"),
                    m.layers.size());
        return bad == 0 ? 0 : 1;
    }

    if (check_expert >= 0) {
        // THE HOST'S ANSWER for one real expert, in double, straight from the file: gate/up/down of expert `check_expert` of layer 0
        // against a realistic MoE input row.  The question this settles: is the routed block ~1e-3 because the KERNELS are wrong at the
        // real 2048/512 widths, or because the ~1e-3 is what these weights actually give?  (The MoE input is the layer's own first
        // normed row - rms(embedding row) * post_norm - which for layer 0 is within a thousandth of what the block really sees, and
        // this is a magnitude question, not a parity one.)
        const strata::qwen36::Layer& L0 = m.layers[0];
        const int N = (int) m.g.n_embd, FF = (int) m.g.n_ff;
        // NOTE: dequantise into FLOAT storage and widen afterwards.  The first version declared these as vectors of double and
        // handed `(float*) x.data()` to ref::dequant, so the floats were written into the first half of a double array and read back as
        // doubles - which produced "gate max 1.4e-12" out of perfectly good weights (max |w| 0.033) and cost a round chasing a defect
        // that was in this check, not in the engine.
        std::vector<float> xf((size_t) N), pf((size_t) N);
        std::vector<double> x((size_t) N), h((size_t) N), pn((size_t) N);
        ref::dequant(m.embed.type(), m.embed.data(*m.file) + (size_t) 760 * ref::row_bytes(m.embed.type(), N), xf.data(), (size_t) N);
        ref::dequant(L0.post_norm.type(), L0.post_norm.data(*m.file), pf.data(), (size_t) N);
        for (int i = 0; i < N; ++i) { x[(size_t) i] = (double) xf[(size_t) i]; pn[(size_t) i] = (double) pf[(size_t) i]; }
        double ss = 0.0;
        for (int i = 0; i < N; ++i) ss += x[(size_t) i] * x[(size_t) i];
        const double rs = 1.0 / std::sqrt(ss / (double) N + (double) m.eps);
        for (int i = 0; i < N; ++i) h[(size_t) i] = x[(size_t) i] * rs * pn[(size_t) i];
        const size_t gu_row = ref::row_bytes(L0.gate_exps.type(), N);
        const size_t d_row = ref::row_bytes(L0.down_exps.type(), FF);
        const size_t gsz = (size_t) FF * gu_row, dsz = (size_t) N * d_row;
        const uint8_t* src = L0.gate_exps.data(*m.file);
        std::vector<float> rowmax((size_t) N);
        auto dot = [&](const uint8_t* slab, size_t rowbytes, int rows, const std::vector<double>& v) {
            std::vector<double> out((size_t) rows, 0.0);
            for (int r = 0; r < rows; ++r) {
                std::vector<float> w((size_t) N);
                ref::dequant(L0.gate_exps.type(), slab + (size_t) r * rowbytes, w.data(), (size_t) N);
                double acc = 0.0;
                for (int i = 0; i < N; ++i) acc += (double) w[(size_t) i] * v[(size_t) i];
                out[(size_t) r] = acc;
            }
            return out;
        };
        // gate and up of this expert: the tensor is [n_embd, n_ff, n_expert], expert outermost
        const uint8_t* g = src + (size_t) check_expert * gsz;
        const uint8_t* u = L0.up_exps.data(*m.file) + (size_t) check_expert * gsz;
        const std::vector<double> gate = dot(g, gu_row, FF, h);
        const std::vector<double> up = dot(u, gu_row, FF, h);
        std::vector<double> act((size_t) FF);
        for (int i = 0; i < FF; ++i) {
            const double gv = gate[(size_t) i];
            act[(size_t) i] = (gv / (1.0 + std::exp(-gv))) * up[(size_t) i];
        }
        // down: [n_ff, n_embd, n_expert]
        const uint8_t* d = L0.down_exps.data(*m.file) + (size_t) check_expert * dsz;
        std::vector<double> y((size_t) N, 0.0);
        double ymax = 0.0, gmax = 0.0;
        for (int r = 0; r < N; ++r) {
            std::vector<float> w((size_t) FF);
            ref::dequant(L0.down_exps.type(), d + (size_t) r * d_row, w.data(), (size_t) FF);
            double acc = 0.0;
            for (int i = 0; i < FF; ++i) acc += (double) w[(size_t) i] * act[(size_t) i];
            y[(size_t) r] = acc;
            ymax = std::max(ymax, std::fabs(acc));
        }
        for (int i = 0; i < FF; ++i) gmax = std::max(gmax, std::fabs(gate[(size_t) i]));
        std::printf("expert %d of layer 0 on the host: gate max %g, down max %g (y[0..3] %g %g %g %g)\n", check_expert, gmax, ymax,
                    y[0], y[1], y[2], y[3]);
        // WHAT the reader sees: the type it decodes with, the tensor's declared size against the stride's arithmetic, and the first
        // block of two experts - a denormal scale here means the wrong bytes, not small weights.
        const char* tn = L0.gate_exps.ok() ? L0.gate_exps.t->type_name() : nullptr;
        std::printf("  gate_exps: type %s (%d), tensor bytes %llu, arithmetic %llu = n_expert %lld * n_ff %lld * gu_row %zu, down bytes %llu\n",
                    tn ? tn : "?", (int) L0.gate_exps.type(), (unsigned long long) L0.gate_exps.bytes(),
                    (unsigned long long) ((uint64_t) m.g.n_expert * gsz), (long long) m.g.n_expert, (long long) FF, gu_row,
                    (unsigned long long) L0.down_exps.bytes());
        for (int e2 : {0, check_expert}) {
            const uint8_t* q = src + (size_t) e2 * gsz;
            std::vector<float> first((size_t) 32);
            ref::dequant(L0.gate_exps.type(), q, first.data(), first.size());
            double fmax = 0.0;
            for (float v : first) fmax = std::max(fmax, (double) std::fabs(v));
            std::printf("  expert %d block 0: bytes %02x %02x %02x %02x ... %02x | scale-as-halves %g | max |w| of the 32 = %g\n", e2, q[0],
                        q[1], q[2], q[3], q[33], (double) (q[0] | (q[1] << 8)), fmax);
        }
        // THE ROUTER'S OWN ANSWER, on the host: the same top-8 from ffn_gate_inp . h, in double.  The engine prints its plan
        // (plan[TOKENT + k]); these two lists must be the same experts in the same order for the token to be routed right.
        {
            const size_t rr = ref::row_bytes(L0.gate_inp.type(), N);
            std::vector<double> lg((size_t) m.g.n_expert);
            std::vector<float> w((size_t) N);
            for (int e = 0; e < (int) m.g.n_expert; ++e) {
                ref::dequant(L0.gate_inp.type(), L0.gate_inp.data(*m.file) + (size_t) e * rr, w.data(), (size_t) N);
                double acc = 0.0;
                for (int i = 0; i < N; ++i) acc += (double) w[(size_t) i] * h[(size_t) i];
                lg[(size_t) e] = acc;
            }
            std::vector<int> order((size_t) m.g.n_expert);
            for (int e = 0; e < (int) m.g.n_expert; ++e) order[(size_t) e] = e;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
            std::printf("  the router's own top-8 on the host:");
            for (int k = 0; k < (int) m.top_k; ++k) std::printf(" %d", order[(size_t) k]);
            std::printf("  (top logits %g %g %g)\n", lg[(size_t) order[0]], lg[(size_t) order[1]], lg[(size_t) order[2]]);
        }
        return 0;
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
