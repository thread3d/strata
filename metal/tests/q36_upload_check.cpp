// metal/tests/q36_upload_check.cpp - the Qwen3.6 device half: upload a synthetic model's dense tensors and prove the
// device holds exactly the file's bytes.
//
// What can go wrong in an upload is quiet: a tensor missed, a buffer bound from a page inside another allocation (two
// MTLBuffers over the same pages, one write invisible to the other's readers - ds4_finish.cpp's rule), a length taken from
// the wrong field, a Q8_0 matrix repacked when the kernels wanted the file's own blocks.  So this compares every uploaded
// buffer against the tensor's payload in the file's mapping, byte for byte, and fails on the first difference with the
// tensor's name.
//
// The expert tensors are deliberately absent: on this card they cannot be resident, and q36_device.cpp says so.
#include "qwen36/q36_device.h"

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

/// Compares one uploaded buffer with the tensor it came from: the same bytes, the same length.
bool same(const strata::qwen36::Model& m, const strata::qwen36::TensorRef& r, const mtl::Buffer& b,
          const char* what, size_t& compared, std::string& why) {
    if (!r.ok()) return true;   // a tensor the layer does not have
    if (b.ptr == nullptr) {
        why = std::string(what) + ": no buffer";
        return false;
    }
    const uint64_t bytes = r.bytes();
    if (b.bytes < bytes) {
        why = std::string(what) + ": buffer is " + std::to_string(b.bytes) + " bytes, tensor is " + std::to_string(bytes);
        return false;
    }
    const uint8_t* src = r.data(*m.file);
    if (std::memcmp(b.ptr, src, (size_t) bytes) != 0) {
        size_t at = 0;
        while (at < (size_t) bytes && ((const uint8_t*) b.ptr)[at] == src[at]) ++at;
        why = std::string(what) + ": differs at byte " + std::to_string(at) + " of " + std::to_string(bytes);
        return false;
    }
    compared += bytes;
    return true;
}

}  // namespace

int main() {
    mtl::Context cx;
    std::printf("device %s (%llu-byte buffers, simd width %d)\n", cx.device_name().c_str(),
                (unsigned long long) cx.max_buffer_length(), cx.simd_width());
    TempDir dir;
    const fs::path f = write_model(dir, "upload");
    if (f.empty()) {
        std::printf("the fixture could not be built\n");
        return 1;
    }

    strata::qwen36::Model m;
    std::string err;
    if (!strata::qwen36::open(f.string(), m, err)) {
        std::printf("open failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("%s", strata::qwen36::describe(m).c_str());

    strata::qwen36::DeviceModel d;
    const bool up = strata::qwen36::upload_dense(cx, m, d, err);
    check(up, "the dense half uploads" + (up ? std::string() : ": " + err));
    if (!up) {
        std::printf("q36_upload_check: FAILURES\n");
        return 1;
    }
    check(d.layers.size() == m.layers.size(), "every layer has device buffers");
    check(d.dense_bytes == m.dense_bytes + m.embed_bytes, "the bytes uploaded are the plan's dense + embedding bytes");
    check(d.dense_bytes > 0 && m.expert_bytes > 0, "the model really does have both a dense half and an expert half");

    // The expert tensors are not on the device, by design - on a 37.8 GB Q8_0 pack they are ~34 of the 34.5 GB and the
    // card is 8.  (The fixture's own experts are small, so what is checked here is that they are still readable in the
    // file's mapping, which is what the streaming path will read, and that nothing was allocated for them.)
    check(m.layers[0].gate_exps.data(*m.file) != nullptr, "the expert tensors are still readable in the file's mapping");
    check(m.expert_bytes > 0 && d.dense_bytes == m.dense_bytes + m.embed_bytes,
          "the expert bytes are counted in the plan and absent from the device");

    // every buffer against its tensor, byte for byte
    size_t compared = 0;
    std::string why;
    bool all = same(m, m.embed, d.embed, "token_embd", compared, why) && same(m, m.output, d.output, "output", compared, why) &&
               same(m, m.output_norm, d.output_norm, "output_norm", compared, why);
    for (size_t i = 0; i < m.layers.size() && all; ++i) {
        const strata::qwen36::Layer& L = m.layers[i];
        const strata::qwen36::DeviceLayer& D = d.layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        all = same(m, L.gate_inp, D.gate_inp, (p + "ffn_gate_inp").c_str(), compared, why) &&
              same(m, L.gate_inp_shexp, D.gate_inp_shexp, (p + "ffn_gate_inp_shexp").c_str(), compared, why) &&
              same(m, L.gate_shexp, D.gate_shexp, (p + "ffn_gate_shexp").c_str(), compared, why) &&
              same(m, L.up_shexp, D.up_shexp, (p + "ffn_up_shexp").c_str(), compared, why) &&
              same(m, L.down_shexp, D.down_shexp, (p + "ffn_down_shexp").c_str(), compared, why) &&
              same(m, L.attn_norm, D.attn_norm, (p + "attn_norm").c_str(), compared, why) &&
              same(m, L.post_norm, D.post_norm, (p + "post_attention_norm").c_str(), compared, why) &&
              same(m, L.qkv, D.qkv, (p + "attn_qkv").c_str(), compared, why) && same(m, L.sgate, D.sgate, (p + "attn_gate").c_str(), compared, why) &&
              same(m, L.ssm_out, D.ssm_out, (p + "ssm_out").c_str(), compared, why) &&
              same(m, L.conv1d, D.conv1d, (p + "ssm_conv1d").c_str(), compared, why) &&
              same(m, L.alpha, D.alpha, (p + "ssm_alpha").c_str(), compared, why) &&
              same(m, L.beta, D.beta, (p + "ssm_beta").c_str(), compared, why) && same(m, L.ssm_a, D.ssm_a, (p + "ssm_a").c_str(), compared, why) &&
              same(m, L.dt_bias, D.dt_bias, (p + "ssm_dt.bias").c_str(), compared, why) &&
              same(m, L.ssm_norm, D.ssm_norm, (p + "ssm_norm").c_str(), compared, why) &&
              same(m, L.q, D.q, (p + "attn_q").c_str(), compared, why) && same(m, L.k, D.k, (p + "attn_k").c_str(), compared, why) &&
              same(m, L.v, D.v, (p + "attn_v").c_str(), compared, why) && same(m, L.o, D.o, (p + "attn_output").c_str(), compared, why) &&
              same(m, L.q_norm, D.q_norm, (p + "attn_q_norm").c_str(), compared, why) &&
              same(m, L.k_norm, D.k_norm, (p + "attn_k_norm").c_str(), compared, why) &&
              same(m, L.eh_proj, D.eh_proj, (p + "nextn.eh_proj").c_str(), compared, why) &&
              same(m, L.enorm, D.enorm, (p + "nextn.enorm").c_str(), compared, why) &&
              same(m, L.hnorm, D.hnorm, (p + "nextn.hnorm").c_str(), compared, why) &&
              same(m, L.shared_head_norm, D.shared_head_norm, (p + "nextn.shared_head_norm").c_str(), compared, why);
        if (!all) break;
    }
    check(all, "every dense buffer holds exactly its tensor's bytes" + (all ? std::string() : ": " + why));
    check(compared == d.dense_bytes, "the bytes compared are the bytes allocated (" + std::to_string(compared) + ")");

    // release has to leave nothing behind that a second upload would duplicate
    strata::qwen36::release(cx, d);
    check(d.layers.empty() && d.dense_bytes == 0, "release empties the device model");

    std::printf("%s\n", g_fail ? "q36_upload_check: FAILURES" : "q36_upload_check: all ok");
    return g_fail ? 1 : 0;
}
