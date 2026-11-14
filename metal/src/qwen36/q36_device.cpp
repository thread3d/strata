// metal/src/qwen36/q36_device.cpp - uploading the dense half of a Qwen3.6 model into shared-storage buffers.
//
// Every upload is a straight copy of the tensor's payload from the file's mapping, so a Q8_0 matrix reaches the device as
// the blocks the file holds and the port's gemv / moe kernels read it unchanged.  What this file adds over a memcpy is
// only the discipline the rest of the port learned on this card:
//
//   * the buffer is allocated by the runtime (mtl::Context::alloc), so a kernel is later handed an OFFSET into it rather
//     than a CPU address - the one rule a Metal 2 discrete card cannot bend (a kernel argument holding contents() is not
//     a GPU address, which is what runtime_hello still measures as 65470 wrong values of 65536);
//   * nothing is bound from a page inside another allocation (ds4_finish.cpp's wrap_span rule), because two MTLBuffers
//     over the same pages make one write invisible to the other's readers;
//   * a tensor whose payload is longer than the device's maxBufferLength is refused here by name, not left to fail at a
//     dispatch.
#include "q36_device.h"

#include <cstring>

namespace strata::qwen36 {
namespace {

// One tensor onto the device.  Empty (and not an error) for a tensor the layer does not have.
bool up(mtl::Context& cx, const Model& m, const TensorRef& r, mtl::Buffer& out, uint64_t& total, std::string& err) {
    if (!r.ok()) return true;
    const uint64_t bytes = r.bytes();
    const uint8_t* src = r.data(*m.file);
    if (src == nullptr) {
        err = std::string(r.name()) + " is not in the file's mapping";
        return false;
    }
    if (bytes == 0) {
        err = std::string(r.name()) + " has no payload";
        return false;
    }
    if (bytes > cx.max_buffer_length()) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s is %.2f GiB, more than this device's %llu-byte buffer limit",
                      r.name(), (double) bytes / 1073741824.0, (unsigned long long) cx.max_buffer_length());
        err = buf;
        return false;
    }
    out = cx.alloc((size_t) bytes);
    if (out.ptr == nullptr) {
        err = std::string("the device refused ") + r.name() + " (" + std::to_string(bytes) + " bytes)";
        return false;
    }
    std::memcpy(out.ptr, src, (size_t) bytes);
    total += bytes;
    return true;
}

}  // namespace

bool upload_dense(mtl::Context& cx, const Model& m, DeviceModel& d, std::string& err) {
    uint64_t total = 0;
    if (!up(cx, m, m.embed, d.embed, total, err)) return false;
    if (!up(cx, m, m.output, d.output, total, err)) return false;
    if (!up(cx, m, m.output_norm, d.output_norm, total, err)) return false;

    // reserve before filling: a Buffer copied by a reallocation would be the same handle twice, and release() frees what
    // is stored
    d.layers.clear();
    d.layers.reserve(m.layers.size());
    for (const Layer& L : m.layers) {
        DeviceLayer dl;
        dl.kind = L.kind;
        const std::pair<const TensorRef*, mtl::Buffer*> dense[] = {
            {&L.gate_inp, &dl.gate_inp},   {&L.gate_inp_shexp, &dl.gate_inp_shexp},
            {&L.gate_shexp, &dl.gate_shexp}, {&L.up_shexp, &dl.up_shexp},
            {&L.down_shexp, &dl.down_shexp}, {&L.attn_norm, &dl.attn_norm},
            {&L.post_norm, &dl.post_norm}, {&L.qkv, &dl.qkv},
            {&L.sgate, &dl.sgate},       {&L.ssm_out, &dl.ssm_out},
            {&L.conv1d, &dl.conv1d},     {&L.alpha, &dl.alpha},
            {&L.beta, &dl.beta},         {&L.ssm_a, &dl.ssm_a},
            {&L.dt_bias, &dl.dt_bias},   {&L.ssm_norm, &dl.ssm_norm},
            {&L.q, &dl.q},               {&L.k, &dl.k},
            {&L.v, &dl.v},               {&L.o, &dl.o},
            {&L.q_norm, &dl.q_norm},     {&L.k_norm, &dl.k_norm},
            {&L.eh_proj, &dl.eh_proj},   {&L.enorm, &dl.enorm},
            {&L.hnorm, &dl.hnorm},       {&L.shared_head_norm, &dl.shared_head_norm},
        };
        for (const auto& [ref, dst] : dense) {
            if (!up(cx, m, *ref, *dst, dl.bytes, err)) {
                release(cx, d);
                return false;
            }
        }
        total += dl.bytes;
        d.layers.push_back(dl);
    }
    d.dense_bytes = total;
    return true;
}

void release(mtl::Context& cx, DeviceModel& d) {
    auto one = [&](mtl::Buffer& b) {
        if (b.handle != nullptr || b.chunks) cx.free(b);
    };
    one(d.embed);
    one(d.output);
    one(d.output_norm);
    for (DeviceLayer& L : d.layers) {
        for (mtl::Buffer* b : {&L.gate_inp, &L.gate_inp_shexp, &L.gate_shexp, &L.up_shexp, &L.down_shexp, &L.attn_norm,
                               &L.post_norm, &L.qkv, &L.sgate, &L.ssm_out, &L.conv1d, &L.alpha, &L.beta, &L.ssm_a,
                               &L.dt_bias, &L.ssm_norm, &L.q, &L.k, &L.v, &L.o, &L.q_norm, &L.k_norm, &L.eh_proj,
                               &L.enorm, &L.hnorm, &L.shared_head_norm})
            one(*b);
    }
    d.layers.clear();
    d.dense_bytes = 0;
}

}  // namespace strata::qwen36
