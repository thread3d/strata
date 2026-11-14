// metal/src/qwen36/q36_dense.cpp - the dense matvec call, once, for every projection of a Qwen3.6 layer.
#include "q36_dense.h"

#include "ref_quant.h"

#include <cstdio>
#include <vector>

namespace strata::qwen36 {
namespace {

// The tile the port runs a shape with (gemv_parity's defaults for the same widths: LPR, rows a group, simdgroups a
// threadgroup, k units a step).  n_in 2048 - q36's hidden - takes the middle set.
struct Cfg {
    int L, R, S, K;
};
Cfg tile_for(int n_in) {
    if (n_in >= 6144) return Cfg{32, 1, 8, 2};
    if (n_in <= 640) return Cfg{8, 1, 4, 1};
    return Cfg{16, 1, 4, 1};
}
int unit_of(int t) {
    switch (t) {
    case ref::Q2_0: case ref::IQ3_S: case ref::IQ3_XXS: case ref::IQ2_XXS: case ref::IQ2_XS: case ref::IQ2_S:
    case ref::BF16: case ref::F32: return 16;
    default: return 32;
    }
}
bool cfg_ok(Cfg c, int type, int n_in) {
    const int U = n_in / unit_of(type);
    return n_in % unit_of(type) == 0 && U % c.K == 0 && U / c.K >= c.L / 2;
}
int rows_per_tg(Cfg c) { return (c.S / c.K) * (32 / c.L) * c.R; }

// which prepared plane a format reads (units.metal XK_*): 0 = raw floats, 1..3 the other kinds
int xkind(int t) {
    switch (t) {
    case ref::Q2_0: return 1;
    case ref::Q4_K: case ref::Q5_K: return 2;
    case ref::Q6_K: case ref::Q3_K: return 3;
    default: return 0;
    }
}

std::string tile_name(int type, Cfg c) {
    return "gemv_" + std::to_string(type) + "_" + std::to_string(c.L) + "_" + std::to_string(c.R) + "_" +
           std::to_string(c.S) + "_" + std::to_string(c.K);
}

}  // namespace

bool dense_init(mtl::Context& cx, DenseKernels& k, std::string& err) {
    if (k.lib >= 0) return true;
    try {
        k.lib = cx.compile({"quant.metal", "units.metal", "gemv.metal"}, "", false);
        k.prep = cx.pipeline(k.lib, "xprep");
    } catch (const std::exception& e) {
        err = std::string("the dense kernels did not compile: ") + e.what();
        return false;
    }
    // xprep writes one thread per 16 values of a token row: the threadgroup is 256 (gemv_parity's dispatch)
    if (k.prep.max_threads < 256) {
        err = "xprep's threadgroup limit is " + std::to_string(k.prep.max_threads) + ", this port dispatches 256";
        return false;
    }
    return true;
}

bool dense_scratch(mtl::Context& cx, DenseScratch& s, int n_in, std::string& err) {
    if (s.xp.ptr != nullptr && s.n_in == n_in) return true;
    if (s.xp.ptr != nullptr) cx.free(s.xp);
    // 4 planes of n_in floats, then the 16-wide sums: (4 + 1/16) rows' worth
    const size_t bytes = (size_t) 4 * (size_t) n_in * 4 + (size_t) (n_in / 16) * 2 * 4;
    s.xp = cx.alloc(bytes);
    if (s.xp.ptr == nullptr) {
        err = "the device refused the activation scratch (" + std::to_string(bytes) + " bytes)";
        return false;
    }
    s.n_in = n_in;
    return true;
}

bool dense_prep(mtl::Context& cx, DenseKernels& k, const mtl::Buffer& x, DenseScratch& s, int n_in, std::string& err) {
    if (!dense_init(cx, k, err)) return false;
    if (s.xp.ptr == nullptr || s.n_in != n_in) {
        if (!dense_scratch(cx, s, n_in, err)) return false;
    }
    mtl::Dispatch prep;
    prep.p = &k.prep;
    prep.args = {mtl::Arg::B(x), mtl::Arg::B(s.xp), mtl::Arg::V((uint32_t) n_in), mtl::Arg::V(1u)};
    prep.tg[0] = 256;
    prep.tg[1] = prep.tg[2] = 1;
    prep.grid[0] = (uint32_t) ((n_in / 16 + 255) / 256);
    prep.grid[1] = prep.grid[2] = 1;
    cx.run({prep});
    return true;
}

bool dense_gemv(mtl::Context& cx, DenseKernels& k, const DenseMatrix& W, const mtl::Buffer& x, mtl::Buffer& y,
                DenseScratch& s, std::string& err, int y_stride) {
    if (!dense_init(cx, k, err)) return false;
    if (W.buf == nullptr || W.n_in <= 0 || W.n_out <= 0) {
        err = "a dense matrix with no buffer or no shape";
        return false;
    }
    if (!dense_prep(cx, k, x, s, W.n_in, err)) return false;
    Cfg c = tile_for(W.n_in);
    if (!cfg_ok(c, W.type, W.n_in)) c = Cfg{32, 1, 4, 1};
    if (!cfg_ok(c, W.type, W.n_in)) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "no gemv tile takes %d values at type %d (the unit is %d)", W.n_in, W.type,
                      unit_of(W.type));
        err = buf;
        return false;
    }
    mtl::Pipeline p;
    const std::string name = tile_name(W.type, c);
    try {
        p = cx.pipeline(k.lib, name, mtl::Constants().i(0, 1));   // (function constant 0 = tokens in flight, 1 here)
    } catch (const std::exception& e) {
        err = "this port has no kernel " + name + " (" + e.what() + ")";
        return false;
    }
    const uint32_t n_in = (uint32_t) W.n_in, n_out = (uint32_t) W.n_out;
    const uint64_t rb = (uint64_t) ref::row_bytes(W.type, W.n_in);
    // the prepared activations: one token, so the planes sit n_in floats apart and the sums after the fourth
    const size_t plane = (size_t) n_in * 4;

    // gemv's arguments, in its own order: W, the prepared plane this format reads, the sums, y, and the shape
    struct GemvArgs {
        uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride;
    } a{n_in, n_out, 1, (uint32_t) rb, n_in,
        (uint32_t) (y_stride > 0 ? y_stride : (int) n_out)};
    mtl::Dispatch d;
    d.p = &p;
    d.args = {mtl::Arg::B(*W.buf, W.off), mtl::Arg::B(s.xp, plane * (size_t) xkind(W.type)),
              mtl::Arg::B(s.xp, plane * 4), mtl::Arg::B(y), mtl::Arg::V(a)};
    d.grid[0] = (uint32_t) ((W.n_out + rows_per_tg(c) - 1) / rows_per_tg(c));
    d.grid[1] = d.grid[2] = 1;
    d.tg[0] = (uint32_t) (c.S * 32);
    d.tg[1] = d.tg[2] = 1;
    cx.run({d});
    return true;
}

}  // namespace strata::qwen36
