// Software reproduction of moe.metal's K-quant down path, compared against ref::dequant.
//
// Transcribed from metal/kernels/moe.metal (XD<F>, dn_body_k, unit_combine) and
// metal/kernels/units.metal (Dec<T_Q6_K>, fmt<T_Q6_K>). Pure CPU: if this matches the
// reference, the kernel *source* is right and the fault is in what the GPU is handed.
//
//   clang++ -O2 -std=c++17 -I metal/src -I metal/include -I third_party/ggml \
//       metal/tests/down_kernel_emu.cpp metal/src/ref_quant.cpp -o /tmp/down_kernel_emu

#include "ref_quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int QK = 256, BB = 210, UNIT = 32;

static inline float rdh(const uint8_t* p) {
    const uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    // fp16 -> fp32
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            man &= 0x3FF;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// Dec<T_Q6_K>::run  (q[UNIT], sc = d1, d2, 32*d1, 32*d2)
static inline void dec_q6k(const uint8_t* row, int u, float* q, float* sc) {
    const uint8_t* b = row + (u >> 3) * BB;
    const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
    const uint8_t* ql = b + 64 * n + 32 * p + 16 * m;
    const uint8_t* qh = b + 128 + 32 * n + 16 * m;
    const int8_t* scs = (const int8_t*)(b + 192);
    const float d = rdh(b + 208);
    for (int w = 0; w < 8; ++w) {
        const uint16_t l = (uint16_t)(ql[2 * w] | (ql[2 * w + 1] << 8));
        const uint16_t h = (uint16_t)(qh[2 * w] | (qh[2 * w + 1] << 8));
        const uint16_t lo = (uint16_t)((l & 0x0F0Fu) | (((h >> (2 * p)) & 0x0303u) << 4));
        const uint16_t hi = (uint16_t)(((l >> 4) & 0x0F0Fu) | (((h >> (2 * p + 4)) & 0x0303u) << 4));
        q[2 * w] = (float)(lo & 0xFF);
        q[2 * w + 1] = (float)(lo & 0xFF00);
        q[16 + 2 * w] = (float)(hi & 0xFF);
        q[16 + 2 * w + 1] = (float)(hi & 0xFF00);
    }
    const float d1 = d * (float) scs[8 * n + m + 2 * p];
    const float d2 = d * (float) scs[8 * n + m + 2 * p + 4];
    sc[0] = d1; sc[1] = d2; sc[2] = 32.0f * d1; sc[3] = 32.0f * d2;
}

// XD<T_Q6_K>::load + xd16(..., kind = 2)
static inline void xd_q6k(const float* h, int u, float* xs, float* xsum) {
    const int p0 = 256 * (u >> 3) + 128 * ((u >> 2) & 1) + 32 * (u & 1) + 16 * ((u >> 1) & 1);
    for (int half = 0; half < 2; ++half) {
        const float* p = h + p0 + (half ? 64 : 0);
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const float v = p[j];
            s += v;
            xs[half * 16 + j] = (j & 1) ? v * 0.00390625f : v;
        }
        xsum[half] = s;
    }
}

// unit_combine<UNIT>
static inline float combine(const float* q, const float* sc, const float* xs, const float* xsum) {
    constexpr int H = UNIT / 2;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = 0; j < H; j += 2) {
        s0 = std::fma(q[j], xs[j], s0);
        s1 = std::fma(q[j + 1], xs[j + 1], s1);
        s2 = std::fma(q[H + j], xs[H + j], s2);
        s3 = std::fma(q[H + j + 1], xs[H + j + 1], s3);
    }
    return sc[0] * (s0 + s1) + sc[1] * (s2 + s3) - (sc[2] * xsum[0] + sc[3] * xsum[1]);
}

}  // namespace

int main() {
    const int N_EMBD = 512;   // the down's row length (n_ff)
    const int ROWS = 8;

    std::vector<uint8_t> blob((size_t) ref::row_bytes(ref::Q6_K, N_EMBD) * ROWS);
    ref::random_blocks(ref::Q6_K, blob.data(), ROWS, N_EMBD, 12345, 0.05f);

    std::vector<float> h((size_t) N_EMBD);
    for (int i = 0; i < N_EMBD; ++i) h[(size_t) i] = std::sin(0.37f * i) * 0.5f + 0.1f * std::cos(0.11f * i);

    const size_t d_row = (size_t) ref::row_bytes(ref::Q6_K, N_EMBD);
    double sref = 0, sdif = 0, maxabs = 0;
    for (int r = 0; r < ROWS; ++r) {
        // reference: dequantise the row (ggml order) and dot with h
        std::vector<float> w((size_t) N_EMBD);
        ref::dequant(ref::Q6_K, blob.data() + (size_t) r * d_row, w.data(), (size_t) N_EMBD);
        double yref = 0;
        for (int k = 0; k < N_EMBD; ++k) yref += (double) w[(size_t) k] * h[(size_t) k];

        // the kernel's way: per unit, decode the codes and slice h
        const uint8_t* row = blob.data() + (size_t) r * d_row;
        float acc = 0.0f;
        const int U = N_EMBD / UNIT;
        for (int u = 0; u < U; ++u) {
            float q[UNIT], sc[4], xs[UNIT], xsum[2];
            dec_q6k(row, u, q, sc);
            xd_q6k(h.data(), u, xs, xsum);
            acc += combine(q, sc, xs, xsum);
        }
        const double diff = (double) acc - yref;
        sref += yref * yref;
        sdif += diff * diff;
        if (std::fabs(diff) > maxabs) maxabs = std::fabs(diff);
        if (r < 3) std::printf("row %d: ref %.9f  kernel-path %.9f  diff %.3e\n", r, yref, (double) acc, diff);
    }
    std::printf("rows %d: rel L2 %.3e, max |diff| %.3e\n", ROWS, std::sqrt(sdif / sref), maxabs);
    return std::sqrt(sdif / sref) < 1e-5 ? 0 : 1;
}
