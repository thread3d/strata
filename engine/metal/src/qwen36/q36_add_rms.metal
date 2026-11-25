// metal/src/qwen36/msl/q36_add_rms.metal - residual + rmsnorm: x += y; h = rms(x) * w, one row per threadgroup.
// Bit-identical to k_add_rms / q36_add_rms on the CUDA side.
#include <metal_stdlib>
using namespace metal;

kernel void q36_add_rms(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device float* h [[buffer(3)]],
    constant int& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint32_t tid [[thread_index_in_threadgroup]],
    uint32_t tg_size [[threads_per_threadgroup]]) {
    const int row = int(threadgroup_position_in_grid.x);
    device float* xr = x + (size_t) row * n;
    const device float* yr = y + (size_t) row * n;
    device float* hr = h + (size_t) row * n;

    float s = 0.0f;
    for (int i = int(tid); i < n; i += int(tg_size)) {
        float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }

    threadgroup float sh[256];
    sh[tid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int o = int(tg_size) / 2; o > 0; o >>= 1) {
        if (tid < (uint32_t) o) sh[tid] += sh[tid + o];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float r = 1.0f / sqrt(sh[0] / float(n) + eps);
    for (int i = int(tid); i < n; i += int(tg_size)) {
        hr[i] = xr[i] * r * w[i];
    }
}
