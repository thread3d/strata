// metal/src/qwen36/msl/q36_add.metal - plain elementwise add: x += y, one element per thread.
// Used for the residual joins in the Qwen3.6 layer loop.
#include <metal_stdlib>
using namespace metal;

kernel void q36_add(
    device float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint32_t n [[buffer(2)]],
    uint32_t tid [[thread_index_in_threadgroup]]) {
    for (uint32_t i = tid; i < n; i += 256u) {
        x[i] += y[i];
    }
}
