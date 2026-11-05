// metal/src/engine/stubs_gyro.cpp - the Gyro (agentionai trellis experts) entry points the Metal engine does not implement:
// the GPU Hadamard rotations of src/kernels/cuda/iq_kernels.cu.  native_expert_supported() (iq_host.cpp) does not list the
// trellis types, so a Gyro pack stops at load with its reason and these are never reached with a supported file.
// Each stops the engine with its name, like stubs.cpp.

#include <cstdint>

#include "strata/kernels/iq_kernels.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

// A stub yields to a real implementation linked beside it: an Intel Mac compiles the x86 CPU kernels
// (src/kernels/cpu/iq_avx2.cpp, iq_avx512.cpp, tq_avx2.cpp, tq_avx512.cpp), which define some of the entry points
// below, and a weak definition lets the real one win instead of colliding with it at link time.
#pragma clang attribute push(__attribute__((weak)), apply_to = function)
namespace strata::kernels {
void hadamard_rows(const float*, float*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::hadamard_rows");
}
void hadamard_rows_f16(uint16_t*, int64_t, int64_t, int, const float*, void*) {
    strata::metal_engine::missing("strata::kernels::hadamard_rows_f16");
}
const float* hadamard_device_signs(const float*, int) {
    strata::metal_engine::missing("strata::kernels::hadamard_device_signs");
    return nullptr;
}
}  // namespace strata::kernels
#pragma clang attribute pop
