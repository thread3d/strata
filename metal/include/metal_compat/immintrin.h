// metal/include/metal_compat/immintrin.h - the x86 intrinsics the host loops use, on arm64: the spin-wait hint and
// the store fence before a host flag is raised (verify.cpp, session.cpp, mtp.cpp, graph.cpp).
#pragma once
#if defined(__aarch64__) || defined(__arm64__)
static inline void _mm_pause() { __asm__ __volatile__("yield"); }
static inline void _mm_sfence() { __asm__ __volatile__("dmb ishst" ::: "memory"); }
#else
// x86_64 (an Intel Mac): the compiler's own header already has both - it declares _mm_pause and _mm_sfence itself, so a
// second definition here is a redefinition - and the CPU lanes use the SSE/AVX intrinsics it brings in.
#include_next <immintrin.h>
#endif
