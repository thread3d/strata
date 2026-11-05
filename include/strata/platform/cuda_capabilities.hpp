// include/strata/platform/cuda_capabilities.hpp - which CUDA runtime entry points the toolkit being used actually has.
//
// The engine is developed and shipped against CUDA 12.3 and newer (the release and aarch64 builds use 13.0).  A Jetson
// AGX Xavier cannot have those: JetPack 5 is the last release for that board and it carries CUDA 11.4, JetPack 6
// dropped Xavier, and CUDA 13 dropped sm_72 (docs/JETSON.md).  This header answers "does this toolkit have it?" once,
// so a call site tests a name instead of repeating a version comparison.
//
// Nothing here changes a CUDA 12.3+ build: every macro is 1 there and every `#if` keeps exactly the code it had.  On an
// older toolkit the fallback is what the engine already does on a device that cannot perform the newer thing - an
// ordinary kernel launch instead of a cluster launch, an ordinary graph edge instead of a programmatic one, a loop of
// cudaMemcpyAsync instead of one batched call - so the result is slower, never different.
#pragma once

#include <cuda_runtime.h>

#if defined(__HIPCC__) || defined(STRATA_HIP_GFX906) || defined(STRATA_USE_HIP)
// AMD (HIP): there is no CUDA version.  Every feature below is off, and every site already has its own HIP path.
#define STRATA_CUDA_RUNTIME_VERSION 0
#else
// CUDART_VERSION, or 0 in the (never seen) case of a toolkit that does not define it: an undefined name is 0 in `#if`.
#define STRATA_CUDA_RUNTIME_VERSION CUDART_VERSION
#endif

// cudaLaunchKernelEx, cudaLaunchConfig_t, cudaLaunchAttribute, cudaOccupancyMaxActiveClusters: thread-block clusters
// (sm_90), programmatic dependent launch, and launch priority.  CUDA 12.0.
#define STRATA_HAVE_CUDA_LAUNCH_EX (STRATA_CUDA_RUNTIME_VERSION >= 12000)

// cudaGraphInstantiate's three-argument form and cudaGraphInstantiateFlagUseNodePriority: CUDA 12.0.  An 11.x toolkit
// has only the five-argument form, and no flags argument on it.  This is that call, once, for both.
#define STRATA_HAVE_CUDA_GRAPH_INSTANTIATE_FLAGS (STRATA_CUDA_RUNTIME_VERSION >= 12000)

inline cudaError_t strata_graph_instantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags) {
#if STRATA_HAVE_CUDA_GRAPH_INSTANTIATE_FLAGS
    return cudaGraphInstantiate(exec, graph, flags);
#else
    (void) flags;   // no flags argument on CUDA 11's entry point, and none of the flags exists there to pass
    return cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0);
#endif
}

// cudaInitDevice (cudaSetDevice before CUDA 12.0; the two differ once a device's flags and priority are given too).
#define STRATA_HAVE_CUDA_INIT_DEVICE (STRATA_CUDA_RUNTIME_VERSION >= 12000)

// The graph side of programmatic dependent launch: cudaStreamGetCaptureInfo_v3 (CUDA 12.3), cudaGraphEdgeData,
// cudaGraphDependencyType*, cudaGraphGetEdges_v2, cudaGraphNodeGetDependencies_v2, cudaKernelNodeAttributePriority and
// cudaFuncGetName.  Below it the verify window's recorder makes ordinary edges, which is what a device without PDL
// gets on any toolkit.
#define STRATA_HAVE_CUDA_PDL_GRAPH (STRATA_CUDA_RUNTIME_VERSION >= 12030)

// cudaKernelNodeAttributePriority (and the priority member of cudaKernelNodeAttrValue) and
// cudaGraphInstantiateFlagUseNodePriority: CUDA 12.0.  STRATA_GRAPH_PRIO does nothing below it; the window graph is
// then captured and instantiated exactly as it is on a device that does not use priorities.
#define STRATA_HAVE_CUDA_NODE_PRIORITY (STRATA_CUDA_RUNTIME_VERSION >= 12000)

// cudaGetDriverEntryPointByVersion: CUDA 12.5 (cudaGetDriverEntryPoint before it, without the version argument).
#define STRATA_HAVE_CUDA_DRIVER_ENTRY_BY_VERSION (STRATA_CUDA_RUNTIME_VERSION >= 12050)

// cudaMemcpyBatchAsync, cudaMemcpyAttributes, cudaMemcpyFlagPreferOverlapWithCompute, cudaMemcpySrcAccessOrderStream:
// CUDA 12.8.
#define STRATA_HAVE_CUDA_MEMCPY_BATCH (STRATA_CUDA_RUNTIME_VERSION >= 12080)

// Two C++20 constructs the CUDA sources use, which nvcc parses only from CUDA 12.0: a templated lambda
// (`[&]<int J>(std::integer_sequence<int, J>) { ... }`, the balanced S26 group in iq_kernels.cu) and `if constexpr`
// inside a device lambda (qsa_prompt_attn.cu, where 11.x reports "expression must have a constant value").  1 means the
// compiler parses them; 0 means the site takes its plain form, which computes the same values.  Clang, and so HIP,
// takes both as C++20 extensions in C++17 mode, which is how the HIP engine builds these files.
#if defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__HIP__) && (STRATA_CUDA_RUNTIME_VERSION < 12000)
#define STRATA_DEVICE_CXX20_SYNTAX 0
#else
#define STRATA_DEVICE_CXX20_SYNTAX 1
#endif
