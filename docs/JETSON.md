# Strata on the NVIDIA Jetson AGX Xavier (Linux, ARM64)

The Jetson AGX Xavier is a small ARM board: 8 Carmel CPU cores, a 512-core Volta-generation GPU of compute capability
7.2 (sm_72), and 32 GB of memory shared between them. This page says how to build Strata's engine on it from a source
checkout, what the tree had to learn to build there, and what the result does not promise. It was tested on one board:
Ubuntu 20.04 (JetPack 5), the CUDA 11.4 toolkit JetPack installs, Linux 5.10.104-tegra.

Xavier is not a machine the maintainers support. [DGX Spark](DGX_SPARK_LINUX.md) is the ARM machine Strata is built
and measured on, and that page says nothing there is known to work on another ARM machine. This page is the same idea
on older ARM hardware: the engine compiles, and it says plainly what was not measured.

## What works, and what does not

- **Builds:** the CUDA engine (`strata`, `strata_kq`) and the tools that need no GPU (`strata-plan`, `strata-gguf`,
  `strata-dequant`). Steps below.
- **Not built for ARM:** the desktop app (the packages are x86-64 and arm64 macOS only); use the Terminal.
- **Starts, and reports the card:** `strata --version` prints the engine version, and `strata-device --selftest`
  reports the Xavier (compute capability 7.2, 30.276 GiB), computes the memory plan and passes its arena check.
- **Not measured here:** a model answering prompts end to end. The engine's kernels are written and measured for
  Ampere and newer, and sm_72 is below the 7.5 floor the engine supports - the build takes the community Pascal/Volta
  path (`-DSTRATA_EXPERIMENTAL_SM60=ON`, #236) to build and to run at all. Treat a first run as a test, not as a
  supported configuration, and read [What the build gives up](#what-the-build-gives-up).
- **Measured here:** the kernel-level tests and micro-benchmarks that need no model, three runs each -
  [below](#what-was-checked-on-one-board), with the raw output in
  [bench/results/jetson-agx-xavier](../bench/results/jetson-agx-xavier/README.md).
- **The models are bigger than the board.** Qwen3.8-Flash-Next packs start at 84 GB and DeepSeek-V4.1-Flash's at 257 GB.
  32 GB of shared memory is the hard ceiling here, so a pack that fits has to be prepared for this board first.

## Why the tree does not build on this board as it comes

Seven things, each stopping the build (or, the last one, a test build's first run) at a different stage. The first
four are the board's tools being older than the tree's assumptions; the next two are the engine using CUDA entry points
and C++20 syntax that CUDA 11.4 does not have; the last one only appears once the binary runs.

| Stop | What it says | Why |
| --- | --- | --- |
| CMake 3.16 | `CMake 3.24 or higher is required` | Ubuntu 20.04 ships 3.16 |
| GCC 9.4 | `fatal error: concepts: No such file or directory` | the sources are C++20 (`<concepts>`, `requires` clauses); GCC 9 has neither, and CUDA 11.4 accepts a host compiler no newer than GCC 10 |
| the CUDA dialect | `Target "strata_core" requires the language dialect "CUDA20"` | nvcc learned `-std=c++20` in CUDA 12.0 |
| the architecture | `Strata needs an NVIDIA GPU of compute capability 7.5 or newer` | Xavier is sm_72; the build guard refuses anything below 7.5 |
| CUDA 12 entry points | `identifier "cudaLaunchKernelEx" is undefined`, and `cudaGraphInstantiate`, `cudaKernelNodeAttributePriority`, `cudaInitDevice`, `cudaGetDriverEntryPoint`'s status argument | the engine is developed against CUDA 12.3+ (the release builds use 13.0); CUDA 11.4 has none of them |
| two C++20 constructs | `expected a "{" introducing a lambda body`, `expression must have a constant value` | nvcc before 12.0 cannot parse a templated lambda (`[&]<int J>(...)`) or `if constexpr` inside a device lambda |
| a CPU-only test build | links, then `std::call_once` throws `std::system_error: Unknown error -1` | glibc 2.31 keeps libpthread separate and Ubuntu's `--as-needed` drops the explicit `-lpthread`, so the pthread calls land on libc's single-threaded stubs; `THREADS_PREFER_PTHREAD_FLAG` asks CMake for `-pthread`, which survives (Ubuntu 22.04+, glibc 2.35, hides this) |

## The toolchain (no administrator rights)

`jetson/toolchain.sh` installs what the build needs and changes nothing system-wide:

```sh
sh jetson/toolchain.sh
. ~/toolchains/strata-env.sh
```

- **CMake and Ninja** through `pip --user`, so they land in `~/.local/bin` (3.24 or newer; 4.x works).
- **GCC 10.5**, Ubuntu's own `gcc-10` packages, downloaded with `apt-get download` and unpacked into
  `~/toolchains/gcc10` with `dpkg-deb -x` rather than installed. GCC 10 is the newest host compiler the CUDA 11.4
  toolkit accepts, and its C++20 support covers everything the host sources use. The unpacked compiler links against
  the board's existing libstdc++6 (Ubuntu 20.04 builds that from GCC 10), so the engine needs no `LD_LIBRARY_PATH` at
  run time.

## Build the engine

```sh
. ~/toolchains/strata-env.sh
sh jetson/build_engine.sh
```

The result lands in `out-jetson/engine`: `strata`, `strata_kq`, the CUDA libraries the engine loads, and a `BUILD.json`
that records the toolkit, the architecture and the experimental flag. `BUILD_DSV41=1` adds the DeepSeek engine,
`ARCHS`, `OUT`, `BUILD` and `JOBS` work as they do in `spark/build_engine.sh`.

The flags that matter, and why this board needs them:

| Flag | Why |
| --- | --- |
| `-DSTRATA_EXPERIMENTAL_SM60=ON` | admits compute capability below 7.5 at configure time and in the engine's runtime device check; the release engine keeps its 7.5 floor |
| `-DCMAKE_CUDA_ARCHITECTURES=72-real` | Xavier's card, and no PTX: the board compiles and runs on the same machine |
| no `-DSTRATA_PORTABLE` | that builds ggml-cpu for an ARMv8.2+dotprod+fp16+i8mm baseline (Grace, Graviton). Carmel has fp16 but **no dotprod and no i8mm**, so those instructions would fault; the build stays `-mcpu=native` |
| `-DSTRATA_BUILD_TESTS=OFF` | the parity tools use further entry points from newer toolkits; they are not part of an engine build |

## What the build gives up

Every one of these is a speed path, and each is what the engine already does on a device that cannot perform it.
None of them changes a computed value.

| Not available on CUDA 11.4 | What the engine does instead |
| --- | --- |
| thread-block cluster launches (`cudaLaunchKernelEx`) | the cluster kernels need sm_90+ anyway; the sampler's one-block kernel and the wide top-k split run instead |
| programmatic dependent launch (PDL) | ordinary launches and ordinary graph edges; PDL is opt-in (`STRATA_DF_PDL`) and needs sm_90 |
| one batched `cudaMemcpyAsync` group (`cudaMemcpyBatchAsync`, CUDA 12.8) | an upload at a time (`STRATA_DMA_BATCH` does nothing) |
| per-node graph priorities (`STRATA_GRAPH_PRIO`) | the window graph is captured and instantiated as it always is |
| the balanced S26 expert grouping (`BD=1`) | the plain per-lane loop: the same sums, one item at a time |
| `cudaInitDevice`'s scheduling flags | `cudaSetDevice` + `cudaSetDeviceFlags`, the same spin policy, set the same way |

The engine takes these paths through `include/strata/platform/cuda_capabilities.hpp`, which names the CUDA version each
entry point arrived in. A CUDA 12.3+ build - every release build, on every other platform - evaluates every one of
those `#if`s to the code it had before, so this board's build is the only one that changes.

## What was checked on one board

One Jetson AGX Xavier (Ubuntu 20.04, CUDA 11.4.315, cuBLAS 11.6.6.84, 30 GiB, MAXN), 2026-11-05, engine 0.1.55 from
this checkout. The raw output is in [bench/results/jetson-agx-xavier](../bench/results/jetson-agx-xavier/README.md).

- **The engine builds and starts:** `strata` and `strata_kq` link, `strata --version` prints `strata 0.1.55`, and
  `strata-device --selftest` reports compute capability 7.2, 30.276 GiB, driver/runtime 11040, prints a memory plan
  and passes its arena check.
- **The CPU test suite passes:** a CUDA-free build of the C++ tests (`-DSTRATA_BUILD_TESTS=ON`) reports
  `100% tests passed, 0 tests failed out of 60` (four are skipped: the AVX2/AVX-512-only ones).
- **Six of the seven GPU parity tests that were built pass** on sm_72: `sampler_parity`, `elementwise_parity`,
  `gdn_rec_parity`, `kv_q8_parity`, `bf16_gemv_parity` and `qsa_parity` (0 failures each; the sampler and top-k
  cluster kernels are compiled out here, so those runs took the fallbacks this page describes). The seventh,
  `gemm_f32_parity`, reports that the `STRATA_DENSE_F32=1` route and the FP16-operand cuBLAS call differ bitwise on
  **all** 1,278 sizes it tested - its own header says that agreement is a property of the cuBLAS build, and
  cuBLAS 11.6.6.84 here does not have it. The FP32 route this board's pre-7.5 default takes is therefore not
  bit-identical to the FP16 one.
- **Memory read bandwidth:** 108 GB/s with a plain `float4` loop and **123 GB/s** with the unrolled probe (median of
  three runs on a 256 MB buffer) against the module's rated 137 GB/s.
- **Dense Q8_0 GEMV** runs at 85 to 95 GB/s on the wide shapes (the head matrix, 675 MB, at 86.5 GB/s) and is
  launch-bound on the narrow ones (320- and 512-row shapes at 31 to 70 GB/s).
- **The Volta prompt-attention path pays only for int8 K/V here:** 1.64x to 1.68x faster than the FP32 fallback at
  32K context, but 0.93x to 0.95x on fp16 K/V and 0.69x to 0.75x on int8+q4 V. The two agree to 1e-6 of scale.
- **Keep `STRATA_GDN_CHUNK` off on this board:** the chunked GDN prompt recurrence is 6.5x to 7.9x slower than the
  token-by-token path at 1K to 8K prompts (457 ms against 69 ms at 8,192 tokens).

Nothing above required a model. No decode or prompt tok/s figure is claimed.

## Limits

- **Compute capability 7.2 is below what the engine supports.** The Pascal/Volta build exists (#236) and is
  community-tested on P100 and V100-class cards, not on an Xavier. The tf32 matrix-multiply path has a portable fp32
  fallback below sm_80 and the tensor-core prompt kernels refuse and fall back, so the arithmetic stays defined; the
  speeds do not carry over.
- **A pack has to fit in 32 GB.** No public pack does as it ships, so a first model run needs a pack prepared for this
  board - and it would be a test, not a measurement.
- **The rest of the GPU test suite was not built here.** Seven parity tools were built and run; the full
  `-DSTRATA_BUILD_TESTS=ON` CUDA build is several hundred targets. The Terminal, the app and the Python server were not
  run either.
- **Report what you ran.** If you take this further, give the pack, the settings, the speed, and what failed.
