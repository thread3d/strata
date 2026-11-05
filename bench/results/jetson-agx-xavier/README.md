# Jetson AGX Xavier: build, test suite and micro-benchmarks

Strata 0.1.55 built from this checkout on one Jetson AGX Xavier with `jetson/build_engine.sh`, CUDA 11.4, on 2026-11-05.
This is a report from hardware the maintainers do not have, in the form
[COMMUNITY_BENCHMARKS.md](../../docs/COMMUNITY_BENCHMARKS.md) asks for: the hardware, the exact commands, three runs
each, and what was not tested. **No model was run** - no public pack fits this board's 32 GB (see
[JETSON.md](../../docs/JETSON.md)) - so there are no decode or prompt tok/s numbers here, only kernel-level
measurements that need no model.

## The machine

| | |
| --- | --- |
| Board | Jetson AGX Xavier (8 Carmel cores), 30 GiB LPDDR4X shared with the GPU, power mode MAXN |
| GPU | Volta-generation, compute capability 7.2 (sm_72), 8 multiprocessors, 30.276 GiB visible to CUDA |
| OS | Ubuntu 20.04.6 LTS, kernel 5.10.104-tegra (JetPack 5) |
| CUDA | 11.4.315 (driver and runtime both report 11040), cuBLAS 11.6.6.84 |
| Compiler | GCC 10.5.0 (the newest host compiler CUDA 11.4 accepts), CMake 4.4.4, Ninja 1.13.2 |
| Engine | Strata 0.1.55, `archs=72`, `experimental_sm60=true`, no PTX, built by `jetson/build_engine.sh` |
| CPU flags | `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop` - no dotprod, no i8mm |

## How it was built

```sh
sh jetson/toolchain.sh                              # cmake/ninja (pip --user) and GCC 10 (unpacked, no root)
. ~/toolchains/strata-env.sh
JOBS=4 sh jetson/build_engine.sh                    # -> out-jetson/engine
```

The engine and `strata_kq` link and run: `strata --version` prints `strata 0.1.55`, and
`strata-device --selftest` reports the card, computes the memory plan and passes its arena check. What had to change in
the tree is in [JETSON.md](../../docs/JETSON.md).

## Test suite

**CPU suite** - a CUDA-free build of the C++ tests (`-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`), `ctest -j2`:

```
100% tests passed, 0 tests failed out of 60
  (4 skipped: expert_multi_test, native_hadamard_test, tq_cpu_parity, tq_cpu_parity_avx2 - x86 AVX2/AVX-512 only)
```

Two of these (`tqp_cpu_test`, `tqp_cpu_test_avx2`) failed with `std::system_error: Unknown error -1` before the
`THREADS_PREFER_PTHREAD_FLAG` fix in this checkout: on glibc 2.31 (Ubuntu 20.04) `--as-needed` drops an explicit
`-lpthread`, the pthread calls then land on libc's single-threaded stubs, and `std::call_once` throws. They pass now.

**GPU parity tests built on the board** (`ninja sampler_parity elementwise_parity gemm_f32_parity gdn_rec_parity
kv_q8_parity bf16_gemv_parity qsa_parity`), each run once:

| Test | Result | What it compares |
| --- | --- | --- |
| `sampler_parity --selftest` | **pass** | the sampled token chain; on this board the cluster kernel is compiled out, so it exercises the one-block fallback |
| `elementwise_parity --selftest` | **pass** | elementwise kernels against their scalar references |
| `gdn_rec_parity` | **pass** (0 failures) | the GDN recurrence against the reference |
| `kv_q8_parity` | **pass** (worst INT8-vs-FP16 error 0.59 quantization steps) | the int8 K/V path |
| `bf16_gemv_parity` | **pass** (0 failures) | BF16 GEMV on this card |
| `qsa_parity` | **pass** (0 failures) | QSA scores, top-k and merge; the cluster top-k kernel is compiled out here |
| `gemm_f32_parity` | **FAIL**: 1,278 of 1,278 sizes differ (first at T=256, 56,911,505,591 outputs in all); the run then hit the 900 s cap while continuing the sweep | the opt-in `STRATA_DENSE_F32=1` route (cublasSgemm on FP32 copies) against the FP16-operand cuBLAS call, bitwise |

`gemm_f32_parity`'s own header says the two routes agreeing "is a property of the cuBLAS build (measured: 12.4.5.8 on a
Tesla P40)". On cuBLAS 11.6.6.84 / aarch64 they do not agree bitwise on any size tested, so on this board the FP32
route the pre-sm_75 default takes is numerically different from the FP16 one by FP32 rounding, not bit for bit. Nothing
in this report measures whether that changes an answer; nobody has run a model here.

## Benchmarks

Every tool below was run three times; the tables give the median of the three and the spread is in the raw files.

**Memory read bandwidth** (`dflash2_bw_bench`, a 256 MB buffer summed with `float4`/`int4` loads):

| Launch shape | median | three runs |
| --- | --- | --- |
| 14,336 blocks, `loop` | 108.4 GB/s | 108.7 / 108.2 / 108.4 |
| 14,336 blocks, `unroll` | 123.2 GB/s | 123.8 / 122.9 / 123.2 |

Xavier AGX's 32 GB module is rated 137 GB/s, so the unrolled probe reaches about 90% of it.

**Dense Q8_0 GEMV** (`q8_gemv_bench --iters 50`, the engine's own `native_mmvq` call, 1 column, medians):

| Matrix (shape) | median time | median | three runs (GB/s) |
| --- | --- | --- | --- |
| GDN qkv 2560->10240 | 325.8 us | 85.5 GB/s | 84.6 / 85.9 / 85.5 |
| GDN gate 2560->6144 | 195.4 us | 85.5 GB/s | 85.7 / 85.5 / 85.0 |
| GDN/QSA out 6144->2560 | 176.7 us | 94.6 GB/s | 94.6 / 95.5 / 94.3 |
| QSA q 2560->12288 | 389.8 us | 85.7 GB/s | 85.7 / 87.3 / 85.7 |
| QSA k/v 2560->512 | 19.9 us | 69.9 GB/s | 70.5 / 69.9 / 68.8 |
| HC down 10240->320 | 41.1 us | 84.8 GB/s | 84.8 / 85.0 / 83.3 |
| HC up 320->10240 | 110.1 us | 31.6 GB/s | 31.6 / 32.2 / 31.4 |
| shared gate/up 2560->640 | 23.3 us | 74.8 GB/s | 74.8 / 75.3 / 73.7 |
| shared down 640->2560 | 34.2 us | 50.9 GB/s | 50.9 / 51.2 / 50.7 |
| head 2560->248320 | 7,805 us | 86.5 GB/s | 86.5 / 87.1 / 84.8 |

Wide shapes sit near the board's read bandwidth; the narrow (320-row and 512-row) shapes are launch-bound.

**Volta prompt attention against the FP32 fallback** (`qsa_prompt_attn_sm70_bench`, ctx 32,768, selection 2051 cells,
2048 queries of 24 heads x 256 dims, medians of three runs):

| KV format | queries | FP32 fallback | Volta path | speedup |
| --- | --- | --- | --- | --- |
| int8 | 2,048 | 222.8 ms | 135.5 ms | **1.64x** |
| int8 | 256 | 28.3 ms | 16.9 ms | 1.68x |
| int8+q4 V (K8V4) | 2,048 | 235.0 ms | 319.6 ms | 0.74x |
| int8+q4 V (K8V4) | 256 | 29.1 ms | 39.6 ms | 0.74x |
| fp16 | 2,048 | 217.1 ms | 230.6 ms | 0.94x |
| fp16 | 256 | 27.1 ms | 29.3 ms | 0.93x |

The two paths agree to 1e-6 of the output scale on this card. On Xavier the Volta prompt-attention path pays only for
**int8** K/V; for fp16 and K8V4 the FP32 fallback is faster, which is worth knowing before enabling it on an sm_72
board.

**Chunked GDN prompt recurrence against token-by-token** (`gdn_chunk_bench`, real GDN geometry, medians of three runs):

| prompt tokens | token-by-token (pipe) | chunked (FMA) | chunked (wmma) |
| --- | --- | --- | --- |
| 1,024 | 7.04 ms | 55.3 ms | 55.4 ms |
| 4,096 | 34.1 ms | 228.2 ms | 228.6 ms |
| 8,192 | 69.4 ms | 457.4 ms | 456.9 ms |

The chunked path is **6.5x to 7.9x slower** than the token recurrence on this board (the two differ by 1.6e-8 at most,
so this is speed, not correctness). `STRATA_GDN_CHUNK` is opt-in; leave it off on an sm_72 card.

## What was not tested

- **No model, no decode or prompt throughput.** No public pack fits 32 GB, and sm_72 is below the engine's 7.5 floor, so
  a first model run is a test, not a measurement.
- **The rest of the test suite.** 7 of the CUDA parity tools were built and run here; the full suite
  (`-DSTRATA_BUILD_TESTS=ON` on the CUDA build) is several hundred targets and was not built on this board.
- **The Terminal, the app and the Python server.** Nothing Python-side was run beyond `setup.py`'s llama.cpp fetch.
- **Power and clocks beyond MAXN**, thermal throttling over a long run, and any multi-day soak. GR3D sat at 99% and the
  GPU reported 60 C during the attention bench; nothing was measured across a sustained load.
- **Clocks were not pinned.** `nvpmodel -q` reports MAXN, but the GPU clock was not fixed, so the numbers are
  comparable within this report only.

## Raw output

`environment.txt`, `device.txt`, `engine.txt`, `tests.txt`, `tests-summary.txt`, and one file per benchmark with all
three runs.
