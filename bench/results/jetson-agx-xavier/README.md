# Jetson AGX Xavier: build, test suite and a benchmark set

Strata 0.1.55 built from this checkout on one Jetson AGX Xavier with `jetson/build_engine.sh`, CUDA 11.4, on 2026-11-05.
This is a report from hardware the maintainers do not have, in the form
[COMMUNITY_BENCHMARKS.md](../../docs/COMMUNITY_BENCHMARKS.md) asks for: the hardware, the exact commands, repeated runs,
and what was not tested. **No model was run** - no public pack fits this board's 32 GB (see
[JETSON.md](../../docs/JETSON.md)) - so there is no decode or prompt tok/s here. What is here is every kernel-level
benchmark in the tree that needs no model: 18 tool variants, 1 discarded warm-up plus **5 measured runs each**, 8,035
individual measurements over 1,611 measured series.

## The machine

| | |
| --- | --- |
| Board | Jetson AGX Xavier (8 Carmel cores), 30 GiB LPDDR4X shared with the GPU, power mode MAXN |
| GPU | Volta-generation, compute capability 7.2 (sm_72), 8 multiprocessors, 30.276 GiB visible to CUDA |
| OS | Ubuntu 20.04.6 LTS, kernel 5.10.104-tegra, L4T R35.5.0 (JetPack 5.1.3) |
| CUDA | 11.4.315 (driver and runtime both report 11040), cuBLAS 11.6.6.84 |
| Compiler | GCC 10.5.0 (the newest host compiler CUDA 11.4 accepts), CMake 4.4.4, Ninja 1.13.2 |
| Engine | Strata 0.1.55, `archs=72`, `experimental_sm60=true`, no PTX, built by `jetson/build_engine.sh` |
| CPU flags | `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm dcpop` - no dotprod, no i8mm |

## How the numbers were taken

- **One warm-up run, then five measured runs** of every tool, in the same order, in one session (`bench/jetson/bench_suite.sh`).
  Every table below gives the **median of the five, the [min-max] range and the coefficient of variation**; a median
  with a wide range is a number to distrust, and the range is printed so that judgement is yours.
- **`tegrastats` sampled every 500 ms for the whole session** (`tegrastats.log`): GPU busy, GPU/CPU/SoC temperatures.
  The heavy tools held the GPU at 99% and the GPU never passed **62 °C**, so no run in this document was
  thermally throttled. GPU clocks were **not pinned** (that needs root): the mode is MAXN and the board was idle apart
  from the suite.
- **Synthetic inputs only.** Each tool allocates its own buffers at the engine's real shapes and formats; none reads a
  model file. The values are therefore meaningless, the timings are not - each tool says so itself.
- Run-to-run spread is **under 1.5% for most series**. Two exceptions are called out where they occur: one
  `dflash2_bw_bench` run of 105.3 GB/s among four of ~123 GB/s (CV 6.5%), and the small kernels whose CV is high
  because their absolute time is a few microseconds.
- Numbers are comparable **within this document**. Another board, another session, or pinned clocks would move them.

## What a decode step costs on this board

The point of measuring the parts is to see which part a token would wait for. These are per-call medians from the
tables below, at the shapes the engine runs a 131k-context decode with:

| part of a decode step | median | where it comes from |
| --- | --- | --- |
| QSA selected attention, 1 row | 0.139 ms | `qsa_rows_bench`, at 131,072-token context |
| QSA scores + top-k + append, 1 row | 0.193 + 0.092 + 0.440 = **0.725 ms** | `qsa_rows_bench` |
| QSA dense attention (the fallback), 1 row | **1.930 ms** | `qsa_rows_bench` - 14x the selected path |
| FFN router chain, 1 row | 0.039 ms a layer, **1.86 ms** over 48 layers | `router_chain_bench` (mmvf+route+doorbell, T=1) |
| FFN router chain, 4 rows | 0.089 ms a layer, **4.29 ms** over 48 layers | the same at T=4 |
| one routed expert group (20 experts) | 0.390 ms at 45 GB/s | `expdec_bench`, IQ2_XXS gate/up + IQ4_NL down |

So on an Xavier the QSA layer alone is a **~1.2k tok/s ceiling** with the selected attention path and a **~0.38k tok/s
ceiling** if the dense fallback is taken, before any expert weights are read. The router chain is the next fixed cost
at ~1.9 ms a token. None of those are measurements of a running model - they are the per-call costs a model would pay.

## Results

### GPU memory read bandwidth (`dflash2_bw_bench`, 256 MB buffer)

| probe | blocks | median [min–max] (CV, n=5) |
|---|---|---|
| loop | 112 | 109.1 GB/s [108.3–109.8] (0.6%) |
| loop | 896 | 107.6 GB/s [106.8–108.9] (0.8%) |
| loop | 3584 | 104.3 GB/s [103.9–105.4] (0.5%) |
| loop | 14336 | 108.1 GB/s [107.6–108.8] (0.4%) |
| unroll | 112 | 106.6 GB/s [106.2–107.0] (0.3%) |
| unroll | 896 | 108.9 GB/s [108.7–109.8] (0.5%) |
| unroll | 3584 | 122.5 GB/s [105.3–123.1] (6.5%) |
| unroll | 14336 | 123.1 GB/s [122.8–123.3] (0.2%) |

### Dense Q8_0 GEMV at the engine's shapes (`q8_gemv_bench`)

| matrix | 1 col GB/s | 2 cols | 3 cols | 4 cols | 6 cols | 1 col µs |
|---|---|---|---|---|---|---|
| GDN gate 2560->6144 | 88 | 66 | 62 | 55 | 44 | 190 |
| GDN qkv 2560->10240 | 88 | 67 | 63 | 56 | 45 | 314 |
| GDN/QSA out 6144->2560 | 95 | 76 | 71 | 63 | 51 | 175 |
| HC down 10240->320 | 84 | 63 | 60 | 56 | 46 | 42 |
| HC up 320->10240 | 32 | 25 | 23 | 20 | 15 | 110 |
| QSA k/v 2560->512 | 71 | 54 | 51 | 46 | 38 | 20 |
| QSA q 2560->12288 | 89 | 67 | 63 | 56 | 44 | 375 |
| head 2560->248320 | 89 | 67 | 64 | 56 | 45 | 7580 |
| shared down 640->2560 | 51 | 42 | 38 | 33 | 25 | 34 |
| shared gate/up 2560->640 | 75 | 57 | 54 | 48 | 39 | 23 |

(GB/s falls as columns are added because the same weights are read once for every column; the 1-column row is
the decode case the engine runs per token.)

### Dense projections, two kernel variants (`b6_mmvq_bench`, A = the engine's, B = the alternative)

| case (1 column) | A ms | B ms | B vs A |
|---|---|---|---|
| gdn gate Q6_K | 0.1991 ms [0.1981–0.1995] (0.3%) | 0.1990 ms [0.1981–0.1999] (0.4%) | -0.1% |
| gdn out IQ4_XS | 0.1004 ms [0.0997–0.1013] (0.7%) | 0.1004 ms [0.0997–0.1012] (0.6%) | +0.0% |
| gdn out Q4_K | 0.1098 ms [0.1082–0.1122] (1.3%) | 0.1101 ms [0.1081–0.1121] (1.3%) | +0.3% |
| gdn out Q6_K | 0.1684 ms [0.1672–0.1705] (0.8%) | 0.1679 ms [0.1673–0.1711] (0.9%) | -0.3% |
| gdn qkv IQ4_XS | 0.2325 ms [0.2314–0.2352] (0.6%) | 0.2327 ms [0.2313–0.2341] (0.5%) | +0.1% |
| gdn qkv Q4_K | 0.2292 ms [0.2278–0.2316] (0.7%) | 0.2290 ms [0.2276–0.2319] (0.8%) | -0.1% |
| gdn qkv Q5_K | 0.2482 ms [0.2464–0.2499] (0.5%) | 0.2478 ms [0.2462–0.2525] (1.0%) | -0.2% |
| gdn qkv Q6_K | 0.3280 ms [0.3236–0.3306] (1.0%) | 0.3256 ms [0.3243–0.3289] (0.6%) | -0.7% |
| head Q5_K | 6.6209 ms [6.5924–6.7259] (0.8%) | 6.6075 ms [6.4715–6.6538] (1.1%) | -0.2% |
| qsa q IQ4_XS | 0.2791 ms [0.2760–0.2843] (1.1%) | 0.2771 ms [0.2759–0.2841] (1.2%) | -0.7% |
| qsa q Q6_K | 0.3885 ms [0.3869–0.3957] (1.0%) | 0.3884 ms [0.3879–0.4013] (1.4%) | -0.0% |
| router BF16 | 0.0296 ms [0.0294–0.0300] (0.8%) | 0.0297 ms [0.0294–0.0299] (0.7%) | +0.3% |
| shexp down IQ4_NL | 0.0379 ms [0.0376–0.0383] (0.7%) | 0.0378 ms [0.0374–0.0382] (0.8%) | -0.3% |
| shexp gate Q4_K | 0.0194 ms [0.0194–0.0196] (0.5%) | 0.0194 ms [0.0192–0.0196] (0.8%) | +0.0% |
| shexp up IQ3_S | 0.0240 ms [0.0239–0.0244] (0.9%) | 0.0241 ms [0.0239–0.0244] (0.9%) | +0.4% |

### Launch cost (`dflash2_launch_bench`, 100 chained kernels)

| shape | median [min–max] (CV) |
|---|---|
| graph 100 kernels 1 blocks | 1.0 µs [1.0–1.1] (3.8%) |
| graph 100 kernels 1024 blocks | 21.1 µs [21.0–21.5] (1.0%) |
| graph 100 kernels 256 blocks | 3.5 µs [3.5–3.5] (0.1%) |
| graph 100 kernels 64 blocks | 1.4 µs [1.4–1.4] (0.4%) |
| idle launch 100 kernels 1 blocks | 0.1 ms [0.1–0.1] (1.4%) |
| idle launch 100 kernels 1024 blocks | 2.2 ms [2.1–2.2] (0.7%) |
| idle launch 100 kernels 256 blocks | 0.4 ms [0.4–0.4] (0.5%) |
| idle launch 100 kernels 64 blocks | 0.2 ms [0.2–0.2] (0.3%) |
| stream 100 kernels 1 blocks | 4.5 µs [4.3–4.8] (3.6%) |
| stream 100 kernels 1024 blocks | 23.6 µs [23.2–26.6] (5.9%) |
| stream 100 kernels 256 blocks | 5.2 µs [5.2–5.8] (5.3%) |
| stream 100 kernels 64 blocks | 4.2 µs [3.9–4.3] (4.1%) |

### Router chain per layer, captured graph (`router_chain_bench`)

| T | composition | per layer (median [min–max] CV) |
|---|---|---|
| T1 | T1 doorbell | 2.5 µs [2.5–2.6] (0.9%) |
| T1 | T1 empty node | 1.0 µs [1.0–1.0] (1.1%) |
| T1 | T1 mmvf | 27.4 µs [27.2–27.4] (0.5%) |
| T1 | T1 mmvf+doorbell | 30.4 µs [30.2–31.0] (1.0%) |
| T1 | T1 mmvf+route | 34.4 µs [34.4–34.8] (0.5%) |
| T1 | T1 mmvf+route+doorbell | 38.8 µs [38.7–39.7] (1.1%) |
| T1 | T1 route | 5.4 µs [5.4–5.4] (0.3%) |
| T1 | T1 route+doorbell | 7.9 µs [7.9–8.0] (0.2%) |
| T2 | T2 doorbell | 2.5 µs [2.5–2.5] (0.7%) |
| T2 | T2 empty node | 1.0 µs [1.0–1.0] (1.1%) |
| T2 | T2 mmvf | 69.5 µs [65.2–70.4] (3.1%) |
| T2 | T2 mmvf+doorbell | 71.5 µs [68.9–76.0] (3.6%) |
| T2 | T2 mmvf+route | 75.2 µs [72.6–78.5] (2.8%) |
| T2 | T2 mmvf+route+doorbell | 79.8 µs [77.6–82.6] (2.4%) |
| T2 | T2 route | 5.4 µs [5.4–5.4] (0.1%) |
| T2 | T2 route+doorbell | 8.1 µs [8.0–8.1] (0.5%) |
| T4 | T4 doorbell | 2.5 µs [2.5–2.5] (0.8%) |
| T4 | T4 empty node | 1.0 µs [1.0–1.0] (0.4%) |
| T4 | T4 mmvf | 77.0 µs [76.6–77.8] (0.5%) |
| T4 | T4 mmvf+doorbell | 81.0 µs [80.1–81.1] (0.5%) |
| T4 | T4 mmvf+route | 84.9 µs [84.7–85.7] (0.5%) |
| T4 | T4 mmvf+route+doorbell | 89.3 µs [87.9–90.2] (0.9%) |
| T4 | T4 route | 5.6 µs [5.6–5.6] (0.3%) |
| T4 | T4 route+doorbell | 8.2 µs [8.2–8.2] (0.4%) |

### VRAM expert decode (`expdec_bench`, groups of 20 experts, `T` tokens)

| cell (T shape gu/down choice) | all kernels | weights read |
|---|---|---|
| T1 real 16/20 def | 390.5 µs [385.5–392.8] (0.7%) | 45.2 GB/s [45.0–45.8] (0.7%) |
| T1 real 16/42 def | 383.3 µs [382.5–402.9] (2.3%) | 34.1 GB/s [32.4–34.1] (2.2%) |
| T1 real 17/20 def | 413.0 µs [403.9–429.7] (2.4%) | 45.2 GB/s [43.5–46.3] (2.3%) |
| T1 real 17/42 def | 404.8 µs [398.2–406.8] (0.8%) | 34.8 GB/s [34.6–35.4] (0.9%) |
| T1 real 18/20 def | 454.8 µs [450.4–468.2] (1.5%) | 47.8 GB/s [46.5–48.3] (1.5%) |
| T1 real 18/42 def | 450.1 µs [445.1–452.4] (0.8%) | 38.1 GB/s [37.9–38.5] (0.7%) |
| T1 real 21/20 def | 506.2 µs [495.3–521.9] (2.1%) | 46.0 GB/s [44.6–47.0] (2.1%) |
| T1 real 21/42 def | 506.4 µs [503.3–511.1] (0.6%) | 36.9 GB/s [36.6–37.1] (0.5%) |
| T1 real 22/20 def | 426.3 µs [423.8–427.7] (0.4%) | 46.2 GB/s [46.1–46.5] (0.4%) |
| T1 real 22/42 def | 411.6 µs [410.7–436.7] (2.7%) | 36.7 GB/s [34.6–36.8] (2.6%) |
| T2 real 16/20 def | 743.5 µs [737.3–749.9] (0.8%) | 47.5 GB/s [47.1–47.9] (0.8%) |
| T2 real 16/42 def | 743.4 µs [736.4–747.6] (0.6%) | 35.1 GB/s [34.9–35.5] (0.7%) |
| T2 real 17/20 def | 795.6 µs [789.3–809.4] (1.1%) | 47.0 GB/s [46.2–47.4] (1.1%) |
| T2 real 17/42 def | 786.7 µs [783.5–806.6] (1.2%) | 35.8 GB/s [34.9–35.9] (1.2%) |
| T2 real 18/20 def | 889.3 µs [864.8–904.0] (1.9%) | 48.9 GB/s [48.1–50.3] (1.9%) |
| T2 real 18/42 def | 887.7 µs [878.2–895.4] (0.8%) | 38.6 GB/s [38.3–39.1] (0.8%) |
| T2 real 21/20 def | 1008.9 µs [993.4–1015.6] (1.0%) | 46.2 GB/s [45.9–46.9] (1.0%) |
| T2 real 21/42 def | 1002.8 µs [994.2–1010.0] (0.6%) | 37.3 GB/s [37.0–37.6] (0.6%) |
| T2 real 22/20 def | 826.0 µs [803.1–828.5] (1.5%) | 47.7 GB/s [47.6–49.1] (1.5%) |
| T2 real 22/42 def | 804.9 µs [799.7–811.6] (0.6%) | 37.5 GB/s [37.2–37.8] (0.6%) |

### Weight-bandwidth-limited expert decode, two formats (`nvfp4_decode_bench`)

| case | best | weights read | of DRAM peak |
|---|---|---|---|
| nvfp4 types 40/40 | 3.0 ms [3.0–3.0] (0.6%) | 59.4 GB/s [59.1–59.9] (0.6%) | 67% of peak |
| iq types 21/20 | 3.3 ms [3.3–3.3] (0.3%) | 45.0 GB/s [44.7–45.0] (0.4%) | 51% of peak |

### Prompt-path routed experts (`mmq_expert_bench`)

| rows/expert, product | time | throughput |
|---|---|---|
| R16 iq3_s gate/up | 1.2 ms [1.2–1.2] (1.2%) | 1.40 TOPS [1.40–1.40] (0.0%) |
| R16 iq4_nl down | 0.6 ms [0.6–0.6] (1.1%) | 1.40 TOPS [1.40–1.40] (0.0%) |
| R16 nvfp4 down | 0.9 ms [0.9–0.9] (1.4%) | 1.00 TOPS [0.90–1.00] (4.6%) |
| R16 nvfp4 gate/up | 1.3 ms [1.3–1.3] (1.4%) | 1.30 TOPS [1.30–1.30] (0.0%) |
| R16 q8_0 down | 0.7 ms [0.7–0.7] (0.7%) | 1.20 TOPS [1.20–1.20] (0.0%) |
| R4 iq3_s gate/up | 1.1 ms [1.1–1.1] (1.4%) | 0.40 TOPS [0.40–0.40] (0.0%) |
| R4 iq4_nl down | 0.5 ms [0.5–0.5] (0.4%) | 0.40 TOPS [0.40–0.40] (0.0%) |
| R4 nvfp4 down | 0.8 ms [0.8–0.8] (0.8%) | 0.30 TOPS [0.30–0.30] (0.0%) |
| R4 nvfp4 gate/up | 1.1 ms [1.1–1.1] (0.9%) | 0.40 TOPS [0.40–0.40] (0.0%) |
| R4 q8_0 down | 0.7 ms [0.6–0.7] (1.1%) | 0.30 TOPS [0.30–0.30] (0.0%) |

### Trellis (Gyro) decode kernels (`tq_pascal_bench`)

| type and shape | 1 column | 8 columns |
|---|---|---|
| Q2_0 8192x2560 | 97.4 µs [96.5–98.0] (0.6%) | 520.5 µs [520.3–520.5] (0.0%) |
| TQ2_T 8192x2560 | 352.8 µs [351.7–374.5] (2.8%) | 767.9 µs [767.4–829.4] (3.5%) |
| TQK6 8192x2560 | 348.6 µs [347.9–374.9] (3.3%) | 765.2 µs [764.7–828.8] (3.7%) |
| TQK7 8192x2560 | 349.5 µs [348.1–349.8] (0.2%) | 765.8 µs [765.2–766.4] (0.1%) |

### QSA decode kernels at a 131k context (`qsa_rows_bench`)

| kernel | T=1 | T=2 | T=4 | each extra row |
|---|---|---|---|---|
| append native_qsa_indexer_append_steps | 5.1 µs [4.8–5.1] (3.5%) | 5.0 µs [4.8–5.0] (2.3%) | 5.2 µs [4.9–5.3] (4.1%) | 0.1 µs [0.0–0.1] (86.6%) |
| attn qsa_decode_attn_batch | 139.0 µs [138.5–140.8] (0.7%) | 248.5 µs [247.9–249.9] (0.3%) | 466.7 µs [464.6–467.9] (0.3%) | 109.2 µs [108.2–109.8] (0.6%) |
| dense qsa_decode_attn_batch | 1930.5 µs [1924.5–1940.8] (0.3%) | 3488.5 µs [3476.4–3528.4] (0.6%) | 6997.3 µs [6979.2–7007.9] (0.2%) | 1688.9 µs [1684.9–1689.1] (0.1%) |
| layer append | 440.0 µs [439.3–446.8] (0.8%) | 671.9 µs [669.0–675.6] (0.4%) | 1191.3 µs [1190.9–1204.1] (0.5%) | 250.3 µs [248.3–254.9] (1.0%) |
| scores qsa_block_scores | 192.9 µs [191.5–194.1] (0.5%) | 312.3 µs [311.7–312.9] (0.1%) | 571.7 µs [571.6–571.8] (0.0%) | 126.3 µs [125.8–126.7] (0.2%) |
| topk qsa_block_topk | 91.9 µs [91.8–92.0] (0.1%) | 95.4 µs [95.4–95.4] (0.0%) | 134.9 µs [132.7–139.7] (2.2%) | 14.4 µs [13.6–16.0] (6.9%) |

### Volta prompt attention against the FP32 fallback (`qsa_prompt_attn_sm70_bench`, 32k context)

| KV format, queries | FP32 fallback | Volta path | speedup |
|---|---|---|---|
| fp16 KV | q2048 | 217.9 ms [217.1–220.4] (0.6%) | 230.5 ms [230.0–231.1] (0.2%) | 0.95× [0.94–0.95] (0.6%) |
| fp16 KV | q256 | 27.1 ms [27.0–27.6] (0.9%) | 28.7 ms [28.6–30.0] (2.0%) | 0.94× [0.90–0.96] (2.3%) |
| int8 KV | q2048 | 225.7 ms [223.2–226.3] (0.6%) | 135.8 ms [135.1–136.7] (0.5%) | 1.66× [1.64–1.67] (0.7%) |
| int8 KV | q256 | 27.8 ms [27.7–29.3] (2.4%) | 16.8 ms [16.8–17.0] (0.4%) | 1.65× [1.65–1.73] (2.1%) |
| int8+q4 V (K8V4) KV | q2048 | 234.7 ms [234.4–235.3] (0.2%) | 318.7 ms [317.8–319.6] (0.3%) | 0.74× [0.73–0.74] (0.6%) |
| int8+q4 V (K8V4) KV | q256 | 29.3 ms [29.0–29.5] (0.7%) | 39.6 ms [39.5–39.7] (0.3%) | 0.74× [0.73–0.74] (0.6%) |

### GDN prompt recurrence (`gdn_chunk_bench`)

| prompt | token-by-token | chunked (FMA) | chunked (wmma) |
|---|---|---|---|
| T1024 | 7.1 ms [7.0–7.4] (2.5%) | 55.3 ms [55.0–55.4] (0.4%) | 55.4 ms [55.1–55.5] (0.3%) |
| T4096 | 34.0 ms [33.9–35.3] (1.7%) | 228.3 ms [227.8–229.1] (0.2%) | 228.4 ms [227.7–229.6] (0.3%) |
| T8192 | 67.7 ms [67.6–69.0] (0.9%) | 456.3 ms [455.1–457.2] (0.2%) | 456.3 ms [455.2–456.9] (0.1%) |

### Drafter kernels (`dflash2_*`)

| bench | case | time | weights read / K/V |
|---|---|---|---|
| small | attention merge+quant only | 93.8 µs [93.0–94.7] (0.8%) | — |
| small | conv (resid) | 5.9 µs [5.9–5.9] (0.5%) | — |
| small | conv_quant | 7.8 µs [7.4–8.2] (3.8%) | — |
| small | conv_rms_quant | 8.5 µs [8.4–8.5] (0.6%) | — |
| small | gemm 640 x 2560 (tiny) | 43.4 µs [43.4–44.0] (0.7%) | — |
| small | kv_ring (3 rows) | 4.7 µs [4.4–4.9] (3.7%) | — |
| small | norm_rope_qk | 5.8 µs [5.8–5.8] (0.1%) | — |
| small | quant_rows (8 x 4096) | 5.3 µs [5.2–5.4] (1.7%) | — |
| small | rms_quant (8 x 2560) | 9.6 µs [9.2–10.3] (4.8%) | — |
| small | rope_table (8) | 6.3 µs [6.2–6.3] (0.2%) | — |
| small | select (7 rows) | 10.1 µs [10.1–10.1] (0.3%) | — |
| small | swiglu_quant (8 x 7680) | 12.0 µs [11.7–12.4] (2.4%) | — |
| small | topk (7 x 106299) | 300.4 µs [300.3–300.5] (0.0%) | — |
| attn | variant 0 | 617.6 µs [616.3–618.2] (0.1%) | 27.3 [27.2–27.3] (0.2%) |
| attn | variant 1 | 348.2 µs [346.5–350.0] (0.4%) | 48.3 [48.1–48.6] (0.4%) |
| attn | variant 2 | 512.8 µs [511.0–514.6] (0.3%) | 32.8 [32.7–32.9] (0.2%) |
| attn | variant 3 | 286.6 µs [283.2–288.1] (0.7%) | 58.7 [58.4–59.4] (0.7%) |
| gemm | down 7680->2560 v1 | 347.8 µs [346.1–349.1] (0.3%) | 60.1 GB/s [59.8–60.4] (0.4%) |
| gemm | down 7680->2560 v2 | 348.0 µs [347.1–349.3] (0.3%) | 60.0 GB/s [59.8–60.2] (0.3%) |
| gemm | down 7680->2560 v3 | 224.2 µs [223.5–225.9] (0.5%) | 93.2 GB/s [92.5–93.5] (0.5%) |
| gemm | down 7680->2560 v4 | 349.0 µs [346.2–350.2] (0.5%) | 59.8 GB/s [59.7–60.3] (0.4%) |
| gemm | down 7680->2560 v5 | 296.7 µs [295.2–298.1] (0.4%) | 70.4 GB/s [70.1–70.8] (0.4%) |
| gemm | down 7680->2560 v6 | 280.2 µs [279.1–283.1] (0.6%) | 74.5 GB/s [73.8–74.8] (0.6%) |
| gemm | down 7680->2560 v7 | 279.5 µs [274.5–282.4] (1.1%) | 74.7 GB/s [74.0–76.1] (1.1%) |
| gemm | down 7680->2560 v8 | 363.5 µs [358.6–367.6] (1.0%) | 57.5 GB/s [56.8–58.2] (1.0%) |
| gemm | down 7680->2560 v9 | 218.9 µs [218.5–221.5] (0.7%) | 95.4 GB/s [94.3–95.6] (0.7%) |
| gemm | fc 12800->2560 v1 | 565.3 µs [562.8–576.3] (1.0%) | 61.6 GB/s [60.4–61.9] (1.0%) |
| gemm | fc 12800->2560 v2 | 564.9 µs [562.4–571.7] (0.6%) | 61.6 GB/s [60.9–61.9] (0.6%) |
| gemm | fc 12800->2560 v3 | 361.8 µs [358.4–363.5] (0.6%) | 96.2 GB/s [95.8–97.1] (0.6%) |
| gemm | fc 12800->2560 v4 | 563.2 µs [557.3–570.1] (1.0%) | 61.8 GB/s [61.1–62.5] (1.0%) |
| gemm | fc 12800->2560 v5 | 471.3 µs [468.9–474.7] (0.6%) | 73.9 GB/s [73.3–74.3] (0.6%) |
| gemm | fc 12800->2560 v6 | 435.1 µs [434.9–445.6] (1.1%) | 80.0 GB/s [78.1–80.1] (1.1%) |
| gemm | fc 12800->2560 v7 | 434.4 µs [432.6–445.6] (1.2%) | 80.2 GB/s [78.1–80.5] (1.2%) |
| gemm | fc 12800->2560 v8 | 584.2 µs [583.7–609.3] (1.9%) | 59.6 GB/s [57.1–59.6] (1.9%) |
| gemm | fc 12800->2560 v9 | 355.9 µs [355.3–393.8] (4.7%) | 97.8 GB/s [88.4–98.0] (4.4%) |
| gemm | gate+up 2560->7680 x2 v1 | 636.0 µs [628.4–679.1] (3.2%) | 65.7 GB/s [61.5–66.5] (3.1%) |
| gemm | gate+up 2560->7680 x2 v2 | 638.7 µs [628.8–641.5] (0.8%) | 65.4 GB/s [65.1–66.4] (0.8%) |
| gemm | gate+up 2560->7680 x2 v3 | 459.9 µs [454.8–461.6] (0.6%) | 90.8 GB/s [90.5–91.9] (0.6%) |
| gemm | gate+up 2560->7680 x2 v4 | 636.4 µs [631.0–653.9] (1.4%) | 65.7 GB/s [63.9–66.2] (1.4%) |
| gemm | gate+up 2560->7680 x2 v5 | 523.8 µs [520.2–524.8] (0.4%) | 79.8 GB/s [79.6–80.3] (0.4%) |
| gemm | gate+up 2560->7680 x2 v6 | 591.9 µs [589.6–648.2] (4.2%) | 70.6 GB/s [64.5–70.9] (3.9%) |
| gemm | gate+up 2560->7680 x2 v7 | 596.4 µs [591.2–683.1] (6.5%) | 70.1 GB/s [61.2–70.7] (6.0%) |
| gemm | gate+up 2560->7680 x2 v8 | 675.3 µs [668.8–684.8] (0.9%) | 61.9 GB/s [61.0–62.5] (0.9%) |
| gemm | gate+up 2560->7680 x2 v9 | 457.8 µs [456.5–459.7] (0.3%) | 91.3 GB/s [90.9–91.5] (0.3%) |
| gemm | kv 2560->1024 x2 v1 | 97.5 µs [95.2–97.9] (1.1%) | 57.1 GB/s [56.9–58.5] (1.1%) |
| gemm | kv 2560->1024 x2 v2 | 97.6 µs [95.6–98.5] (1.1%) | 57.1 GB/s [56.6–58.3] (1.1%) |
| gemm | kv 2560->1024 x2 v3 | 70.4 µs [69.2–71.5] (1.3%) | 79.2 GB/s [77.9–80.5] (1.3%) |
| gemm | kv 2560->1024 x2 v4 | 98.4 µs [95.3–99.0] (1.6%) | 56.6 GB/s [56.2–58.4] (1.6%) |
| gemm | kv 2560->1024 x2 v5 | 86.9 µs [84.4–87.0] (1.3%) | 64.1 GB/s [64.0–66.0] (1.3%) |
| gemm | kv 2560->1024 x2 v6 | 91.6 µs [89.9–92.6] (1.1%) | 60.8 GB/s [60.1–62.0] (1.2%) |
| gemm | kv 2560->1024 x2 v7 | 90.9 µs [90.2–92.9] (1.2%) | 61.3 GB/s [59.9–61.8] (1.3%) |
| gemm | kv 2560->1024 x2 v8 | 100.6 µs [100.2–102.3] (0.8%) | 55.4 GB/s [54.5–55.6] (0.8%) |
| gemm | kv 2560->1024 x2 v9 | 69.2 µs [68.8–70.1] (0.7%) | 80.5 GB/s [79.5–81.0] (0.7%) |
| gemm | o 4096->2560 v1 | 188.1 µs [187.6–190.2] (0.5%) | 59.2 GB/s [58.6–59.4] (0.5%) |
| gemm | o 4096->2560 v2 | 189.0 µs [186.7–191.0] (0.8%) | 58.9 GB/s [58.3–59.7] (0.8%) |
| gemm | o 4096->2560 v3 | 130.8 µs [129.8–132.1] (0.7%) | 85.2 GB/s [84.4–85.8] (0.6%) |
| gemm | o 4096->2560 v4 | 189.3 µs [188.3–190.3] (0.4%) | 58.9 GB/s [58.5–59.2] (0.5%) |
| gemm | o 4096->2560 v5 | 166.4 µs [165.9–167.4] (0.3%) | 67.0 GB/s [66.6–67.2] (0.3%) |
| gemm | o 4096->2560 v6 | 162.3 µs [161.9–165.3] (0.9%) | 68.7 GB/s [67.4–68.8] (0.9%) |
| gemm | o 4096->2560 v7 | 162.3 µs [159.0–164.0] (1.3%) | 68.6 GB/s [67.9–70.1] (1.3%) |
| gemm | o 4096->2560 v8 | 197.0 µs [194.8–198.7] (0.8%) | 56.5 GB/s [56.1–57.2] (0.8%) |
| gemm | o 4096->2560 v9 | 129.0 µs [128.4–130.5] (0.7%) | 86.3 GB/s [85.3–86.8] (0.7%) |
| gemm | qkv 2560->(4096,1024,1024) v1 | 268.4 µs [266.6–315.5] (7.6%) | 62.3 GB/s [53.0–62.7] (6.9%) |
| gemm | qkv 2560->(4096,1024,1024) v2 | 269.2 µs [267.1–339.1] (11.1%) | 62.1 GB/s [49.3–62.6] (9.7%) |
| gemm | qkv 2560->(4096,1024,1024) v3 | 197.5 µs [194.5–218.3] (4.8%) | 84.6 GB/s [76.6–85.9] (4.5%) |
| gemm | qkv 2560->(4096,1024,1024) v4 | 268.9 µs [264.0–275.0] (1.5%) | 62.1 GB/s [60.8–63.3] (1.5%) |
| gemm | qkv 2560->(4096,1024,1024) v5 | 230.5 µs [223.7–233.8] (1.6%) | 72.5 GB/s [71.5–74.7] (1.6%) |
| gemm | qkv 2560->(4096,1024,1024) v6 | 255.6 µs [248.5–260.8] (1.8%) | 65.4 GB/s [64.1–67.2] (1.8%) |
| gemm | qkv 2560->(4096,1024,1024) v7 | 254.1 µs [246.5–260.4] (2.2%) | 65.8 GB/s [64.2–67.8] (2.2%) |
| gemm | qkv 2560->(4096,1024,1024) v8 | 282.4 µs [276.3–285.5] (1.6%) | 59.2 GB/s [58.5–60.5] (1.6%) |
| gemm | qkv 2560->(4096,1024,1024) v9 | 195.0 µs [192.1–199.2] (1.4%) | 85.7 GB/s [83.9–87.0] (1.4%) |

### Thermals and clocks during the measured runs (median of the five windows)

| bench | GPU busy (med) | GPU busy (max) | GPU max | CPU max | SoC max |
|---|---|---|---|---|---|
| b6_mmvq_bench | 99% | 99% | 62 °C | 54 °C | 55 °C |
| dflash2_attn_bench | 8% | 98% | 46 °C | 46 °C | 45 °C |
| dflash2_bw_bench | 98% | 99% | 46 °C | 46 °C | 44 °C |
| dflash2_gemm_bench | 47% | 99% | 48 °C | 46 °C | 45 °C |
| dflash2_launch_bench | 98% | 99% | 46 °C | 46 °C | 45 °C |
| dflash2_small_bench | 97% | 99% | 46 °C | 45 °C | 44 °C |
| expdec_bench | 99% | 99% | 62 °C | 56 °C | 56 °C |
| gdn_chunk_bench | 99% | 99% | 56 °C | 48 °C | 49 °C |
| mmq_expert_bench | 0% | 97% | 48 °C | 48 °C | 47 °C |
| nvfp4_decode_bench | 17% | 98% | 48 °C | 48 °C | 47 °C |
| nvfp4_decode_bench_iq | 0% | 98% | 48 °C | 48 °C | 46 °C |
| q8_gemv_bench | 11% | 99% | 58 °C | 50 °C | 51 °C |
| qsa_prompt_attn_sm70_bench | 99% | 99% | 58 °C | 51 °C | 51 °C |
| qsa_rows_bench | 99% | 99% | 57 °C | 50 °C | 51 °C |
| router_chain_bench_T1 | 78% | 99% | 50 °C | 47 °C | 47 °C |
| router_chain_bench_T2 | 92% | 99% | 52 °C | 48 °C | 48 °C |
| router_chain_bench_T4 | 76% | 99% | 53 °C | 48 °C | 48 °C |
| tq_pascal_bench | 1% | 99% | 48 °C | 46 °C | 45 °C |

## What the numbers say

1. **Memory read bandwidth is 123.1 GB/s** with the unrolled probe and 109.1 GB/s with a plain `float4` loop, against
   the module's 137 GB/s rating - 90% and 80%. This is the ceiling every weight-bound kernel below is measured against.
2. **Decode GEMV comes close to it on the wide matrices** (89-95 GB/s for the head, the QSA output and the
   GDN/QSA output), and falls away on the narrow shapes: 31.7 GB/s on the 320-row `HC up` matrix, which is
   launch-bound rather than bandwidth-bound. Adding columns costs throughput, as the same weights are re-read.
3. **Expert decode - the path a real token spends its time in - is at 45-59 GB/s**, i.e. 37-48% of what the board can
   read: 45.2 GB/s for IQ2_XXS gate/up with IQ4_NL down, 59.4 GB/s for the NVFP4 pair. The Q2_0-down pairs are the
   slowest per byte (34.1 GB/s for 16/42). (`nvfp4_decode_bench` prints its own "% of DRAM peak" against a 88 GB/s
   reference, which is not this board's peak; the GB/s column is the comparable one.)
4. **The trellis (Gyro/TQ) decode kernels are 3.6x slower per column than Q2_0** for the same shape - 352.8 us against
   97.4 us at 8192x2560, 1 column - so a TQ pack is a poor fit for this board's GPU path even though it is smaller.
5. **Volta prompt attention pays only for int8 K/V**: 1.66x faster than the FP32 fallback at 2048 queries, but 0.95x
   on fp16 and 0.74x on int8+q4 V. The ranges overlap between runs, so the ordering is solid and the size of each
   effect is small.
6. **The chunked GDN prompt recurrence is 6.5-7.9x slower than token-by-token** (456 ms against 68 ms at 8,192 tokens).
   `STRATA_GDN_CHUNK` is opt-in; leave it off on an sm_72 card. The two paths agree to 2.2e-8, so this is speed alone.
7. **Graph capture is what makes small kernels viable**: a 1-block kernel in a captured graph costs 1.03 us against
   4.48 us streamed - 4.3x - while a 1024-block kernel costs 21.1 us either way.
8. **The two dense-projection kernel variants in `b6_mmvq_bench` are within 1% of each other here** (A and B in the
   table, every case), so on this card the choice between them does not matter; on the cards they were written for it
   does.
9. **Prompt-path routed experts reach 1.4 TOPS** at 16 rows an expert (`mmq_expert_bench`, IQ3_S gate/up and
   IQ4_NL down), 0.4 TOPS at 4 rows - the prompt path is far behind the decode path's bandwidth.

## Test suite

**CPU suite** - a CUDA-free build of the C++ tests (`-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`), `ctest -j2`:

```
100% tests passed, 0 tests failed out of 60
  (4 skipped: expert_multi_test, native_hadamard_test, tq_cpu_parity, tq_cpu_parity_avx2 - x86 AVX2/AVX-512 only)
```

Two of these (`tqp_cpu_test`, `tqp_cpu_test_avx2`) failed with `std::system_error: Unknown error -1` before the
`THREADS_PREFER_PTHREAD_FLAG` fix in this checkout: on glibc 2.31 (Ubuntu 20.04) `--as-needed` drops an explicit
`-lpthread`, the pthread calls then land on libc's single-threaded stubs, and `std::call_once` throws. They pass now.

**GPU parity tests built on the board**, each run once:

| Test | Result | What it compares |
| --- | --- | --- |
| `sampler_parity --selftest` | **pass** | the sampled token chain; the cluster kernel is compiled out here, so this exercises the one-block fallback |
| `elementwise_parity --selftest` | **pass** | elementwise kernels against their scalar references |
| `gdn_rec_parity` | **pass** (0 failures) | the GDN recurrence against the reference |
| `kv_q8_parity` | **pass** (worst INT8-vs-FP16 error 0.59 quantization steps) | the int8 K/V path |
| `bf16_gemv_parity` | **pass** (0 failures) | BF16 GEMV on this card |
| `qsa_parity` | **pass** (0 failures) | QSA scores, top-k and merge; the cluster top-k kernel is compiled out here |
| `gemm_f32_parity` | **FAIL**: 1,278 of 1,278 sizes differ (first at T=256, 56,911,505,591 outputs in all) | the opt-in `STRATA_DENSE_F32=1` route against the FP16-operand cuBLAS call, bitwise |

`gemm_f32_parity`'s own header says the two routes agreeing "is a property of the cuBLAS build (measured: 12.4.5.8 on a
Tesla P40)". On cuBLAS 11.6.6.84 / aarch64 they do not agree bitwise on any size tested, so the FP32 route this board's
pre-7.5 default takes is numerically different from the FP16 one by FP32 rounding, not bit for bit. Nothing here
measures whether that changes an answer.

## What was not tested

- **No model, no decode or prompt tok/s.** No public pack fits 30 GiB, and sm_72 is below the engine's 7.5 floor, so a
  first model run is a test, not a measurement. Every number above is a kernel call in isolation; a model adds
  scheduling, the CPU pool and the expert cache on top.
- **No CPU expert benchmark on this board.** `kq_cpu_bench` is built only for x86 (`if(STRATA_CPU_X86)` in the
  top-level CMakeLists), so the CPU lane's expert rows have no benchmark here.
- **Clocks were not pinned**, so the absolute numbers belong to this session. The spread columns bound how much that
  matters for each series.
- **One board, one session, one build.** Nothing was repeated on another Xavier, and no long soak; the GPU never
  passed 62 °C, so nothing here is a statement about sustained thermals.
- **Not run at all:** the Terminal, the app, the Python server, and every benchmark that needs a pack
  (`native_expert_bench`, `pa_bench`, `bench.sh`).

## Reproduce it

```sh
sh jetson/toolchain.sh && . ~/toolchains/strata-env.sh    # CMake 4.4.4, Ninja, GCC 10 - no root
JOBS=4 sh jetson/build_engine.sh                          # the engine
cmake -S . -B build-jetson -DSTRATA_BUILD_TESTS=ON        # adds the parity tools and benchmarks
ninja -C build-jetson b6_mmvq_bench router_chain_bench tq_pascal_bench nvfp4_decode_bench mmq_expert_bench \
      expdec_bench dflash2_gemm_bench dflash2_small_bench dflash2_attn_bench dflash2_launch_bench dflash2_bw_bench \
      q8_gemv_bench qsa_rows_bench qsa_prompt_attn_sm70_bench gdn_chunk_bench
N=5 sh bench/jetson/bench_suite.sh                        # 1 warm-up + 5 measured runs, ~32 minutes here
python3 bench/jetson/bench_agg.py                         # metrics.csv + summary.csv (the statistics)
python3 bench/jetson/bench_report.py                      # the tables in this file
```

The raw output of every run is in `runs/<tool>/run<NN>.txt` (`run00` is the discarded warm-up), the clocks and
temperatures in `tegrastats.log`, each run's window in `manifest.tsv`, and the statistics the tables came from in
`summary.csv`.
