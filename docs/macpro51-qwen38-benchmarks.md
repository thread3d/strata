# Qwen3.8-Flash-Next quantisation benchmarks on macpro51 (RX 580, Metal 2)

_Generated 2026-11-19 00:36 UTC by `bench/macpro51/run_bench_qwen38.py` after every completed run. Every number below was produced on the machine named here._

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata` 0.1.59-metal (`metal/` build, Ninja Release), branch `macpro51` |
| Branch / commit at the last run | `macpro51` / `57cc869362` |

The engine's Metal decode window reads the model's GGUF in place. The RX 580 cannot hold a 40–160 GB pack in 8 GB of VRAM, so the dense side (norms, hyper-connection projections, the shared expert and the attention/GDN projections) is read once into host memory and the routed experts stream from the SSD through an arena; a token's experts are looked up after its router and the next layer's are read ahead. Nothing is copied into a second file.

## What ran

Qwen3.8-Flash-Next (`qwen4exp`) is the shape the engine compiles for: 48 trunk layers (36 gated-DeltaNet, 12 full-attention QSA with the 4×128 indexer), 4 hyper-connection streams of rank 320, and 512 routed experts of width 640 with 10 used per token, plus the 28.8 GB PLE n-gram table in the second shard. Every pack below is the same architecture; only the expert encodings differ.

| Pack | Quantisation | Source | On disk | Result |
| --- | --- | --- | ---: | --- |
| `Q2_0` | Q2_0 (GSQ-RCO) | ISTA-DASLab | 61.9 GiB | 3 runs, 1.05 tok/s decode (steady state) |
| `IQ3_XXS` | IQ3_XXS (GSQ-RCO) | ISTA-DASLab | 70.6 GiB | 2 runs, 0.90 tok/s decode (steady state) |
| `IQ3_S` | IQ3_S (GSQ-RCO) | ISTA-DASLab | 77.9 GiB | 3 runs, 0.82 tok/s decode (steady state) |
| `UD-IQ1_M` | UD-IQ1_M (Unsloth v3.1) | unsloth | 63.3 GiB | queued |
| `UD-Q2_0` | UD-Q2_0 (Unsloth v3.1) | unsloth | 5.7 GiB | queued |
| `UD-Q4_K_XL` | UD-Q4_K_XL (Unsloth) | unsloth | 103.7 GiB | 2 runs, 0.62 tok/s decode (steady state) |
| `UD-Q5_K_XL` | UD-Q5_K_XL (Unsloth) | unsloth | 147.4 GiB | 3 runs, 0.57 tok/s decode (steady state) |
| `UD-Q6_K_XL` | UD-Q6_K_XL (Unsloth) | unsloth | 157.5 GiB | 2 runs, 0.54 tok/s decode (steady state) |
| `IQ2_XS` | refused on this branch | — | 63.4 GiB | the window has no kernel for its IQ1_M gate/up experts: `mtl: no function moe_gu_29_32_1_4` |

Not on disk yet (not attempted, and not counted as a failure): `UD-IQ2_XS`, `UD-Q2_K_XL`, `UD-IQ3_XXS`, `UD-Q3_K_XL`, `UD-IQ4_XS`, `Q8_0`, `Gyro-S`, `Gyro-M`.

## Method

- **One process per run.** `strata` opens the shard, builds the pack metadata once, reads the dense side and the expert arena, then times the prompt and the decode separately. Load time is reported on its own and is not part of either throughput.
- **Fixed workload.** The same 32 token ids for every pack (the Qwen 248 320-entry vocabulary, a varied English sentence rather than repeated ids, so routing is not narrowed), read one at a time, then 32 greedy tokens.
- **Mixed order.** Each round contains every downloaded pack once, in a fresh shuffled order (seed in the run's `meta.json`). No pack is always first or last, so thermal drift, page-cache state and SSD behaviour are spread across the matrix.
- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs (rounds 2+); round 1 is each pack's first pass and is priced separately. Individual runs are the dots on the charts.
- **Streaming.** `STRATA_METAL_STREAM=1` is set for every run: the pack cannot be resident in 8 GB of VRAM, and forcing the streamed path keeps every pack on the same code path.

## Decode throughput

![Decode throughput by quantisation](benchmarks/macpro51/qwen38-decode.svg)

| Pack | Steady runs | Decode, steady (mean ± sd) | All runs (mean) | Steady range | Prefill, steady (mean ± sd) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | 2 | 1.05 ± 0.000 | 1.05 | 1.05 – 1.05 | 3.34 ± 0.006 |
| `IQ3_XXS` | 2 | 0.898 ± 0.004 | 0.898 | 0.895 – 0.901 | 2.58 ± 0.129 |
| `IQ3_S` | 2 | 0.817 ± 0.001 | 0.815 | 0.817 – 0.818 | 2.41 ± 0.011 |
| `UD-Q4_K_XL` | 2 | 0.615 ± 0.002 | 0.615 | 0.614 – 0.617 | 1.89 ± 0.034 |
| `UD-Q5_K_XL` | 2 | 0.574 ± 0.000 | 0.573 | 0.574 – 0.574 | 1.81 ± 0.004 |
| `UD-Q6_K_XL` | 2 | 0.540 ± 0.000 | 0.540 | 0.540 – 0.540 | 1.40 ± 0.013 |

## Prompt (prefill) throughput

![Prefill throughput by quantisation](benchmarks/macpro51/qwen38-prefill.svg)

The prompt is read one token at a time on this port, so prefill here is per-token latency through the same expert stream, not the batched GEMM a CUDA card runs. It is comparable across these packs, not to a desktop GPU's numbers.

## Size against speed

![Decode throughput against pack size](benchmarks/macpro51/qwen38-size-vs-decode.svg)

A token reads only its 10 routed experts per layer, so a bigger file is not automatically slower: what matters is the bytes one token actually touches, the encoding's decode cost (Q2_0 and the i-quants trade a wider row read for cheap decode), and how much of the dense side stays resident.

## The bytes a token reads

![Decode throughput against the largest expert blob](benchmarks/macpro51/qwen38-decode-vs-blob.svg)

| Pack | Largest expert blob | Decode, steady |
| --- | ---: | ---: |
| `Q2_0` | 1.38 MB | 1.05 ± 0.000 tok/s |
| `IQ3_XXS` | 2.33 MB | 0.898 ± 0.004 tok/s |
| `IQ3_S` | 2.66 MB | 0.817 ± 0.001 tok/s |
| `UD-Q4_K_XL` | 3.99 MB | 0.615 ± 0.002 tok/s |
| `UD-Q5_K_XL` | 4.43 MB | 0.574 ± 0.000 tok/s |
| `UD-Q6_K_XL` | 5.22 MB | 0.540 ± 0.000 tok/s |

Every token pulls its ten routed experts per layer from the SSD, so the blob one expert costs is the number a token actually pays for; the pack file is mostly experts a token never asks for.  Two packs whose blobs are the same size land at nearly the same rate, which is why the Unsloth 4.5-bit set is slower than the 2-bit one even though their files differ by less than 2x.


## What the numbers say

- **Fastest decode:** `Q2_0` at 1.05 tok/s; **slowest:** `UD-Q6_K_XL` at 0.54 tok/s (1.95× between them).
- **Run-to-run spread:** coefficient of variation from 0.0% (`Q2_0`) to 0.5% (`IQ3_XXS`).
- **Fastest prompt read:** `Q2_0` at 3.34 tok/s; slowest `UD-Q6_K_XL` at 1.40 tok/s.

The first round is the first pass over each pack (cold page cache, cold GPU clocks); the later rounds are the steady state. The difference is the price of a first run, not a property of the pack: it averages +0.5% over these 6 packs.

| Pack | Round 1 (tok/s) | Rounds 2+ (tok/s) | Change |
| --- | ---: | ---: | ---: |
| `Q2_0` | 1.05 | 1.05 | +0.6% |
| `IQ3_XXS` | 0.89 | 0.90 | +0.7% |
| `IQ3_S` | 0.81 | 0.82 | +1.0% |
| `UD-Q4_K_XL` | 0.61 | 0.62 | +0.5% |
| `UD-Q5_K_XL` | 0.57 | 0.57 | +0.5% |
| `UD-Q6_K_XL` | 0.54 | 0.54 | -0.0% |

## Run order (mixing)

![Every run in execution order](benchmarks/macpro51/qwen38-run-order.svg)

The sweep is resumable: a `(round, pack)` pair already in `results-qwen38.jsonl` is skipped, so an interrupted or rebooted machine costs one run rather than the series.

## Per-run log

| # | Round | Pack | Prefill s | Decode s | Decode tok/s | Note |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 1 | `UD-Q5_K_XL` | 17.29 | 56.03 | 0.571 | first round (cold) |
| 2 | 1 | `UD-Q4_K_XL` | 16.64 | 52.14 | 0.614 | first round (cold) |
| 3 | 1 | `Q2_0` | 9.82 | 30.60 | 1.05 | first round (cold) |
| 4 | 1 | `IQ3_S` | 13.54 | 39.52 | 0.810 | first round (cold) |
| 5 | 1 | `IQ3_XXS` | 12.46 | 35.77 | 0.895 | first round (cold) |
| 6 | 1 | `UD-Q6_K_XL` | 21.94 | 59.25 | 0.540 | first round (cold) |
| 7 | 2 | `UD-Q6_K_XL` | 22.23 | 59.27 | 0.540 | later round |
| 8 | 2 | `IQ3_S` | 12.80 | 39.11 | 0.818 | later round |
| 9 | 2 | `UD-Q5_K_XL` | 17.12 | 55.77 | 0.574 | later round |
| 10 | 2 | `UD-Q4_K_XL` | 16.23 | 51.87 | 0.617 | later round |
| 11 | 2 | `IQ3_XXS` | 11.61 | 35.53 | 0.901 | later round |
| 12 | 2 | `Q2_0` | 9.27 | 30.43 | 1.05 | later round |
| 13 | 3 | `UD-Q5_K_XL` | 17.07 | 55.75 | 0.574 | later round |
| 14 | 3 | `IQ3_S` | 12.89 | 39.18 | 0.817 | later round |
| 15 | 3 | `Q2_0` | 9.30 | 30.43 | 1.05 | later round |

## Limits recorded here

- The pack's second shard is the 28.8 GB PLE n-gram table. It is read from the SSD a few rows at a time and is never loaded whole; its size counts in the download, not in VRAM.
- Metal 2 on this card has no device addresses; kernels take buffer + offset, and the engine hands every weight that way. No Metal 3 feature is used.
- The output head is 512 wide in the draft path and 248 320 wide in the sample; greedy decoding is used throughout so the comparison is not a sampler's.

