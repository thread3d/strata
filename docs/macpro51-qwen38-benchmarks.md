# Qwen3.8-Flash-Next quantisation benchmarks on macpro51 (RX 580, Metal 2)

_Generated 2026-11-19 03:34 UTC by `bench/macpro51/run_bench_qwen38.py` after every completed run. Every number below was produced on the machine named here._

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata` 0.1.59-metal (`metal/` build, Ninja Release), branch `macpro51` |
| Branch / commit at the last run | `macpro51` / `1d58c15b0a` |

The engine's Metal decode window reads the model's GGUF in place. The RX 580 cannot hold a 40–160 GB pack in 8 GB of VRAM, so the dense side (norms, hyper-connection projections, the shared expert and the attention/GDN projections) is read once into host memory and the routed experts stream from the SSD through an arena; a token's experts are looked up after its router and the next layer's are read ahead. Nothing is copied into a second file.

## What ran

Qwen3.8-Flash-Next (`qwen4exp`) is the shape the engine compiles for: 48 trunk layers (36 gated-DeltaNet, 12 full-attention QSA with the 4×128 indexer), 4 hyper-connection streams of rank 320, and 512 routed experts of width 640 with 10 used per token, plus the 28.8 GB PLE n-gram table in the second shard. Every pack below is the same architecture; only the expert encodings differ.

| Pack | Quantisation | Source | On disk | Result |
| --- | --- | --- | ---: | --- |
| `Q2_0` | Q2_0 (GSQ-RCO) | ISTA-DASLab | 61.9 GiB | 5 runs, 1.06 tok/s decode (steady state) |
| `IQ3_XXS` | IQ3_XXS (GSQ-RCO) | ISTA-DASLab | 70.6 GiB | 5 runs, 0.91 tok/s decode (steady state) |
| `IQ3_S` | IQ3_S (GSQ-RCO) | ISTA-DASLab | 77.9 GiB | 5 runs, 0.82 tok/s decode (steady state) |
| `UD-Q2_0` | UD-Q2_0 (Unsloth v3.1) | unsloth | 65.7 GiB | runs, but its greedy answer diverges from every other pack's |
| `UD-IQ2_XS` | UD-IQ2_XS (Unsloth v3.1) | unsloth | 65.0 GiB | queued |
| `UD-Q4_K_XL` | UD-Q4_K_XL (Unsloth) | unsloth | 103.7 GiB | 5 runs, 0.62 tok/s decode (steady state) |
| `UD-Q5_K_XL` | UD-Q5_K_XL (Unsloth) | unsloth | 147.4 GiB | 5 runs, 0.58 tok/s decode (steady state) |
| `UD-Q6_K_XL` | UD-Q6_K_XL (Unsloth) | unsloth | 157.5 GiB | 5 runs, 0.53 tok/s decode (steady state) |
| `IQ2_XS` | refused on this branch | — | 63.4 GiB | the window has no kernel for its IQ1_M gate/up experts: `mtl: no function moe_gu_29_32_1_4` |
| `UD-IQ1_M` | refused on this branch | — | 63.3 GiB | runs, but its forward is degenerate (`76 76 1544 7509 4654 76 88 76`, first-token logits at cosine 0.16 to Q2_0's); the residual is healthy after layer 0 (0.177; Q2_0 0.181), still healthy after layer 1's attention (0.173; Q2_0 0.174, UD-Q4 0.170), and 15.1 after layer 1's expert dispatch (UD-Q4 0.348), then stays there.  The PLE block's own outputs (keyn/qn/gated/normW) match the healthy packs to three digits, the experts pass `expert_parity` on this pack's real data (rel L2 1.3e-7), and layer 1 uses the same expert types as layer 0 - so the fault is inside layer 1's routed/shared expert path (plan, gu, swiglu, down, combine), not the quantisation |

Not on disk yet (not attempted, and not counted as a failure): `UD-Q2_K_XL`, `UD-IQ3_XXS`, `UD-Q3_K_XL`, `UD-IQ4_XS`, `Q8_0`, `Gyro-S`, `Gyro-M`.

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
| `Q2_0` | 4 | 1.06 ± 0.002 | 1.06 | 1.06 – 1.06 | 3.31 ± 0.025 |
| `IQ3_XXS` | 4 | 0.907 ± 0.002 | 0.907 | 0.904 – 0.909 | 2.65 ± 0.015 |
| `IQ3_S` | 4 | 0.821 ± 0.001 | 0.821 | 0.820 – 0.822 | 2.40 ± 0.009 |
| `UD-Q4_K_XL` | 4 | 0.619 ± 0.004 | 0.619 | 0.613 – 0.622 | 1.92 ± 0.017 |
| `UD-Q5_K_XL` | 4 | 0.577 ± 0.000 | 0.577 | 0.577 – 0.577 | 1.81 ± 0.011 |
| `UD-Q6_K_XL` | 4 | 0.532 ± 0.002 | 0.532 | 0.529 – 0.533 | 1.42 ± 0.007 |

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
| `Q2_0` | 1.38 MB | 1.06 ± 0.002 tok/s |
| `IQ3_XXS` | 2.33 MB | 0.907 ± 0.002 tok/s |
| `IQ3_S` | 2.66 MB | 0.821 ± 0.001 tok/s |
| `UD-Q4_K_XL` | 3.99 MB | 0.619 ± 0.004 tok/s |
| `UD-Q5_K_XL` | 4.43 MB | 0.577 ± 0.000 tok/s |
| `UD-Q6_K_XL` | 5.22 MB | 0.532 ± 0.002 tok/s |

Every token pulls its ten routed experts per layer from the SSD, so the blob one expert costs is the number a token actually pays for; the pack file is mostly experts a token never asks for.  Two packs whose blobs are the same size land at nearly the same rate, which is why the Unsloth 4.5-bit set is slower than the 2-bit one even though their files differ by less than 2x.


## What the numbers say

- **Fastest decode:** `Q2_0` at 1.06 tok/s; **slowest:** `UD-Q6_K_XL` at 0.53 tok/s (1.99× between them).
- **Run-to-run spread:** coefficient of variation from 0.0% (`UD-Q5_K_XL`) to 0.7% (`UD-Q4_K_XL`).
- **Fastest prompt read:** `Q2_0` at 3.31 tok/s; slowest `UD-Q6_K_XL` at 1.42 tok/s.

The first round is the first pass over each pack (cold page cache, cold GPU clocks); the later rounds are the steady state. The difference is the price of a first run, not a property of the pack: it averages +0.0% over these 6 packs.

| Pack | Round 1 (tok/s) | Rounds 2+ (tok/s) | Change |
| --- | ---: | ---: | ---: |
| `Q2_0` | 1.06 | 1.06 | +0.0% |
| `IQ3_XXS` | 0.91 | 0.91 | -0.1% |
| `IQ3_S` | 0.82 | 0.82 | -0.1% |
| `UD-Q4_K_XL` | 0.62 | 0.62 | +0.4% |
| `UD-Q5_K_XL` | 0.58 | 0.58 | +0.1% |
| `UD-Q6_K_XL` | 0.53 | 0.53 | -0.3% |

## Run order (mixing)

![Every run in execution order](benchmarks/macpro51/qwen38-run-order.svg)

The sweep is resumable: a `(round, pack)` pair already in `results-qwen38.jsonl` is skipped, so an interrupted or rebooted machine costs one run rather than the series.

## Per-run log

| # | Round | Pack | Prefill s | Decode s | Decode tok/s | Note |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 1 | `UD-Q6_K_XL` | 22.17 | 60.03 | 0.533 | first round (cold) |
| 2 | 1 | `UD-Q4_K_XL` | 16.90 | 51.90 | 0.617 | first round (cold) |
| 3 | 1 | `UD-Q2_0` | 9.98 | 39.07 | 0.819 | answer diverges |
| 4 | 1 | `Q2_0` | 9.47 | 30.21 | 1.06 | first round (cold) |
| 5 | 1 | `IQ3_S` | 13.25 | 38.94 | 0.822 | first round (cold) |
| 6 | 1 | `IQ3_XXS` | 11.91 | 35.26 | 0.908 | first round (cold) |
| 7 | 1 | `UD-Q5_K_XL` | 17.19 | 55.51 | 0.576 | first round (cold) |
| 8 | 2 | `Q2_0` | 9.41 | 30.19 | 1.06 | later round |
| 9 | 2 | `UD-Q6_K_XL` | 21.74 | 60.06 | 0.533 | later round |
| 10 | 2 | `UD-Q4_K_XL` | 16.20 | 52.22 | 0.613 | later round |
| 11 | 2 | `UD-Q5_K_XL` | 17.09 | 55.45 | 0.577 | later round |
| 12 | 2 | `IQ3_S` | 12.97 | 39.02 | 0.820 | later round |
| 13 | 2 | `UD-Q2_0` | 9.72 | 39.05 | 0.819 | answer diverges |
| 14 | 2 | `IQ3_XXS` | 11.70 | 35.23 | 0.908 | later round |
| 15 | 3 | `UD-Q4_K_XL` | 16.26 | 51.55 | 0.621 | later round |
| 16 | 3 | `IQ3_S` | 12.87 | 39.01 | 0.820 | later round |
| 17 | 3 | `UD-Q2_0` | 9.80 | 39.03 | 0.820 | answer diverges |
| 18 | 3 | `Q2_0` | 9.30 | 30.30 | 1.06 | later round |
| 19 | 3 | `UD-Q6_K_XL` | 21.92 | 60.22 | 0.531 | later round |
| 20 | 3 | `IQ3_XXS` | 11.72 | 35.28 | 0.907 | later round |
| 21 | 3 | `UD-Q5_K_XL` | 17.13 | 55.44 | 0.577 | later round |
| 22 | 4 | `UD-Q2_0` | 9.76 | 39.01 | 0.820 | answer diverges |
| 23 | 4 | `UD-Q5_K_XL` | 17.32 | 55.47 | 0.577 | later round |
| 24 | 4 | `UD-Q4_K_XL` | 16.03 | 51.47 | 0.622 | later round |
| 25 | 4 | `UD-Q6_K_XL` | 21.71 | 60.43 | 0.529 | later round |
| 26 | 4 | `IQ3_XXS` | 11.59 | 35.20 | 0.909 | later round |
| 27 | 4 | `Q2_0` | 9.31 | 30.16 | 1.06 | later round |
| 28 | 4 | `IQ3_S` | 12.97 | 38.99 | 0.821 | later round |
| 29 | 5 | `UD-Q5_K_XL` | 17.11 | 55.48 | 0.577 | later round |
| 30 | 5 | `IQ3_S` | 12.97 | 38.92 | 0.822 | later round |
| 31 | 5 | `Q2_0` | 9.44 | 30.18 | 1.06 | later round |
| 32 | 5 | `UD-Q2_0` | 9.83 | 39.00 | 0.820 | answer diverges |
| 33 | 5 | `IQ3_XXS` | 11.73 | 35.41 | 0.904 | later round |
| 34 | 5 | `UD-Q6_K_XL` | 21.87 | 60.07 | 0.533 | later round |
| 35 | 5 | `UD-Q4_K_XL` | 15.96 | 51.48 | 0.622 | later round |

## Limits recorded here

- The pack's second shard is the 28.8 GB PLE n-gram table. It is read from the SSD a few rows at a time and is never loaded whole; its size counts in the download, not in VRAM.
- Metal 2 on this card has no device addresses; kernels take buffer + offset, and the engine hands every weight that way. No Metal 3 feature is used.
- The output head is 512 wide in the draft path and 248 320 wide in the sample; greedy decoding is used throughout so the comparison is not a sampler's.

