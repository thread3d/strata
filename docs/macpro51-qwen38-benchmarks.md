# Qwen3.8-Flash-Next prompt-throughput benchmarks on macpro51 (RX 580, Metal 2)

_Generated 2026-11-18 18:48 UTC by `bench/macpro51/run_bench_qwen38.py` after every completed run. Every number below was produced on the machine named here._

> **The engine does not answer yet.** Its forward pass goes non-finite at the first full-attention layer, so
> there is no decode rate to report and none is claimed here. What this page measures is the prompt path: the
> whole model is opened, the dense side and the expert arena are built, and the 32-token prompt is read through
> the same streaming kernels a decode window uses. Those bytes and that work are real. The blocker and the
> evidence are in [qwen38-metal-macpro51.md](qwen38-metal-macpro51.md); this page becomes the decode matrix
> the day the query projection is fixed.

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata` 0.1.59-metal, branch `macpro51` |
| Branch / commit at the last run | `macpro51` / `ff7ed691aa` |

## What ran

| Pack | Quantisation | Source | On disk | Runs | Prompt, steady (mean ± sd) | All runs (mean) | Steady range |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Q2_0` | Q2_0 (GSQ-RCO) | ISTA-DASLab | 61.9 GiB | 3 | 4.72 ± 0.034 | 4.74 | 4.69 – 4.74 |
| `IQ2_XS` | IQ2_XS (GSQ-RCO) | ISTA-DASLab | 63.4 GiB | — | — | — | — |
| `IQ3_XXS`* | IQ3_XXS (GSQ-RCO) | ISTA-DASLab | 70.6 GiB | 2 | 2.86 ± 1.13 | 2.86 | 2.06 – 3.66 |
| `IQ3_S`* | IQ3_S (GSQ-RCO) | ISTA-DASLab | 77.9 GiB | 2 | 3.15 ± 0.272 | 3.15 | 2.96 – 3.35 |
| `UD-IQ1_M` | UD-IQ1_M (Unsloth v3.1) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-Q2_0` | UD-Q2_0 (Unsloth v3.1) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-IQ2_XS` | UD-IQ2_XS (Unsloth v3.1) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-Q2_K_XL` | UD-Q2_K_XL (Unsloth v3.1) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-IQ4_XS` | UD-IQ4_XS (Unsloth) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-Q4_K_XL` | UD-Q4_K_XL (Unsloth) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-Q5_K_XL` | UD-Q5_K_XL (Unsloth) | unsloth | 0.0 GiB | — | — | — | — |
| `UD-Q6_K_XL` | UD-Q6_K_XL (Unsloth) | unsloth | 0.0 GiB | — | — | — | — |
| `Q8_0` | Q8_0 (Unsloth) | unsloth | 0.0 GiB | — | — | — | — |

\* fewer than two steady-state runs (rounds 2+) so far, so this row uses every completed run.

## Method

- **One process per run.** The engine reads the dense side (2.36 GB for Q2_0), fills the expert arena (2.2 GB),
  compiles its window kernels, then reads the 32-token prompt as one chunk through the prompt path.
- **The same 32 token ids** for every pack (the Qwen 248 320-entry vocabulary, a varied English sentence).
- **Mixed order.** Each round contains every downloaded pack once, in a fresh shuffled order (seed in
  `bench/macpro51/meta-qwen38.json`), so a drift in the machine is spread rather than assigned to one pack.
- **Error bars.** Mean ± 1 sample standard deviation over the steady-state runs (rounds 2+). Round 1 is each
  pack's first pass on a cold page cache and is kept separately. The dots on the chart are the individual runs.
- **Streaming.** `STRATA_METAL_STREAM=1`: the pack cannot be resident in 8 GB of VRAM, and every pack is measured
  on the same streamed path. The prompt path and the decode window read the same experts from the same arena.

## Prompt throughput

![Prompt throughput by quantisation](benchmarks/macpro51/qwen38-prefill.svg)

## Size against prompt speed

![Prompt throughput against pack size](benchmarks/macpro51/qwen38-size-vs-prompt.svg)

A prompt token reads the same 10 routed experts per layer a decode token does, so the pack's encoding and the
bytes a token actually touches decide this number more than the file size does.

## What the numbers say

- **Fastest prompt read:** `Q2_0` at 4.72 tok/s; **slowest:** `IQ3_XXS` at 2.86 tok/s (1.65× between them).
- **Run-to-run spread:** coefficient of variation from 0.7% (`Q2_0`) to 39.6% (`IQ3_XXS`).

The first round is the first pass over each pack (cold page cache, cold clocks); the later rounds are the
steady state. The round-1 and steady numbers are kept apart in the table and the per-run log.

## Run order (mixing)

![Every run in execution order](benchmarks/macpro51/qwen38-run-order.svg)

The sweep is resumable: a `(round, pack)` pair already in `results-qwen38.jsonl` is skipped, so an interrupted
or rebooted machine costs one run rather than the series.

## Per-run log

| # | Round | Pack | Prompt tokens | Prompt s | Prompt tok/s | Note |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 1 | `Q2_0` | 31 | 6.48 | 4.79 | first round (cold) |
| 1 | 1 | `IQ3_XXS` | 31 | 15.07 | 2.06 | first round (cold) |
| 2 | 1 | `IQ3_S` | 31 | 10.47 | 2.96 | first round (cold) |
| 3 | 1 | `IQ2_XS` | — | — | — | failed: mtl: shared-event waits 2320 times, 33.3 ms (0.014 ms a wait) |
| 4 | 2 | `IQ2_XS` | — | — | — | failed: mtl: shared-event waits 2320 times, 32.7 ms (0.014 ms a wait) |
| 5 | 2 | `IQ3_XXS` | 31 | 8.47 | 3.66 | later round |
| 6 | 2 | `IQ3_S` | 31 | 9.26 | 3.35 | later round |
| 7 | 2 | `Q2_0` | 31 | 6.54 | 4.74 | later round |
| 8 | 3 | `Q2_0` | 31 | 6.61 | 4.69 | later round |

