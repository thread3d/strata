# Benchmarks on macpro51 (2026-11-22)

The Intel Mac in `docs/MAC.md` ("An Intel Mac"), branch `macpro51`, built from the branch tip:
`strata 0.1.59-metal`. Results now include a **llama.cpp reference on the same machine**, which is
what turns "it runs" into a statement about correctness.

| Family | Path | Result |
|---|---|---|
| Qwen3.6-35B-A3B | `build-metal/strata-q36-metal` | **runs and is correct**: 5 of 7 packs, 0.86 to 1.11 s per step, all 5 now match llama.cpp |
| Qwen3.8-Flash-Next | `build-metal/strata` | **stops**: needs a 4.07 to 5.13 GiB Metal buffer; this card's limit is 3.5 GiB |
| DeepSeek V4.1-Flash | `build-metal/strata_ds4` | **refuses by design**: no Metal 3 device addresses on macOS 11 |

## Qwen3.6-35B-A3B, checked against llama.cpp

23-token context, 24 greedy tokens, identical token ids fed to both engines so they share one
tokenisation. llama.cpp returns the same 24 ids on **every** pack (bar one near-tie at the last
token), which is the behaviour of a faithful engine.

| Pack | gate/up, down | strata vs llama.cpp | s/step (mean of 3) |
|---|---|---|---|
| `Q8_0` | Q8_0, Q8_0 | **matches** | 1.253 |
| `UD-Q6_K_XL` | Q6_K, Q8_0 | **matches** | 1.172 |
| `Q6_K` | Q6_K, Q6_K | **matches** | 1.051 |
| `Q4_K_M` | Q4_K, Q6_K | **matches** | 1.008 |
| `UD-Q4_K_XL` | Q4_K, Q5_K | **matches** | 0.936 |
| `UD-Q2_K_XL` | — | refused: the embedding is Q5_K | — |
| `UD-Q8_K_XL` | — | refused: the expert down tensor is BF16 | — |

All five now agree with llama.cpp, 23 of 24 tokens identical; the 24th is `25562` on the two
Q6_K-down packs, the same near-tie llama.cpp's own `Q4_K_M` shows. That was not always true: until
2026-11-22 the three K-quant-down packs diverged at token 4, on a fault first recorded on 2026-11-16
that `be4cbc78f` did not fix. It is fixed now — `moe_down_k` took its simdgroup index and lane from
the hardware attributes instead of mapping them through `sg32`/`lane32` like the other MoE kernels
do, so on this card's 64-lane simdgroups the row formula overlapped and never wrote rows 24-31 of
every threadgroup: a quarter of the output, and a relative L2 of exactly 0.5.

**Speed.** On the identical 47-step workload, wall clock including load:

| Pack | strata (Metal 2) | llama.cpp (CPU) |
|---|---|---|
| `UD-Q4_K_XL` | 48.6 s | 15.1 s |
| `Q4_K_M` | 50.3 s | 17.5 s |
| `Q6_K` | 53.7 s | 16.3 s |
| `UD-Q6_K_XL` | 57.7 s | 14.3 s |
| `Q8_0` | 61.4 s | 15.5 s |

A CPU build of llama.cpp beats the Metal path by 2.9 to 4.0 times. That is the readback fault
showing up as cost, and it means these Metal timings say nothing about what the card could do.

These figures are the ones taken after the down fix, on a freshly rebooted machine. They are about
10% higher than the pre-fix run — but `Q8_0`, whose down kernel the fix does not touch, moved by
the same amount, so that shift is machine state and not the fix.

![Same workload, wall clock](docs/benchmarks/macpro51/same-workload-wall.svg)
![Seconds per step](docs/benchmarks/macpro51/decode-steps.svg)

## What this is not

One prompt, one length, greedy only, no perplexity or KL measurement — a sharp but narrow test. No
Qwen3.8 or DeepSeek output exists for this machine. The llama.cpp build here has no Metal backend,
so its speed is a CPU reference. These are porting-validity figures on a 2010 Mac Pro with a Radeon
RX 580 (8 GB, Metal 2), not product performance and not comparable to the Apple Silicon or CUDA
numbers in `docs/MAC.md`.

Every number, refusal message and command is in
[`docs/macpro51-2026-11-22.md`](docs/macpro51-2026-11-22.md). The figures are produced, not
transcribed: `bench/macpro51/run_matrix.py` and `run_reference.py` write the two JSON files,
`charts.py` and `charts_reference.py` draw the charts from them, and the reference side is
`bench/macpro51/llama_ref.cpp`, built against llama.cpp (the build line is in that page).

The earlier macpro51 pages (`docs/macpro51-test-results.md` and the three
`docs/qwen36-metal-*-benchmarks*.md` files) predate `be4cbc78f`, disagree with each other, and carry
a superseded banner; none of their token tables can be reproduced by the engine now.
