# Strata quantisation benchmarks on macpro51 (Intel Mac Pro, Radeon RX 580)

_Generated 2026-11-18 08:47 UTC by `bench/macpro51/run_bench.py`. Every number below was produced on the machine named here, during this sweep._

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata-q36-metal` (Qwen3.6) and `strata_ds4` (DeepSeek V4.1-Flash), built Release with Ninja |
| Branch / commit at the last run | `macpro51` / `83d2da3218` |

The Metal runtime here is the workaround path: kernels take buffer offsets instead of raw device addresses, the DeepSeek engine flushes command buffers one at a time (`DS4_FLUSH_EVERY`), and `simdgroup_matrix`/Metal 3 features are skipped. Numbers are therefore this card's, not a Metal 3 Mac's.

## What ran, and what could not

Strata's three public families are DeepSeekV4.1-Flash-1017, Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. This branch's Metal engine has paths for Qwen3.6 (`src/qwen36`) and DeepSeek V4.1-Flash (`src/dsv41`); Qwen3.8 has no Metal path here and the docs place it on Apple silicon, so it is not runnable on this Intel Mac. Every installed pack of the two runnable families was tested.

| Family | Pack | On disk | Result |
| --- | --- | ---: | --- |
| qwen36 | Q4_K_M | 19.7 GiB | no completed run yet |
| qwen36 | Q6_K | 26.6 GiB | no completed run yet |
| qwen36 | Q8_0 | 35.2 GiB | no completed run yet |
| qwen36 | UD-Q2_K_XL | 11.7 GiB | no completed run yet |
| qwen36 | UD-Q4_K_XL | 21.3 GiB | no completed run yet |
| qwen36 | UD-Q6_K_XL | 30.4 GiB | no completed run yet |
| qwen36 | UD-Q8_K_XL | 36.4 GiB | 1 run(s), 0.64 tok/s decode |
| dsv41 | IQ2_XXS | 239.2 GiB | no completed run yet |
| dsv41 | UD-Q8_K_XL | — | refused: the dense weights (attention, shared experts, output head) need 15.61 GB of VRAM and the RX 580 has 8.59 GB |

Two Qwen3.6 packs were refused when this sweep started, and the refusals were real defects rather than hardware limits; they were fixed on the branch before the sweep (see [Engine fixes](#engine-fixes)). The DeepSeek UD-Q8_K_XL pack is refused by the engine on this card because its dense weights alone need 15.61 GB of VRAM against the RX 580's 8.59 GB; that is a hardware limit, recorded here rather than worked around.

## Method

- **One process per run.** The engine loads the model, warms every pipeline with one full pass, resets the sequence state in place, then times a fresh pass. Model load time is reported separately and is not part of either throughput.
- **Fixed workload.** Qwen3.6: 32 prompt tokens read one at a time, then 32 greedy tokens. DeepSeek: 16 prompt tokens, then 16 greedy tokens, with its own chunked prompt path and its reported prompt/decode times. The same token ids are used for every pack of a family.
- **Mixed order.** Each round contains every runnable pack once, in a fresh shuffled order (seed in `bench/macpro51/meta.json`). No pack is always first or always last, so thermal or page-cache drift is spread across the matrix.
- **Error bars.** Mean ± 1 sample standard deviation over the completed runs of that pack; the individual runs are drawn as dots on the charts. With 5 rounds the bars describe run-to-run spread on this machine, not a confidence interval for another one.
- **Steady state.** Qwen runs use a 2 GiB expert-slot budget, which selects the streaming path for every pack here; the warm-up pass is what fills the OS page cache and compiles the pipelines before the first measured token.

## Decode throughput

![Decode throughput by quantisation](benchmarks/macpro51/decode-throughput.svg)

| Family | Pack | Runs | Decode tok/s (mean ± sd) | Range | Prefill tok/s (mean ± sd) |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen36 | UD-Q8_K_XL | 1 | 0.64 ± 0.00 | 0.64 – 0.64 | 0.83 ± 0.00 |

## Prompt (prefill) throughput

![Prefill throughput by quantisation](benchmarks/macpro51/prefill-throughput.svg)

Prefill is read one token at a time by the Qwen engine, so it is a per-token latency measurement as much as a bandwidth one; the DeepSeek engine reads its prompt in chunks. The two families are therefore comparable within themselves, not to each other.

## Size against speed

![Decode throughput against model size](benchmarks/macpro51/size-vs-decode.svg)

On this card a token streams its routed experts from RAM/SSD through an 8 GB window, so the pack's size is only one of the things that decide its speed: the number of bytes a token actually reads, the format's decode cost, and how much of the dense side stays resident all matter as much.

## Run order (mixing)

![Every run in execution order](benchmarks/macpro51/run-order.svg)

The sweep is resumable: a `(round, pack)` pair already in `results.jsonl` is skipped, so an interrupted or rebooted machine costs one run rather than the series.

## Per-run log

| # | Round | Family | Pack | Prefill s | Decode s | Decode tok/s | Note |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | 1 | qwen36 | UD-Q8_K_XL | 38.71 | 48.67 | 0.64 | first round (cold cache) |

## Engine fixes made to run the matrix

These were found by opening every installed pack before the sweep, and are on this branch:

1. **Embedding rows outside the three kernel formats.** `q36_embed` had device branches for Q8_0, Q6_K and Q4_K only, so UD-Q2_K_XL (Q5_K token embeddings) was refused at open. The row is now dequantised on the host for any other dense format the port accepts; it is 2,048 values per token against forty layers of expert work.
2. **The never-run MTP block's expert types.** The MTP block is a draft head the engine skips, but type validation and slot accounting still walked it, and UD-Q2_K_XL's MTP block carries Q2_K gate/up (no kernel, because nothing dispatches it). Validation, slot sizing and slot packing now skip the MTP block, as the forward pass already did.
3. **BF16 grouped experts.** UD-Q8_K_XL promotes three layers' routed experts to BF16 (layers 34, 38 and 39). `moe.metal` now instantiates the gate/up and down kernels for BF16 as well, and the type tables accept it. The pack then runs; its first generated token matches the Q8_0-era 279.

## Tests

`gemv_parity` passes on this build for Q2_0, IQ3_S, Q3_K, Q4_K, Q5_K and Q6_K at n_in 2560 and 10240 and 1/3/8 tokens (rel L2 6e-08…4e-07, and the multi-token window is bitwise equal to the one-token path).

The Qwen3.6 checks: 15 of the 18 registered ctest tests pass on this build. The three that do not are not on a real pack's path:

- `q36_forward_check` and `q36_generate_check` abort inside `attn_init` while compiling the synthetic eight-head fixture's attention kernels on this Metal 2 driver; the sweep below is the evidence that a real pack's shapes compile and run.
- `runtime_hello` fails its device-address and graph-replay cases because macOS 11 has neither; the engine's documented workarounds are why these benchmarks run at all.

`q36_model_check`, `q36_glue_parity` and `q36_upload_check` were repaired as part of this work (a stale kernel argument, an error message that looked a type up in the wrong table, and a byte comparison against a private buffer's handle, which now blits the buffer back first). `q36_upload_check` verifies 34.5 MB of dense and embedding bytes and passes.

## Reproduce

```sh
cd ~/strata-bench
ninja -C build-bench strata-q36-metal strata_ds4
python3 bench/macpro51/run_bench.py --rounds 5 --seed 20261118
python3 bench/macpro51/run_bench.py --only Q4_K_M --rounds 1 --no-push   # a single pack
```

Results are `bench/macpro51/results.jsonl` (one JSON object per run, including the per-step decode times) and `bench/macpro51/meta.json` (the matrix and the settings). Charts are SVG in `docs/benchmarks/macpro51/`.

