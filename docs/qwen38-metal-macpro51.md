# Qwen3.8-Flash-Next on macpro51: the Metal bring-up

_Measured 2026-11-18 on the machine below. This page records what runs, what does not, and the evidence for each. It is
a bring-up record, not a benchmark: no throughput number here is a usable quantisation comparison yet._

> The pack matrix this page feeds is [macpro51-qwen38-benchmarks.md](macpro51-qwen38-benchmarks.md), which the sweep
> rewrites after every completed run. The page below says why that sweep has not started.

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata` 0.1.59-metal, built from the `macpro51` branch with Ninja, Release |
| Model | `~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{00001,00002}-of-00002.gguf` (37.6 GB + the 28.8 GB PLE table) |

The pack is the ISTA-DASLab GSQ-RCO Q2_0 set: `general.architecture = qwen4exp`, 48 trunk layers (36 GDN, 12 QSA at
3, 7, … 47), 512 routed experts of width 640 with 10 used, 4 hyper-connection streams of rank 320, and the PLE n-gram
table in the second shard. Its tensor table matches the shape the Metal engine compiles for (`kCompiledGeometry`), and
`tools/iq_pack.py` builds the metadata pack beside it (`native_experts.txt`, `index.txt`, `dense.bin`, `tokenizer/`).

## What runs

The engine opens the pack, builds the pack metadata, reads the dense side and the expert arena, compiles its kernels,
reads the prompt, and reaches the decode window. Measured on the card:

| step | measurement |
| --- | --- |
| dense side read into host memory | 789 tensors, 2.36 GB, 1.8 s |
| expert arena | 2.20 GB, 1578 slots of the profile's top experts, filled in 1.0 s (2.25 GB/s) |
| decode window ready (kernels compiled) | 41.3 s the first time, 3.4 s once the pipeline cache is warm |
| prompt of 31 tokens | 6.22 s → 5.0 tok/s, read as one chunk through the engine's prompt path |
| one decode window (1 token) | 1.09 – 2.81 s, of which the GPU's own work is 1.04 – 1.12 s |
| expert reads per decode window | 731.6 MB in 2,049 reads (Q2_0), hit rate 0.75 |

So the Metal 2 path on this card is alive end to end: it streams a 37.6 GB model through an 8 GB card, it reads the
prompt at 5 tok/s, and it returns a decode window in about a second. The **answers are not usable**: every head logit
is non-finite, so the sampler returns `2147483647` (its "no finite maximum" value).

## Where the forward first goes non-finite

The window's residual stream is checked after every layer (`STRATA_WIN_LAYERCHECK`, a host read of `R` at each
streaming segment boundary). On Q2_0:

| layer | kind | residual |
| ---: | --- | --- |
| 0 | GDN | finite, max 0.19 – 0.51 |
| 1 | GDN + PLE | finite, max 0.67 – 0.69 |
| 2 | GDN | finite |
| 3 | **QSA (first full-attention layer)** | **non-finite** |
| 4 … 47 | | non-finite from here on |

At layer 3 the query buffer `qcur` (24 heads × 256) and the indexer query `qidx` (4 × 128) are already non-finite
before the attention is computed; `part` (the attention partials) inherits them. That places the first failure in the
full-attention layer's preparation (`qsa_prep`: the `attn_q` / `indexer.q_proj` GEMVs, the q/k RMS norms and RoPE), not
in the GDN layers, the hyper-connections, the MoE, the head or the sampler.

## What was ruled out

Every check below was run on this card and passed, so none of them explains the non-finite query:

- **The hyper-connection kernels** (`gr_norm`, `gr_down`, `gr_up`): `build-bench/gr_parity` and `gr2_parity` **ALL PASS**
  for 1, 3 and 8 tokens, apply 0 and 1, Q8_0 and BF16 (relative L2 1e-07).
- **The GDN layer**: `gdn_parity` **ALL PASS** (relative L2 1.6e-07), including the state/history contract.
- **The prompt path and the fused hyper-connection read**: `prompt_parity` **ALL PASS**.
- **Q2_0 expert and GEMV kernels**: `pair_parity` **ALL PASS**, including the "Q2_0 raw-x GEMV 640 → 2560" case.
- **The RoPE table**: built on the host as `cos/sin[p][32]` with `base = qwen4exp.rope.freq_base` (1e7), all finite.
- **The F32 weights** (all RMS/norm tensors, including `attn_q_norm`, `attn_k_norm`, `indexer.q_norm`): a scan of every
  F32 tensor in the file after the dense load found **0 non-finite values**.
- **The Q2_0 row layout**: `blk.3.attn_q.weight` occupies exactly 12,288 × 720 bytes in the file, the row size
  `ref::row_bytes(Q2_0, 2560)` the GEMV uses, so the rows are not misaligned.
- **The 64-lane hazards** the other kernels needed: every kernel the window dispatches derives its group and lane
  from `sg32`/`lane32`; `qsa_prep` uses `sum32` for its reductions.

The narrowest open question is therefore inside the QSA preparation at layer 3 on this card: the query projection is
non-finite while its inputs (the residual, the attention norm weights, the RoPE table) are finite. The next probe
saved for that is `w/window-instrumented.cpp` (a copy of `window.cpp` with per-layer buffer dumps), and the first thing
to read with it is `qfull` (the `attn_q` GEMV output) *before* `qsa_prep`, which the shipped window overwrites in place.

## Engine changes made for this card

These are on the branch, all behind `STRATA_GPU_METAL`, so the CUDA/HIP paths are unchanged:

1. **A native pack needs `--native` only.** The shared host code required `--spec T (T >= 2)` and `--prefill CHUNK`
   for a native `--pack` before it could start. The Metal decode window reads the model's GGUF itself, one token
   window at a time, and has neither a device residency table for `--spec` nor a batched prompt path of its own, so a
   native pack is now accepted with `--native SHARD1` alone there.
2. **A spec window does not need the device residency table on Metal.** The window streams its own experts and keeps
   its own state, so the `thits.d_res == nullptr` refusal is a CUDA/HIP check; it is compiled out on Metal.
3. **A token id outside the vocabulary is refused, not dereferenced.** A head whose logits are all non-finite returns
   `0x7fffffff` from its argmax; the window then read that row from the embedding tensor and crashed (SIGSEGV). It now
   stops with `the token id … is outside the vocabulary` instead.

The pack tools also needed two Python 3.8 fixes (`Path.write_text(..., newline=...)` is 3.10's), which is the Python
macOS 11 ships: `tools/strata_tokenizer.py` and `tools/iq_pack.py` now open the file themselves.

## The packs on disk

Downloaded for this work into `~/models/qwen38/` (one folder per pack, the 28.8 GB PLE shard hard-linked between them
because it is the same bytes):

| pack | size | state |
| --- | ---: | --- |
| `Q2_0` | 66.4 GB | verified, runs to the QSA layer (above) |
| `IQ2_XS` | 68.0 GB | verified |
| `IQ3_XXS` | 75.8 GB | verified |
| `IQ3_S` | 83.6 GB | downloading |

`bench/macpro51/run_bench_qwen38.py` is the sweep for them: one process per run, 32 prompt ids + 32 greedy tokens, a
fresh shuffled order each round, mean ± 1 sample standard deviation over the steady rounds, an SVG chart and a
commit+push after every run. It is ready and waiting on the fix above.

## Reproduce

```sh
cd ~/strata-bench
python3 tools/iq_pack.py --gguf ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --out ~/models/qwen38/Q2_0/pack
STRATA_METAL_STREAM=1 STRATA_WIN_LAYERCHECK=1 ./build-bench/strata \
    --pack ~/models/qwen38/Q2_0/pack \
    --native ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --tokens 3,1,42498,456,11,303,1330,2716,22157,11,3069,279,12515,5686,6105,2261,279,1834,321,2438,506,41564,13,4844,1910,3069,279,16820,5686,18016,3043,279 \
    --max-new 1 --max-context 256 --spec 1 --suffix-draft 0 --lookup-max-t 0 --prefill 32
```

`STRATA_WIN_LAYERCHECK=1` is the per-layer residual check; it is a bring-up diagnostic and not meant for a benchmark
run (each cut waits for the GPU and reads 10,240 floats).
