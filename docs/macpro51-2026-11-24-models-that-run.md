# The downloaded models that would not run — macpro51, 2026-11-24

> **Paths moved on 2026-11-25:** `metal/` is `engine/metal/`, `src/` is `engine/src/`,
> `bench/macpro51/` is `terminal/bench/macpro51/`. See
> [macpro51-2026-11-25-merge.md](macpro51-2026-11-25-merge.md) for the mapping and the test record.

At `f2349f144` (the "every model on the box exercised" pass) four Qwen3.6/Qwen3.8 files were refused by the
engine and six Qwen3.8 directories were recorded as "not convertible here". This page is the follow-up: what
each refusal actually was, the six commits that fixed them, and the matrix that results.

**Branch `macpro51`, `c69388179` → `99e2d6c7b`.** Engine `strata 0.1.59-metal`, clean rebuild, `ctest` 17
passed / 1 skipped (`runtime_hello`) / 0 failed at each commit. Everything below was run on the machine; the
records are `bench/macpro51/results-all-models.json`.

## Result in one line

**All 7 Qwen3.6 packs and all 17 Qwen3.8 directories now run.** Before: 5 of 7 and 11 of 17. The two things
still not run are not models — `mmproj-F16.gguf` is a CLIP vision projector (`architecture is 'clip'`) and
`strata_ds4` needs `STRATA_DS4_ALLOW_UNSTABLE_METAL=1`, which resets the card (see the workspace AGENTS.md).

## The six fixes

| commit | what | evidenced by |
|---|---|---|
| `528573a98` | **Qwen3.6 UD-Q8_K_XL: BF16 experts.** `blk.34`/`blk.38` down and all of `blk.39`'s experts are BF16; neither `GU_SET` nor `DN_SET` instantiated it. `Dec<T_BF16>`/`fmt<T_BF16>` already existed, so it was two instantiation sets plus the host gate. | `expert_parity` BF16/BF16 h 1.40e-07 y 1.07e-07, Q8_0/BF16 1.51e-07/1.08e-07, BF16/Q8_0 1.40e-07/1.13e-07; the pack returns llama.cpp's 24 tokens id for id, 1.289 s/step |
| `1d447cf51` | **Qwen3.6 UD-Q2_K_XL part 1: `q36_embed` Q5_K**, plus a **Q6_K decode bug** the new parity case found (group 1 took the high nibble of `ql[l+32]`; ggml takes the low one — `ggml-quants.c:1954`). | every format at rel L2 0.000e+00 against `ref::dequant` on tokens 0 and 63 of a 64-row table; Q6_K pack A/B identical 24 tokens |
| `1debdf282` | **Qwen3.6 UD-Q2_K_XL part 2: Q2_K gate/up** (`blk.40`). Q2_K existed nowhere in the tree — added to `ref_quant`, `quant.metal`/`units.metal`, `GU_SET` and the host gate. | `dsu_ref_vs_ggml` Q2_K 0 of 131072 values differ from ggml; `expert_parity --gu Q2_K --dn Q3_K --ff 512 --embd 2048` h 1.54e-07 y 2.71e-07; the pack returns llama.cpp's 24 tokens, 0.874 s/step |
| `c313adcce` | **`metal_repack.py` could not repack these files at all**: its `TYPE_SIZE` table had no IQ1_M (type 29) or trellis (144/145/146) entries, so it died with a bare `KeyError: 29`. Sizes from `ggml-common.h`'s static_assert and `tools/gguf_reader.py`'s tested table. | repack of IQ2_XS: 1223 tensors, 36.54 GiB, "SHA-256 identical to the source shards" |
| `144eb9ddd` | **The six "unconverted" Qwen3.8 packs did not need converting.** Each carries every non-PLE tensor in one shard, and that shard is what `native_experts.txt` names as the file its absolute offsets are in. The harness demanded a `-metal.gguf` and skipped them; it now takes `--native` from that header line. | five of the six run through the engine directly, each returning `303 7472` on the fixed 5-token prompt |
| `99e2d6c7b` | **Qwen3.8 IQ2_XS: IQ1_M gate/up** (type 29, `blk.8/13/37`). A value is `dl * (grid[j] + delta)`; with the format's `UNIT` of 16 a unit is exactly two 8-value groups, so `X<T_IQ1_M>` supplies the two per-8 sums (`unit_combine` subtracts `sc.z*xsum[0] + sc.w*xsum[1]`) and `Dec<T_IQ1_M>` the two deltas — the existing contract, no new kernel. | `expert_parity --gu IQ1_M --dn Q2_0 --embd 2560 --ff 640` h 1.45e-07 y 1.17e-07; the pack returns `303 7472` |

## The Qwen3.8 six, measured

The engine command `bench/macpro51/run_all_models.py` builds (now with the fixed native-file rule), run by hand
for each pack: 5-token prompt `3 1 42498 456 11`, `--max-new 2`, DFlash2 attached. Decode is quoted where it
was recorded; the sweep's own per-pack numbers are in `bench/macpro51/results-all-models.json`:

| pack | native GGUF | output | decode |
|---|---|---|---|
| IQ2_XS | shard 1 (39.2 GB) | `303 7472` | 0.67 tok/s |
| IQ3_XXS | shard 1 (47.0 GB) | `303 7472` | 0.56 tok/s |
| IQ3_S | shard 1 (54.8 GB) | `303 7472` | — |
| Q2_0 | shard 1 (37.6 GB) | `303 7472` | — |
| Gyro-M | the single TQ2_T file (91.9 GB) | `303 7472` | 0.45 tok/s |
| Gyro-S | the single TQK6/TQK7 file (58.5 GB) | `303 7472` | 0.49 tok/s |

`303 7472` is what Q8_0 and every ≥3-bit Qwen3.8 pack returns on this prompt; the low-precision Unsloth packs
(`UD-IQ1_M`, `UD-IQ2_XS`, `UD-Q2_0`, `UD-Q2_K_XL`) are the ones that give the other continuation, `1544 76`.
So the six land with the high-precision group, including the 2 bpw `Q2_0` and the trellis `Gyro-M`/`Gyro-S`.

## Why "not convertible here" was wrong

`docs/macpro51-2026-11-24-all-models.md` recorded the six as holding "no `pack/` and no `*-metal.gguf`". Both
halves were misleading: every one of them has a complete `pack/` (index.txt, dense.bin, native_experts.txt,
tokenizer/, 1.5 GB of float tensors), and none needs a repack because shard 1 *is* a complete dense GGUF for
this engine. What was actually missing was (a) the harness's willingness to use that shard and (b) the IQ1_M
expert decode for the three layers of IQ2_XS. The repack tool's type table was a real defect too, but for
these files it was only ever on the "nice to have" path.

## What is still not covered

- **Draft acceptance is still 0 of 7** on every Qwen3.8 pack — unchanged by this work and the subject of
  `docs/macpro51-2026-11-23-dflash2.md`.
- **Two typed refusals remain in the Qwen3.6 engine, both now unused on this box**: `--spec 1` reports the
  native-pack rule, and `IQ1_S` as an *expert* type would say "no function `moe_gu_19_...`" (no file here uses
  it as an expert; it is listed in `expert_parity` so a synthetic blob can be built).
- **The fp16 expert path** (`STRATA_METAL_F16`) was not extended to BF16/Q2_K/IQ1_M; it is compile-checked only
  (`msl_dump --kq --f16`).
- **The Qwen3.6 K-quant embed width** assumption: `q36_embed` needs an embed width that is a whole number of
  256-value blocks for the K-quants; no pack here has one that is not.

## Reproduce

```sh
cd ~/projects/Strata/strata
PATH=/usr/local/bin:$PATH cmake --build build-metal -j 24
(cd build-metal && /usr/local/bin/ctest)                       # 17 passed, 1 skipped
./build-metal/dsu_ref_vs_ggml                                  # Q2_K and the 11 others vs ggml
./build-metal/expert_parity --gu Q2_K --dn Q3_K --ff 512 --embd 2048
./build-metal/expert_parity --gu IQ1_M --dn Q2_0 --embd 2560 --ff 640
python3 bench/macpro51/run_all_models.py                       # every model on the box
```

The llama.cpp oracle for the Qwen3.6 packs is `bench/macpro51/run_reference.py` (`~/llama_ref`), same ids
in, ids out.

## After the fixes: the full sweep and the llama.cpp regression

`python3 bench/macpro51/run_all_models.py`, one engine process per model, 26 records in
`bench/macpro51/results-all-models.json` (commit `93a93aab3`):

- **Qwen3.6: 7 of 7 packs rc 0**, and all seven return the *same* 8 tokens on the harness's text prompt
  (`271 248068 198 8160 579 264 7047 1817`). The harness now parses the Qwen3.6 engine's `tokens:` line as
  well as the Qwen3.8 one's `output:`, so those sequences are in the record instead of "0 tok".
- **Qwen3.8: 17 of 17 directories rc 0** - the eleven from before plus the six above, which return
  `303 7472` like `Q8_0`; the four low-bit Unsloth packs keep their own `1544 76`.
- `mmproj-F16.gguf` is still refused, correctly: `architecture is 'clip'`.
- `deepseek-v4.1-flash`: `--pack-info` needs `--pack` (a bare call prints usage and exits 2), so the record
  now carries the real report - 544 G parameters, 132.1 GB of experts at 1.94 bpw, dense 124.73 GB, two
  IQ4_NL engram tensors on the CPU lane, 6 files present. It is still not *run*: that needs
  `STRATA_DS4_ALLOW_UNSTABLE_METAL=1`, which resets this card.

**Regression against the oracle** (`tokens/dsu_ref_vs_ggml`-independent: the 23-token id contract of
`bench/macpro51/run_reference.py`, compared with `results-reference.json`): `Q4_K_M`, `UD-Q4_K_XL`,
`UD-Q6_K_XL` and `Q8_0` are **24 of 24 ids identical** to llama.cpp; `Q6_K` is 23 of 24, the difference
being the documented `25562`/`14162` near-tie at the last id. So the three engine fixes did not move any
previously-working pack.

## Prompt scaling on the two packs that could not load before

Prompt built by cycling the 23-token context (so the continuations echo it - these are *timing* points,
not a quality test), 8 new tokens, `--n 8`:

| pack | prompt tokens | wall | s/step | derived s per prompt token |
|---|---|---|---|---|
| UD-Q2_K_XL (11.7 GB) | 8 | 12.50 s | 0.781 | - |
| | 128 | 117.17 s | 0.862 | 0.87 |
| | 512 | 568.59 s | 1.093 | 1.18 |
| UD-Q8_K_XL (36.4 GB) | 8 | 47.22 s | 2.951 | - |
| | 128 | 211.08 s | 1.552 | 1.54 |

A 2048-token run on `UD-Q2_K_XL` was stopped after **27 minutes** without finishing, which is consistent
with the 512 point's rate and rising. Two things this says that the engine had not recorded: the Qwen3.6
prompt path costs **~0.9-1.2 s per prompt token**, i.e. 4-5x the Qwen3.8 engine's 243 ms/token
(`docs/macpro51-2026-11-24-suite.md`), and the cost per token *rises* with length rather than amortising.
`UD-Q8_K_XL` is ~2x `UD-Q2_K_XL` at the same length - it streams 33.8 GB of BF16/Q8_0 experts against
~7 GB. Both packs were previously unloadable, so neither had a long-prompt measurement before this pass.

The scaling sweep (`--only scale/threads` on the Qwen3.8 side) is unchanged by this work and still stands
in `docs/macpro51-2026-11-24-suite.md`.
