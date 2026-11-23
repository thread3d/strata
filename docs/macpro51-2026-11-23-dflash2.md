# macpro51: the DFlash2 drafter on a 64-lane SIMD GPU, 2026-11-23

Companion to `docs/macpro51-2026-11-22.md`, which is the record for the Qwen3.6 engine on this
machine. This page covers one thing: **DFlash2** (`--draft dflash2`, the block drafter for
Qwen3.8-Flash-Next), which the Metal port carried but had never run here.

The short version: **the drafter's kernels are ported to this card and verified against the CPU;
the engine loads a real drafter pack, binds it, and enters the speculative loop; it then stops one
step later, on a token the *verifier* never wrote — not on the drafter.** No end-to-end output and
no acceptance figure exist yet, and nothing here is a performance number.

## The machine and the build

| | |
|---|---|
| Host | MacPro5,1 ("macpro51"), macOS 11.7.11, 2 × Xeon X5675, ~96 GB DDR3 |
| GPU | AMD Radeon RX 580 8 GB, Metal 2, **64-lane simdgroups** |
| Branch / commit | `macpro51`, `e04d9f70b` (the commit this page describes) |
| Build | `PATH=/usr/local/bin:$PATH cmake --build build-metal -j 24` |

## Where the drafter was broken

`metal/src/engine/msl/dflash2.metal` was written for an Apple GPU, where a simdgroup is 32 lanes.
This card's is 64 (`threads_per_simdgroup` measured by the runtime, `mtl_rt.mm:75`), so
`[[simdgroup_index_in_threadgroup]]` runs 0..3 where the code wanted 0..7 and
`[[thread_index_in_simdgroup]]` runs 0..63 where the code wanted 0..31. The kernels therefore
produced half the row tiles they were gridded for and reduced across two logical groups at once.

`df2_bench` checks `df2_mm4` against a plain CPU dot product, and said so:

```
before   df2_mm4_4_8 check: max |gpu - cpu| 5.08 (max |value| 5.08)  FAIL
after    df2_mm4_4_8 check: max |gpu - cpu| 3.35e-07 (max |value| 5.08)  OK
```

The fix is the same pattern the K-quant MoE fix (`a2ad1da3e`) and `metal/kernels/mma.metal` already
use: take the group and lane from the runtime prologue — `sg32(tid, sg)`, `lane32(tid, lane)`,
`sum32(v)` — which fold back to the built-ins on a 32-lane GPU. Applied to every kernel that had
assumed 32 lanes: `df2_rms`, `df2_rms_h`, `df2_norm_rope`, `df2_topk_part`, `df2_topk_merge`,
`df2_select`, `df2_select_q4`, `df2_mm`, `df2_mm4`, `df2_mma4`, `df2_mma4s`, `df2_mma8`.

## The matrix units cannot run here at all

The drafter's Q4_0 GEMM and its draft head used `simdgroup_matrix` exclusively. On this
card/driver **a compute pipeline for any `simdgroup_matrix` kernel cannot be created**:

```
mtl: pipeline df2_mma4_2_4:
Linking two modules of different data layouts: '' is '' whereas '1' is
'e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-...'
SC compilation failure
```

This is not specific to DFlash2: `mma_parity` fails the same way on the engine's own
`gemv_mma_30_2` (`SC compilation failure`, `There is a call to an undefined label`). So it is a
property of Metal 2 on macOS 11.7 with this AMD driver, not of the drafter.

## What the port does instead

`df2_bind` asks `Context::simd_width()` (the device's `threads_per_simdgroup`) and, when it is not
32, takes a scalar path:

* the pack's Q4_0 matrices are stored as flat 16-byte code blocks (not fragment order), the draft
  head's Q8_0 rows as flat `int8[row][K]` with separate half scales;
* the block GEMMs and the draft head dispatch the scalar `df2_mm4` / `df2_mm` kernels instead of
  `df2_mma4s` / `df2_mma8`;
* it says so once, at bind.

```
strata metal: DFlash2: 64-lane simdgroups, so the matrix units are not used: the scalar Q4_0 / Q8_0 GEMMs and flat pack layouts
strata metal: DFlash2 drafter ready (1103 MiB: pack 725, 106299-token draft head as Q8_0), in 260 ms
```

An Apple GPU (32 lanes) keeps the fragment-order pack and the matrix-unit path; the scalar branch
is never taken there.

`df2_draft` also refuses an anchor token outside the vocabulary, instead of indexing the embedding
table with it (the principle of `9aac01969`).

## What is verified

* `df2_bench`'s Q4_0 check against the CPU: **3.35e-07** on values up to 5.08 in magnitude.
* The 8-row block runs: **102 dispatches, 93.3 ms GPU median** — one iteration on a just-woken
  machine. This is a bring-up figure, not a measurement, and it says nothing about a working
  drafter, which it is not yet.
* The drafter pack builds and loads. `tools/dflash2_pack.py` turns
  `incoai/Qwen3.8-Flash-Next-DFlash2` (1.2 GB bf16 safetensors; block 8, 5 layers, hidden 2560,
  conv group 16, selector rank 256 / top-16, mask token 248077, taps 5/15/25/35/45) into a
  760 MB DF2P pack of 81 tensors; the engine repacks 482 MiB of matrices and 243 MiB of the rest.
* The engine accepts it. With the Qwen3.8 `UD-IQ1_M` native pack:

  ```
  strata generate: DFlash2 drafter on (…/dflash2.pack, graphs captured in 0 ms), no MTP layer is loaded (--draft-fallback none)
  ```

  That line has never been printed on this machine before: it means `--draft dflash2
  --draft-fallback none` is no longer refused, the MTP layer is not loaded, and the speculative
  loop starts.

## What blocks it

The first draft is issued with **`anchor = 2147483647`** (`n_new = 1`, `pos0 = 4`): the verify
window's row-0 token is never written, so the drafter is handed an unset id. Before the guard that
commit added, the drafter indexed the embedding table with it and the process died with SIGSEGV at
a wild address; it now stops with

```
strata generate: dflash2: the anchor token 2147483647 is outside the vocabulary (248320 rows)
```

The fault is **upstream of the drafter, and it is now located precisely**: the first verify window
(`T = 1`, its one input token correct — `window: 11`) produces an all-NaN forward pass. Traced in
that order:

```
predraft: p=4 T=1 a=0 outv: 2147483647 | window: 11
predraft-logits: copy=1 nan=248320 min=1e+30 max=-1e+30     # every logit of the window is NaN
df2 taps: 12800 values, nan 12800                           # ... and so is the whole forward: the taps dflash2 reads
```

So the sampler is right to leave its sentinel (`0x7fffffff` is `sample.metal`'s "no candidate" value,
and its argmax skips NaN): the model's own output for that window is NaN. An earlier reading that
called this a *readback* fault was wrong — the verifier reports faithfully what the sampler wrote.

The window is `T = 1` because the first speculative round has no drafts to widen it. Further runs
show it is **not** the single row:

* the forward really executes — 1194 dispatches, 1257 ms of GPU for that window (`--spec 8`), so it
  is a numeric failure, not a skipped dispatch;
* padding the first window to **two** rows (a diagnostic, row 0 is unaffected by row 1) still gives
  all-NaN taps and logits, so the one-row path is not the culprit;
* the pack's experts are not the problem either: `native_experts.txt` (parsed, not guessed) says
  layer 0 is gate/up type 16 (**IQ2_XXS**) and down type 42 (**Q2_0**), with IQ2_S / IQ4_NL on other
  layers — and `expert_parity --gu IQ2_XXS --dn Q2_0 --embd 2560 --ff 7680` **passes** on this card:
  `h rel L2 1.46e-07, y rel L2 1.83e-07`.

So the NaN is in the model's own forward (the `qwen4exp` pieces this family adds — hyper-connections,
PLE, QSA), not in the experts and not in any DFlash2 code. Two further notes from the dig: the
engine's non-finite guard (`nonfinite_hit`, issue #879) cannot fire on this port because
`logits_nonfinite_rows` is a `missing()` stub in `metal/src/engine/stubs.cpp:244`, and the CLI loop
has no such check at all; and `T_IQ1_S` / `T_IQ1_M` exist in `quant.metal` but have no `GU_SET` /
`DN_SET` instantiations in `moe.metal`, so a pack whose experts really are IQ1 would find no kernel
(this pack's are not).

### Where the NaN actually starts (a per-stage probe)

`STRATA_NAN_PROBE` (a `gr_nan` kernel dispatched after each stage, reading the stage's own buffer -
the patch is in the workspace as `nan-probe.patch`, not committed) reports how many of a stage's
values are non-finite. The first window of this run is **4 rows** (the draft is asked for one row;
the verify window is four), and the answer is:

```
embeddings     0/10240
hc bias bo     0/20480      inj[0] 0/32   inj[1] 0/32
L0 entry    1024/40960   first at 31232        <- 31232 = 3 * 10240 + 512
L0 after attn 10240/40960 first at 30720        <- 30720 = 3 * 10240
layer 0 .. 47 10240/40960 (each, first at 30720)
```

Reading that against a row width of `HC * N = 4 * 2560 = 10240`: **rows 0, 1 and 2 of the window are
clean at every stage; only the last row goes non-finite.** At layer 0's entry it holds 1024 NaN
values starting 512 floats into the row; once layer 0's attention half has run, that whole row is
NaN and every later layer keeps only that row broken.

So this is not "the forward is broken" - it is **the last row of the verify window**, whose state is
bad before the first layer runs. The PLE block's inputs (history, value, conv weights) probe clean;
its `qn` buffer reads NaN but its consumers are finite, which is a probe-ordering artefact, not a
finding worth chasing.

The next thing to look at is therefore how the window writes that last row's residual before layer 0
- the embedding broadcast and the hyper-connection bias for row `Tn - 1` - and why exactly floats
512..1536 of its 10240 are the ones that arrive non-finite.

Earlier narrower readings (kept for the record):

* `STRATA_ADMIT_MERGE_FIRST=1` does not widen the first window in a CLI run (the merge path is for
  batch admission);
* a **single-token prompt** (`--tokens 11`, no prefill chunk at all) fails identically, so the
  prompt pass is not the source.

The run is also far over its memory cap (15.3 GB against 8.6 GB) with a 0.0 expert-cache hit rate,
which is not ruled out as a contributor. Next step: bisect the `qwen4exp` forward itself (the layer
dump that would do it, `--dump-layers`, disables the device residency table, which `--spec` requires,
so it needs either a `--batch`/serve run or a separate probe).

Two smaller things seen and not fixed:

* the CUDA card heuristic fires on Metal — `--draft dflash2: on a 12 GB NVIDIA card the MTP drafter
  was faster …` — because `cudaGetDeviceProperties` is a shim here (`compute capability 0.0`);
* the run is far over the memory cap it plans for: `the process holds 15.3 GB … cap 8.6 GB - OVER
  THE CAP`, with the expert arena at 2.0 GB and the drafter's 1.1 GB on top.

## Not run, and therefore not claimed

* **No end-to-end output.** Not one token has been produced with DFlash2 drafting on this machine.
* **No acceptance rate.** The selector's picks have never been checked against a verify window.
* **No `check_metal_dump.py` comparison.** That is the real oracle (the torch port of
  `DFlash2DraftModel` in `tools/dflash2/dflash2_ref.py`); the engine's `STRATA_DF2_DUMP` path exists
  and was reached, but the round it dumps never completes. Torch 2.2.2 is available on the
  MacPro71 side if the dump is to be checked there.
* **No other pack.** Only `UD-IQ1_M` was tried; `Q8_0` is 133 GB and the other UD packs were not
  exercised.
* **No performance claim.** 93.3 ms is one cold iteration of a block, not a round, and not a
  working one.

## Commands

```sh
cd ~/projects/Strata/strata

# the kernel check the port is judged by (Q4_0 GEMM against the CPU)
./build-metal/df2_bench metal/src/engine/msl

# one stage only, one iteration
DF2B_ONLY=block DF2B_ITERS=1 ./build-metal/df2_bench metal/src/engine/msl

# build the drafter pack (numpy only; ~1 min)
python3 tools/dflash2_pack.py ~/models/dflash2-drafter/model.safetensors \
    ~/models/dflash2-drafter/config.json ~/models/dflash2-drafter/dflash2.pack

# the engine run this page is about (stops on the anchor token)
./build-metal/strata --pack ~/models/qwen38/UD-IQ1_M/pack \
    --native ~/models/qwen38/UD-IQ1_M/UD-IQ1_M-metal.gguf \
    --native-dense-gguf ~/models/qwen38/UD-IQ1_M/UD-IQ1_M-metal.gguf \
    --ple-gguf ~/models/qwen38/UD-IQ1_M/Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf \
    --expert-profile data/expert-profile.bin --expert-cache auto \
    --tokens 3,1,42498,456,11 --max-new 2 --max-context 512 \
    --draft dflash2 --draft-model ~/models/dflash2-drafter/dflash2.pack \
    --draft-fallback none --mtp-draft-vocab data/draft_vocab.bin \
    --spec 8 --suffix-draft 0 --lookup-max-t 0 --prefill 5
```

`--expert-profile` and `--expert-cache` are not optional: `--spec` refuses to run without the
device residency table, and that table only exists when the cache is planned from a profile
(`data/expert-profile.bin` ranks all 48 × 512 pairs, the Qwen3.8 geometry).
