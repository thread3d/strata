# macpro51: the DFlash2 drafter on a 64-lane SIMD GPU, 2026-11-23

Companion to `docs/macpro51-2026-11-22.md`, which is the record for the Qwen3.6 engine on this
machine. This page covers one thing: **DFlash2** (`--draft dflash2`, the block drafter for
Qwen3.8-Flash-Next), which the Metal port carried but had never run here.

The short version: **the drafter's kernels are ported to this card, the model runs end to end with
the drafter attached, and the drafter's forward now agrees with the torch port on the same context
(cos 0.9992-0.9995 on the block's eight final rows, the selector's walk 7 of 7). It still drafts
nothing the verifier accepts — 0 of 105 — and the section "The `d.part` probe" says why that is now
a question about the context the drafter is fed, not about its arithmetic.** Nothing here is a
performance number. Sections are dated; the older ones are the trail, newest last.

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

## It runs (2026-11-23, `c22a8a6bd`)

Qwen3.8-Flash-Next produces tokens on this card, with the DFlash2 drafter attached:

```
prompt  : 3 1 42498 456 11
output  : 1544 76 398 8709 96302 17776 78 76
decode                   8 tokens in 16671.8 ms  ->  0.48 tok/s
prefill                  4 tokens in 26956.7 ms  ->  0.15 tok/s
dflash2                  206.9 ms/round drafting (7 rounds, GPU 204.1), 1103 MiB of VRAM
speculation              8 rounds of 8, drafts accepted 0 of 49 (0.000), 1.00 tokens per round
```

The fix was one defect, described above: **the dense arena's host reads did not go through
`Context::host_ptr`**. The arena is two overlapped buffers on this card, and `W_.ptr + offset` is
chunk 0's base only, so `token_embd.weight` — which sits at exactly `stride` (3.76 GiB), the first
tensor in the second chunk — was dequantised out of unrelated memory. Fifteen host reads (ten in
`window.cpp`, four in `dflash2.inc`, one in `prompt_chain.inc`) now ask the context for the pointer.

Two independent oracles say the arithmetic around that fix is sound. `df2_bench` checks both of the
drafter's GEMMs against plain CPU dot products - `df2_mm4` (Q4_0) at 3.35e-07 and, new in `fff3188e0`,
`df2_mm` (Q8_0) at 4.31e-06. And `tools/dflash2/check_metal_dump.py` against the torch port agrees
with the engine's selector walk on **7 of 7 picks** (probabilities within 0.01) and on `hsel` within
0.085.

## The drafter still drafts badly

Acceptance is **0 of 49**, and the same torch check says why: the block's eight final rows come out
with **cosine similarity 0.55-0.74** against the fp32 reference (the check's bar is 0.98), while their
norms are right (0.96-0.99). So the block computes values that are the right size and the wrong
direction — the selector then walks a wrong-but-confident tree, which is exactly what a run of
`0.64 0.82 0.27 0.83 ...` probabilities with nothing accepted looks like.

What it is *not*, each checked: the GEMMs (both pass above); the lane mapping (every kernel in
`dflash2.metal` takes its group and lane from `sg32`/`lane32`, 12 of 12, and no `simd_sum` is left);
the pack (clean, and the Q8_0 draft head requantised from the target's own rows); and the context
(the ring is filled and advancing - `kring[0]` 4096 of 4096 non-zero, one context row per round at
positions 4, 5, 6, ...).

So the next step is to compare the block's *own* intermediates against torch, layer by layer. The
dump carries the final rows and the head's logits but nothing in between, so `df2_draft` needs the
same kind of stage dump the window now has.

## Where the drafter's forward stands (2026-11-23, `9de180eea`)

The port is not finished: acceptance is 0 of 49. What is known, measured stage by stage with
`STRATA_DF2_STAGE_DUMP` (eleven buffers of layers 0 and 1, each copy behind a barrier) against
`tools/dflash2/dflash2_ref.py` on the same taps:

| stage | layer 0 | layer 1 |
|---|---|---|
| `xn` after the input norm | 1.000000 | 0.9998 |
| `xc` after the conv | 0.9998 | 0.9996 |
| the query after `q_norm` + rope | 0.99991 | 0.99975 |
| `k_proj` / the block's K | 0.99997 | 0.99989 |
| `v_proj` / the block's V | 0.99969 | 0.99966 |
| **the attention's output** | 0.9998 | **0.909** |
| `o_proj` | 0.9999 | 0.927 |
| `h` after the attention half | 0.999997 | 0.998 |
| `xn2` / `xc2` | 0.9999 / 0.9996 | 0.987 / 0.889 |
| gate / up / swiglu | 0.9996 / 0.9996 / 0.9992 | 0.898 / 0.891 / 0.787 |
| `mlp` | 0.9995 | 0.827 |

Two things follow. First, **every operator is correct on this card**: layer 0 is exact at all
eleven stages. Second, at layer 1 **every operand of the attention is right to about 3e-4**
(xn, xc, q, k, v) while its output is wrong by 0.09, so the fault is in *which keys are folded in* -
the attention's chunk/mask bookkeeping - not in any projection, norm, rope or GEMM.

A cross-check guards against the trap I fell into twice: engine layer 0's `ao` scores 0.9998
against the reference's layer 0 and 0.013 against its layer 1, and engine layer 1's `ao` 0.9086
against layer 1 and 0.013 against layer 0. Each slot is provably its own layer.

The open leads, in the order worth trying:

* `d.part`, the per-chunk (m, l, O) buffer the merge reads over - the last unmeasured one;
* the ring rows the **prompt** path filled (`Prefill` -> `df2.context` -> `df2_kv_list`), as distinct
  from the rows the decode rounds fill; substituting zeros (0.48), layer 0's keys (0.27) or layer
  0's keys for just those rows (0.879) all make the result worse than the engine's own 0.909, so
  that context is *nearly* right;
* `df2_attn_merge`'s `nchunks` against `df2_attn_part`'s placeholder rows for chunks past the end.

## The `d.part` probe: the merge was reading the layer before's statistics (2026-11-23, `f708fe8ae`+)

The lead this page ended on was `d.part`, the per-chunk (m, l, O) buffer `df2_attn_merge` reads over.
Two probe changes were needed before it could be read: `STRATA_DF2_RING_DUMP` (the context K/V rows
the block's attention actually reads, per layer, plus the `hctx` / `kc` / `vc` the context path last
wrote), and a move of the stage-13 slot — it was dumped *before* the attention ran, so it held the
previous layer's `d.part` (zeros at layer 0). It is now dumped after the merge.

Run at round 3 (`s = 8`, `anchor 8709`, one chunk of 16 keys), against `tools/dflash2/dflash2_ref.py`
on the same taps:

* **The ring is right, on every layer.** `kring[l]` / `vring[l]` rows 0..7 against the port's
  `Kc` / `Vc`: cos **0.999986-0.999993** for layers 0 and 1 and 0.99999 for layers 2-4 (V 0.99994+).
  Both the rows the prompt filled (0-4) and the rows the decode rounds filled (5-7) are right, so
  the context path, the ring slots and the rope at the context positions are not the fault. This
  closes the "ring rows" lead.
* **`d.part` is right too** — layer 1's chunk-0 O against the port: cos **0.99935** — **but the
  merged `ao` was not.** Rebuilding `ao` from the engine's *own* dumped part gives cos 1.0 at layer
  0 and the same 0.877 at layer 1: the engine's attention output disagreed with the buffer its own
  merge had just read.
* The disagreement had a shape: every 128-float head block of `ao` matched the part's O in
  *direction* exactly (cos 1.00000) but some blocks were scaled by `l1 / l0`. The merge had divided
  the fresh O by the **previous layer's `l`** — 32 of 32 sampled `(kvh, qi)` entries in row 0 whose
  `l0` and `l1` differ divided by `l0`.

**Root cause: the part kernel and the merge are two dispatches in one `MTLDispatchTypeConcurrent`
encoder**, separated only by `memoryBarrierWithScope:MTLBarrierScopeBuffers`. On this card that
barrier does not hold the reader back. Each 130-float struct is written by 64 threads (thread 0
writes `m` and `l`, threads d write O[d] and O[d+64]), so a reader that starts before the writer's
threadgroups retire can see thread 0's `(m, l)` still old while the other threads' O are new — a torn
struct, not a whole stale buffer. **It is scheduling, not arithmetic**: inserting one extra dispatch
between the two changed layer 1's `ao` (max |diff| 12.6) while every value through layer 0 stayed
bit-identical. Layer 0 was clean because nothing had read `part` before it.

The fix is `run_graph(ds, count, serial)` in `metal/src/mtl_rt.mm`: one serial compute encoder, so
the dispatches run in order. The drafter's three graphs (the context update, the block, the lineage
rebuild) take it; the rest of the engine does not (the window under `STRATA_METAL_STREAM=1` already
runs segmented with shared-event handshakes between segments, which is a real GPU-side dependency).
The block's list is barrier-dense already, so the cost is small: **226.3 ms/round against 218.0**
(GPU 220.6), and the target model's own tokens are untouched — only the drafter's graphs changed.

After it:

| | before | after |
|---|---|---|
| layer 1 `ao` against the port | 0.877 | **0.99976** |
| layer 1 `ao` against its own `d.part` | 0.877 | **1.0** |
| the block's 8 final rows against the port | 0.55-0.74 | **0.9992-0.9995** |
| selector walk (torch, on the engine's own rows) | 7 of 7 | 7 of 7 |
| `tools/dflash2/check_metal_dump.py` | fails the row bar (0.98) | **ok** |

**Acceptance is still 0 of 105** (`16 rounds of 8, drafts accepted 0 of 105 (0.000)`). That figure
is now informative in a way it was not before: the drafter's forward is faithful to the torch port
on the same context, so the disagreement is no longer inside the drafter. It is between the drafter
and the verifier, and only two things can carry it: **the taps** the drafter is fed (the target
model's hidden states at layers 5/15/25/35/45, folded by `df2_tap`) or **the target model's own
argmax** on this card. `df2_tap`'s fold, `(Σ_c R_c + bo·Σ_c g_c) / hc` with `g_c = 2/(1+exp(-inj_c/hc))`,
is the same math `hcf.metal` documents for the target's own hyper-connection read, so the formula is
not the suspect; whether the *right* rows are being tapped (and whether qwen4exp's forward itself is
right here) is. That needs an oracle for the target model's hidden states, which this machine does
not have yet.

What the dump says about the drafts themselves (round 3, `s = 8`, anchor `8709`): the seven proposals
are **tokens 8, 14, 14, 15, 15, 15, 15** with probabilities 0.344, 0.577, 0.947, 0.648, 0.269, 0.344,
0.445 — while the target's own continuation from that anchor is `96302 17776 78 76`. They are not
near misses; they are unrelated to the continuation. `vmap` maps the draft vocabulary to the target's
(identity for the first rows, so those *are* token ids), the anchor and mask embeddings are sane
(norm 0.25 each, distinct), and the tap wiring was then checked end to end: `taps_enable` fills
`tap_slot_` with -1 and maps layers 5/15/25/35/45 to slots 0-4, `df2_tap` folds
`(Σ_c R_c + y·Σ_c g_c)/hc` with `g_c = 2/(1+exp(-inj_c/hc))`, and the injection index is the one that
pairs with the sublayer output (the toggle happens before the tap). All of that is right, so the
context the drafter is handed is *internally* correct and only its **provenance** is questionable:
it is the target model's hidden states, and the target model has no oracle here.

Next, in order: dump the target's `h` at the start of layer 6 and compare it with the tap written for
layer 5 — they must be the same vector, and if they are not the tap is being taken at the wrong point.
If they agree, the taps are right and the question moves to qwen4exp's forward itself.

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

### The root cause: the dense arena does not hold the file's bytes

Chasing the last row's 1024 NaN values to their source ends outside the kernels: the engine's
**token embedding is not the one in the pack**.

The dequantiser is not at fault. `dsu_ref_vs_ggml` compared `ref::dequant` against ggml's `to_float`
for IQ4_XS, IQ3_S and IQ4_NL (the three types these packs use for their embeddings, and the only
ones that check had never covered) and all three match **bit for bit**, 0 of 131,072 values
differing. The converter is not at fault either: the same scan of `token_embd.weight`'s fp16 scales
in the pack reads **0 of 2,483,200 blocks non-finite** (once the file's `general.alignment` of 16384
is honoured; with the wrong alignment the same scan reports a meaningless ~2%).

But the **engine's** own copy of that table does not match. Its tensor table says the embedding sits
at arena offset 4,037,634,048 (type 23 = IQ4_XS, 337,715,200 bytes = 248,320 x 1,360, so the sizes
agree), yet the bytes there are not the file's:

```
file  at token_embd:  21 80 0f cf da 4c c8 f0 ...
arena at that offset: 1a d2 9c 4b 74 69 55 58 ...
file  row 456:        59 00 f0 f1 c0 01 50 88 ...
arena row 456:        09 2a 67 f0 cf 53 ec 11 ...
```

A scan of the arena's own table finds **45,247 of 248,320 rows carrying an "Inf/NaN" fp16 scale
(49,380 of 2,483,200 blocks — 1.99%)**, which is what random bytes look like: 1.99% is about the
rate at which arbitrary 16-bit patterns have an all-ones exponent. The file's own table has none.

So the native dense path fills its 4.38 GB arena with something that is **not** the tensor the pack
holds at that offset — the offsets and sizes the loader assigns do not line up with the file. Every
downstream symptom follows: token 456 (the last row of the first prompt window) dequantises to 1024
NaN values, that row poisons the hyper-connection bias `bo_` and `inj_` that the *next* window
reads, and from there the whole decode is NaN. That is also why the MoE parity and the dequantiser
both pass while the model cannot produce a token.

The sharpest clue: **the offset the engine uses lands inside another tensor.** The file's 1,223
tensors have `token_embd.weight` at 528,515,072 (relative to the data section), and nothing sits at
4,037,634,048 — it falls about 3 MB *inside* `blk.43.attn_output.weight` (4,034,641,920 + 12,902,592).
The bytes the window reads there match neither the file at that offset nor at that offset plus the
data base, so the dense buffer is not a straight copy of the file into the pack's own layout either.

`src/core/native_dense.cpp` is the CUDA path (`cudaMalloc` + `cudaMemcpy` per tensor, with shape
checks against the canonical table), so the Metal window's fill is a different code path — that is
the file to read next: how the Metal window fills its 4.38 GB dense buffer and where it gets each
tensor's source offset and length. Two candidate mistakes, both cheap to confirm once the code is in
view: the destination offset used as the source offset, or a per-type row-size that disagrees with the
file (which would shift every payload after the first tensor of that type).

### Where the NaN shows up stage by stage (a per-stage probe)

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

* **No end-to-end tokens from DFlash2 drafting** — the model produces tokens, but every draft is
  rejected (0 of 105), so no speculative speedup exists. (Written before the model ran; kept as the
  record of that state. The acceptance figure and the `check_metal_dump.py` comparison now exist,
  see "The `d.part` probe".)
* **No acceptance rate above zero.** The selector's walk matches torch 7 of 7 on the engine's own
  rows, and the drafts are still all rejected.
* **No comparison of the taps** with an independent implementation of qwen4exp's hidden states —
  that is the open question the acceptance figure now points at.
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
