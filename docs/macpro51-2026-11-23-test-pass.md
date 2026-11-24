# macpro51: a full test pass at `5360e6292`, 2026-11-23

"Full" on this page means: build the branch, run the whole `ctest` suite, drive each model family's
CLI path as far as this machine allows, and run the diagnostics the port is judged by. Every number
is from that run's own output on macpro51; nothing was inferred. Timings are single cold runs and
are not performance claims.

**The short version: the engine works. The suite was 12 passed / 6 failed when the pass started;
each failure is explained below and all six are now resolved - five fixed, one turned into a
documented skip - so the suite is 17 passed, 1 skipped, 0 failed.** The first version of this page
also called `df2_bench` a hang; that was wrong, and the section below says what it really costs.

## The machine and the build

| | |
|---|---|
| Host | MacPro5,1 ("macpro51"), macOS 11.7.11, 2 × Xeon X5675, ~96 GB DDR3 |
| GPU | AMD Radeon RX 580 8 GB, Metal 2, **64-lane simdgroups** |
| Branch / commit | `macpro51`, `5360e6292` + the six commits in "What the pass changed" |
| Build | `PATH=/usr/local/bin:$PATH cmake --build build-metal -j 24` → clean |
| Engine | `strata 0.1.59-metal` |

The binaries in `build-metal/` are newer than the installed `/usr/local/bin/strata*` copies (those
are 2026-11-22 09:54); this pass used the build tree.

## The suite: 17 passed, 1 skipped (74 s)

| # | Test | At `5360e6292` | Now |
|---|---|---|---|
| 1 | `q36_model_check` | Failed | **Passed** |
| 2 | `q36_attn_parity` | Passed | Passed |
| 3 | `q36_glue_parity` | Failed | **Passed** |
| 4 | `q36_rope_parity` | Passed | Passed |
| 5 | `q36_router_parity` | Passed | Passed |
| 6 | `q36_dense_parity` | Passed | Passed |
| 7 | `q36_attn_layer_parity` | Passed | Passed |
| 8 | `q36_gdn_layer_parity` | Passed | Passed |
| 9 | `q36_moe_shared_parity` | Passed | Passed |
| 10 | `q36_expert_blob_check` | Passed | Passed |
| 11 | `q36_moe_routed_parity` | Passed | Passed |
| 12 | `q36_layer_check` | Passed | Passed |
| 13 | `q36_bind_check` | Passed | Passed |
| 14 | `q36_forward_check` | SIGSEGV | **Passed** |
| 15 | `q36_stream_check` | Passed | Passed |
| 16 | `q36_generate_check` | SIGABRT | **Passed** |
| 17 | `q36_upload_check` | Failed | **Passed** |
| 18 | `runtime_hello` | Failed (4/4) | **Skipped** (77, see below) |

## The six failures, and what each one was

`377fdec0d` ("make every supported quant generate on the RX 580", 2026-11-18) introduced two engine
changes involved here - the private-storage allocator and the `q36_embed` type argument. Three of the
tests were written against the API before it.

| Test | What it reported | Cause | Fixed by |
|---|---|---|---|
| `q36_upload_check` | `token_embd: differs at byte 0 of 16885760`, `0 bytes compared` | `memcmp` through `Buffer::ptr`, which `alloc(bytes, src)` sets to the buffer handle as a sentinel (`metal/src/mtl_rt.mm:860`: *"Private storage has no CPU address; non-null sentinel passes upload checks"*). | `Context::read_back` - blit Private → shared staging, wait, copy - and `same()` reads through it: 34 571 920 bytes now compare, the figure the port's docs recorded before the allocator changed. |
| `q36_forward_check` | SIGSEGV, no output | the test `memset`/`ones()`es **through** that sentinel, over the Objective-C object header | `Context::stage_write` (shared staging → blit → Private), used by `zero()`/`ones()`. |
| `q36_generate_check` | SIGABRT | same | same |
| `q36_glue_parity` | `q36_embed … rel L2 1.000e+00` | the test dispatches `q36_embed` with **4** args; the kernel takes **5** (`etype` at `buffer(4)`), so no branch ran and nothing was written. The engine passes it (`q36_forward.cpp:170`). | the 5th argument, `ref::Q8_0`; the three cases are bit-exact (rel L2 0.000e+00). |
| `q36_model_check` | `token_embd.weight is ?, q36_embed only supports …` (wants `Q4_1`) | `ref::type_name` had no `Q4_1` (ggml id 3) case | `Q4_1` added to the id enum, `type_name`, `block_elems` (32) and `block_bytes` (20); the refusal now reads `token_embd.weight is Q4_1, …`. |
| `runtime_hello` | `RESULT: 4 FAILED`, 65470/65536 wrong | the program tests a CUDA-shaped contract: a kernel argument carries a device **address**, and a mapped host buffer is read and written in place. That needs `MTLBuffer.gpuAddress` (macOS 13). Big Sur has none, so the runtime hands kernels `contents()`, nil for GPU-only storage, and every check runs against address zero. | a platform guard: print the reason, return 77, and `SKIP_RETURN_CODE 77` in CMake. ctest reports it **Skipped**, not failed. macOS 13+ still runs all four checks. |

## `df2_bench`: slow, not hung

The first version of this page called the 8-row context update a hang. It is not. `time_it` pays the
verify-window stand-in graph - 18 `df2_mm4` dispatches over a 60000 × 12800 Q4_0 matrix, written for
about 13 ms on a faster GPU - once per warm-up round (30) and once per timed round
(`DF2B_ITERS`, default 200). Measured here:

```
stand-in graph: 1794.7 ms a round; a stage warms up 30 and times 200, about 413 s.
DF2B_STANDIN=0 leaves it out.
```

So a default stage needs ~7 minutes and the whole run ~41, and a sample of the process shows the
main thread parked in `-[_MTLCommandBuffer waitUntilCompleted]` on a command buffer that is simply
still running.

The graph itself is healthy and now measurable in seconds:

```
DF2B_STANDIN=0 DF2B_ONLY=ctx DF2B_ITERS=5      (2.35 s for all three stages)
  context update, 1 rows   7.110 ms
  context update, 3 rows   7.533 ms
  context update, 8 rows   9.456 ms
```

`DF2B_TRACE=1` additionally runs a stage as growing prefixes and prints each dispatch; all 22 run.
With `DF2B_SKIP=mm` the three stages complete at 0.080 / 0.174 / 0.428 ms, and variant `4_4` finishes
just as well once the stand-in is out of the way (7.124 / 7.994 / 11.182 ms, 2.37 s for all three) -
the earlier "variant-dependent hang" was the same warm-up cost against a shorter alarm.

## The app on real models

Measured before the fixes above; none of them touches a kernel, a graph or an engine path.

### Qwen3.6-35B-A3B — `strata-q36-metal`, 23 prompt ids → 24 generated, `--text`

| Pack | s/step | Last tokens | vs llama.cpp reference |
|---|---|---|---|
| Q4_K_M | 1.359 | `… 2972 25562` | **matches**, 24/24 |
| UD-Q4_K_XL | 1.299 | `… 2972 14162` | **matches**, 24/24 |
| Q6_K | 1.558 | `… 2972 25562` | 1-23 match; token 24 is the documented 25562/14162 near-tie (reference: 14162) |
| UD-Q6_K_XL | 1.624 | `… 2972 14162` | **matches**, 24/24 |
| Q8_0 | 1.805 | `… 2972 14162` | **matches**, 24/24 |
| UD-Q2_K_XL | — | — | refused: `token_embd.weight is Q5_K, q36_embed only supports Q8_0, Q6_K and Q4_K` |
| UD-Q8_K_XL | — | — | refused: `blk.34.ffn_down_exps.weight is BF16, and this port has no grouped expert down kernel for that type (it takes: Q8_0, Q5_0, Q5_1, IQ4_NL, IQ4_XS, IQ3_S and Q2_0)` |

Q8_0's full sequence, for the record:

```
271 248068 198 8160 579 264 7047 1817 25 271 16 13 220 2972 2014 53983 2570 5396 64700 198 256 471 2972 14162
```

Reference: `bench/macpro51/results-reference.json` (llama.cpp, same 23 ids). The s/step figures are
one cold run per pack (experts streamed from the HDD RAID); the stored warm matrix is 0.90-1.17
s/step, so 1.30-1.81 here should not be read as a regression without a warmed re-run.

### Qwen3.8-Flash-Next UD-IQ1_M + DFlash2 — the documented command

```
strata metal: DFlash2: 64-lane simdgroups, so the matrix units are not used …
prefill 4 tokens in 1 chunks, 27222.7 ms (0.1 tok/s)
speculation  2 rounds of 8, drafts accepted 0 of 7 (0.000), 1.00 tokens per round
prompt  : 3 1 42498 456 11
output  : 1544 76
decode                   2 tokens in 3516.5 ms  ->  0.57 tok/s
resident footprint after the fill: the process holds 13.0 GB (15.3 GB resident), planned 15.8 GB,
cap 8.6 GB - OVER THE CAP
```

It runs and emits tokens, and **draft acceptance is still 0** (0 of 7 in this 2-round run; the
drafter page's 0 of 105 stands). Prefill is very slow (27.2 s for 4 tokens, TTFT 32.9 s), and the run
is over the memory cap as before.

### DeepSeek V4.1 — `strata_ds4`

`--pack-info` works: `40 layers x 384 experts, 6 used; dim 5120, ff 2304, vocab 129280`, with
per-layer IQ1_M / IQ2_XXS mixes. Generation refuses by design (rc=1): *"this Mac's Metal has no
device addresses (macOS 13's gpuAddress) and no residency set… It is not run here."*
`STRATA_DS4_ALLOW_UNSTABLE_METAL=1` was not set.

### CLI surface

`build-metal/strata` is generate-only on this port: `strata bench run`, `strata serve` and
`strata help` all exit 2 with `unknown argument: …`. `strata-q36-metal --help` and `--version` also
exit 2; `strata --version` prints `strata 0.1.59-metal`. The standard `strata bench run` protocol is
therefore not available on this machine.

## Diagnostics

| Check | Result |
|---|---|
| `expert_parity --gguf` on the five runnable Qwen3.6 packs (`--ff 512 --embd 2048`) | all **PASS** — Q8_0 h 1.43e-07/y 1.03e-07; Q6_K 3.11e-07/2.06e-07; Q4_K_M 2.87e-07/2.48e-07; UD-Q4_K_XL 2.78e-07/1.74e-07; UD-Q6_K_XL 3.06e-07/1.05e-07 |
| `dsu_ref_vs_ggml` | **11/11** formats bit-exact vs ggml `to_float` (0 of 131 072 values differ) |
| `down_kernel_emu` | rel L2 **1.989e-07**, max abs diff 7.818e-08 |
| `msl_dump --kq`, `--kq --f16` | **all libraries compile** (core, gdn, qsa, sample, ple, prompt, prompt_gdn, prompt_qsa, mtp, mtp_chain, dflash2, coop_probe) |
| `df2_bench` | see its section: the graph is healthy; the default run is ~41 minutes because of the stand-in |

## Not run here

* The benchmark protocol (`strata bench run`) — not present in this binary (exit 2).
* `--dn-prep` — the GPU-wedge combination, and pointless since the K-quant down fix.
* `STRATA_DS4_ALLOW_UNSTABLE_METAL=1` — reset risk; needs the owner's say-so.
* Logit-distance re-measurement (`logit_distance.py`) — would need fresh logit dumps.
* The full default `df2_bench` matrix (about 41 minutes; `DF2B_STANDIN=0` is the quick form).

## What the pass changed

| Commit | Subject |
|---|---|
| `538508e66` | q36_glue_parity passes q36_embed's type argument |
| `a19fd0579` | name Q4_1 in ref::type_name, with its block geometry |
| `e6354ceff` | exact host read/write of Private buffers, and the upload check on it |
| `5317020ca` | the forward and generate checks write through the staged path |
| `2e158bc7d` | df2_bench reports what the stand-in costs, and can leave it out |
| `310666f47` | runtime_hello skips on a Metal without device addresses |

Still open, and out of this pass: DFlash2 acceptance (0 of 105) and the Qwen3.8 memory cap.

## The two out-of-scope issues, researched: what the evidence now says (2026-11-23, after `acefc3bd8`)

### 1. Qwen3.8 over the cap: not a double count, and where the 13.2 GB is

The report was "the process holds 13.0 GB (15.3 GB resident), planned 15.8 GB, cap 8.6 GB - OVER THE
CAP", with a 1.0-2.0 GB expert arena and a ~1.3% hit rate. Two explanations were worth testing: the
budget double counts memory, or it charges clean file pages that this 96 GB machine could reclaim.

`STRATA_MEM_DEBUG=1` prints the components behind `TASK_VM_INFO`:

```
[mem] footprint 13.23 rss 13.26 internal 13.23 external 0.03 compressed 0.00
strata metal stream: 4.61 GB allocated before the arena (the process holds 13.23 GB, 13.26 GB resident)
strata metal stream: expert arena 2.02 GB, 1152 slots in 2 size classes
```

So: **13.23 GB is anonymous, 0.03 GB is file-backed.** No file cache is being charged, and nothing is
counted twice — the engine's comment that `phys_footprint` is anonymous memory is correct. A `vmmap
-summary` of the live process agrees with the engine's own number (physical footprint 14.3 GB at the
later phase, `IOAccelerator` 15.1 GB size / 14.9 GB resident), and it shows the anonymous total is
spread over the Metal allocations, the 2.34 GB native dense set, the prompt path's own buffers and the
arena.

The 8.6 GB figure it is compared against is `MTLDevice.recommendedMaxWorkingSetSize` — the driver's
recommendation for an 8 GB card under macOS 11 — so the run is genuinely over the card's working set,
and the arena is clamped to its floor. The avoidable terms, in descending size, are the prompt path's
~4.34 GB of matrix-unit buffers (a path `df2_bind` already gates off on this 64-lane card), the
2.34 GB native dense set (needed) and the arena itself. **Freeing the prompt buffers is the one change
that would move the arena from 1.0 GB to ~5 GB; it is not done here**, because the prompt path is
shared with the CUDA/HIP builds and needs its own measurement.

### 2. DFlash2 acceptance: what is ruled out, and the experiment that remains

Established before this pass: the drafter's forward agrees with the torch port (block rows cos
0.9992-0.9995, selector walk 7 of 7, `check_metal_dump.py` ok), and the proposals are target ids
`8/14/14/15/15/15/15` against a target continuation of `96302 17776 78 76`. Re-checked here:

- The mapping order looked like the prime suspect, and it is **not** a bug. `df2_topk_merge` applies the
  draft-vocabulary map before `df2_select` uses those ids as codebook indices (`succ_cb + t * rank`,
  `pred_cb + prev * rank`, `prev = out_tok[i]`), while the torch reference walks on unmapped `cand`
  ids. The reason is the tables: the drafter's codebooks are `candidate_selector.predecessor_codebook`
  `[248320, 256]` and the successor table likewise, i.e. **target**-vocabulary rows, while the engine's
  Q8_0 draft head has 106299 rows. The map has to be applied before the selector indexes those tables;
  the reference only skips it because it walks the full 248320-row head. So the early map is correct
  and the walk genuinely matches the reference.

- What the dump cannot settle is provenance: the engine and the reference receive the same taps, so
  agreement on them says the drafter is faithful, not that the context is right. The next experiment
  is still the one the page names: dump the target's `h` at the start of layer 6 and compare it with
  the tap written for layer 5; they must be the same vector.

That one experiment is bounded (a small probe and one run), and it was not run here. Acceptance stays
**0**, and this page does not claim otherwise: the drafter is faithful, its proposals on this
context are not, and the context is the only term left unverified.

## Update: the acceptance question moved to the draft head (2026-11-23, after `176a98e58`)

The section above ends by naming the tap provenance as the experiment to run. The dump reader
(`tools/dflash2/read_engine_dump.py`, `0f9e54bf0`) makes that no longer the first thing to do, and this
records why so the next reader does not start where this one did.

- **The taps are real data.** A real round's dump: 64 000 tap values, **0 zero, 0 NaN**, RMS 0.2883, per
  slot RMS 0.17–0.45, block rows RMS 3.50, selector hidden RMS 3.13. "The context is empty" is not the
  fault.
- **The drafter's head is the fault.** Per row: `max 10.6077 at 12, mean 0.0010, sd 0.0825, and exactly
  **16 of 106299** entries above 1e-6` — on every row. Only the first 16 of the head's 106 299 rows are
  written, so the top-16 is drawn from ids 0–15 and the proposals can only be those ids
  (`[10, 2, 12, 15, 11, 15, 11]` on that round). **Acceptance is 0 by construction**, and no amount of
  context correctness can change it while this holds. It also explains why "the drafter agrees with the
  torch reference" settled nothing: both read the same 16 written rows and ignore the other 106 283.
- **Ruled out as the cause** (each with a measurement, not an argument): the tile
  (`STRATA_METAL_GEMV_NR=2`, `rpt` 8 → 16, still 16 rows); `Window::gemv`
  (`STRATA_GEMV_DEBUG=1` over a full round: 1625 dispatches, unique `n_out` 640 / 2560 / 6144 / 10240 /
  12288 / 248320 — the 106 299-row head never appears); and the `df2_mma8_2_4` / `gemv` head branches in
  `dflash2.inc` (a probe at both never fired).
- **So the live head path is `df2_mm`** (`dflash2.inc:723`), and its grid is where the 16 rows come from.
  That is the next measurement, and it is the reason this page no longer points at the taps first.

One more observation, recorded because it may matter beyond the drafter: changing the tile moved the
*target's* output (`1544 76` → `98282 6084`), which a correctly tiled memory-bound GEMM should not do.

The instruments this left behind, all env-guarded: `STRATA_DF2_DUMP` (+ `_ROUND`, now known to count from
0 and to need `--prefill 1`, + `_TRACE`), `STRATA_GEMV_DEBUG`, `STRATA_MEM_DEBUG`, and the reader itself.

### Round 14: the matvec is cleared, the head's own bytes are next

`df2_mm` (the live head path, `dflash2.inc:723`) is not the fault. Its row coverage was read end to end:
`row0 = (tg.x * SG + sg) * R` with `R=2, SG=8`, and `grid[0] = ceil(rows / (SGN * R))` = `ceil(106299 / 16)`
= **6644**, i.e. 106 304 rows covered — every row. The instantiated name (`df2_mm_2_8`) matches the host's
`df2_mm_<R>_<SGN>`, and the kernel's row addressing (`qs + row * a.K`, K-byte flat Q8_0 rows) is what the
repacked head expects.

The written-row count was also measured under three different launch shapes and did not move:

| setting | rows a threadgroup | non-zero logits of 106299 |
|---|---|---|
| `STRATA_DF2_MM` 2,8 (default) | 16 | **16** |
| `STRATA_DF2_MM` 1,4 | 4 | **16** |
| `STRATA_METAL_GEMV_NR=2` (window path) | — | **16** |

A count that is invariant under the tile cannot be a property of the launch geometry, and the geometry
provably covers the rows. So the missing rows are most likely **empty upstream**: the draft head is
requantised to Q8_0 into `d.hq` / `d.hsc` before this kernel runs (the comment at `dflash2.inc:353` — "the
head rows of the draft vocabulary requantized to Q8_0 … in df2_mm's layout"), and if that fill stops after
the first 16 rows then every logit past row 16 is zero, the argmax is confined to ids 0–15, and the
proposals (and acceptance) follow. That also explains why every tile experiment above returned 16.

**Next probe:** the fill loop and the allocation sizes of `d.hq` / `d.hsc` (`dflash2.inc`, around 353–372),
printed against `d.nd`. That is the last unmeasured quantity in this chain.

### Item 10: the fp16 MoE kernels run and pass (2026-11-23, `5d7178727`)

The earlier pages call the fp16 MoE path "compile-checked only", so this was re-run on the branch tip
rather than trusted:

```
EXPERT_F16=1 expert_parity --gu Q2_0 --dn Q2_0 --ff 512 --embd 2048
device AMD Radeon RX 580; gate/up Q2_0, down Q2_0, blob 884736 bytes; 1 token(s)
parity Q2_0/Q2_0 ntok 1 gu 32,2,4 dn 8,2,4: plan ok (10 groups),
  h rel L2 1.14e-03 max 1.27e-03, y rel L2 8.72e-05 max 1.94e-04  PASS
```

Both halves pass against the fp16 tolerance (4e-3), and the numbers reproduce the ones recorded on
2026-11-22 to three digits, so the 64-lane lane fix is exercised on `moe_gu16` / `moe_down16` too and has
not regressed. `msl_dump --kq --f16` remains the compile check; this is the run.

### Item 13: what `df2_bench`'s defaults cost on this card

`df2_bench` defaults to `DF2B_ITERS=200` and interleaves the verify-window stand-in graph before every
timed round. That stand-in was written for about 13 ms on a faster GPU and measures **1794.7 ms** here,
so each timed stage runs 30 + 200 rounds of it: **about 413 s a stage, about 41 minutes for the default
run**, which reads as a hang (the process sits in `waitUntilCompleted` the whole time).

Since `2e158bc7d` the tool prints that cost before the first stage:

```
stand-in graph: 1794.7 ms a round; a stage warms up 30 and times 200, about 413 s. DF2B_STANDIN=0 leaves it out.
```

For this machine the useful forms are:

| goal | invocation |
|---|---|
| the graph itself, in seconds | `DF2B_STANDIN=0 DF2B_ITERS=5 ./build-metal/df2_bench metal/src/engine/msl` (all three context stages in 2.35 s; 8 rows 9.456 ms) |
| one stage only | add `DF2B_ONLY=ctx` (or `block`, `head`) |
| where a stage stops | add `DF2B_TRACE=1`, which runs the stage as growing prefixes and prints every dispatch |
| the full matrix | leave the defaults and give it ~41 minutes |

### Item 5: where the 13.2 GB is, phase by phase

Three measurements of the same workload (Qwen3.8 `UD-IQ1_M` + DFlash2), each with the number it came
from, so the table can be checked rather than believed:

| phase | what is held | measured |
|---|---|---|
| after the dense load (arena + native set) | `mtl: arena 4.38 GB as 2 overlapped buffers`, `302 canonical tensors skipped: served natively` (2.34 GB) | `vmmap -summary`: physical footprint **9.4 GB**, `IOAccelerator` **6.8 GB** |
| at planning, before the expert arena | 4.61 GB of Metal allocations + native dense + prompt buffers | `[mem] footprint 13.23 rss 13.26 internal 13.23 external 0.03 compressed 0.00` |
| after the arena fill | the above + `arena 2.02 GB, 1152 slots` | `[mem] footprint 12.97 rss 15.31 internal 15.27`; `vmmap`: footprint 14.3 GB, `IOAccelerator` 15.1 GB size / 14.9 GB resident, total resident 17.8 GB |

What the table settles:

- **`external` is 0.03–0.04 GB** in both `[mem]` samples: the 39.2 GB of mapped pack, the 320 M-row PLE
  table and every other clean file page are *not* charged. The 13.2 GB is anonymous, so there is no
  double count and no file cache being counted against the arena.
- **`IOAccelerator` is the volume**: 6.8 GB after the dense load, 14.9 GB resident once the arena and the
  prompt buffers exist. That is the driver's accounting of the buffers this process owns, not a leak of
  the engine's own ledger (`cx_->allocated()` is 4.61 GB at planning) — the gap between the two is the
  native dense set, the prompt path's own buffers and the arena, all of which the engine allocates
  deliberately.
- **The 8.6 GB it is compared with** is `recommendedMaxWorkingSetSize` for an 8 GB card under macOS 11,
  so the clamp is real, and the arena floor of 1.0–2.0 GB follows from it.

The avoidable term remains the prompt path's ~4.34 GB of matrix-unit buffers (item 6), which this
64-lane card cannot use at all.

### Item 4: a longer acceptance sample (2026-11-23, `4d9549abb`)

The 0-acceptance figure quoted on this page came from a 2-round run (0 of 7). This is the same command at
`--max-new 8`, so 8 rounds of 8:

```
speculation              8 rounds of 8, drafts accepted 0 of 49 (0.000), 1.00 tokens per round
output  : 1544 76 398 8709 96302 17776 78 76
decode                   8 tokens in 16489.5 ms  ->  0.49 tok/s
```

49 drafts offered, none accepted, and the eight tokens are the sequence the drafter page records
(`1544 76 398 8709 96302 17776 78 76`), so the target is behaving as documented while every proposal
misses. With the logits measurement (16 of 106299 head rows written) this is no longer a mystery to be
sampled further: a head that ranks only ids 0–15 cannot propose `96302`, so 0 is the expected rate, not a
low one. The sample is recorded here to close the "0 of 7 is a small sample" objection, not as open
evidence.

### Item 9: the Q6_K near-tie cannot be quantified with the shipped binaries (2026-11-23)

The claim to check is that Q6_K's 24th token is the 25562/14162 near-tie llama.cpp shows. Measuring it
needs the logits at that position, and this is what the tree offers:

- the engine actually used here, `build-metal/strata-q36-metal`, takes `--model --prompt --n --text` (the
  invocation `bench/macpro51/run_matrix.py` uses) and has printed nothing but ids and text;
- `metal/q36_engine.cpp` *does* print `qwen36: top-5 logits …` with values (line 378), but **no binary in
  `build-metal` contains that string**, so that engine is not part of this build;
- the main `strata` binary has `--dump-logits`, but it is the qwen4exp engine and cannot load a
  Qwen3.6-35B-A3B pack.

So the gap is not measurable as the branch stands: it needs either `metal/q36_engine.cpp`'s engine wired
into the build, or a `--dump-logits` added to `strata-q36-metal`. Both are small pieces of work, and
neither was done here. The honest statement therefore stays as it was - *the 24th token is the documented
25562/14162 near-tie between Q6_K and Q8_0* - with no number attached, and this page does not claim a gap.

Recorded so the next attempt does not begin by looking for a flag that does not exist in this build.

### Item 12: the installed binaries now match the branch tip (2026-11-23 15:19)

`/usr/local/bin/strata*` were from 2026-11-22 09:54, a day behind `build-metal/`. With the owner's
approval they were reinstalled from the build tree:

```
before  6 832 240  Nov 22 09:54  /usr/local/bin/strata
          525 224  Nov 22 09:54  /usr/local/bin/strata-q36-metal
        6 183 496  Nov 22 09:54  /usr/local/bin/strata_ds4
after   6 866 712  Nov 23 15:19  /usr/local/bin/strata
          525 712  Nov 23 15:19  /usr/local/bin/strata-q36-metal
        6 183 896  Nov 23 15:19  /usr/local/bin/strata_ds4
        /usr/local/bin/strata --version  ->  strata 0.1.59-metal
```

All three timestamps and sizes moved, so the copies are the ones this page's measurements were taken
with rather than the previous day's. `/usr/local/bin` is owned by `threaded`, so no elevation was needed;
the binaries themselves stay out of git, which is why this lands as a note rather than a build artefact.

### Item 7: the default `df2_bench` completes, in 78 minutes (2026-11-23, `10b49e7ae`)

One complete default run (`DF2B_ITERS=200`, nothing else set), `rc=0`:

```
real      4679.81        user 82.78   sys 2.26      -> 78.0 minutes wall
context update, 1 rows     22 dispatches | GPU  7.093 ms | wall  7.418 ms
context update, 3 rows     22 dispatches | GPU  7.509 ms | wall  7.847 ms
context update, 8 rows     22 dispatches | GPU 10.225 ms | wall 10.573 ms
block (8 rows, 5 layers)  102 dispatches | GPU 93.224 ms | wall 93.819 ms
head + top-16 + selector    5 dispatches | GPU 68.516 ms | wall 68.926 ms
FULL ROUND, n_new = 1     129 dispatches | GPU 168.861 ms | wall 169.539 ms
FULL ROUND, n_new = 3     129 dispatches | GPU 169.380 ms | wall 170.029 ms
FULL ROUND, n_new = 8     129 dispatches | GPU 172.816 ms | wall 173.508 ms
```

**This supersedes the "about 41 minutes" figure in the item 13 section.** The tool's own estimate
(`stand-in graph: 1777.4 ms a round … about 409 s` a stage) is a *floor*, not the stage time: eight
stages at that rate would be ~55 minutes, and the measured run took 78. So the printed estimate
understates the default run by roughly a third on this card, and the number to quote is **78 minutes**.

What the run does establish, and why the estimate misses it: every stage's own graph is milliseconds
(7–173 ms), while the wall time is dominated by the 230 stand-in rounds of 1.78 s each — i.e. ~410 s of
stand-in per stage, which is where the ~55 minutes comes from, plus the per-round host gap and the
per-stage warm-up beyond the counted rounds. The graph numbers above are the useful output of the run,
and they match the quick form measured earlier (8 rows 9.456 ms then, 10.225 ms now; `block` 93.1 then,
93.2 now; `head` 68.0 then, 68.5 now), so the two paths agree.

It also confirms the run that once looked like a hang is simply long: the process sits in
`waitUntilCompleted` for most of 78 minutes, `user` time is 82 s.

## Round 16: what the repaired head still cannot do, and what that rules out (2026-11-23)

The head now fills all 106299 rows, so the remaining question is no longer "can the head reach the right
token" but "why does the drafter rank the target's own tokens so low". Three things were measured; all
three are negative results, and together they narrow the gap to one link.

**1. The 5-token prompt is not the reason.** Every acceptance figure on this page came from the 5-token
prompt `3 1 42498 456 11`. The same command with a 24-token prompt (that sequence plus the model's own
continuation) at `--max-new 8`:

```
speculation              8 rounds of 8, drafts accepted 0 of 49 (0.000), 1.00 tokens per round
output  : 639 1320 96302 76 76 78 88 13535
decode                   8 tokens in 16644.9 ms  ->  0.48 tok/s
```

0 of 49 again, from a context five times longer, so "the drafter had nothing to go on" is out.

**2. The tap layer is not off by one — the other convention is worse.** HF's `hidden_states[i]` is the
*input* of decoder layer `i` (the output of layer `i-1`), while the engine taps a layer's *completed
output*, so `target_layer_ids` `[5, 15, 25, 35, 45]` could be meant as layers `[4, 14, 24, 34, 44]`. A
temporary `STRATA_DF2_TAP_SHIFT=-1` build (not committed) tested exactly that, and the truth ranks got
worse on 5 of 7 rows:

| draft row | rank, tap = output of layer l | rank, tap = output of layer l-1 |
|---|---|---|
| 0 | 457 | 779 |
| 1 | 2084 | 9152 |
| 2 | 2282 | 2887 |
| 3 | 15620 | 47886 |
| 4 | 33206 | 68550 |
| 5 | 91 | 293 |
| 6 | 10 | 14 |

Acceptance was 0 of 49 in both, and the target's own output was unchanged
(`1544 76 398 8709 96302 17776 78 76`), which is the expected behaviour: the taps feed the drafter only.

**3. The head is not context-blind.** `tools/dflash2/cmp_dumps.py` (new) compares two dumps. Between the
two tap sets above, the taps differ by cosine 0.90–0.96 (adjacent layers are this correlated — which is
why experiment 2 is a weak discriminator), the block's rows by 0.79–0.89, and the head's logits by
0.84–0.88 **with the argmax changing on 5 of 7 rows** (62/62, 16/12, 12/16, 17/16, 17/17, 11/17, 11/17).
So the logits do follow their input; the frequent ids that keep winning (`11`, `12`, `16`, `17`, `62`)
win because they are frequent, not because the head ignores `x`.

**What is left.** The drafter's own arithmetic is verified against the torch port on the same taps (block
rows cos 0.9992–0.9995, selector walk 7 of 7), its head now covers its whole vocabulary, and its logits
respond to the context. The one unverified link is the **fold that turns the target's hyper-connection
streams into the vector the drafter was trained on**: `df2_tap` writes
`(Σ_c R_c + bo·Σ_c g_c) / hc` with `g_c = 2/(1+exp(-inj_c/hc))`, and nothing on this machine can confirm
that this is the quantity the community drafter expects — the reference port consumes the engine's own
taps, so it agrees by construction. Two probes follow from that, in order:

* dump the target's `R'` at the *start of layer 6* and the tap written for layer 5 into the same buffer,
  and compare them: they must be bitwise the same vector. If they are not, the tap read is racing the
  layer's writes, which is the defect class this card already demonstrated once in `df2_attn_part` /
  `df2_attn_merge`.
* if they are the same, the formula itself is the variable: try the `g`-weighted read
  `Σ_c g_c (R_c + bo) / Σ_c g_c` and the plain `Σ_c R_c / hc`, and compare truth ranks. Ranks respond to
  tap changes (experiment 2 moved them by 2–4×), so they are the metric to use, not acceptance.

Not claimed: that any of these is the cause, that acceptance can be made non-zero on this card, and any
timing as a performance number. The drafter's cost is unchanged at ~0.48 tok/s decode with speculation
offering nothing.

## Round 17: the tap conventions are not the discriminator (2026-11-23)

Round 16 named the tap fold as the last unverified link. This round tested every axis of it that can be
reached without an outside oracle, and none of them moves acceptance. All numbers are the documented
Qwen3.8 `UD-IQ1_M` dflash2 run at `--max-new 8` (8 rounds of 8), and "rank" is
`tools/dflash2/rank_true_token.py`: where the token the target actually emitted ranks among the head's
logits.

| tap convention / head | truth ranks (7 rows) | acceptance |
|---|---|---|
| output of layer l, gathered 106299-row Q8_0 head (default) | 457 / 2084 / 2282 / 15620 / 33206 / 91 / **10** | 0 of 49 |
| the target's whole 248320-row head (`STRATA_DF2_HEAD=native`) | 521 / 2544 / 2746 / 24615 / 61052 / 92 / **10** (of 248320) | 0 of 49 |
| output of layer l-1 (`STRATA_DF2_TAP_SHIFT=-1`) | 779 / 9152 / 2887 / 47886 / 68550 / 293 / 14 | 0 of 49 |
| output of layer l+1 (`STRATA_DF2_TAP_SHIFT=+1`) | 99 / 920 / 3462 / 15651 / 27275 / 154 / 14 | 0 of 49 |
| plain mean of the streams, no `bo` term (`STRATA_DF2_TAP_PLAIN=1`) | 382 / 3156 / 2563 / 20201 / 51856 / 86 / **7** | 0 of 49 |

(The three `STRATA_DF2_TAP_*` switches were temporary experiment builds and are **not** in the tree; only
the defaults are committed.)

What that says, and the one real divergence it found:

- **The fold is not a Metal invention.** `src/kernels/cuda/dflash2.cu`'s `tap_gr_kernel` is the same
  formula as the port's `df2_tap`, term for term: `(Σ_c R_c + bo·Σ_c g_c)/hc`, `g_c = 2/(1+exp(-inj_c/hc))`.
  The port reproduces the CUDA **decode** path exactly.
- **But CUDA's prefill path uses a different fold.** `sycl/src/prefill/prefill.cpp:4950` calls
  `df2_tap_rows` — commented "this layer's completed output, mean of the streams" — which is
  `Σ_c R_c / hc` with **no `bo` term**; its kernel is `tap_rows_kernel` in the same CUDA file. So CUDA
  emits one quantity for a prompt's rows and another for a decode window's, and the Metal port applies
  the decode fold on both paths. That is a real divergence (worth aligning whichever way the drafter
  wants), and the `STRATA_DF2_TAP_PLAIN` row above is what the prefill fold does: rank 7 on the last row
  (the best single rank seen) but 51 856 on row 4, and no acceptance. So it is not the fix either, and
  the divergence is recorded rather than guessed at.
- **The gathered draft vocabulary is irrelevant to acceptance.** With the whole 248320-row head the
  proposals are *identical* (`[626, 28, 16, 17, 18, 471, 16]`) and the ranks scale to the same weakness,
  so the `--mtp-draft-vocab` list is not standing between the drafter and a draft.
- **Layer indexing is not it, in either direction.** -1 is clearly worse; +1 is better on 3 rows
  (99, 920, and 14) and worse on 2. No shift changes acceptance.

**Net effect on the objective's question.** Every link that can be checked against a reference on this
machine now matches it: the drafter's arithmetic against the torch port (block rows cos 0.9992-0.9995,
walk 7 of 7), the tap fold against the CUDA decode kernel (identical), the head's coverage against its own
size (106299 of 106299 after the Round 15 fix), and the vocabulary path against the whole head (identical
proposals). Acceptance is still 0, so the term that remains is the *content* of the tap vectors — whether
the target's post-layer, stream-folded hidden states are what this drafter was trained on. Nothing on this
machine can answer that: the only target pack that fits in 8.6 GB is IQ1_M (1.06 bpw experts), no
qwen4exp reference implementation is available to produce an independent `h`, and the reference port for
the drafter consumes the engine's own taps, so it agrees by construction.

The probes that would answer it, in the order I would run them: (1) the same drafter and prompt against a
less quantized target on a machine that can hold one, and compare the truth ranks — a jump would put the
blame on IQ1_M, no jump on the fold/layer semantics; (2) a write-race check on the tap read (dispatch
`df2_tap` again at the start of the next layer, where `R_` is untouched, and diff), the one sequencing
defect class this card has already shown; (3) the community's own target-extraction snippet, to settle the
fold and the layer indexing by construction rather than by rank.

Not claimed: acceptance, a speculative speedup, or that the taps are wrong — only that no convention
reachable on this machine changes the outcome, and that the port now matches its reference everywhere it
can be checked.

## Round 18: the taps are reproducible, and the matrix-unit prompt path is gated off on this card (2026-11-23)

### 1. Item 3's last mechanical suspect is retired: the taps are bit-reproducible

Two runs of the documented Qwen3.8 `UD-IQ1_M` dflash2 command (`--max-new 2 --prefill 1`, dump round 0),
separated by a **reboot** (so cold page cache and different scheduling), compared with
`tools/dflash2/cmp_dumps.py`: **cos 1.000000 on all 5 tap rows, all 8 block rows and all 7 logits rows,
|d| rms 0.0000, identical argmax on every row.** A `df2_tap` read racing the layer's producers would not
reproduce bit for bit across a reboot, so the tap kernel reads settled data. That was the second of the
three probes Round 17 listed; it is now closed, and the tap content question is left to the other two.

### 2. Item 6: the 4.34 GB matrix-unit prompt buffers are real, allocated up front, and a loss on this card

The engine names the allocation itself, so this no longer needs `vmmap` archaeology:

```
strata metal: prompts on the matrix units in chunks of 4096 (experts v2, QSA v1, matmul2d layout other;
              4.34 GB of buffers, in the expert arena while a prompt runs)
strata metal stream: the arena is small: the prompt's 4.34 GB of buffers are their own allocation,
                     not a region of its slots
```

It is `PromptGemm` (`metal/src/engine/window.cpp:2869`+), whose buffers `prompt_gemm_init` allocates. Two
things were **not** what the earlier page assumed:

- **The path is not dead on this card.** `prompt_gemm_on()` looked only at `STRATA_METAL_PROMPT_GEMM` and
  `n >= 64`, so a 90-token prompt really does run it (prefill 32.1 s, 2.80 tok/s). Its "matmul2d layout
  other" is the scalarised layout of the matrix units, and it works. Gating it off therefore needed a
  measurement, not the assumption that a 64-lane card cannot use it.
- **On this card it costs both time and memory.** Same 90-token prompt, identical output `4960 78`:

| | prefill 90 tokens | footprint after fill | expert arena |
|---|---|---|---|
| PromptGemm on (old default) | 32.10 s (2.80 tok/s) | 15.3 GB | 2.02 GB, 1152 slots |
| `STRATA_METAL_PROMPT_GEMM=0` (window path) | 25.30 s (3.56 tok/s) | 14.2 GB | 1.00 GB, 637 slots |

So the path now requires the same 32-lane simdgroup the drafter already requires
(`df2_bind`): `prompt_gemm_hw()` is `cx_->simd_width() == 32`, and both the load-time init in
`stream_arena` and the on-demand selection in `prompt_run_rows` consult it.

**The first attempt crashed, and that is the useful part of the record.** Gating only the load-time init
left `prompt_run_rows` with its own env-only branch: for `n >= 64` it called `prompt_gemm_init` lazily with
no headroom left and died with **SIGSEGV at address 0x21 inside `prompt_run_rows`**. A gate that covers only
the allocation is not a gate. With both sites consulting the same predicate, the same 90-token prompt takes
the window path:

```
64-lane simdgroups: the matrix-unit prompt path is off (the window path is faster and smaller here)
expert arena 1.00 GB, 637 slots ... resident footprint after the fill: the process holds 14.2 GB
prefill                  90 tokens in 26357.2 ms  ->  3.41 tok/s
output  : 4960 78
```

26.36 s against the 25.30 s `PROMPT_GEMM=0` control (the 4% is run-to-run spread on this machine), the same
tokens, 14.2 GB, and no crash. Net on this card: **~18% faster prompt prefill and 1.1 GB less resident
memory**, on buffers that until now were allocated at load and — for every prompt under 64 rows, which is
every draft run on this page — never touched.

What this does **not** do: grow the expert arena. The arena went from 2.02 GB to 1.00 GB in the measured
runs, because the planner's slot reservation and the footprint feedback move together; the 4.34 GB were
partly arena-funded, so the win shows up as resident memory and prefill time, not as a bigger arena. Making
the arena take the freed bytes is a separate change to the planner, and it is not attempted here.

One stale message found on the way, recorded not fixed: for a native IQ pack without `--spec`, the engine
still prints that the alternative source "`--draft dflash2` with an external `--draft-model`, is CUDA/HIP
only and is not built for this backend" (`src/program/generate.cpp:4818`). It is false on this branch — the
same command with those flags runs the drafter here — and it is the kind of line that sends the next reader
down the wrong path.

## Round 19: the warm Qwen3.6 matrix reproduces, and the cold run is the outlier (2026-11-24)

Item 8 asked whether the stored warm matrix (`0.90-1.17 s/step`) could be reproduced, because this session's
single runs measured 1.30-1.81. It can, and the gap was the page cache, not the engine.

A fresh 3-run sweep (`python3 bench/macpro51/run_matrix.py --runs 3`, engine `build-metal/strata-q36-metal`,
24 tokens decoded per run, 21 runs, rc=0) measured every pack once on a cold cache and twice after it:

| pack | run 1 (cold) | run 2 | run 3 | mean of the warm runs | previously committed |
|---|---|---|---|---|---|
| Q4_K_M | 1.494 | 0.981 | 0.984 | **0.982 ± 0.002** | 0.980 ± 0.005 |
| UD-Q4_K_XL | 1.717 | 0.903 | 0.901 | **0.902 ± 0.001** | 0.905 ± 0.003 |
| Q6_K | 1.971 | 1.017 | 1.017 | **1.017 ± 0.000** | 1.017 ± 0.002 |
| UD-Q6_K_XL | 2.086 | 1.083 | 1.078 | **1.081 ± 0.004** | 1.097 ± 0.002 |
| Q8_0 | 2.283 | 1.171 | 1.173 | **1.172 ± 0.001** | 1.172 ± 0.003 |
| UD-Q2_K_XL | refused | | | | refused (Q5_K embed) |
| UD-Q8_K_XL | refused | | | | refused (BF16 down) |

So the warm matrix reproduces to **0.5-1.5%** on every pack, and both refusals reproduce with the engine's
own wording (`token_embd.weight is Q5_K, q36_embed only supports Q8_0, Q6_K and Q4_K`;
`blk.34.ffn_down_exps.weight is BF16, and this port has no grouped expert down kernel for that type`).

The one number that changes is the *cold* one, and it is worth writing down because it is what made this
session's earlier single runs look like a regression: **the first run of a pack costs 1.5-1.95x its warm
time** (1.494/0.981 = 1.52, 1.717/0.903 = 1.90, 1.971/1.017 = 1.94, 2.086/1.081 = 1.93, 2.283/1.172 = 1.95).
That is disk and page cache, not the GPU: each pack is a 19.7-36.4 GB file and its first pass reads it.

**A tooling fix came out of this, and it is the reason the numbers above are the warm ones.** The first
3-run sweep was about to be published as `mean 1.153 ± 0.295` (Q4_K_M) — a mean the machine never sustains
and an sd that is really just "one cold run". The repository's own protocol settles it:
`docs/BENCHMARK_PROTOCOL.md:96-97`, "Include warm-up time in `required_duration_s`, but **exclude warm-up
requests from performance and quality summaries**". So round 1 is now tagged `"phase": "warmup"` in
`results-qwen36.json` and excluded by `charts.py`; the CSV's `runs` column says 2, not 3, and the tooltip
says which. The cold run stays in the JSON, so nothing is hidden — it is excluded from the summary, as the
protocol says, not deleted.

Reproduce with:

```sh
rm -f bench/macpro51/results-qwen36.json        # the sweep resumes from existing records: it skips (pack, round) pairs it already has
python3 bench/macpro51/run_matrix.py --runs 3    # ~25 min here, 21 engine processes
python3 bench/macpro51/charts.py
```

## Round 20: the arena cannot be grown by freeing the prompt path, and a less-quantised target does not fix drafting (2026-11-24)

### 1. Item 6, second half: the arena is capped by the card's working set, not by the prompt buffers

Round 18 freed ~2.3 GB and the expert arena did not move (1.00 GB, 637 slots, both runs). The reason is in
`arena_budget.h:114-117`, which takes `max(1 GB, min(gpu, ram, foot))` with
`gpu = working_set - footprint_base - 1 GB` and `footprint_base = max(allocated, the process's footprint)`.
The log line never said which term bound, so the arithmetic was re-derived by hand each time — including the
Round 17 note that freeing 4.34 GB "would move the arena from 1.0 to ~5 GB", which is not possible. The three
bounds are now one shared struct (`ArenaTerms`) and the line names the binding one:

```
strata metal stream: only 1.0 GB left for the expert arena (the GPU working set 8.6 GB, RAM 137.4 GB, 4.6 GB allocated): expect a low hit rate
strata metal stream: the arena is capped by the GPU working set: working set 8.6 - base 10.9 = -3.3,
                     RAM 121.5 - keep = 100.3, cap 8.6 - base = -2.8 GB (base = max(4.6 allocated, 10.9 held))
```

Both the GPU and the cap terms are **negative**: the process holds 10.9 GB against an 8.6 GB working set, so
the arena is on its 1 GB floor and nothing freed elsewhere reaches it. The gate did shrink the base —
13.23 GB before it, 10.9 GB in this run, footprint 14.2 -> 11.9 GB — but the arena stayed at 1.00 GB, which is
what the formula says it must. To grow the arena on this machine you would have to shrink the 10.9 GB itself
(the 2.34 GB native dense set, the ~1.5 GB of dense pack weights, the drafter's 1.1 GB), or the driver would
have to recommend a larger working set; it will not, on an 8 GB card under macOS 11.

### 2. Item 3: the oracle was run — the target's precision is not the explanation

Round 17 left the tap *content* as the only unverified term, with two candidate causes: the target being
1-bit (IQ1_M) puts its hidden states off the drafter's training distribution, or the tap semantics are wrong.
The machine has a shelf of less-quantised Qwen3.8 packs after all (the earlier page's "only `UD-IQ1_M` was
tried; `Q8_0` is 133 GB" was about `Q8_0` alone), so the same drafter and prompt were run against
**`UD-IQ2_XS`** — the same layout (`pack/`, `*-metal.gguf`, the PLE shard), a 47 GB dense GGUF against 39 GB:

| head of the drafter, same taps machinery | truth ranks (7 rows) | acceptance |
|---|---|---|
| target = UD-IQ1_M | 457 / 2084 / 2282 / 15620 / 33206 / 91 / **10** | 0 of 7 |
| target = UD-IQ2_XS | 283 / 4957 / 4131 / 18475 / 47429 / 132 / **7** | 0 of 7 |

Mixed on individual rows, **no improvement in kind**: the truths still sit at ranks 100s-10 000s, the head
still prefers the same frequent punctuation/digit ids (`62`, `16`, `12`, `17`, `11`), and drafting still
accepts nothing. The target itself behaves consistently — both packs emit `1544 76` for this prompt.

And the two packs' taps are almost the same vectors, so the precision difference barely reaches them:

```
tap rows  cos 0.967 - 0.994   (rms 0.227/0.226, 0.265/0.268, 0.335/0.336, 0.320/0.317, 0.283/0.279)
block rows cos 0.960 - 0.987
logits     cos 0.957 - 0.974, argmax identical on rows 0 and 1
```

**That is a negative result for the quantisation hypothesis, and it is the useful kind:** a two-bit-better
target moves the hidden states by 0.6-3.3% in cosine and changes nothing about the drafted answer. So the
remaining explanation is the *relationship* between the taps this engine produces and the vectors the
drafter was trained on — the fold and the layer semantics — not the target's precision. The probes that would
settle that are the ones Round 17 named and this page still has not been able to run: the community's
target-extraction snippet, or an independent `h` for qwen4exp to compare the tap against.

Recorded because it cost a run and will be needed again: `UD-IQ2_XS` runs on this engine with the same
command, one pack swap (`--pack`, `--native`, `--native-dense-gguf`, `--ple-gguf` -> the `-00003-of-00003`
shard), 0.50 tok/s decode, 4.8 s prefill for 4 tokens.

## Round 21: the drafter's own capture patch settles the fold — and it was already measured (2026-11-24)

Round 17 and 20 both ended by naming "the community's target-extraction snippet" as the way to settle the tap
semantics. It exists, and it is authoritative: the drafter's model card
(`incoai/Qwen3.8-Flash-Next-DFlash2`, the pack this branch loads) ships `patches/qwen38-dflash2-capture.patch`
for SGLang, and the patch states the contract in four lines:

```python
assert list(layer_ids) == [5, 15, 25, 35, 45], layer_ids
...
if i in getattr(self, "_dflash_completed_layer_ids", ()):
    # DFlash2 was trained on completed, zero-based layer outputs,
    # averaged over the four hyper-connection streams.
    assert hidden_states.shape[-1] == self.hc_count * self.hidden_size
    aux_hidden_states.append(hidden_states.unflatten(-1, (self.hc_count, self.hidden_size)).mean(dim=-2))
...
print("Q38_DFLASH_CAPTURE completed_layers=[5,15,25,35,45] hc_mean=True features=12800", flush=True)
```

Three things follow, and two of them are already measured on this machine:

- **The layer list is right.** "completed, zero-based" `[5, 15, 25, 35, 45]` is what `tap_slot_` does after each
  layer completes, so the Round 16 layer-shift experiments were measuring a deviation from a correct choice.
- **The gather is right.** Five slots of 2560 = the patch's `features=12800`, and `fc.weight` consumes exactly
  that (verified in Round 15's dump: the drafter's context is 5 x 2560).
- **The fold is a plain mean over the four streams — no `bo` term.** `mean(dim=-2)` over the hyper-connection
  streams is `Σ_c R_c / hc`. The engine's `df2_tap` writes `(Σ_c R_c + bo·Σ_c g_c) / hc`, and so does the CUDA
  **decode** kernel `tap_gr_kernel`; only the CUDA **prefill** kernel `tap_rows_kernel` matches the patch.
  **That variant was already measured** (Round 17, `STRATA_DF2_TAP_PLAIN=1`): truth ranks
  382 / 3156 / 2563 / 20201 / 51856 / 86 / **7** against the default's 457 / 2084 / 2282 / 15620 / 33206 / 91 /
  **10**, acceptance 0 of 49 either way. So adopting the patch's fold changes the ranks by noise and does not
  produce a draft.

What that leaves is narrower than before, and it is **not** the fold, the layer list or the target's precision
(Round 20). What is still unverified is whether the engine's `R_` at the moment `df2_tap` reads it is the
*completed layer output* in the patch's sense — the patch reads `hidden_states` at one specific point inside
the layer, and a tap that fires one sublayer early or late would carry a neighbouring quantity, which a
stream-mean of the right form would not reveal. That is Round 17's probe 2 in its only remaining form:
dump `R_` at the start of layer 6 and compare it, vector for vector, with the tap written for layer 5.

**And a note on the `bo` term, because the two CUDA paths disagree with each other:** on the evidence of this
patch, `tap_rows_kernel` is the correct one and the port (like CUDA's decode path) is the deviant. Aligning
`df2_tap` to the plain mean is therefore a faithfulness fix worth making even though it does not move
acceptance — but it should be made together with the measurement above, not instead of it, so the change
lands with evidence rather than with a citation.

External sources are recorded by URL only; nothing above was taken on trust from them, and the numbers are
this machine's.
