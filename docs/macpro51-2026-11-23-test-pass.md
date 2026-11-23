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
