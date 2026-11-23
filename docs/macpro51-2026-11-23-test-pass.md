# macpro51: a full test pass at `5360e6292`, 2026-11-23

"Full" on this page means: build the branch, run the whole `ctest` suite, drive each model family's
CLI path as far as this machine allows, and run the diagnostics the port is judged by. Every number
is from that run's own output on macpro51; nothing was inferred. Timings are single cold runs and
are not performance claims.

**The short version: the engine works. 12 of the 18 tests pass, and five of the six failures are
stale tests or this port's readback design, not engine faults. One new defect: `df2_bench` hangs in
its 8-row context update.**

## The machine and the build

| | |
|---|---|
| Host | MacPro5,1 ("macpro51"), macOS 11.7.11, 2 × Xeon X5675, ~96 GB DDR3 |
| GPU | AMD Radeon RX 580 8 GB, Metal 2, **64-lane simdgroups** |
| Branch / commit | `macpro51`, `5360e6292` (a docs-only commit on top of `0256a2ce1`); tree clean |
| Build | `PATH=/usr/local/bin:$PATH cmake --build build-metal -j 24` → `ninja: no work to do` |
| Engine | `strata 0.1.59-metal` |

The binaries in `build-metal/` are newer than the installed `/usr/local/bin/strata*` copies (those
are 2026-11-22 09:54); this pass used the build tree.

## The suite: 12 of 18, 74.27 s

| # | Test | Result |
|---|---|---|
| 1 | `q36_model_check` | **Failed** |
| 2 | `q36_attn_parity` | Passed 0.26 s |
| 3 | `q36_glue_parity` | **Failed** |
| 4 | `q36_rope_parity` | Passed 0.32 s |
| 5 | `q36_router_parity` | Passed 17.91 s |
| 6 | `q36_dense_parity` | Passed 0.21 s |
| 7 | `q36_attn_layer_parity` | Passed 0.89 s |
| 8 | `q36_gdn_layer_parity` | Passed 0.50 s |
| 9 | `q36_moe_shared_parity` | Passed 0.20 s |
| 10 | `q36_expert_blob_check` | Passed 0.01 s |
| 11 | `q36_moe_routed_parity` | Passed 25.82 s |
| 12 | `q36_layer_check` | Passed 0.36 s |
| 13 | `q36_bind_check` | Passed 0.48 s |
| 14 | `q36_forward_check` | **SIGSEGV** 2.65 s |
| 15 | `q36_stream_check` | Passed 20.03 s |
| 16 | `q36_generate_check` | **Aborted** 2.61 s |
| 17 | `q36_upload_check` | **Failed** 0.48 s |
| 18 | `runtime_hello` | **Failed** 0.19 s |

## The six failures

`377fdec0d` ("make every supported quant generate on the RX 580", 2026-11-18) introduced both engine
changes involved here — the private-storage allocator and the `q36_embed` type argument. The affected
tests were last edited in `be4cbc78f` (2026-11-21) and were not updated with it.

| Test | Observed | What it is |
|---|---|---|
| `q36_model_check` | one check fails: `token_embd.weight is ?, q36_embed only supports Q8_0, Q6_K and Q4_K`; it wants `Q4_1`. The other ~24 checks are `ok`. | `ref::type_name()` (`metal/src/ref_quant.cpp:20-31`) has no `Q4_1` case, so a real-but-unlisted ggml type prints `?`. Cosmetic message defect. |
| `q36_glue_parity` | `q36_embed token 0/63/7 … rel L2 1.000e+00 FAIL`; every `q36_add_rms`/`q36_add` case passes. | The test dispatches `q36_embed` with **4** args; the kernel takes **5** (`etype` at `buffer(4)`), so no branch runs and nothing is written. The engine passes it (`metal/src/qwen36/q36_forward.cpp:170`). Stale test. |
| `q36_forward_check` | SIGSEGV with no stdout (buffered). | The test `memset`/`ones()`es through `Buffer::ptr` on buffers from `upload_dense`. On this port `Context::alloc(bytes, src)` returns a **Private** buffer and sets `r.ptr = r.handle` as a sentinel (`metal/src/mtl_rt.mm:860`: *"Private storage has no CPU address; non-null sentinel passes upload checks"*). Writing floats over the object header crashes. |
| `q36_generate_check` | SIGABRT after `device AMD Radeon RX 580`. | Same cause (same `zero()`/`ones()` helpers). |
| `q36_upload_check` | `token_embd: differs at byte 0 of 16885760`, then `the bytes compared are the bytes allocated (0)`. | `memcmp(b.ptr, src, bytes)` reads the buffer object, not tensor bytes (sentinel again). The `(0)` is fallout from `same()` returning early. The test cannot pass on this port as written. |
| `runtime_hello` | `RESULT: 4 FAILED`, `65470/65536 wrong values`. | Readback of GPU-private storage — the pre-existing, documented limitation, not new. |

So: the four `q36_*` failures are test-side (three from the sentinel, one missing argument), the
`Q4_1` one is a one-line gap in a message table, and `runtime_hello` is the known readback design.

## The app on real models

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

The full Q8_0 sequence, for the record:

```
271 248068 198 8160 579 264 7047 1817 25 271 16 13 220 2972 2014 53983 2570 5396 64700 198 256 471 2972 14162
```

Reference: `bench/macpro51/results-reference.json` (llama.cpp, same 23 ids). The s/step figures are
one cold run per pack (experts streamed from the HDD RAID); the stored warm matrix is 0.90-1.17
s/step, so 1.30-1.81 here should not be read as a regression without a warmed re-run.

### Qwen3.8-Flash-Next UD-IQ1_M + DFlash2 — the documented command

```
strata metal: DFlash2: 64-lane simdgroups, so the matrix units are not used …
strata metal: DFlash2 drafter ready (1103 MiB …)
prefill 4 tokens in 1 chunks, 27222.7 ms (0.1 tok/s)
speculation  2 rounds of 8, drafts accepted 0 of 7 (0.000), 1.00 tokens per round
dflash2      231.281 ms/round drafting (1 rounds, GPU 216.892)
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
| `df2_bench metal/src/engine/msl` | **hangs** — see below |

### `df2_bench` hangs, reproducibly

The plain invocation blocks. Sampling the process puts the main thread in
`mtl::Context::run_graph → -[_MTLCommandBuffer waitUntilCompleted] → __psynch_cvwait`: a command
buffer that never completes (12+ minutes, 14 s CPU, 0 % CPU).

Stage isolation with `DF2B_ONLY=… DF2B_ITERS=1`, each bounded by an alarm:

* `ctx` — prints `context update, 1 rows 7.118 ms` and `3 rows 7.515 ms`, then **blocks at 8 rows**;
  the alarm killed it (rc=142).
* `block` — OK, 93.103 ms of GPU.
* `head` — OK, 68.032 ms of GPU.
* Both of its own kernel checks pass in every run: `df2_mm4_4_8 3.35e-07 OK`, `df2_mm_4_8 (Q8_0)
  4.31e-06 OK`.

So the hang is the **8-row context-update graph** specifically. The card is healthy afterwards
(`q36_attn_parity` and `q36_dense_parity` still pass), and the engine's own 8-row verify window ran
in the Qwen3.8 test above, so this is the bench's own graph, not the card and not the engine path.

## Not run here

* The benchmark protocol (`strata bench run`) — not present in this binary (exit 2).
* `--dn-prep` — the GPU-wedge combination, and pointless since the K-quant down fix.
* `STRATA_DS4_ALLOW_UNSTABLE_METAL=1` — reset risk; needs the owner's say-so.
* Logit-distance re-measurement (`logit_distance.py`) — would need fresh logit dumps.

## Follow-ups, in order

1. Fix the `df2_bench` 8-row context-update hang. It is the check the DFlash2 page calls "the kernel
   check the port is judged by"; the barrier/lane layout is the first suspect on a 64-lane card.
2. Update the four stale `q36_*` tests for this port: give `q36_upload_check`/`forward_check`/
   `generate_check` a host-visible readback path or skip them here, and pass `etype` in
   `q36_glue_parity`. Add `Q4_1` to `ref::type_name`.
3. Correct `docs/qwen36-metal.md`: it still reports `q36_upload_check` and `q36_model_check` as
   "all ok", which is no longer true at this commit.
4. Rebuild/reinstall `/usr/local/bin/strata*` if the installed copies are meant to track the branch.
