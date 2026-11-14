# Qwen3.6-35B-A3B (Q8_0) on Metal: what "no backend" means, and the port's plan

Written 2026-11-14, after the branch moved to upstream `13eedf55` (Strata 0.2.13, engine 0.1.59).
Everything below is read from the code at that commit unless it is marked **measured**, in which case it
was run on the Mac Pro (2 x 6-core Xeon X5675, 24 threads, 96 GB, Radeon RX 580 8 GB, Metal 2,
macOS 11.7.11) on this date, on the merged tree, and the output is quoted.

## 1. The gate the user hit

There is no "Qwen3.6 is unsupported" message, because Qwen3.6 support exists — on the CUDA and HIP
builds only. The gate is a compile-time define:

* `src/program/generate.cpp:3398` — `#if defined(STRATA_HAVE_QWEN35)`: the **family switch** reads the
  file's architecture before any engine state exists and hands a `qwen35moe` file to
  `strata::qwen35::run()` (`src/qwen35/q36_engine.cu`, entry via `include/strata/qwen35/engine.hpp`).
* `STRATA_HAVE_QWEN35=1` is set in exactly one place: `cmake/strata_gpu_engine.cmake:518`, the CUDA/HIP
  engine target. `metal/CMakeLists.txt` never sets it, and `src/qwen35` is CUDA throughout — it opens
  with `#include <cublas_v2.h>`, `<cuda_fp16.h>`, `<cuda_runtime.h>` and builds against
  `strata::kernels::native_mmvq` / `native_moe` / `native_router` / `fused_gdn` / `prefill::mmq`.
* `sycl/src/program/generate.cpp:1782` records the same fact for the SYCL build: "`src/qwen35` is CUDA
  and HIP only".

So on the Mac build the switch is compiled out, the file falls through to the Flash-Next (Qwen3.8)
engine, and that engine refuses it on architecture. `docs/MODELS.md:269` states it plainly: "the Intel
Arc engine and the Mac engine do not run it". That is the "no backend" the user saw: not a missing
flag, a missing implementation.

## 2. What already works on this card (measured today)

The kernels a Qwen3.6 port needs are Strata's own MSL in `metal/kernels/`, not ggml's (`GGML_METAL` is
forced OFF in `metal/CMakeLists.txt:83`). The parity harnesses for the parts of that set a MoE
transformer shares were run on the merged tree today and all pass:

    gemv_parity         ALL PASS   (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K..Q6_K, IQ4_NL, IQ4_XS, IQ2/3,
                                    BF16/F32; worst rel L2 3.54e-07 over Q6_K's 20 tiles)
    gdn_parity          ALL PASS   (y rel L2 1.6e-07, state 7.8e-08, history exact,
                                    verify-leaves-state and partial-commit bitwise)
    expert_parity       PASS       (Q2_0 gate/up and down, 10 groups, h rel L2 1.88e-07)
    ds4_dq_parity       PASS       (Q4_K, Q5_K, Q6_K, Q8_0, BF16 vs ggml's dequantize_row)
    ds4_route_guard_parity  all ok (incl. the router NaN flag added upstream this round)
    ds4_pnorm_parity    ALL PASS   (hc_pre_norm rel L2 7.9e-08)

That matters because Q8_0 is the format the user asked for, and it is in `gemv_parity`'s type list:
the dense Q8_0 matvec is a tested building block here, not something to write.

## 3. What actually has to be built

Three things, in dependency order. None of them is a kernel rewrite.

### (a) A q36 host driver for the Metal runtime

`src/qwen35/q36_engine.cu` is 3,970 lines of host code: GGUF load and geometry checks, expert blobs
repacked at load, the layer loop, chunked prefill, the decode window, the MTP draft layer, sampling,
`--serve` (README/GEN/T/DONE/STOP/PP), `--tokens` and the bench modes. Its Metal counterpart has to be
written against `metal/src/mtl_rt.h` (`mtl::Context`: `compile`, `pipeline`, `alloc`, `wrap`, `run`,
`Dispatch`, `Arg`) exactly as `metal/src/engine/window.cpp` is for Flash-Next.

The dispatch discipline is not optional on this card — `runtime_hello` still fails three of its four
checks here and MAC.md's numbers (65470, 65536, 65470 wrong values of 65536) still reproduce:

* no device pointers inside kernel arguments (a Metal 2 `contents()` is not `gpuAddress()`);
* no mapped-buffer readback;
* no CapturedGraph replay with live data.

The DeepSeek port's precedent is `metal/src/engine/ds4_finish.cpp`: job structs carry offsets into
bound allocations, and ranges are bound from the allocation, never from a page inside it
(`cpu_for_wrap` / `wrap_span` / `span_off`).

### The kernel inventory is now complete (rounds 120-122)

A layer of qwen35moe is: `h = rms(x) * norm`; `y = mixer(h)`; `x += y`; `h = rms(x) * post_norm`; `y = moe(h)`; `x += y`. Every
step of that now has a kernel that is parity-tested on this card:

| step | kernel | verified |
|---|---|---|
| residual + rmsnorm | `q36_add_rms` (`metal/src/qwen36/msl/q36_glue.metal`) | `q36_glue_parity`, 4.6e-08 at n=2048 |
| qk-norm + RoPE | `q36_head_norm_rope_{f32,bf16}` (`q36_rope.metal`) | `q36_rope_parity`, 6.6e-08; bf16 KV exact |
| attention | `q36_attn` (`metal/src/qwen36/msl/q36_attn.metal`) | `q36_attn_parity`, 3.8e-07 at 16/2x256, 37 rows |
| GDN mixer | `gdn_step_t` (`metal/kernels/gdn.metal`) | `gdn_parity`, y 1.6e-07 |
| router (256, top-8) | `moe_route` (`metal/kernels/moe.metal`, its `n_expert == 256` path; `ROUTE_K` from `GEO_TOPK`) | `expert_parity`, `moe_mm_parity` |
| grouped experts, Q8_0 | `moe_gu` / `moe_down` (`metal/kernels/moe.metal`) | `expert_parity --gu Q8_0 --dn Q8_0`, h 1.52e-07 |
| shared expert gate+up | `gemv_pair` (silu fused, `metal/kernels/gemv.metal`) | `gemv_parity` |
| shared expert down | `gemv` | `gemv_parity`, Q8_0 rows |
| combine + gated shared | `moe_combine_sh` (`metal/kernels/glue.metal`) | `moe_mm_parity` |

So the remaining work is the **driver** that puts them in that order, not arithmetic.

### (b) A dense-attention kernel for q36's shapes

`metal/kernels/qsa*.metal` is Flash-Next's attention: an indexer selects ~2048 cells out of a long
context (`qsa_score`, `qsa_topk`, `qsa_attn`, `qsa_merge`). Qwen3.6 has no indexer: 16 query heads over
2 KV heads, dense causal attention, split-K over 256-cell chunks with a log-sum-exp merge (q36_engine.cu's
own description). `qsa_merge`'s LSE merge is the same arithmetic; the score/selection half is not
wanted at all. This is the one genuinely new MSL file.

### (c) Qwen3.6's own geometry and layer algebra

`metal/src/engine/geo.h` compiles the kernel set for one shape (`strata::core::kCompiledGeometry`,
`cmake -DSTRATA_GEOMETRY[_FILE]`), with Flash-Next's value as every `#ifndef GEO_*` default. Flash-Next
and Qwen3.6 do not share a shape:

| | Flash-Next (what `geo.h` is written for) | Qwen3.6-35B-A3B |
|---|---|---|
| residual | hyper-connections, 4 streams, low rank 320 | plain pre-norm residual, no HC |
| experts | 512 routed, 10 per token | 256 routed, 8 per token, plus a shared expert with a sigmoid gate |
| attention | indexer, 24 q heads / 2 kv, head width 256 | no indexer, 16 q heads / 2 kv (from the GGUF: `qwen35moe.attention.head_count` = 16) |
| mixer stack | 36 GDN + 12 full-attention | GDN + full attention per `full_attention_interval`, plus an MTP draft layer |

So `geo.h` needs a Qwen3.6 column, and the layer drivers need q36's algebra (a plain residual, the
shared-expert gate, the q36 router rules) rather than Flash-Next's HC/PLE/QSA steps. The primitives
underneath — `gemv*`, `moe_route/moe_plan/moe_gu/moe_down/moe_combine`, `gdn_step_t`, `sample`,
`dense_mm`/`mma` — are reusable, and (a)/(b) are what put them together.

## 4. Progress, slice by slice

| # | slice | state |
|---|---|---|
| 1 | **The family switch on Metal says what it means.** `model_family_not_supported()` now picks its bracket by build (`STRATA_GPU_METAL`), so the Mac build says "the Mac (Metal) engine has no path for that family" instead of naming a SYCL port that is not the one refusing. | **done** — `2c80f101`, built, `strings` checked |
| 2 | **`coop_k4()` on a Metal without MPP.** The probe asked for `coop_layout_16_64_32` unconditionally, on a kernel the port's `STRATA_HAVE_MPP` guards leave out; the lookup threw before any model was touched. It now answers "not the M5's layout" (the M4 path), as `metal/tests/coop_probe.h` always did. | **done** — `861c2a2b`, built |
| 3 | **A q36 host half**: `metal/src/qwen36/q36_model.{h,cpp}` reads a Qwen3.6 GGUF and decides everything decidable from the header — geometry, layout over every shard, the kernel gate (head 256, state 128, conv 4, ≤512 experts, top-k ≤15), every tensor per layer, the embedding/dense/expert split, one expert's slab and the expert bytes a decoded token must read — and refuses, naming the tensor and its type, any weight the port's kernels do not instantiate. `metal/tests/q36_model_check.cpp` drives it on tiny synthetic files with no GPU. | **done** — `f0ecdb1e`, built and `all ok` |
| 4 | **One verified numeric rung** at q36's real widths: the Q8_0 dense matvec against a CPU decode of the same file. (`gemv_parity` already covers Q8_0, and `expert_parity --gu Q8_0 --dn Q8_0` was measured on this card today: h rel L2 1.52e-07 at one token, 1.47e-07 at eight.) | next |
| 5 | **The dense-attention MSL**: `metal/src/qwen36/msl/q36_attn.metal` — one 32-lane group a query head, online softmax over the window, the output gate fused, bf16 rows read without a bfloat type, correct on 32- and 64-lane simdgroups. `metal/tests/q36_attn_parity.cpp` compares it against a double reference at the family's 16/2/256 and at small widths, gated and not, over 0, 1, 37, 300 and 2048 rows. | **done** — `948c8e7a`, built, `ALL PASS` |
| 6 | **The device half**: `metal/src/qwen36/q36_device.{h,cpp}` uploads every dense tensor into runtime-allocated buffers, byte for byte (a Q8_0 matrix keeps the file's own blocks), and deliberately does **not** allocate the 34 GB of experts — they stay in the file's mapping for the streaming path. `metal/tests/q36_upload_check.cpp` compares every buffer against its tensor and fails on the first differing byte. | **done** — `a6b56f0b`, built, `all ok` (34,571,920 bytes compared) |
| 7 | **The dense matvec call**, once: `metal/src/qwen36/q36_dense.{h,cpp}` — `DenseMatrix`/`DenseScratch`/`DenseKernels` and `dense_gemv()`, the tile choice, the prepared-activation planes and the argument order for every projection of every layer, refusing a missing kernel by name instead of dispatching empty. `q36_dense_parity` checks it against a CPU dequantisation at q36's widths (2048x64, 2048x512, 512x32, 320x16). | **done** — `c87cc089`, built, `ALL PASS` |
| 8 | **The GDN layer** (30 of the 40): `metal/src/qwen36/q36_gdn_layer.{h,cpp}` — four projections (alpha/beta RAW, because `gdn_step_t` applies softplus + dt + ssm_a itself), the step kernel committing state and history, the output projection. `q36_gdn_layer_parity` checks it against `gdn_parity`'s own reference at 1 and 3 tokens, plus the four projections. | **done** — `4501d511`, built, `ALL PASS` (y 2.3e-07) |
| 9 | **The MoE block**, then the **layer loop** that alternates them with `q36_add_rms`: `q36_add_rms` → mixer (`gdn_step_t` or `q36_attn`) → `q36_add_rms` → MoE (`moe_route`/`moe_gu`/`moe_down`/`moe_combine_sh` + `gemv_pair`/`gemv` for the shared expert) → residual; then the expert streaming path, chunked prefill, the decode window, and `--serve`. | next |

Both landed checks are `ctest` targets (`5946af6a`): `ctest -R q36_` runs the host half with no GPU and the kernel against
its reference.

Each lands as its own commit on `macpro51`, built on the Mac before it is pushed.

## 5. What is not claimed here

* No Qwen3.6 model has run on this card. The Q8_0 pack (37.8 GB, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`)
  is **not** downloaded here; the only model on the machine is the DeepSeek IST-DAS 2.00 pack.
* The parity results in §2 are for the shared kernels on synthetic shapes, not for Qwen3.6 tensors.
* Nothing above says the port will be fast. The card's measured decode is ~0.57 tok/s on DeepSeek 2.00
  bpw with ~1 s a prompt token (NEXT.md rounds 96-106); a 37.8 GB Q8_0 model has less weight traffic
  per token than that pack, but no measurement exists.

## 6. The next slice in detail: wiring one attention layer

Written down because it is the first place the measured parts must agree on buffer layouts, and the one decision that is not
obvious from any single file.

**The arithmetic**, in q36_engine.cu's order, with the kernels this port has:

| step | call | note |
+|---|---|---|
| `qg = W_q . h` | `dense_gemv` | `attn_q` is `[n_embd, 2 * n_head * head_dim]`: the queries first, **then the output gate** |
| `k = W_k . h`, `v = W_v . h` | `dense_gemv` | `[n_embd, n_kv_head * head_dim]` each |
| `q = rope(rms(q_head) * q_norm)` | `q36_head_norm_rope_f32` | grid `n_head` threadgroups of `head_dim` |
| `k = rope(rms(k_head) * k_norm)` | `q36_head_norm_rope_bf16` | written straight into the KV ring |
| `v -> bf16` | **a kernel that does not exist yet** | `q36_store_bf16(device const float*, device uint16_t*, uint n)` — five lines; `q36_head_norm_rope_bf16` cannot stand in, it would rotate `v` |
| `a = attn(q, k, v)` | `q36_attn` | `sigmoid(gate)` is already folded in |
| `y = W_o . a` | `dense_gemv` | `attn_output` is `[n_head * head_dim, n_embd]` |

**RESOLVED (round 127):** the ring is **token-major** and `q36_attn` reads it that way (`(r * KVH + kvh) * QD`), so one
`q36_head_norm_rope_bf16` / `q36_store_bf16` dispatch writes a whole token's KV heads. The value half of an append is
`q36_store_bf16`, now beside the attention kernel and checked bit for bit. The reasoning below is kept because it is what the
decision cost, and the same trade may come up again.

**The decision that is not obvious: the KV ring's layout.** `q36_attn` indexes the rows head-major —
`kb + (kvh * n_kv + r) * head_dim` — which is what its parity program builds. A per-token append wants token-major —
`(r * n_kv_head + kvh) * head_dim` — because that is the only layout in which one `q36_head_norm_rope_bf16` dispatch can write
all of a token's KV heads, contiguous, at the row the token belongs to (the kernel writes head `h` at `h * head_dim` from the
base it is given). Two ways out, both honest:

* **token-major ring, and change `q36_attn`'s two index expressions** (one line each, plus the parity program's reference
  which already computes `(kvh * NKV + r) * QD`). One dispatch an append, the layout every KV cache in the port uses.
* **head-major ring, and dispatch the append once per KV head** with a base pointer per head (`raw + kvh * QD`, `out +
  (kvh * cap + row) * QD`, grid 1). No kernel change, `n_kv_head` dispatches a token — 2 here, which on this card is 2 x
  ~0.4 ms of dispatch cost per token, and dispatch cost is exactly what rounds 92-97 of this port spent its time merging away.

The first is the better trade and the one to do; it costs one kernel edit and its parity reference.

**Two things to thread through rather than hard-code**: the layer's `eps` (currently a literal in `q36_attn_layer`'s draft -
it must come from the model, `qwen35moe.attention.layer_norm_rms_epsilon`, which `q36_model.h` already reads into
`Model::eps`) and the KV ring's `cap` (the window the run is given, not the model's maximum).

**The test that would catch a wiring error** needs no model: tiny Q8_0 matrices at the family's shapes (`n_embd` 2048,
`n_head` 16, `n_kv_head` 2, `head_dim` 256), one token, `n_kv` 1..3, and a CPU reference that dequantises the same bytes
(`ref::dequant`) and runs the same steps in double — the pattern `q36_dense_parity` and `q36_rope_parity` already use.

## 7. The next slice in detail: the GDN layer (30 of the 40)

The kernel is already there and already parity-checked (`gdn_parity`: y 1.6e-07, state 7.8e-08, history exact, verify and
partial-commit bitwise). What is not settled is what feeds it. Written down from the code, so the next round starts at the
decision rather than the search.

**`gdn_step_t`'s contract** (`metal/kernels/gdn.metal:57`), in buffer order:

| # | buffer | note |
|---|---|---|
| 0 | `state` | fp32, per layer, carried across tokens |
| 1 | `hist` | the conv history, `[channel][3]` a layer |
| 2 | `qkv` | the projection, `channels` = `(2 * h_k + h_v) * 128` = 8192 for this family |
| 3 | `alpha` | **already transformed** (see below) |
| 4 | `beta` | the same |
| 5 | `z` | the gate projection's output, `h_v * 128` = 4096 |
| 6 | `conv_w` | `ssm_conv1d.weight` |
| 7 | `dt` | `ssm_dt.bias` |
| 8 | `ssm_a` | `ssm_a` |
| 9 | `gamma` | `ssm_norm.weight` |
| 10 | `y` | out: `h_v * 128` |
| 11 | `GdnArgs{h_k, h_v, channels, commit, eps, plane}` | `commit` 1 to write state+history, 0 to verify; `plane` 0 for y only |
| 12 | `hist_out` | |
| 13 | `rinfo` | optional, a function constant |

Launched with **512 threads** a threadgroup, `h_v` threadgroups.

**CORRECTED (round 132) - the answer to question 1 below was WRONG, and the correction is the important part.**

`gdn_step_t` **does** transform `alpha` and `beta` itself. It carries `dt` and `ssm_a` as its own arguments and uses them
(`gdn.metal:65` `alpha[..] + dt[h]`, `:69` `exp(sp * ssm_a[h])`, `:70` `sigmoid(beta[..])`), so the layer passes the **raw**
`ssm_alpha.weight · h` and `ssm_beta.weight · h` projections. The `k_gdn_ab` / `k_gdn_gb` transform on the CUDA side is the
*reference* path's shape, not the Metal kernel's contract; feeding pre-transformed values would apply softplus, `dt` and `ssm_a`
twice - an over-decayed state, no NaN, and nothing to see until a model generated nonsense. The `q36_gdn_ab` kernel added in
round 131 is removed (`930cfa2e`).

**What round 130 got right** (kept for the record, with question 1 superseded):

1. **`gdn_step_t` does not transform `alpha`/`beta`; the caller does.** The CUDA engine has a kernel for exactly this,
   `k_gdn_ab` (`q36_engine.cu:183`), and its rule is two lines:

       alpha[j] = softplus(W_alpha[j] . x + dt_bias[j]) * ssm_a[j]
       beta[j]  = sigmoid (W_beta [j] . x)

   (read off `k_gdn_ab`'s body: `alpha[...] = softplus(acc + dt[r]) * ssm_a[r]` for the alpha row, `beta[...] = sigm(acc)` for
   the beta row). So the port needs one small kernel - call it `q36_gdn_ab` - taking the two projections, `ssm_dt.bias` and
   `ssm_a`, and writing the `h_v`-wide `alpha` and `beta` rows. Two rows a token, no reduction; it belongs beside
   `q36_add_rms` in `q36_glue.metal`.

2. **`gdn_step_t`'s instantiations pin the inner block, not the head counts.** It is `gdn_step_t<4>` (`gdn_step`) and
   `gdn_step_t<8>` (`gdn_step8`) - the template parameter is IB, the state rows a thread keeps - while `h_k` and `h_v` are
   `GdnArgs` run-time fields. The state width is `constexpr int S = 128` inside the kernel, which is exactly this family's
   `ssm.state_size`, so q36's 16 K groups / 32 V heads are inside what the kernel takes.

3. **`gdn_step_t` is launched with 512 threads a threadgroup, `h_v` threadgroups** (the CUDA `k_gdn_seq_cols` dispatch is the
   same shape), `commit` 1 to write the state and history, `plane` 0 for a plain `y`.

So the layer is now fully specified: `dense_gemv` for `attn_qkv` (→ 8192 = `channels`), `attn_gate` (→ 4096 = `z`),
`ssm_alpha`, `ssm_beta`; `q36_gdn_ab` for alpha/beta; `gdn_step_t`; `dense_gemv` on `ssm_out`; `q36_add_rms`. **DONE (round 131)**: `q36_gdn_ab` is in `q36_glue.metal` and parity-checked (alpha 5.7e-08, beta 2.7e-08), so the layer is now
only its wiring and a composition test.

The layer around it is short: `dense_gemv` on `attn_qkv.weight` → `qkv`, on `attn_gate.weight` → `z`, on `ssm_alpha.weight` and
`ssm_beta.weight` → the alpha/beta projections; `gdn_step_t`; `dense_gemv` on `ssm_out.weight` → the layer's output; then
`q36_add_rms` for the residual. The composition test follows the pattern that found three bugs last round: a CPU reference
over the same bytes, and the intermediates (the projections, the state after one token, y) compared one boundary at a time.

## 8. The MoE block: what is pinned, and what is left

**Measured (round 15).** `moe_route` at q36's shape works, and two requirements that are not in the file's metadata are now
pinned by a test (`q36_router_parity`, 9/9 on the machine):

* the logits are read at a **512-float row stride whatever the expert count**, and only the first 256 values (`lane + 32*i`,
  `i < 8`) are looked at - so a 256-expert model places its router output at 512 floats a token and leaves the rest alone (the
  test fills that half with 50.0 and proves no id comes from it);
* `ROUTE_K` comes from `GEO_TOPK` in the host prelude, so the library must be compiled with `#define GEO_TOPK 8`.

**The rest of the block's contract, read off `expert_parity`** (which drives every one of these kernels and passes on this card):

| step | kernel | arguments |
|---|---|---|
| plan | the plan kernel (K.plan) | `{ids, wts, plan, MoeArgs}`, 512 threads, no grid (one threadgroup) |
| gate/up | `moe_gu` | `{W, prepared x plane, h, plan, MoeArgs, s16}` - **indirect grid** from `plan + P_IGU*4` |
| down | `moe_down` | `{W, h or prepared h, ye, plan, MoeArgs, s16}` - **indirect grid** from `plan + P_IDN*4` |
| combine | `moe_combine_sh` | `{ye, plan, ysh, sg, bo, GlueArgs}`; `plan[P_TOKENT + t*topk + j]` is entry j's row in ye |

`P_TOKENT = 532` (gr2.metal names it "moe.metal's plan layout"); the plan buffer also carries the indirect grid sizes and the
per-expert entry table, which is why the two expert dispatches read their threadgroup count from it.

**What is left for the MoE block**: the shared expert (`gemv_pair` for gate+up with the SiLU fused, then `gemv` for the down),
its gate (`ffn_gate_inp_shexp`, a vector dot), and wiring the three routed experts steps plus the combine with the router above
- then the layer loop that alternates `q36_add_rms`, the GDN or attention mixer, `q36_add_rms` and this block.

## 9. The MoE block: two of its parts are already verified, and how

**The shared expert's gate+up is `gemv_pair`, and `pair_parity.cpp` already checks it against double** - so the MoE block does
not need a kernel here, it needs the wiring. The engine's own call is the recipe (`metal/tests/token_chain.cpp:564-580`):

    name   "gemv_pair_<type_gate>_<type_up>"   (a function constant 0 = n_tok)
    args   { W + gate_off,  XF + plane*xkind(tgs.type),
             W + up_off,    XF + plane*xkind(tus.type),
             XF + plane*4,                       // the 16-wide sums
             XH,                                 // h, raw (the consumer reads raw activations)
             GemvArgs{N, NFF, ntok, row_bytes(tgs.type, N), N, NFF},
             rb_up = row_bytes(tus.type, N) }
    grid   NFF / 4 threadgroups, 128 threads

with the silu(g) * u fused in, so the down projection follows as a plain `dense_gemv` on `XH`.

**`XF` is a prepared-activation buffer** - the same `DenseScratch.xp` the port's `dense_gemv` already fills - and `xkind()` is
`q36_dense.cpp`'s (0 raw, 1 Q2_0, 2 Q4_K/Q5_K, 3 Q6_K/Q3_K). So the shared expert is: `dense_gemv`'s prep already run, one
`gemv_pair` dispatch, one `dense_gemv` for the down, and the gate is a vector dot of `ffn_gate_inp_shexp` against the same
normed row (a `gemv` with n_out = 1 through the same machinery).

**DECIDED (round 18): a blob IS a slot** - `metal/src/qwen36/q36_expert_blob.h` holds the layout arithmetic and `pack()`, the
one function that knows both layouts, checked by `q36_expert_blob_check` (11/11 on the machine). The note below is what the
decision was made from.

**What the routed half needs**: the experts must reach the kernel as **per-expert blobs**
(`moe_gu` reads expert e at `W + e * gu_stride + gate_off / up_off`, the down at `+ down_off`), which is not the file's layout -
its three stacked tensors are expert-major, so a slice is contiguous *within* one tensor but the three are separate. Repacking
into the blob is a copy per expert (`expert_parity` does it on the host: `blob = down_off + n_embd * d_row`,
`up_off = n_ff * gu_row`, `down_off = 2 * n_ff * gu_row`), and on this card that copy *is* what the streaming path will do when
it fills a slot - which is why the blob layout and the slot layout must be the same thing, decided once, in the streaming
design rather than here.

## 10. The layer that alternates them: what the parts do not own

Starting `q36_layer.{h,cpp}` (the layer: norm → mixer → norm → MoE → residual) found that the parts between them do not carry
three things a layer needs.  Written here so the next attempt starts with them rather than rediscovering them:

* **the two norm weights**, `attn_norm.weight` and `post_attention_norm.weight`, as *device buffers with offsets* - the device
  half (`q36_device.h`) uploads them into `DeviceLayer`, but no part's weight struct takes them;
* **the normed row `h`** (n_embd floats) - every part takes it as an input and none of them owns it;
* **the shared gate** (one float: `ffn_gate_inp_shexp · h`, a `dense_gemv` with n_out = 1, which this port's tiles do take) and
  **the MoE's output** (n_embd), which the combine writes and the residual then adds.

And the primitive the chain itself needed: **a plain `x += y`**, because two of the layer's four joins are not followed by a
norm - the MoE's output goes back into the residual, and the norm that follows belongs to the *next* layer, whose
`q36_add_rms` fuses it with its own mixer.  `q36_add` is that four-line kernel (bit for bit in `q36_glue_parity`).

So the layer is: `join(h, attn_norm, add = not first)` → mixer (`attn_forward` or `gdn_forward`, its output read from the
part's own scratch) → `join(h, post_norm, add)` → `shared_forward` → `dense_gemv(gate_inp_shexp → 1)` → `routed_forward(h,
ysh, sg → moe_out)` → `q36_add(x, moe_out)`.  Nothing new is computed; the work is buffers and order.


## 11. CORRECTION to round 29 (round 30)

Round 29's note that "tied logits make the router pick outside its window" is **wrong** and is retracted here.  The test had passed
`n_expert = 8` to `moe_route`, and the kernel picks its window from that field: 256 selects `route_tok<8>` (a 256-value window),
anything else `route_tok<16>` (512 values).  With 8 the kernel read 256 zeros past the logits the test had written, and returned
legitimate indices for the window it was given.  `q36_stream_check` now routes 256 experts with the eight real candidates on top
and checks the ids reach the host as 0..7, in order.

The lesson, which is the general one: **ask the suspected code with the arguments a real model would give it** before recording a
finding against it.

## 12. What is now verified against the CUDA engine, line by line

Every item below was checked against `src/qwen35/q36_engine.cu` (or measured on the real pack) rather than inferred, because the
synthetic parity tests compare each part with a reference THIS PORT WROTE - so a misreading of the rule agrees with itself and passes.
The output is still not coherent, so this list is the map of what is NOT the defect.

| rule | CUDA engine | port | verdict |
|---|---|---|---|
| the residual norm | `k_add_rms`: `x += y`; `r = rsqrtf(ss/n + eps)`; `h = x * r * w` | `q36_add_rms`: identical | **identical** |
| the MoE block's tail | `k_moe_out`: `x += sum_k w_k part_k` in pick order, FMA after the first product, `+ sigmoid(sgate) * shared`; then `h = rms(x) * wn` with the NEXT layer's norm | `q36_add` (plain add) then the next layer's first join with `add = false` | **equivalent** |
| the q projection's layout | `qg[(t*H+head)*2*HD]` for q, `+HD` for the gate - per head `[q \| gate]` | was contiguous halves; fixed (round 44) | **fixed** |
| the key projection | `attn_k` is a plain `[n_embd, KVH*HD]`, no gate | query stride was wrongly used on it; fixed (round 46) | **fixed** |
| the in-file MTP block | a draft head, evaluated separately | was run as trunk layer 41; fixed (round 46) | **fixed** |
| the embedding row | `token_embd` row `t`, `t * row_bytes` | `q36_embed`, compared with the file's own bytes on the real pack | **bit-exact (rel L2 0.000e+00)** |
| the attention layer | - | `q36_attn_layer_parity` at pos 0, 3 and 37 | **1.1e-07 .. 1.5e-07** |
| the geometry read from the pack | - | 40 trunk + 1 MTP, 30 GDN / 10 attention, 16q/2kv x 256, 256 experts top-8, 3.19 MiB a slab | **matches** |

Still suspects, unverified against anything but this port's own references: **the GDN layer's q/k/v split and conv state** (its parity test
uses a 2-group/4-head shape, not the pack's 16/32), **the routed MoE's expert order** (the router's ids against the slot each expert
loaded into), and **the final norm with the output projection's orientation**.

The comparison that decides them is one command: `metal/tests/q36_real_compare.sh PACK` (round 49) - the same prompt ids on both sides,
and the first token that disagrees is the first part that is wrong.

## 13. Where this stands, and how to continue

**What works** (all measured on the Mac Pro, Radeon RX 580, macOS 11.7.11): the Qwen3.6-35B-A3B Q8_0 pack (37.8 GB) opens, uploads its
2.52 GiB dense half in 3.8 s, streams its 32.67 GiB of experts through an 8-slot pool refilled per layer from the file, and generates tokens
at ~1.0 s a step with 64.6 MiB of engine state.  Entry point: `build-metal/strata-q36-metal --model PACK --tokens a,b,c --n N --text`.

**What does not**: the tokens are not text.  The engine runs and is not correct.

**What is verified against the real pack** (never wrong, unlike the three retracted findings):

| check | command | result |
|---|---|---|
| the embedding row | `--check-embed N` | bit-exact, rel L2 0.000e+00 |
| the output projection on a 248,320-row matrix | `--check-output K` | worst rel 4.6e-07 |
| every dense binding of a layer, norms included | `--check-bind L` | 0 wrong of 13 |
| one expert's gate/up/down vs a double reference | `--check-expert E --h-file F` | agrees |
| the routed MoE at the pack's shapes (2048/512/256/8) | `q36_moe_routed_parity` | 2.2e-07, arena == resident bit-exact |
| the reference's own tokenization compared with ours | `tests/q36_real_compare.sh PACK` | **disagrees from the first token** |

**The three real defects found, all in the GDN step** (function constants 26, 27, 28 in `kernels/gdn.metal`, all undefined-by-default so no
other family changes, all set by `q36_gdn_layer.cpp`): the output gate is `silu(z)` not `sigmoid(z)`; the key is normed and NOT scaled by
`1/sqrt(D)`; the scale lands once (on q), not twice (q and o).  Each was found by diffing `src/qwen35/q36ref.cu`'s eight-line statement of
the step against the kernel — **that is the method to keep using, and it should be continued on the attention layer next.**

**The rule that separated real findings from three false ones**: a comparison against **the file's own bytes** or against a **double
reference computed on the same inputs** has never been wrong; every false finding (rounds 29, 56, 60) came from reading an *internal buffer
or field* and assuming its meaning.  `s.ye`, the plan's slots, and a synthetic blob's "file" all lie by default.

**Where to look next**, in order:

1. **the attention layer** (10 of 40) against `q36ref.cu`'s block: `rms(.)*q_norm/k_norm`, NEOX over the first 64 with base 1e7, GQA 16/2,
   `y = W_o (softmax(q k^T / 16) v * sigmoid(gate))` — note the sigmoid here, unlike the GDN's silu;
2. **the same treatment for the MoE**: `top-8, renormalised` (verified: weights sum 1.0000) and `sigmoid(w_shexp_gate . h2)`;
3. **the layer loop's composition** (`q36_forward.cpp`): it skips the MTP block, adds each layer's MoE with a plain `q36_add`, and runs the
   final norm once - all checked against `q36ref.cu`'s structure, none checked against a second implementation;
4. a **host reference for one whole layer** on the real weights (the `--check-expert` pattern taken one level up) - the only measurement that
   would localise a defect inside a layer rather than at its edges.

**The comparison that decides every fix**: `metal/tests/q36_real_compare.sh PACK` — llama.cpp tokenizes the prompt, this engine runs those
ids greedily, llama.cpp runs the same text; the first token that disagrees is the first part that is wrong.
