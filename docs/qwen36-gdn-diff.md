# WIP: the GDN's SiLU output gate (round 64, reverted in round 63's tree)

The defect is confirmed (NEXT.md round 63): qwen35moe needs `y = rms(o) * gamma * silu(z)` and `metal/kernels/gdn.metal` implements
`sigmoid(z)`. The change was attempted and reverted because it did not compile; this file is what the next attempt needs.

## What was tried

1. `metal/kernels/gdn.metal`, near the top:
       constant bool GDN_SILU_C [[function_constant(26)]];
       constant bool gdn_silu = is_function_constant_defined(GDN_SILU_C) && GDN_SILU_C;
2. the gate line (gdn.metal:~168):
       const float sg = 1.0f / (1.0f + exp(-zz));
       const float yv = ((rs * o) * gamma[j]) * (gdn_silu ? zz * sg : sg);
3. `q36_gdn_layer.cpp`: `k.step = cx.pipeline(k.lib, "gdn_step", mtl::Constants().i(2, 1).b(26, true));`
4. `q36_gdn_layer_parity.cpp`'s reference: `... * zz * sg` instead of `... / (1.0 + exp(-zz))`.

## Why it failed

The pipeline compile said `use of undeclared identifier 'gdn_silu'` - **the file has the gate line MORE THAN ONCE** (a second variant -
the `XH16`/plane path, or a second `gdn_step_t` body), and the replacement hit every occurrence while the `constant bool` declaration only
exists once, in a scope the later body does not see. `grep -c "1.0f + exp(-zz)" metal/kernels/gdn.metal` gives the count; the fix is to
put the declaration where every use can see it (file scope, outside any `#if`), or to declare it again in the second region.

## The discipline that must come with the fix

`gdn_parity` must gain a **silu case** (it currently tests the sigmoid step, which is why upstream's test never caught this), and
`q36_gdn_layer_parity`'s reference must use silu - which step 4 above did. Then `metal/tests/q36_real_compare.sh PACK` answers whether the
output becomes text.

Tree is green at the revert; nothing of this is committed.

## ADDENDUM (round 65): a second GDN defect, found the same way

`metal/kernels/gdn.metal:132-144` applies the `1/sqrt(S)` scale to BOTH q and k:

    const float nv = (rs * cv) * scale;
    if (tid < 128) qs[tid] = nv;  else ks[tid - 128] = nv;

`q36ref.cu:220-221` applies it to the query only:

    q[i] = qi / sqrtf(qn + eps) / sqrtf((float) D);
    k[i] = ki / sqrtf(kn + eps);

Fix: `else ks[tid - 128] = rs * cv;` (no `scale`), plus a `gdn_parity` case that pins k's scale and a reference in
`q36_gdn_layer_parity` to match. Settle the disagreement against llama.cpp (`ggml_l2_norm` and the delta-net graph in `qwen3next.cpp`)
before landing, exactly as the silu question was settled.


## STATE AFTER ROUND 67: three defects fixed, the step's core verified, the conv still to diff

Landed (function constants, undefined by default, so no other family changes):
    26  silu(z) output gate                     (round 64)
    27  the key is normed, not scaled           (round 66)
    28  the scale lands once (on q), not twice   (round 67)
All three verified: `q36_gdn_layer_parity` 4.85e-07, state 2.34e-07, `ctest -R q36_` 17/17.

Verified as matching the reference by reading, not by measurement:
    the decay `exp(softplus(alpha + dt) * ssm_a)`                 kernel line ~130   == q36ref:224
    `beta = sigmoid(beta)`                                        ~131                == :226
    `S = g*S + k*delta`, `delta = (v - g*(S k))*b`                ~160                == :231-236
    `o = rms(o) * gamma * silu(z)`                                ~168-171            == :249
    the conv is 4 taps with a 3-deep history, silu'd              header comment      == :200-209

STILL TO DIFF (in the order worth doing):
    1. ~~the CONV's tap order~~ CHECKED (round 69): the kernel uses `{hst[0], hst[1], hst[2], in}` - current input on
       tap 3, history 0..2 oldest first, depth 3 - which is exactly q36ref.cu's `w[c*K + K-1]` / `w[c*K + k]` / `state[k]`
       rotation.  No defect.
    2. ~~the conv history's depth~~ CHECKED (round 69): 3 on both sides.
    3. the attention layer (the other 10 of 40): rms(.) * q_norm/k_norm, NEOX over the first 64, GQA 16/2,
       `y = W_o (softmax(q k^T / 16) v * sigmoid(gate))` - and the gate: q36ref says SIGMOID for attention (contrast
       with the GDN's silu, which is what made round 63 hard to see).
    4. the MoE's `top-8, renormalised` (verified on the real pack: weights sum 1.0000) and the shared expert's
       `sigmoid(w_shexp_gate . h2)`.

The method that found three defects in three rounds: read q36ref.cu's header (eight lines), read the kernel, and compare
OPERATOR BY OPERATOR - then fix behind a function constant with the reference changed to match, so the fix is measured.
