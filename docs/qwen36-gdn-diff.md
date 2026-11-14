# The GDN step against the reference: what the diff found, and the two "fixes" that were wrong

`metal/kernels/gdn.metal` is shared with the DeltaNet families, and Qwen3.6-35B-A3B (qwen35moe) differs from them in the step's output gate.  This
document records the operator-by-operator diff of the kernel against `src/qwen35/q36ref.cu`'s statement of the family's step, and - more usefully -
the two changes that diff produced which were NOT defects, and how that was finally established.

## The gate: `silu(z)`, not `sigmoid(z)` - right, and landed

The family's step ends `o = rms(o) * ssm_norm * silu(z)`; the kernel implemented the DeltaNet `sigmoid(z)`.  Three sources agreed
(llama.cpp's `ggml_silu` in `build_norm_gated`, this repository's `include/strata/kernels/fused_gdn.hpp:35` which names qwen35moe for
`fused_gdn_step_norm_silu`, and `q36ref.cu`), so the kernel gained a **function constant 26**, undefined by default - every other caller of
`gdn.metal` keeps the semantics it was built with - and `q36_gdn_layer.cpp` asks for it:

    constant bool GDN_SILU_C [[function_constant(26)]];      // file scope: the first attempt put it beside XH16, inside a guard,
    constant bool gdn_silu = is_function_constant_defined(GDN_SILU_C) && GDN_SILU_C;   // and the step body could not see it
    const float sg = 1.0f / (1.0f + exp(-zz));
    const float yv = ((rs * o) * gamma[j]) * (gdn_silu ? zz * sg : sg);

Verified by the layer's parity test against a silu reference (4.85e-07), and by the whole-model comparison below.

## ADDENDUM (round 65): a "second defect" that was NOT a defect - withdrawn

`gdn.metal` applies the `1/sqrt(S)` scale to **both** q and k, where `q36ref.cu:220-221`'s prose puts it on the query only:

    q[i] = qi / sqrtf(qn + eps) / sqrtf((float) D);
    k[i] = ki / sqrtf(kn + eps);

On that reading the port switched the key's scale off (function constant 27) and, two rounds later, the output's scale too (constant 28).
**Both were wrong.**  The comparison that settles it is llama.cpp's, not the prose's:

    with 27 and 28 both off:   layer 0  cos 0.3355   ||ours|| 5.017  vs ||llama|| 1.172
    with the kernel's defaults: layer 0  cos 0.9999   ||ours|| 1.174  vs ||llama|| 1.172
                                layer 5  cos 1.0000
                                layer 10 cos 1.0000

The kernel was right in both places all along, and each "fix" made the model wronger.  The measurements that would have shown it in one run
(`tools/qwen35/q36_llama_dump.cpp` and `Q36_DUMP_LAYERS`, compared per layer) took fifty rounds to reach.

### The one change that was right

| constant | change | verdict |
|---|---|---|
| 26 `GDN_SILU_C` | the output gate is `silu(z)`, not the DeltaNet `sigmoid(z)` | **right** |
| 27 `GDN_KSCALE_C` | "the key is normed, not scaled" | **wrong - withdrawn** |
| 28 `GDN_OSCALE_C` | "the scale lands once, on q" | **wrong - withdrawn** |

`q36_gdn_layer.cpp` sets **only constant 26**.  27 and 28 are left at the kernel's defaults, which is what the model wants; the constants still
exist, so a future family that does need the DeltaNet form can ask for it.

### What the operator-by-operator diff did establish (all still true)

    the decay `exp(softplus(alpha + dt) * ssm_a)`            kernel ~130   == q36ref:224
    `beta = sigmoid(beta)`                                   ~131           == :226
    `S = g*S + k*delta`, `delta = (v - g*(S k))*b`           ~160           == :231-236
    `o = rms(o) * gamma * silu(z)`                           ~168-171       == :249
    the conv: 4 taps, history 3 deep, oldest first, current input on tap 3     == :200-209
    q and k L2-normed, both scaled by 1/sqrt(D); the output rms-normed and silu-gated

and separately verified on the real pack, each by its own measurement:

* the **embedding** row, bit-exact against the file;
* the **output projection**, worst rel 4.6e-07 on a 248,320-row real matrix;
* every **dense binding** of a layer, including its norms, byte-exact;
* the **router's** top-8, renormalised so the weights sum to 1.0000;
* the **carried GDN state** after a step, six digits against a double reference;
* the **attention layer**'s projections and output (the factor of 16 that looked like a defect was the host's misplaced scale);
* the **per-layer buffers**: every layer has its own state, history and ring, and the kinds are `l % 4 == 3`, as the reference states.

### The lesson this document is here to record

Every parity test in this port passed while the model was wrong, because **each test's reference was written by the same hand that misread the
prose** - so both sides agreed.  A self-consistent check cannot find a shared misreading.  What can is a comparison against an authority that
shares no reading with the port: for this family, llama.cpp's own dumped layer outputs, which is `tools/qwen35/q36_llama_dump.cpp` in this
repository and was there from the start.
