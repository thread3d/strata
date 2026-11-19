# The Unsloth v3.1 packs on macpro51: a forward that is wrong inside layer 1

_Measured 2026-11-19 on the machine in [qwen38-metal-macpro51.md](qwen38-metal-macpro51.md). This page exists so the
defect does not have to be re-derived: it is the one thing that keeps four downloaded packs out of the benchmark
matrix, and every other explanation has been measured and excluded._

## What happens

Four packs — `IQ2_XS` (ISTA) on a missing kernel, and the three Unsloth v3.1 sets `UD-Q2_0`, `UD-IQ1_M`, `UD-IQ2_XS`
in the forward — either refuse to start or answer `76 76 1544 7509 …` where every pack that works answers
`271 248068 198 760 1156 6587 264 61446`. The harness's answer gate excludes them from the matrix; this page is
about the three that run.

## It is systematic to the revision, not to a pack

The first-token logits of the four v3.1 sets agree **with each other** at cosine 0.96–0.98 (UD-Q2_0/UD-Q2_K_XL
0.975, UD-Q2_0/UD-IQ2_XS 0.984, UD-Q2_0/UD-IQ1_M 0.982, UD-Q2_K_XL/UD-IQ2_XS 0.983), and none of them agrees with the
working Q2_0 (0.16–0.31).  Four packs with different expert encodings (IQ2_XXS, IQ2_S, IQ3_XXS, IQ2_XS, Q2_0, Q8_0
mixes) converging on the same wrong logits means the fault is **shared by the v3.1 revision** and pack-independent
once it fires — which is why the answer is the same `76 76 1544 7509 …` for all of them.  A weight- or
offset-specific mistake in one pack could not do that.

## The Q6_K head is not the cause (measured)

The refused sets' output head is a Q6_K tensor, the only one no answering pack uses — so it was tested directly:
`XA_` (the final normed row the head reads) and the logits were dumped from a UD-Q2_0 run, and the head's rows were
decoded in Python from the pack's own GGUF and dotted with that row:

| id | engine logit | Python Q6_K dot | difference |
| ---: | ---: | ---: | ---: |
| 76 | 13.2727 | 13.2727 | 0.0000 |
| 271 | 0.4965 | 0.4965 | 0.0000 |
| 1506 | 10.8821 | 10.8821 | 0.0000 |
| 1692 | 10.4968 | 10.4968 | 0.0000 |
| 1605 | 10.3406 | 10.3406 | 0.0000 |

The head decodes and multiplies correctly, so the wrong logits come from a wrong `XA_` — the forward — and not from
this tensor.  (The engine prints "experimental native Q5_K head" for it; the byte count it reports, 521,472,000, is
248,320 rows of 2,100 bytes, which is Q6_K's row size, so the label is stale, not the arithmetic.)

## The one path only these packs use: a Q6_K output head

Every pack that answers has a **Q5_K** (ISTA Q2_0) or **Q8_0** (UD-Q4/Q5/Q6) `output.weight`; every v3.1 set has a
**Q6_K** head.  That is the only tensor type the refused packs share and no answering pack exercises, and it fits the
logit evidence above: four Q6_K quantisations of the same head, decoded by the same (possibly wrong) code, give four
*similar* garbage logit vectors (0.96–0.98 with each other) that are unrelated to the correct logits (0.16–0.31).
A weight- or offset-specific mistake in one pack cannot do that; a wrong decode of one format can.

Two caveats keep this a hypothesis rather than a conclusion: `gemv_parity --one Q6_K 2560 512` did not finish in the
time allowed here (a separate thing to look at), and the residual-vector tap described below still shows layer 1
diverging — if the head is the whole story, that tap is measuring an artefact of its own cut placement.

## Where it breaks: layer 1, and only layer 1

Same prompt, same window, greedy, `STRATA_WIN_RDUMP` on the residual rows:

| cut | cosine vs Q2_0 | residual norm, UD-Q2_0 | residual norm, Q2_0 | max |
| --- | ---: | ---: | ---: | ---: |
| after layer 0 | **0.9945** | 0.375 | 0.389 | 0.084 |
| after layer 1 | **0.188** | **38.4** | 1.10 | **19.58** |

One row (the sampling window is one row), so this is not a batch artefact. Layers 2–47 never recover: the
`STRATA_WIN_LAYERCHECK` series stays near 22.

## What is *not* the cause (each measured)

- **The expert format.** The refused packs' types (16/20, 16/42, 22/20, 22/42, 22/6) all appear in the answering
  `IQ3_XXS` and `IQ3_S` packs, which answer correctly.
- **The quant decode.** `expert_parity --gu IQ2_XXS --dn Q2_0` passes on the refused pack's **own**
  `-metal.gguf` layer-0 experts (h rel L2 1.33e-07), and at every GU tile (32,1,4 / 32,2,4 / 32,4,4 / 16,1,4 /
  8,1,4).
- **The token embedding.** A full search of UD-Q2_0's 248,320 IQ4_XS rows for UD-Q4_K_XL's Q8_0 row 814 lands on
  **id 814 at cosine 0.9967**; the tokenizers and all but one metadata field are identical across the revisions.
- **The PLE block.** Its outputs (`keyn`, `qn`, `gated`, `normW`) match the healthy packs to three significant
  digits, and swapping in Q2_0's PLE shard changes nothing.
- **The n-gram table, the prompt path, the tile screen, the constant table, fp16 experts.** A swapped table, `--prefill 1`,
  four `STRATA_METAL_MOE_GU` tiles, `STRATA_METAL_CTAB=0` and
  `STRATA_METAL_F16=1` all give the same wrong answer.
- **The fold kernels as a 64-lane reduction bug.** `gr.metal`/`gr2.metal` already take `sg32`/`lane32`, reduce with
  `sum32` and write all eight `part[]` slots.

## What is left

At layer 1's cut the fold's own inputs match a working pack while the residual it produces cannot be made from
them (`STRATA_WIN_EXPDUMP`):

| | UD-Q2_0 | Q2_0 |
| --- | ---: | ---: |
| `bo` (attention output folded) rms | 0.063 | 0.129 |
| `ysh` / `ye` rms | 0.089 / 0.0041 | 0.088 / 0.0037 |
| injects (gate 0..2) | 3.5 / 8.2 | 14.1 / 26.0 |
| `hc_ffn_norm.weight` first values | 0.107 … | 0.0199 … |
| **residual produced: rms / max** | **0.390 / 22.6** | **0.021 / 0.393** |

With |`bo`| ≤ 0.54 times a gate ≤ 2 folded onto a 0.375-norm residual the result must be under 1, so the fold
**reads or writes the wrong data** for this pack rather than computing a different function — the same signature as
the `gr_norm2` reduction that started this port. The two things not yet tapped:

1. the `W_` offset / tensor identity the window binds for `hc_ffn_norm.weight` and `hc_ffn_up` in a `--pack` run
   (pack `dense.bin` layout vs the GGUF's tensor table),
2. the `inj_[cur_inj_]` parity the fold reads, across windows.

A tap that prints the bound offset and the first values of `w_norm` as the kernel receives them would name it in one
run; `--gguf`/`--layer` in `metal/tests/expert_parity.cpp` shows the harness for that kind of check.

## Reproduce

```sh
cd ~/strata-bench
D=~/models/qwen38/UD-Q2_0
STRATA_METAL_STREAM=1 STRATA_WIN_LAYERCHECK=1 ./build-bench/strata --pack "$D/pack" \
  --native "$D/UD-Q2_0-metal.gguf" --ple-gguf "$D/Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf" \
  --native-dense-gguf "$D/UD-Q2_0-metal.gguf" \
  --tokens 814,20139,11,303,1330,2716,22157,11 --max-new 1 --max-context 512 --spec 1 \
  --suffix-draft 0 --lookup-max-t 0 --prefill 8
```
