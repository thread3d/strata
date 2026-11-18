# Qwen3.8-Flash-Next on macpro51: the Metal bring-up

_Measured 2026-11-18 on the machine below. Qwen3.8-Flash-Next now runs on the RX 580: it reads a prompt, decodes
coherent tokens, and the quantisation matrix is in [macpro51-qwen38-benchmarks.md](macpro51-qwen38-benchmarks.md).
This page records the bring-up: what it took to get there, the two defects that blocked it, and what is still out of
reach._

## The machine

| | |
| --- | --- |
| Mac | MacPro5,1, 2 × 6-core Xeon X5675 (12 cores / 24 threads), 128 GB DDR3 ECC |
| GPU | AMD Radeon RX 580, 8 GB, Metal 2 (no Metal 3, no bf16, no device addresses) |
| OS | macOS 11.7.11 (Big Sur, 20G1443) |
| Engine | `strata` 0.1.59-metal, built from the `macpro51` branch with Ninja, Release |
| Model | `~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-{00001,00002}-of-00002.gguf` (37.6 GB + the 28.8 GB PLE table) |

The pack is the ISTA-DASLab GSQ-RCO Q2_0 set: `general.architecture = qwen4exp`, 48 trunk layers (36 GDN, 12 QSA at
3, 7, … 47), 512 routed experts of width 640 with 10 used, 4 hyper-connection streams of rank 320, and the PLE n-gram
table in the second shard. Its tensor table matches the shape the Metal engine compiles for (`kCompiledGeometry`), and
`tools/iq_pack.py` builds the metadata pack beside it (`native_experts.txt`, `index.txt`, `dense.bin`, `tokenizer/`).

## What runs

The engine opens the pack, builds the pack metadata, reads the dense side and the expert arena, compiles its kernels,
reads the prompt, and reaches the decode window. Measured on the card:

| step | measurement |
| --- | --- |
| dense side read into host memory | 789 tensors, 2.36 GB, 1.8 s |
| expert arena | 2.20 GB, 1578 slots of the profile's top experts, filled in 1.0 s (2.25 GB/s) |
| decode window ready (kernels compiled) | 3.4 – 41 s depending on the kernel cache |
| prompt of 31 tokens | 9.3 s → 3.33 tok/s, read as one chunk through the engine's prompt path |
| decode | 1.06 tok/s (32 greedy tokens, one window each, ± 0.001 over four steady runs) |
| expert reads per decode window | 3.1 GB in 2,213 reads (Q2_0, with the shipped profile; hit rate 1.9%) |

And it answers. With the pack's own tokenizer for the prompt, a 16-token greedy run on Q2_0 continues

```
\n\n thinking\nThe user wants a concise explanation in two short sentences about why the
```

which is the model reasoning about the question it was asked. The three quantisations the branch runs (Q2_0,
IQ3_XXS, IQ3_S) decode at 1.06, 0.90 and 0.84 tok/s; the matrix is [here](macpro51-qwen38-benchmarks.md).

## The first non-finite value: found and fixed

The forward used to go non-finite at the first full-attention layer (layer 3): `qcur`, `qidx` and every
projection after them were NaN, while the residual stream stayed finite. A runtime probe at the norm that
feeds that layer (a cut inserted after `gr_read(hc_attn)`) showed the norm's own inputs finite — the
residual 1.26, `bo` 2.38, the inject 25.9 — and its output NaN. That is the signature of a bad reduction:
the kernel's total is a sum over threadgroup partials, and the partials were not all written.

`gr_norm2` (the HC read the window uses by default, `STRATA_METAL_LDB`) declared `threadgroup float part[8]`
but took its group and lane from the hardware simdgroup. A 256-thread threadgroup on this card is four
64-lane simdgroups, not eight 32-lane ones, so only `part[0..3]` were written and the total added four words
of threadgroup memory left over from earlier dispatches. When those held a NaN bit pattern, `rsqrt` of the
total was NaN and the normed row — and every projection after it — was NaN; the fold into the residual is a
separate store, so the residual stayed finite. That also explains why the first bad layer moved between
builds: the stale words depend on which kernels ran before.

The fix is the same one the rest of the port already carries: `gr_norm2` takes `sg32`/`lane32` and reduces
with `sum32`. Measured after it, on Q2_0:

| check | result |
| --- | --- |
| residual after all 48 layers | finite, max 0.44 at layer 3 (`STRATA_WIN_LAYERCHECK`) |
| 3 greedy tokens | 1.00 tok/s, output `32946 13 248046`, no non-finite values |
| prompt (31 tokens) | 3.2 tok/s |

The second defect surfaced right after it: the **fourth decode window** died with SIGSEGV at address 0x0 in a
worker thread. The one-shot loop starts its expert-adaptation thread when `(rounds + 1) % --adapt-every` is due —
the fourth window at the default interval — and `adapt()` indexes `host_res`, the host residency table, which the
Metal window never has (it streams its own experts). The thread read address 0. It is now started only when that
table is non-empty (all three spawn sites), and 8, 16 and 32 tokens decode end to end.

A third, quieter defect was in the benchmark, not the engine: the prompt ids were Qwen3.6's. The two vocabularies
share a size (248 320) but not their order, so those ids render as punctuation in Qwen3.8 and the model was
answering gibberish. The harness now encodes its prompt with the pack's own tokenizer
(`tools/strata_tokenizer.py`), which is also why the 16-token sample above is coherent.

## What was ruled out along the way

Every check below was run on this card and passed, so none of them was the cause (the list is kept because
it is the fastest way for the next person to stop re-testing them):

- **The hyper-connection kernels**: `gr_parity` and `gr2_parity` **ALL PASS** for 1, 3 and 8 tokens, apply 0
  and 1, Q8_0 and BF16. (The parity test passes *because* fresh threadgroup memory is zero there — the
  fault needs a stale NaN in the four words that were never written.)
- **The GDN layer**: `gdn_parity` ALL PASS; **the prompt path**: `prompt_parity` ALL PASS; **the Q2_0 expert
  and GEMV kernels**: `pair_parity` ALL PASS, including "Q2_0 raw-x GEMV 640 → 2560".
- **The RoPE table** (cos/sin, base 1e7) and **every F32 tensor** in the file (all norms incl.
  `blk.3.attn_q_norm`): 0 non-finite values.
- **The Q2_0 weight data**: `blk.3.attn_q.weight` occupies exactly 12,288 × 720 bytes and every one of its
  491,520 block scales is finite (max 0.88).
- **The device's simdgroup width**: 64 lanes (the runtime's own probe).

## Engine changes made for this card

These are on the branch, all behind `STRATA_GPU_METAL`, so the CUDA/HIP paths are unchanged:

1. **A native pack needs `--native` only.** The shared host code required `--spec T (T >= 2)` and `--prefill CHUNK`
   for a native `--pack` before it could start. The Metal decode window reads the model's GGUF itself, one token
   window at a time, and has neither a device residency table for `--spec` nor a batched prompt path of its own, so a
   native pack is now accepted with `--native SHARD1` alone there.
2. **A spec window does not need the device residency table on Metal.** The window streams its own experts and keeps
   its own state, so the `thits.d_res == nullptr` refusal is a CUDA/HIP check; it is compiled out on Metal.
3. **A token id outside the vocabulary is refused, not dereferenced.** A head whose logits are all non-finite returns
   `0x7fffffff` from its argmax; the window then read that row from the embedding tensor and crashed (SIGSEGV). It now
   stops with `the token id … is outside the vocabulary` instead.
4. **`gr.metal`'s `gr_norm2` takes its 32-lane group from the thread index** (`sg32`/`lane32`/`sum32`), like every
   other kernel in the window. This was the non-finite forward; see above.
5. **No adaptation thread without a host residency table.** `adapt()` indexes `host_res`; the Metal window has none.
   All three spawn sites (the one-shot loop and the two serve loops) now require it, which also stops the fourth
   decode window's SIGSEGV.
6. **A dense side larger than one Metal buffer.** The RX 580's `maxBufferLength` is 3.8 GB and an Unsloth UD-* pack's
   dense side is 4.7–6 GB, so `ExpertStream::load_dense` now reads it into an anonymous host blob and wraps that as
   overlapping chunks (every GPU binding selects its chunk; the host reads still index the base pointer). This is what
   makes the Unsloth packs runnable on this card at all.

The pack tools also needed two Python 3.8 fixes (`Path.write_text(..., newline=...)` is 3.10's), which is the Python
macOS 11 ships: `tools/strata_tokenizer.py` and `tools/iq_pack.py` now open the file themselves.

## The packs on disk

Downloaded for this work into `~/models/qwen38/` (one folder per pack, the 28.8 GB PLE shard hard-linked between them
because it is the same bytes):

| pack | size | state |
| --- | ---: | --- |
| `Q2_0` | 66.4 GB | verified; **1.06 tok/s decode**, 3.33 tok/s prompt |
| `IQ3_XXS` | 75.8 GB | verified; 0.90 tok/s decode, 2.63 tok/s prompt |
| `IQ3_S` | 83.6 GB | verified; 0.84 tok/s decode, 2.53 tok/s prompt |
| `IQ2_XS` | 68.0 GB | verified, but **refused on this branch**: some gate/up experts are IQ1_M and `moe.metal` has no `moe_gu_29_*` |

### Multi-shard Unsloth packs

An Unsloth UD-* pack is 3–6 shards whose dense tensors and experts are cut across them, so the window (which opens one
GGUF) needs `metal/tools/metal_repack.py` first. That path now works, and `bench/macpro51/prep_unsloth.sh <PACK>` runs
all of it: repack (verified tensor by tensor), `iq_pack --compat-bf16` (the sets keep the HC projections as Q8_0; the
conversion rounds only those), then a smoke run. Two details matter:

- pass the repacked file as `--native` **and** `--native-dense-gguf`; with `--native` alone the engine adds the PLE
  shard to the dense list and the shard-split check refuses a repacked file beside it;
- `iq_pack --compat-bf16` needs llama.cpp's `gguf-py` (`STRATA_GGUF_PY=<llama.cpp>/gguf-py`) and PyYAML.

Measured on UD-Q4_K_XL (111 GB downloaded, 76.9 GiB repacked) **while the rest of the set was still downloading**:
0.61 tok/s decode and 1.57 tok/s prompt (the download competes for the SSD), with the same first tokens as Q2_0
(`\n\n<think>\nThe user wants a concise explanation…`) — the 4.5-bit pack answers the same way the 2-bit one does.

Gyro is one file with the n-gram table inside, and its TQ expert kernels are the `stubs_gyro` path on this branch —
check before promising it.

`bench/macpro51/run_bench_qwen38.py` is the sweep: one process per run, the pack's own 32-token prompt + 32 greedy
tokens, a fresh shuffled order each round, mean ± 1 sample standard deviation over the steady rounds, SVG charts and
a commit+push after every run. The matrix is [macpro51-qwen38-benchmarks.md](macpro51-qwen38-benchmarks.md).

## Reproduce

```sh
cd ~/strata-bench
python3 tools/iq_pack.py --gguf ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --out ~/models/qwen38/Q2_0/pack
STRATA_METAL_STREAM=1 ./build-bench/strata \
    --pack ~/models/qwen38/Q2_0/pack \
    --native ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf ~/models/qwen38/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --tokens 814,20139,11,303,1330,2716,22157,11,3069,279,12515,5686,6105,2261,279,1834,321,2438,506,41564,13,4844,1910,3069,279,16820,5686,18016,3043,279,32946,13 \
    --max-new 16 --max-context 512 --spec 1 --suffix-draft 0 --lookup-max-t 0 --prefill 32
```

The ids are that sentence in the **pack's** vocabulary; `tools/strata_tokenizer.py`'s `Tokenizer.from_gguf` encodes
them. `STRATA_WIN_LAYERCHECK=1` adds the per-layer residual check (a bring-up diagnostic: each cut waits for the
GPU and reads 10,240 floats, so never in a timed run).
