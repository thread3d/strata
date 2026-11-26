# Inkling-2-Mini and Hy4-Flash on macpro51: what it took, and how fast it goes

Status: **partial**. The standard protocol cannot complete on this machine — the 16K cases exceed the
protocol's own 1800 s request budget (measured, below). The numbers that did complete are recorded here.

macpro51 is the 2010/2012 Mac Pro (Xeon X5675, 96 GB DDR3, **Radeon RX 580 8 GB, Metal 2, macOS 11.7**)
whose Qwen3.6/Qwen3.8 port is in `macpro51-2026-11-25-sweep.md`. Neither shape engine had ever been
built or run there. Both are written for Apple silicon; the difference that matters is that **this
card's simdgroups are 64 lanes wide, not 32**.

## What was broken, and is fixed

Nine defects stood between "will not compile" and "answers correctly" for Inkling-2-Mini. Fixes are in
`ea91bd694`, `21ecf5474` and `8a51e304b`.

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | `build_mac.sh` left `$INC` and the zstd list unquoted | `$HOME` is `/Volumes/Sonnet Raid/…`, so clang got `"/Volumes/Sonnet"` + `"Raid/…"`; nothing compiled | quote them (array for the includes) |
| 2 | `Context::wrap` used `newBufferWithBytesNoCopy` on a **read-only private** mapping | Metal 2 refuses it, so every `.strata` tensor failed to wrap; a malloc'd buffer of the same size worked | `mprotect(PROT_WRITE)` on the MAP_PRIVATE range (copy-on-write, file untouched) and retry, copy last |
| 3 | Section 4 limited to **4** Metal buffers | SQ-20 refused: `14950 MiB exceeds 4 buffers of 3584 MiB` | `kMaxSec4Chunks` 4 → 8 |
| 4 | `simdgroup_matrix` pipelines created unconditionally | this card cannot compile or link them, and `pipeline()` calls `exit(3)` | `mat_ok = simd_width() == 32`; matrix pipelines skipped, matrix-free `mv` kernels, batch/slots capped at 8, DSpark off |
| 5 | Apple's `simd_sum` / `simd_prefix_exclusive_sum` span the whole hardware simdgroup | on 64 lanes every sum mixed two logical 32-lane groups; selfcheck logits came in tied pairs | 32-lane redefinitions (shuffle xor/up, offsets < 32) |
| 6 | 22 kernels took `sg`/`lane` from `simdgroup_index_in_threadgroup` / `thread_index_in_simdgroup` | `sg` ran 0..3 where 0..7 was meant, so threadgroup slots were written by the wrong groups — output fluent but wrong | derive from the thread index (`sg = tid >> 5`, `lane = tid & 31`) |
| 7 | `--serve` advertised in the header, never implemented | the server could not drive the engine | implemented READY / GEN / T / DONE |
| 8 | serve split `GEN` ids on whitespace only | the server writes `id,id,id`, so a whole request became one id (`atoi("1775,149884,…")` = 1775) and every templated request was a one-token prompt | accept the comma-separated field |
| 9 | serve printed `T` only after the whole generation | the server's silence watchdog killed the engine at 301 s (`exit -9`) and TTFT was the end of the reply | `on_token`/`on_prefill` callbacks; `T`/`PP` stream and flush |

Hy4-Flash needed no engine work: `strata-hy4-metal` already implements the full line protocol
(`INFO`/`READY`/`RESUME`/`PP`/`T`/`DONE`, conversation slots, SSD spill) and builds clean when
`HY4_DIR` points at a space-free directory whose `wt/engine` symlinks this checkout.

## That it is correct, not merely running

Inkling-2-Mini SQ-10, through the HTTP server with its chat template, greedy:

| prompt | answer |
|---|---|
| What is the capital of France? Answer in one word. | **France's capital is Paris.** |
| Write a Python function that adds two numbers. | ` ```\ndef add(a, b)` |
| In C programming, a pointer is a variable that stores the | ` the pointer is a variable that stores the` |
| The quick brown fox jumps over the lazy dog … towards | ` the fox.\n\nThe quick brown` |

The tokenizer conversion is exact: the converted `vocab.json` matches HF `tokenizers` on all 200,058
entries, and "What is 2+2? Answer in one short sentence." encodes to the same ids.

## Speed, and why the standard protocol cannot finish

Measured on a fresh boot at context 20480 (`--max-context`), which is the smallest that fits the
protocol's 16K prompts:

| step | prompt tok/s | decode tok/s | time to first token |
|---|---|---|---|
| warmup (128 cap) | 9.60 | 1.43 | 389.0 s |
| performance-short | 9.00 | 1.40 | 4.7 s |
| performance-prompt-4k | 8.62 | 1.30 | **397.2 s** (fits) |
| performance-prompt-16k | 7.27 | — | **1,894.5 s → timeout** |

`terminal/strata_cli/bench_standard.py:48` sets `REQUEST_TIMEOUT_S = 1800.0`, and `--request-timeout`
is refused for the standard suite ("the protocol fixes the workload"). The prefill rate **decays with
context** (9.6 → 8.6 → 7.3 tok/s) because attention grows with the sequence, so a 13,775-token prompt
needs ~1,840 s of prefill plus ~365 s of decode. The three 16K cases and the two 13.8K sanity cases
therefore fail by timeout; everything at 4K and below passes.

**Why prefill is slow, measured not guessed.** `STRATA_INK_FMT_HIST=1` (added to
`ink_metal_common.h`) prints the trunk make-up: `fmt101 (F32) 500 tensors/0.14 GiB, fmt100 (Q8ROW)
133/1.01 GiB, fmt102 (BF16) 12/0.04 GiB, fmt6 (S6) 1/0.33 GiB`. The model's 9.75 GiB is almost all
experts; the dense trunk is ~1.5 GiB and **one 8-token prefill chunk reads essentially all of it**, in
about 1.4 s — roughly **1 GB/s, PCIe-shaped**, because the card's 3.5 GiB `maxBufferLength` forces
host-mapped buffers. The scaling fix is a tiled GEMM with BM=32..64, so weights are read once per 64
tokens instead of once per 8; the present ceiling comes from the `mat_ok` cap of 8, which exists because
the matrix-free `mv` kernels handle M ≤ 8. That kernel is **not implemented here**, and no number in
this page is extrapolated from it.

## Standard benchmark results

<!-- the suite is still running; the table is filled from
     docs/bench-results/2026-11-25-macpro51-inkling-sq-10/README.md + results.json when it finishes -->

| case | result |
|---|---|
| warmup | truncated, 1.43 tok/s decode, TTFT 389 s |
| short (×3) | truncated, ~1.40 tok/s decode, TTFT ~4.7 s |
| 4K (×3) | truncated, 8.6 tok/s prompt, 1.30 tok/s decode, TTFT ~397 s |
| 16K (×3) | **failed: timeout at 1800 s** (first token at 1,894 s) |
| sanity 12 | pending (the 13.8K cases R3/R4 will time out like 16K) |

## 22 checked tasks

Not yet run to completion. The first attempt used `BENCH_MAX_TOKENS=400` and was abandoned: the first
coding task spent the whole 400 tokens thinking and never emitted the function (`FAIL ... out 400`), so
that cap manufactures failures. It is re-run at 1200-1500 tokens with `BENCH_TIMEOUT`, and the deviation
from the 2026-11-22 Radeon report (6000 tokens, `reasoning_effort: high`) will be stated with the result.

## The crash on 2026-11-25, for the record

The machine was unreachable from ~13:00 and was power-cycled the next morning. The panic log says:

```
panic(cpu 14): userspace watchdog timeout: no successful checkins from
com.apple.WindowServer in 120 seconds
... AMDRadeonX4000 (4.6.21) loaded ... uptime ~6.5 h
```

That is a window-server hang escalated by the userspace watchdog with the AMD driver loaded — the GPU
wedge this box is known for, most likely from ~10.1 GiB of model plus 2.3 GiB of KV on an 8 GiB card that
also drives the display. The context is now 20480 (KV ~1.44 GiB) and the engine has run for hours since.

## Not working

- **Inkling-2-Mini SQ-20** loads (18.40 GiB over 5 buffers) but its logits are degenerate (all −30,
  top-1 = 0); its SF4/MX8 formats are the difference from SQ-10/SQ-15, which are correct.
- **The Unsloth GGUF builds** cannot be read by this runner (`.strata` only, "bad .strata magic").
- **DSpark** is off: its kernels are matrix based and this card cannot run them.
- **Hy4-Flash** is built, its tokenizer converted, its config written, and `--info` reads SQ-64
  (`arch=hy4flash layers=42 hidden=3072 vocab=120832 experts=512 drafter=dfly2 images=1`) — but it has
  **not been run**, because the GPU has been busy with the Inkling sweep.
