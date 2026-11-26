# Inkling-2-Mini and Hy4-Flash on macpro51: what it took, and how fast it goes

Status: **partial** — Inkling-2-Mini SQ-10 runs and answers correctly, and its standard run cannot complete because the 16K cases exceed the protocol's own 1800 s request budget (measured below). Hy4-Flash builds and cannot load on this card (Section 1 needs a 5.92 GB buffer; the limit is 3.8 GB). The 22-task sweep did not fit in the time available.
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

## Standard benchmark results — partial, and why

Raw evidence: `docs/bench-results/2026-11-26-macpro51-inkling-sq-10/` (README + console log). There is
**no `results.json`**, because the runner writes it at the end and the run cannot reach the end in a
budget that exists.

| case | prompt tok/s | decode tok/s | first token | outcome |
|---|---:|---:|---:|---|
| warmup | 9.60 | 1.43 | 389.0 s | truncated |
| short (×3) | 9.00 / 9.01 / 8.95 | 1.40 | 4.7 s | truncated |
| 4K (×3) | 8.62 / 8.63 / 8.60 | 1.30 | 397 s | truncated |
| 16K (×3) | 7.27 / 7.27 / 7.26 | — | **1,894.5 / 1,894.9 / 1,896.5 s** | **failed: timeout** |
| sanity I1-I3 | 9.3-10.0 | 1.40 | 5 s | truncated at the 2048 cap (= failed) |

The 16K failure is deterministic to within 2 s across three runs, and the prefill rate *decays* with
context (9.6 → 8.6 → 7.3 tok/s), so it is not a marginal miss that a little luck would fix.

## 22 checked tasks — attempted twice, not completed

Both attempts are documented because the failures are informative:

1. `BENCH_MAX_TOKENS=400`: the first coding task spent all 400 tokens thinking and never emitted the
   function (`FAIL ... out 400`). A cap that low manufactures failures, so the attempt was abandoned
   rather than reported.
2. `BENCH_MAX_TOKENS=800, BENCH_REASONING=low`: the first task reached **710 of 800 tokens after 879 s**
   and was still thinking, with decode down to 0.8 tok/s. At 10-17 minutes a task the 22-task sweep needs
   roughly 6 hours, and the runner writes its JSON only at the end, so a partial run yields no artifact.

No pass/fail figure is reported for the tasks. The honest statement is that this model's thinking budget
and this machine's 0.8-1.4 tok/s decode make the sweep a multi-hour run that did not fit alongside the
standard suite in the time available.

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
- **Hy4-Flash SQ-64 builds, inspects, and cannot load on this card.** `strata-hy4-metal` compiles clean,
  its tokenizer converts, and `--info` reads it (`arch=hy4flash layers=42 hidden=3072 vocab=120832
  experts=512 drafter=dfly2 images=1`), but the load dies at

  ```
  [gpu] AMD Radeon RX 580 metal4=0 maxBuffer 3.8 GB workingSet 8.6 GB
  mtl: wrap 5920620544 failed
  ```

  With a 62 GB file on an 8.6 GB working set, `hy4_metal_main.cpp:238` selects the cold-expert path and
  then wraps **Section 1 (5.92 GB) as a single Metal buffer** (line 252); this card's `maxBufferLength`
  is 3.8 GB, and the copy fallback fails for the same reason. Every tensor accessor reads through that
  one `wbuf`, so the fix is to split Section 1 across several buffers and remap the accessors — the
  chunked-buffer design Inkling already has. That is a port, not a configuration change, and it was not
  attempted: the Hy4 kernels are the same Apple-silicon vintage, so it would very likely surface the same
  32-lane/`simdgroup_matrix` defects that took nine fixes on the Inkling side.
