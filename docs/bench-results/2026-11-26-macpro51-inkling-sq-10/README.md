# macpro51: Inkling-2-Mini SQ-10, standard protocol `standard-v1` — PARTIAL

**This is not a conforming result and there is no `results.json`.** The run cannot finish on this
machine, and the reason is measured, not guessed: the protocol gives each request
`REQUEST_TIMEOUT_S = 1800.0` (`terminal/strata_cli/bench_standard.py:48`), and a 13,775-token ("16K")
request needs about 1,895 s for its first token here. `--request-timeout` is refused for the standard
suite ("the protocol fixes the workload"), so there is no legitimate way to widen it.

The raw console log of the run is [`standard-v1-console.log`](standard-v1-console.log) beside this file.

## What was run

- Machine: macpro51, MacPro5,1, Xeon X5675, 96 GB DDR3, **Radeon RX 580 8 GB**, macOS 11.7, Metal 2.
- Engine: `strata-inkling-metal` 0.1.67 (this branch's own build; see
  `docs/macpro51-2026-11-26-inkling-hy4.md` for the nine fixes it needed on this card).
- Model: `Inkling-2-Mini-SQ-10.strata`, context `--max-context 20480` (the smallest that holds a 16K
  prompt plus a 512-token reply; 32768 was reduced after a GPU wedge, see the page).
- Server: `terminal/serve/server.py --engine strata --config …/config-sq10.json --port 8095`.
- Command: `strata bench run --url http://127.0.0.1:8095 --track custom --suite standard-v1`.

## Measured

| step | prompt tok/s | decode tok/s | first token | outcome |
|---|---:|---:|---:|---|
| warmup (128 cap) | 9.60 | 1.43 | 389.0 s | truncated |
| short-r1 | 9.00 | 1.40 | 4.7 s | truncated |
| prompt-4k-r1 | 8.62 | 1.30 | 397.2 s | truncated |
| prompt-16k-r1 | 7.27 | — | **1,894.5 s** | **failed: timeout** |
| short-r2 | 9.01 | 1.40 | 4.7 s | truncated |
| prompt-4k-r2 | 8.63 | 1.30 | 396.8 s | truncated |
| prompt-16k-r2 | 7.27 | — | **1,894.9 s** | **failed: timeout** |
| short-r3 | 8.95 | 1.40 | 4.7 s | truncated |
| prompt-4k-r3 | 8.60 | 1.30 | 397.9 s | truncated |
| prompt-16k-r3 | 7.26 | — | **1,896.5 s** | **failed: timeout** |
| sanity-I1 | 9.81 | 1.40 | 4.8 s | truncated at the 2048 cap |
| sanity-I2 | 9.96 | 1.40 | 5.6 s | truncated |
| sanity-I3 | 9.32 | 1.40 | 5.5 s | truncated |
| sanity-I4 | — | — | — | started, not finished (run stopped) |

Two things are worth reading off this table:

1. **The 16K failure is deterministic, not flaky**: 1,894.5 / 1,894.9 / 1,896.5 s across three runs.
2. **The prefill rate decays with context** (9.6 → 8.6 → 7.3 tok/s), because attention cost grows with
   the sequence, so lengthening the budget would not rescue a marginal case either.

The run was stopped during sanity-I4. The sanity checks are capped at 2048 tokens and the model uses
them for thinking; I1, I2 and I3 each hit that cap and were marked truncated, which the protocol counts
as a failed check. Completing the remaining nine checks would have taken about four more hours and could
not have changed a single verdict, so the machine went to the Hy4 and 22-task work instead.

## Why it is this slow, and what would fix it

`STRATA_INK_FMT_HIST=1` shows the dense trunk is only ~1.5 GiB (Q8ROW 1.01 GiB, F32 0.14, BF16 0.04):
the model's 9.75 GiB is almost all experts. One 8-token prefill chunk reads essentially all of the trunk
in ~1.4 s, i.e. about **1 GB/s**, which is PCIe-shaped because the card's 3.8 GB `maxBufferLength`
forces host-mapped buffers.

The scaling fix is a tiled GEMM with BM=32..64 so each weight is read once per 64 tokens instead of once
per 8. The present ceiling is the matrix-free `mv` kernels' M ≤ 8 and the `mat_ok` cap that follows from
it. That kernel is **not** implemented on this branch, and nothing in this file is extrapolated from it.

## Not measured

- The 12 sanity checks (I4 onward) and their pass/fail count.
- The 22 checked tasks: they were attempted twice and abandoned (see the page); with this model's
  thinking budget each task needs 10-17 minutes and truncates at any cap that fits the time available.
- Anything for Hy4-Flash: its engine cannot load on this card (Section 1 is 5.92 GB as a single buffer,
  the card's limit is 3.8 GB).
