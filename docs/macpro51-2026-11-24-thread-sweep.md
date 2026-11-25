# CPU expert-pool worker sweep on macpro51 — 2026-11-24

> **Paths moved on 2026-11-25:** `metal/` is `engine/metal/`, `src/` is `engine/src/`,
> `bench/macpro51/` is `terminal/bench/macpro51/`. See
> [macpro51-2026-11-25-merge.md](macpro51-2026-11-25-merge.md) for the mapping and the test record.

> **Superseded for numbers by the post-merge sweep (2026-11-25):**
> [macpro51-2026-11-25-sweep.md](macpro51-2026-11-25-sweep.md) - all seven packs run, 4-7 % faster.

Varying `--pool-workers` from 1 to the machine's ceiling, three runs per point, on the branch tip
(`f42506b47`, tag `working2` + the harness). Records: `bench/macpro51/results-threads.json` (39 records).
Charts: `docs/benchmarks/macpro51/thread-sweep.svg` and `.csv`.

**Result: the worker count changes nothing measurable, and the reason is in the data — the CPU expert pool
is never given any work to do.**

## What was varied, and what was measured

`--pool-workers N` sizes the CPU expert pool ("N pool workers on logical processors 1..N, the host thread on
0"). The machine is a 2×6-core Xeon X5675: **12 physical cores, 24 logical**. Points: **1..12** (the physical
ceiling the pool is built for) plus **24** (the logical maximum, deliberately oversubscribed).

Workload: a fixed 260-token prompt (`3 1 42498 456 11 1544 76 398 8709 96302 17776 78 76` repeated) with a
2-token decode, on `UD-IQ1_M` with the DFlash2 drafter attached. A long prompt is deliberate: the prompt path
is where the engine gives the idle CPU a share of the experts ("the CPU share is ON by default for prompt
chunks up to 3072 tokens"), so this is the workload in which threads *could* matter.

| `--pool-workers` | runs | mean prefill (tok/s) | sd | min–max | mean prefill (ms) | decode (ms) |
|---|---|---|---|---|---|---|
| 1 | 3 | 3.937 | 0.015 | 3.920–3.950 | 65 812 | 3551 |
| 2 | 3 | 3.940 | 0.000 | 3.940–3.940 | 65 706 | 3537 |
| 3 | 3 | 3.947 | 0.012 | 3.940–3.960 | 65 667 | 3532 |
| 4 | 3 | 3.940 | 0.000 | 3.940–3.940 | 65 677 | 3530 |
| 5 | 3 | 3.953 | 0.006 | 3.950–3.960 | 65 525 | 3532 |
| 6 | 3 | 3.953 | 0.006 | 3.950–3.960 | 65 510 | 3529 |
| 7 | 3 | 3.937 | 0.012 | 3.930–3.950 | 65 733 | 3533 |
| 8 | 3 | 3.943 | 0.029 | 3.910–3.960 | 65 724 | 3542 |
| 9 | 3 | 3.947 | 0.006 | 3.940–3.950 | 65 647 | 3534 |
| 10 | 3 | 3.930 | 0.010 | 3.920–3.940 | 65 910 | 3542 |
| 11 | 3 | 3.937 | 0.006 | 3.930–3.940 | 65 776 | 3535 |
| 12 | 3 | 3.937 | 0.006 | 3.930–3.940 | 65 773 | 3537 |
| 24 | 3 | 3.937 | 0.006 | 3.930–3.940 | 65 760 | 3536 |

Spread across the whole sweep: **0.6% on prefill (3.930–3.953 tok/s)** and **0.6% on decode
(3529–3551 ms)**, i.e. smaller than the run-to-run standard deviation at several points. Going from one
worker to twenty-four — a 24× change in pool size — moves the mean prefill rate by 0.016 tok/s.

## Why it is flat: the pool is idle, and the engine says so

Every record carries the engine's own accounting of what the CPU pool did, and in **all 39 runs** it is zero:

```
prefill ...   experts streamed 0 (0 by DMA, host 0.0 ms), resident 0, on the CPU 0 (share 0.00)
verify window wait for rings ... CPU experts 0.00 distinct / 0.00 routed per layer
```

`cpu_prefill_experts = 0` and `cpu_prefill_share = 0.0` for every worker count; `cpu_decode_routed = 0.0`
everywhere. The CPU share is *armed* (the engine prints the policy line) but takes **no experts** on this
configuration, because there is nothing for it to take: the experts are streamed to the GPU from the SSD, or
served from the GPU cache, and the machine has 137 GB of RAM with ~50 GB free, so the low-memory path that
would hand rows to the CPU never triggers.

Two further attempts to make the pool work, both measured, both also zero:

- **`--mmap-experts --resident-cpu-experts`** (the "keep the experts the GPU cache does not fit" mode): it
  built a 30.27 GiB pageable RAM cache (64–79 s to prepare, 75.9 GB resident), and the verify window still
  reported `CPU experts 0.00 distinct / 0.00 routed per layer`, with decode unchanged at 0.50 tok/s.
- **Longer prompt**: 259 tokens instead of 4 — the share still took 0 experts.

So the invariance is not a plateau: the pool has no work on this port, and a sweep of its size cannot move
anything. The chart is a flat line because the code path is not exercised, and the numbers to quote are the
ones above, not a curve.

## What this does not say

- It does **not** say the pool is useless. On a machine where the experts genuinely do not fit in RAM and
  have to be computed on the CPU (a small-RAM PC, the `--resident-experts` setup the help text describes),
  the worker count should matter, and this sweep cannot see that case because this box has too much RAM to
  enter it.
- It does **not** say the flag is broken: the engine reported `N pool workers on logical processors 1..N`
  for every N tried, so the flag took effect; it simply had nothing to feed.
- All 39 runs produced the **same two output tokens** (`4960 4960`), so nothing here changes results — which
  is the one thing a thread count must never do.

## Reproduce

```sh
cd ~/projects/Strata/strata
python3 bench/macpro51/run_thread_sweep.py --runs 3          # 39 runs, ~60 min, resumable
python3 bench/macpro51/charts_threads.py                     # -> docs/benchmarks/macpro51/thread-sweep.{csv,svg}
```

The chart plots mean prefill tokens/second per worker count with ±1 standard deviation error bars; the CSV
also carries min/max, mean prefill ms and mean decode ms. Tooltips name each point's runs and range.
