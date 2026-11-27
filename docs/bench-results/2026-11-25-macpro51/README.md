# 2026-11-25-macpro51 — the 22 checked tasks on Inkling-2-Mini SQ-10

Run 2026-11-26 20:57 → 2026-11-27 05:02 (488 min) on macpro51 (MacPro5,1, RX 580 8 GB, Metal 2,
64-lane simdgroups), engine `strata-inkling-metal` 0.1.67 at commit `6e418682e`, model
`Inkling-2-Mini-SQ-10.strata`, server `terminal/serve/server.py` on :8095, context **20480** (the
first tasks run with a real 20480-token KV — earlier builds silently clamped it to 4096, see
`docs/macpro51-2026-11-26-inkling-hy4.md`).

Harness: `tasks/bench_models.py`, the 22 tasks from the 2026-11-22 Radeon report, unchanged:

```
BENCH_URL=http://127.0.0.1:8095/v1/chat/completions \
BENCH_MAX_TOKENS=1200 BENCH_REASONING=low BENCH_TIMEOUT=1800 \
  python3.11 bench_models.py inkling-sq-10-c8
```

`BENCH_MAX_TOKENS` is 1200, not the report's 6000, because decode is ~1.4 tok/s here: 6000 tokens
is 71 min per task. `BENCH_TIMEOUT` is 1800 rather than the script's 900 for the same reason (a
900 s cap manufactures failures; a 400-token cap already did, see the handoff).

## Result: 3 of 22 — and this is not a capability measurement

| group | passed | of |
|---|---|---|
| code | 0 | 6 |
| reason | 1 | 6 |
| tool | 0 | 4 |
| reading | 0 | 3 |
| format | 2 | 3 |
| **all** | **3** | **22** |

**19 of the 22 rows ended with `finish=length`, i.e. the 1200-token cap, not a wrong answer.** The
three passes are exactly the three rows that ended with `finish=stop`. The model spends the cap in
the thinking phase (`think_chars` 2 200–6 100 on most rows) and the reply never reaches the answer,
so a FAIL here mostly means "did not finish", not "answered wrongly".

Per-group medians (s): code first token 1 058 / total 1 382; reason 1 379 / 1 379; tool 1 144 /
1 490; reading 2 467 / 3 456; format 36 / 107. The reading group is the expensive one: its prompts
are long and prefill runs at 5–9 tok/s, so its first token arrives after ~41 min.

Reproduce a single row: `results-bench-inkling-sq-10-c8.json` holds `reply_head` (the first 1500
characters) beside each verdict, so a truncated row can be read rather than guessed at.

## What this run did establish

- The engine, the server and the harness can sustain 8 h of mixed requests on this card at real
  context without wedging (unlike the 2026-11-26 13:00 panic, which was at ctx 32768).
- Every group is prefill- or thinking-bound, not answer-bound: even the two format tasks that pass
  finish in 38–107 s, while anything needing reasoning runs into the token cap.
- A follow-up run with a bounded thinking budget (`BENCH_THINKING_BUDGET`) is the way to measure
  capability here, but the first such attempt hung the GPU during the server's "wrapping up the
  thinking" continuation — see the handoff before repeating it.
