# Every model on macpro51, exercised — 2026-11-24

> **Paths moved on 2026-11-25:** `metal/` is `engine/metal/`, `src/` is `engine/src/`,
> `bench/macpro51/` is `terminal/bench/macpro51/`. See
> [macpro51-2026-11-25-merge.md](macpro51-2026-11-25-merge.md) for the mapping and the test record.

> **Superseded for the refusals and the six skipped packs (`99e2d6c7b`).** Both Qwen3.6 refusals below are
> fixed (BF16 experts; Q5_K embedding + Q2_K gate/up) and the six Qwen3.8 directories recorded as "not
> convertible here" now run — they never needed converting. See
> `docs/macpro51-2026-11-24-models-that-run.md`. The tables below are the state *before* those fixes.

A full pass over everything in `~/models`, on the branch tip after a **clean rebuild** of `build-metal`
(`cmake --build build-metal --clean-first -j 24`, 254/254 steps, `rc=0`) and with `ctest` green
(17 passed, 1 skipped, 55.8 s). One engine process per model, driven by
`bench/macpro51/run_all_models.py`; the records are `bench/macpro51/results-all-models.json`.

Short fixed workload: the Qwen3.6 packs decode 8 tokens from a 96-character prompt; the Qwen3.8 packs run
the documented DFlash2 command (prompt `3 1 42498 456 11`, `--max-new 2`, `--prefill 1`) with the drafter
attached. Nothing here is a performance measurement — the numbers only say whether a model runs and how
long it takes to get through that workload on this card.

## Qwen3.6-35B-A3B (MoE, 35B total) — 5 of 7 packs run

| pack | size | result |
|---|---|---|
| Q4_K_M | 19.7 GB | runs, 0.967 s/step |
| UD-Q4_K_XL | 21.3 GB | runs, 0.907 s/step |
| Q6_K | 26.6 GB | runs, 1.01 s/step |
| UD-Q6_K_XL | 30.4 GB | runs, 1.082 s/step |
| Q8_0 | 35.2 GB | runs, 1.177 s/step |
| UD-Q2_K_XL | 11.7 GB | **refused**: `token_embd.weight is Q5_K, q36_embed only supports Q8_0, Q6_K and Q4_K` |
| UD-Q8_K_XL | 36.4 GB | **refused**: `blk.34.ffn_down_exps.weight is BF16, and this port has no grouped expert down kernel for that type` |
| `mmproj-F16.gguf` | 0.8 GB | **refused**: `architecture is 'clip', this engine requires 'qwen4exp'` (it is the vision projector, not a model) |

The two refusals are the documented ones, unchanged, and they are the engine's own words. Token sequences
for the five that run are the 24-token runs recorded in `bench/macpro51/results-qwen36.json` (all five
return the sequence llama.cpp returns); this sweep's records carry the step times but not the token text,
because the Qwen3.6 engine prints `tokens:` where the Qwen3.8 one prints `output:` and the harness only
parsed the latter — a harness gap, recorded rather than papered over.

## Qwen3.8-Flash-Next (the large MoE: 48 blocks, 512 experts) — **all 11 native packs run**

| pack | metal GGUF | result |
|---|---|---|
| UD-IQ1_M | 36.5 GB | runs, prefill 5614 ms, drafts accepted 0 of 7 |
| UD-Q2_0 | 38.9 GB | runs, prefill 4583 ms, 0 of 7 |
| UD-IQ2_XS | 43.8 GB | runs, prefill 4502 ms, 0 of 7 |
| UD-Q2_K_XL | 47.3 GB | runs, prefill 4716 ms, 0 of 7 |
| UD-IQ3_XXS | 51.1 GB | runs, prefill 4998 ms, 0 of 7 |
| UD-Q3_K_XL | 56.5 GB | runs, prefill 4848 ms, 0 of 7 |
| UD-IQ4_XS | 60.4 GB | runs, prefill 4821 ms, 0 of 7 |
| UD-Q4_K_XL | 76.9 GB | runs, prefill 5002 ms, 0 of 7 |
| UD-Q5_K_XL | 96.8 GB | runs, prefill 5609 ms, 0 of 7 |
| UD-Q6_K_XL | 106.9 GB | runs, prefill 5824 ms, 0 of 7 |
| Q8_0 | **124.6 GB** | runs, prefill 6009 ms, 0 of 7 |

Every one of them loads its native pack, streams experts from the SSD, prefills, and decodes — **including
the 124.6 GB Q8_0**, on an 8 GB card, by streaming. Draft acceptance is 0 of 7 on all of them, which is the
same result Rounds 15–23 reached by other routes and is not a per-pack effect: the drafter's proposals miss
whatever the target's precision.

**Six more Qwen3.8 directories are present but were not convertible with what is on disk**: `Gyro-M`,
`Gyro-S`, `IQ2_XS`, `IQ3_S`, `IQ3_XXS`, `Q2_0` hold the multi-shard GGUF weights but no `pack/` (native
expert pack) and no `*-metal.gguf`, so there is nothing for `--pack`/`--native` to point at. Recording that
as "not convertible here" rather than "does not run": converting them is `tools/`' job, not the engine's.

## DeepSeek V4.1 Flash

`strata_ds4 --pack-info` runs and reports; the model itself is **not run**, by design: it needs
`STRATA_DS4_ALLOW_UNSTABLE_METAL=1`, which resets this card (see the workspace AGENTS.md — do not set it
without asking). Recorded, not attempted.

## What the passes say about the system

- **Loading and streaming scales past the card.** The largest pack tried, 124.6 GB, runs on 8 GB of VRAM
  and 137 GB of RAM with the expert arena at its 1 GB floor (Round 20 explains why it cannot be bigger).
  Prefill for the same 4-token prompt grows 4.50 s → 6.01 s from the 43.8 GB pack to the 124.6 GB one, which
  is first-touch I/O, not a kernel difference.
- **Two families, two engines, both behaving as documented.** `strata-q36-metal` handles Qwen3.6 with two
  typed refusals; `strata` handles Qwen3.8 with the DFlash2 drafter attached and produces tokens on every
  native pack.
- **The one thing that does not work anywhere is drafting.** 0 of 7 accepted on all 11 packs. That is
  consistent with the Rounds 15–24 conclusion: the fold, the layer list, the feature gather, the tap instant
  and the target's precision are each settled, and what is left needs an independent `h` for qwen4exp.

## Reproduce

```sh
cd ~/projects/Strata/strata
PATH=/usr/local/bin:$PATH cmake --build build-metal --clean-first -j 24   # 254/254, ~6 min here
(cd build-metal && /usr/local/bin/ctest)                                   # 17 passed, 1 skipped
python3 bench/macpro51/run_all_models.py                                   # ~35 min, one process per model
```

Two harness bugs were found and fixed while running this, both worth knowing: the PLE shard must be found by
*reading its GGUF header* for `per_layer_token_embd.weight` (its size differs per pack — 28.8 GB in
UD-IQ1_M, 12.1 GB in UD-Q4_K_XL, 34.0 GB in Q8_0), and a size rule silently skipped four packs until it was
replaced.
