# The post-merge sweep — macpro51, 2026-11-25

Branch `macpro51` at `c482dbd9a` (tag `working4`), engine `strata 0.1.67-metal`, built as
`build-merge` with `-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0` and the pinned vendored ggml
(`~/strata/third_party/strata-ggml`) — see [macpro51-2026-11-25-merge.md](macpro51-2026-11-25-merge.md).

Five stages, one engine process per run, all five finished `rc=0`, 07:29 → 09:48 CET:

| stage | command | records |
|---|---|---|
| Qwen3.6 matrix | `run_matrix.py --runs 2` | 14 |
| llama.cpp reference | `run_reference.py --runs 2` | 10 |
| suite (Qwen3.8) | `run_suite.py --runs 2` | 28 |
| CPU-pool thread sweep | `run_thread_sweep.py --runs 3` | 39 |
| all models | `run_all_models.py` | 26 |

## Result in one line

**All seven Qwen3.6 packs now run, and the five that ran before are 4–7 % faster.** Round 1 is the
page-cache warm-up, round 2 is the measurement, as `BENCHMARK_PROTOCOL.md` prescribes.

| pack | run 1 s/step | run 2 s/step | mean wall | llama.cpp wall | strata / llama.cpp |
|---|---|---|---|---|---|
| UD-Q2_K_XL | 0.945 | **0.733** | 44.4 s | — | — |
| UD-Q4_K_XL | 1.217 | **0.849** | 55.7 s | 16.1 s | 3.46× |
| Q4_K_M | 0.987 | **0.931** | 48.9 s | 18.4 s | 2.66× |
| Q6_K | 1.236 | **0.977** | 56.5 s | 17.8 s | 3.18× |
| UD-Q6_K_XL | 1.528 | **1.016** | 66.5 s | 15.3 s | 4.35× |
| Q8_0 | 1.560 | **1.093** | 67.9 s | 14.7 s | 4.62× |
| UD-Q8_K_XL | 1.763 | **1.132** | 75.2 s | — | — |

- **`UD-Q2_K_XL` and `UD-Q8_K_XL` run here for the first time.** Before the merge the first was
  refused for its Q5_K embedding and the second for a BF16 expert down; both now decode.
- Against the pre-merge run-2 numbers (0.981 / 0.903 / 1.017 / 1.083 / 1.171 s/step for
  Q4_K_M / UD-Q4_K_XL / Q6_K / UD-Q6_K_XL / Q8_0) the same packs are 5 / 6 / 4 / 6 / 7 % faster.
  That is one warm pair of runs on each side, not a controlled A/B: same box, different binary,
  and the pre-merge numbers came from a `build-metal` of 2026-11-24.
- **The tokens agree across all seven packs for 24 of 24 steps except the last**, which is the known
  `25562` / `14162` near-tie (Q4_K_M, Q6_K and UD-Q2_K_XL give 25562; the rest 14162). The whole
  24-token sequence before it is identical, so the new packs produce the same text as the rest.
- llama.cpp (CPU + BLAS, no Metal backend, `8bed7cfb9`) covers the five packs it can read, 14.7–18.4 s
  for the same 24 tokens. The 2.7–4.6× gap is the port's readback cost, as documented before; it is
  not what the card could do.

## Suite: Qwen3.8 scaling, threads, edges

`UD-IQ1_M` native pack with the DFlash2 drafter (`dflash2.pack`), 2 new tokens:

| case | prefill | wall |
|---|---|---|
| prompt 8 | 7 tokens in 4 440–4 478 ms (1.56–1.58 tok/s) | 28.3 s |
| prompt 32 | 31 tokens in 10 423–10 763 ms (2.88–2.97 tok/s) | 34.1–34.9 s |
| prompt 128 | 127 tokens in 36 387–36 666 ms (3.46–3.49 tok/s) | 60.8–60.9 s |
| prompt 512 | 511 tokens in 148 354–149 415 ms (3.42–3.44 tok/s) | 172.9–173.9 s |
| prompt 2048 (one run, by design) | 2 047 tokens in 629 338 ms (3.25 tok/s) | 654.1 s |

A case name is one more than the tokens the engine prefills (the first sampled token is not
prefilled): "prompt 8" prefills 7, "prompt 2048" prefills 2 047. An earlier version of this table
printed one run's milliseconds beside the other run's token rate, and 1.8 tok/s for prompt 8; both
are corrected here from `results-suite.json`.

The edge block behaves: `1-token prompt`, `verify block 2`, `no GPU expert cache`, `no CPU pool`,
`pool-workers 64` and `max-context 4096` all run; the four invalid cases
(`prompt 64` with `--max-context 16`, `--max-new 0`, `--spec 1`, `--spec 64`) and an unknown flag are
refused, which is what they are for.

**The suite's draft counter is not 0 of 7 in every case** — an earlier version of this page said it
was, and the run's own `results-suite.json` says otherwise. It reads **0 of 7 at an 8-token prompt,
2 of 7 at 32 and 128, and 7 of 7 at 512 and 2048**. That is the prompt, not the drafter: `run_suite.py`
builds every prompt by repeating one 17-token cycle (`BASE`), so at 512 and 2048 tokens the next
tokens are the cycle's next tokens and target and drafter agree on them trivially. Re-run of the
512-token case on the current build (2026-11-25): `speculation 2 rounds of 8, drafts accepted 7 of 7
(1.000), 4.50 tokens per round`, output `42498 456` (the cycle continuation), 0.50 tok/s decode,
3.39 tok/s prefill. Speed does not improve across these cases (0.54 / 0.54 / 0.51 / 0.50 / 0.47 tok/s
at 8 / 32 / 128 / 512 / 2048), so this is not evidence the drafter helps on real text. The real-text
result is unchanged at **0 of 105** ([macpro51-2026-11-23-test-pass.md](macpro51-2026-11-23-test-pass.md)),
and the open question remains the context the drafter is fed.

## Thread sweep

39 runs over `--pool-workers` 1–12 and 24 on the same Qwen3.8 pack: **99.3 s to 101.2 s, mean 100.3 s** —
flat. On this card the CPU expert pool is not the lever; the window's GPU/readback path is.

## All models

26 records: 8 `qwen36` entries (the 7 packs plus `mmproj-F16.gguf`, which the engine refuses with
`architecture is 'clip'` — it is a vision projector, not a model), all 17 `qwen38` directories, and
one `deepseek-v4.1-flash` record (the ds4 engine still refuses without
`STRATA_DS4_ALLOW_UNSTABLE_METAL=1`, and that flag resets the card).

## What the layout move needed in the harness

The 2026-11-25 merge moved `bench/macpro51` to `terminal/bench/macpro51` and `data/` to
`terminal/data/`, which broke three things in the sweep scripts. All are fixed in this commit:

- `REPO = HERE.parents[1]` counted directories, so it resolved to `terminal/`; the runners now find
  the clone by its `.git`.
- the engine binary was hardcoded as `build-metal/…`; it is now `build-merge` by default with
  `STRATA_BENCH_BUILD=<dir>` to override.
- `--expert-profile data/expert-profile.bin` and `--mtp-draft-vocab data/draft_vocab.bin` were
  relative, so every Qwen3.8 case was refused with "the draft vocabulary … does not exist"; they are
  absolute `terminal/data/…` paths now.

The runners resume from their own `results-*.json`, skipping `(case, round)` pairs already recorded —
useful after an interruption, but it means a sweep must start from cleared result files (or it
finishes in zero seconds, as this one first did). The CSV/SVGs beside the results were regenerated.
