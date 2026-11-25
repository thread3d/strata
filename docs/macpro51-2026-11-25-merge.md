# The main merge — macpro51, 2026-11-25

**Branch `macpro51`, merge commit `4c5b58da3`** (parents `43b0b04da` + `2af30ccaa`), annotated tag
`working4`. Engine `strata 0.1.67-metal`. The branch was **676 ahead / 2448 behind** `origin/main`
when the merge started (the 2448 only became visible after a fetch: the remote-tracking ref had been
stale since 2026-11-23); it is now **0 behind / 677 ahead**. Full suite on this machine: **22 passed,
2 skipped, 0 failed**.

## What main had done since the branch forked

Two things turned this from a text merge into a migration:

- **The tree moved.** `metal/` → `engine/metal/`, `src/` → `engine/src/`,
  `include/strata/` → `engine/include/strata/`, `tests/` → `engine/tests/`,
  `bench/` → `terminal/bench/`, `apps/` → `app/`. The root `CMakeLists.txt` is gone; the engine builds
  from `engine/CMakeLists.txt` (CUDA/HIP) or `engine/metal/CMakeLists.txt` (Metal, a CMake project of
  its own, as `metal/` used to be).
- **DeepSeek V4.1 was retired.** `DEEPSEEK41.md` and `DEEPSEEK_MAC.md` on main say "Strata no longer
  runs DeepSeek V4.1-Flash. Version 0.2.13 was the last one that set it up and started it", and the
  whole `dsv41` / ds4 source tree is gone.

The merge was done with rename detection enabled (`-c merge.renameLimit=100000`): without it git
paired none of the moved files and reported 117 conflicts. With it, 82 were "added by us" (the files
git relocated correctly, mostly the q36 engine), 17 were real content merges and 16 were
"modified by us, deleted by them" (the ds4 engine).

## What this branch keeps

- **The Qwen3.6-35B-A3B Metal engine**, now at `engine/metal/src/qwen36/` (host side) with its checks
  in `engine/metal/tests/`. Built as `strata_q36_model` / `strata-q36-metal`; 17 q36 checks are
  registered with ctest.
- **The 64-lane simdgroup port.** The `lane32` / `sg32` / `sum32` / `max32` / `prefix_excl32` prologue,
  `STRATA_SIMD_WIDTH`, `STRATA_MSL_GRID_FENCE` and `STRATA_HAVE_MPP` in `engine/metal/src/mtl_rt.mm`,
  and the kernels that take their lane/group through them. main has no such helpers (its kernels are
  written for a 32-lane Apple simdgroup); the RX 580's simdgroups are 64 lanes wide.
- **The DeepSeek ds4 Metal port**, restored from `13eedf552` (the merge base, the last commit that
  carried it) into the new layout: `engine/src/dsv41` (51 files), `engine/include/strata/dsv41` (6),
  `engine/metal/src/engine/{ds4_*,exl3_legacy_stub}.cpp`, `engine/metal/src/engine/msl/ds4_*.metal`,
  `engine/metal/tests/`, `engine/metal/include/metal_compat/ds4_*`, `data/dsv41` and
  `packaging/mac/build_dsv41_engine.sh`. The `strata_ds4` target and its `STRATA_DS4_RUNTIME` object
  libraries are back in `engine/metal/CMakeLists.txt`. This was a deliberate choice — main retired the
  engine; the branch keeps its Metal port.
- **Kernel registrations are the union of both sides:** main's newer TQP5/6/7 and NVFP4 kernels plus
  the port's BF16 / IQ1_M / IQ3_XXS expert registrations (the UD-Q8_K_XL pack's bf16 experts).

## `generate.cpp`: an accidental revert was dropped

`be4cbc78f` ("Complete benchmarking work…", the 94-file `git add -A` commit of 2026-11-21) changed
`generate.cpp` by **174+/1028−**. That was not port work: before it the file was base plus the
branch's four real commits (19+/3−), and the 177 "added" lines it brought are older variants of
mainline code (`>= 48` layer hardcodes, older `stage_room()`, older `--draft-cap` help). It also
clobbered two of the branch's own fixes. So the merge takes **main's file** and re-applies the four
real changes: `15f0fb59b` (native/IQ run path, out-of-vocabulary refusal), `d5c7b86ac` +
`0730145be` (refuse `--spec` on a pack with no MTP layer, and the corrected question it asks) and
`ed441a17c` (no adaptation thread without a host residency table).

## Path map

| before 2026-11-25 | now |
|---|---|
| `metal/src`, `metal/kernels`, `metal/tests` | `engine/metal/src`, `engine/metal/kernels`, `engine/metal/tests` |
| `src/...` (engine host code, dsv41) | `engine/src/...` |
| `include/strata/...` | `engine/include/strata/...` |
| `bench/macpro51/` | `terminal/bench/macpro51/` |
| `docs/benchmarks/macpro51/` | unchanged |
| root `CMakeLists.txt`, `metal/CMakeLists.txt` | `engine/metal/CMakeLists.txt` (own CMake project) |

Docs written before this merge still use the old paths in their command lines; the mapping above
applies to them.

## Building on this machine

```sh
cmake -S engine/metal -B build-merge -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
      -DMETAL_CPP_DIR=~/projects/metal-cpp/metal-cpp \
      -DSTRATA_GGML_DIR=~/strata/third_party/strata-ggml \
      -DSTRATA_BUILD_TESTS=ON
ninja -C build-merge -j 24          # exit 0, 0 errors
```

Two details worth keeping:

- **`-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0` is required.** `engine/metal/CMakeLists.txt` defaults to
  15.0; with that target `__builtin_available(macOS 15.0, *)` is compile-time true, so the
  residency-set path runs on Big Sur and `runtime_hello` dies with an unrecognised selector
  (`-[BronzeMtlDevice newResidencySetWithDescriptor:error:]`). With 11.0 the check is a runtime one
  and the test returns 77 (skipped).
- **ggml source matters for the quant golden.** `-DSTRATA_GGML_DIR=~/projects/llama.cpp` (8bed7cfb9)
  builds, and every test passes except `quant_parity`: its leg 1 compares input digests against
  `engine/tests/quant/golden_x86.txt`, which was made with the pinned vendored ggml, and the iq1_m
  group's rows differ. Pointing at the vendored tree (`~/strata/third_party/strata-ggml`) makes it
  **pass, 283 group checks, 0 failures**. Both configurations are recorded below; the build above uses
  the vendored tree.

## Tests, 2026-11-25

`ctest --test-dir build-merge -j 4` — **22 passed, 2 skipped, 0 failed** (58.7 s):

- 17 q36 checks pass (`q36_model_check`, `q36_dense_parity`, `q36_attn_parity`,
  `q36_attn_layer_parity`, `q36_gdn_layer_parity`, `q36_glue_parity`, `q36_rope_parity`,
  `q36_router_parity`, `q36_moe_shared_parity`, `q36_moe_routed_parity`, `q36_expert_blob_check`,
  `q36_layer_check`, `q36_bind_check`, `q36_forward_check`, `q36_stream_check`, `q36_generate_check`,
  `q36_upload_check`), plus `moe_tile_grid`, `nvfp4_decode_model_test`, `quant_selftest`, `type_geometry_oracle`
  and `quant_parity`.
- skipped: `runtime_hello` (77: Big Sur has no `MTLBuffer.gpuAddress`) and `prompt_workspace_growth`.
- With `-DSTRATA_GGML_DIR=~/projects/llama.cpp` instead: 22 passed, 2 skipped, 1 failed
  (`quant_parity`, the iq1_m input digest above).

Diagnostics that are not part of ctest, run on the merged tree against the RX 580:

| check | result |
|---|---|
| `dsu_ref_vs_ggml` | 12 of 12 bit-exact (Q8_0, IQ4_XS, IQ3_S, IQ4_NL, …) |
| `down_kernel_emu` | rel L2 1.989e-07, max abs 7.818e-08 |
| `expert_parity --gu Q8_0 --dn Q6_K --ff 512 --embd 2048` | h 1.51e-07, y 2.31e-07 — PASS |
| `expert_parity --gu IQ2_XXS --dn Q2_0 --embd 2560 --ff 7680` | h 1.46e-07, y 1.83e-07 — PASS |
| `msl_dump --kq`, `msl_dump --kq --f16` | every library compiles |

## Charts

`terminal/bench/macpro51/charts*.py` regenerated `docs/benchmarks/macpro51/` from the same
`results-*.json` records. Only `same-workload-wall.csv` / `.svg` moved, by tenths of a second
(Q4_K_M 49.3 → 49.8 s, Q6_K 52.4 → 52.6, UD-Q6_K_XL 57.0 → 56.5, Q8_0 60.6 → 60.8); the committed
CSV had been produced by an older revision of `charts_reference.py`. The scripts also had to stop
counting parent directories to find the repo root (that broke when `bench/` moved under `terminal/`):
they now locate the clone by its `.git`. Their usage lines say `terminal/bench/macpro51/...`.

## Not done in this pass

- No model pack was loaded and no inference or benchmark was re-run after the merge. The last measured
  behaviour of each model is the pre-merge state in
  [macpro51-2026-11-24-models-that-run.md](macpro51-2026-11-24-models-that-run.md).
- `strata_ds4` still refuses to run on this card by design (it needs
  `STRATA_DS4_ALLOW_UNSTABLE_METAL=1`, which resets the GPU); the restored port is compile- and
  link-verified, not run.
- main's Metal keep-alive (#6078) is enabled in the runtime constructor; on the 8.6 GB working set here
  it can pin arena memory, and `STRATA_METAL_KEEPALIVE_MS=0` turns it off. The port does not add its
  own buffers to the kept-alive set.
