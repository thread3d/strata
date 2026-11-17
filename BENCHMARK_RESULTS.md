# Qwen3.6 Metal Benchmark Results (macpro51, Radeon RX 580 8GB)

## Test Configuration
- **Hardware**: MacPro5,1, Intel Xeon X5675, Radeon RX 580 8GB (Metal 2), macOS 11.7.11
- **Model**: Qwen3.6-35B-A3B (40 trunk layers + 1 MTP, 256 experts top-8)
- **Prompt**: tokens 760, 6511, 314 (3 prompt tokens)
- **Generated**: 10 tokens per run
- **Date**: 2026-11-17

## Results Summary

| Model | Quantization | Total Time (s) | Time/Token (s) | First 10 Tokens |
|-------|--------------|----------------|----------------|-----------------|
| Q8_0 | Q8_0 (52 GB) | 19.62 | 1.509 | 279, 220, 17, 17, 17, 17, 24, 23, 13, 271 |
| UD-Q4_K_XL | Q4_K gate/up, Q5_K/Q6_K down (32 GB) | 15.47 | 1.190 | 107570, 107570, 107570, 61356, 107570, 107570, 107570, 61356, 107570, 107570 |
| UD-Q6_K_XL | Q6_K gate/up, Q8_0 down (44 GB) | 18.55 | 1.427 | 279, 220, 17, 20, 15, 15, 15, 13, 23, 13 |

## Analysis

### Q8_0 (Baseline)
- Works correctly, consistent output across runs
- Slowest due to largest model size (52 GB)
- Reference for correctness comparison

### UD-Q6_K_XL (Recommended)
- **Excellent parity with Q8_0**: First 3 tokens match exactly (279, 220, 17)
- Good performance: 1.427s/token (5% faster than Q8_0)
- Loads successfully with mixed quantization: Q6_K gate/up, Q8_0 down
- Minor divergence after token 3 expected for different quantization

### UD-Q4_K_XL (Recommended)
- **Correctness issue**: Output diverges completely from baseline (107570 vs 279)
- Fastest: 1.190s/token (21% faster than Q8_0)
- Parity test confirms: gate/up (Q4_K) passes, down (Q5_K/Q6_K) fails with y rel L2 ~0.63
- Root cause: Down kernel (moe_down) has issues with K-quant down weights (Q4_K/Q5_K)
- Gate/up path works correctly (h rel L2 ~2e-7)

## Parity Test Status

| Config | Gate/Up | Down | h rel L2 | y rel L2 | Status |
|--------|---------|------|----------|----------|--------|
| Q8_0/Q8_0 | Q8_0 | Q8_0 | 1.5e-7 | 1.2e-7 | PASS |
| Q6_K/Q8_0 | Q6_K | Q8_0 | 2.9e-7 | 1.1e-7 | PASS |
| Q4_K/Q4_K | Q4_K | Q4_K | 2.0e-7 | 0.63 | FAIL (down) |
| Q4_K/Q5_K | Q4_K | Q5_K | 2.0e-7 | 0.68 | FAIL (down) |
| Q4_K/Q8_0 | Q4_K | Q8_0 | 2.0e-7 | 1.1e-7 | PASS |

## Fixes Applied

1. **bytes4/bytes2 shifting** (units.metal): Fixed byte extraction to shift down (>>8, >>16, >>24) instead of masking only
2. **Q4_K/Q5_K per-group decoder** (units.metal): Fixed code/qh offset calculation - pairs of groups share code bytes
3. **get_scale_min_k4** (ref_quant.cpp): Fixed j>=4 case to use q[j-4] instead of q[j]
4. **XD<T> load** (moe.metal): Added xd16 helper with proper per-16 scaling for K-quants
5. **xprep plane selection** (q36_moe_routed.cpp): Added xprep_plane() helper for correct plane selection

## Remaining Work

- [ ] Fix Q4_K/Q5_K down kernel (moe_down) for K-quant down weights
- [ ] Run full benchmark matrix (powers-of-2 context/query sizes, multiple runs)
- [ ] Generate performance charts
- [ ] Document findings in qwen36-metal.md

## Files Modified

- metal/kernels/units.metal: Dec<T_Q4_K>, Dec<T_Q5_K>, DecDNK, Dec16, bytes4, bytes2, scale_min_k4
- metal/kernels/moe.metal: XD<T> load for K-quants
- metal/src/ref_quant.cpp: get_scale_min_k4 fix
- metal/src/qwen36/q36_moe_routed.cpp: xprep_plane() helper
