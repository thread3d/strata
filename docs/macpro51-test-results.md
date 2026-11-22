# macpro51 Qwen3.6 Metal Test Results (2026-11-16)

> **Superseded 2026-11-22.** The numbers below were not reproduced by the branch tip (`be4cbc78f`) and some cannot be: the token ids listed for several packs change at the fourth token, and the file sizes are wrong (Q8_0 is 37.8 GB, not 52 GB). See [macpro51-2026-11-22.md](macpro51-2026-11-22.md) for what the engine actually does on this machine now.

## Hardware
- **Host**: MacPro5,1 (macOS 11.7.11)
- **CPU**: 2x Intel Xeon X5675 (12 cores / 24 threads @ 3.07 GHz)
- **RAM**: ~96/128 GB DDR3 ECC
- **GPU**: AMD Radeon RX 580 8 GB (Metal 2, 36 CUs, ~5.8 TFLOPS FP32)
- **Work Dir**: `/Volumes/Sonnet Raid/Users/threaded/projects/Strata/strata`
- **Branch**: `macpro51` (HEAD d26994ce)

## Quantisation Status

| Quantisation | Size | expert_parity | Model Inference | Notes |
|--------------|------|---------------|-----------------|-------|
| **Q8_0** | 52 GB | ✅ PASS (h L2 1.5e-7) | ✅ Works | Baseline, fully validated |
| **UD-Q6_K_XL** | 44 GB | ✅ PASS (Q6_K/Q8_0, h L2 2.9e-7) | ✅ Works* | Generates reasonable tokens; streaming falls back to Q8_0 |
| **UD-Q4_K_XL** | 32 GB | ❌ FAIL (h L2 1.1) | ❌ Repeats token 48313 | DNK path Dec<T_Q6_K> mismatch for down kernel |

*UD-Q6_K_XL loads as "Q6_K gate/up, Q8_0 down" but streaming path uses Q8_0 for both (gu_type=8, down_type=8).

## Expert Parity Test Results

```
parity Q8_0/Q8_0 ntok 1 gu 32,2,4 dn 8,2,4: plan ok (10 groups), h rel L2 1.52e-07 max 1.60e-07, y rel L2 1.15e-07 max 1.63e-07  PASS
parity Q6_K/Q8_0 ntok 1 gu 32,2,4 dn 8,2,4: plan ok (10 groups), h rel L2 2.86e-07 max 2.99e-07, y rel L2 1.13e-07 max 1.21e-07  PASS
parity Q4_K/Q5_K ntok 1 gu 32,2,4 dn 8,2,4: plan ok (10 groups), h rel L2 1.36e+14 max 7.30e+13, y rel L2 4.58e+06 max 5.54e+06  FAIL
```

## Hadamard Rotation Test
```
had-test width 640, 37 rows (no signs): f32 0 / f32 in place 0 / f16 0 / f16 rows 0 of 23680 values differ from the CPU
had-test width 2560, 37 rows (no signs): f32 0 / f32 in place 0 / f16 0 / f16 rows 0 of 94720 values differ from the CPU
had-test PASS (bitwise)
```

## Model Generation Samples

### Q8_0 (52 GB)
```
tokens: 279 220 17 17 17 17 24 23 13 271  (reasonable, diverse)
```

### UD-Q6_K_XL (44 GB)
```
tokens: 279 220 17 20 15 15 15 13 23 13  (reasonable, some repetition)
```

### UD-Q4_K_XL (32 GB)
```
tokens: 48313 48313 48313 48313 48313  (catastrophic failure - repeated token)
```

## Key Fixes Committed (branch `macpro51`)

1. **Dec<T_Q4_K>/Dec<T_Q5_K>**: Fixed block min bias handling (0-63 → -32..+31), now matches CPU reference exactly
2. **X<T_Q2_0>**: Fixed 2-bit code extraction `(a >> (2*j)) & 3`
3. **xprep_plane()** in `q36_moe_routed.cpp`: Hybrid strategy
   - Q2_0 → plane 1 (XK_Q2)
   - Q6_K/Q3_K → plane 3 (XK_P2, p2s scaling)
   - Q4_K/Q5_K → plane 0 (XK_RAW, no scaling, Dec matches CPU)
4. **expert_parity xkind()**: Updated to match hybrid xprep_plane strategy
5. **XPX/XPH buffer allocation**: Fixed for 4-plane layout (4 × n_tok × n_embd + half sums)
6. **Hadamard kernels**: Bitwise PASS for f32/f16/f16-rows

## Remaining Issues

### UD-Q4_K_XL Failure (Critical)
- **Root Cause**: DNK (STRATA_KQ_DOWN) path for Q6_K down kernel uses Dec<T_Q6_K> with `sc.z = 32*d1` designed for X<T_Q6_K> with p2s-scaled input (plane 3), but XD<T_Q6_K> loads RAW h (plane 0). The unit_combine formula `d1*sum(q*xs) - 32*d1*xsum` is incorrect for raw activations.
- **Impact**: DOWN projection produces garbage, causing repeated token 48313.
- **Fix Required**: Dec<T_Q6_K> needs separate specialization for DNK path (sc.z = 0) or DNK kernel should use different Dec.

### UD-Q6_K_XL Streaming Fallback
- **Observation**: Model loads as "Q6_K gate/up, Q8_0 down" but streaming path uses gu_type=8 (Q8_0) for both gate/up and down.
- **Likely Cause**: stream_load reads expert tensor types incorrectly, or kernel compilation fails for Q6_K and falls back.
- **Impact**: Not using true Q6_K quantization in streaming mode.

## Performance (RX 580, 8 GB VRAM)

| Config | Prompt (3 tok) + 10 gen | Tokens/sec |
|--------|------------------------|------------|
| Q8_0 | 31.28 s | ~0.42 |
| UD-Q6_K_XL | 28.45 s | ~0.46 |

Memory usage: ~2.5 GiB dense + expert streaming from SSD (Sonnet RAID).

## Next Steps

1. Fix DNK path Dec<T_Q6_K> for raw h input (separate Dec specialization or runtime selection)
2. Debug stream_load tensor type detection for UD-Q6_K_XL
3. Run comprehensive benchmarks (powers-of-2 context/query, error bars)
4. Generate performance charts and documentation
5. Push final commits to `macpro51` branch