# Scaling Investigation — NaN Isolation Tests
Date: 2026-03-17

## Key Finding: NaN is GPU-SPECIFIC, not architectural

CPU at 768-dim with lr=1e-4 + rk4=16: CONVERGES (loss 4.37 → 2.86 at iter 100)
GPU at 768-dim with lr=1e-4 + rk4=16: NaN at iter 50-100

Same model, same seed, same settings. The difference is the compute backend.
The GPU matvec shader produces slightly different floating point results
than CPU due to FMA/order-of-operations differences. At 768-dim these
differences compound through RK4 integration and cause divergence.

## Test Matrix

| Test | Dim | LR | RK4 | Backend | Result |
|---|---|---|---|---|---|
| 1 | 768 | 1e-4 | 8 | GPU | NaN iter 50 |
| 2 | 768 | 3e-4 | 16 | GPU | NaN iter 50 |
| 3 | 768 | 1e-4 | 16 | GPU | NaN iter 100 |
| 4 | 768 | 1e-4 | 16 | GPU (fixed pool) | NaN iter 100 |
| 5 | 768 | 1e-4 | 16 | GPU (seed=123) | NaN iter 100 |
| 6 | 768 | 1e-4 | 16 | CPU | **CONVERGES** (2.86 at iter 100) |

## Root Cause: TWO ISSUES FOUND

### Issue 1: Buffer pool data caching (FIXED)
The pool cached input data by pointer identity via `ensure_data()`. But input
slices (`x`, `w_flat`) are stack-allocated or freshly flattened — same pointer
can point to different data between dispatches. The GPU served stale data,
corrupting the forward pass. Fix: remove `ensure_data`/`data_ref`, use fresh
`storage_buf()` for all inputs. Only cache scratch/uniform/staging buffers.

Evidence: with data caching, NaN at iter 50. Without data caching, NaN at iter 250.

### Issue 2: f32 accumulation error (PARTIALLY FIXED with Kahan)
The matvec shader accumulates 768 multiply-adds in a raw loop. Kahan
compensated summation reduces error from O(n) to O(1). This extends
stability from iter 50 → 200 for forward pass. But the NaN still occurs
at curriculum transitions (24-band opening) suggesting the issue also
exists in backward pass or is amplified by the sudden band change.

Evidence: forward Kahan extends life from 50 → 200. CPU never NaN's.

### Issue 3: Curriculum transition shock (REMAINING)
At iter 166 (768-dim, 500-iter run), 24 bands open simultaneously.
The model's ODE dynamics change abruptly, producing large gradients
that the optimizer amplifies. At lower dims this recovers. At 768-dim
the GPU accumulation errors compound through the gradient explosion.
CPU handles it because its accumulation is exact (sequential f32).

Possible fixes: gentler curriculum ramp, gradient norm monitoring,
per-band gradient clipping, or Kahan in ALL shaders (backward too).

## Potential Fixes

1. **Kahan summation in shader** — compensated accumulation reduces float error
2. **Mixed precision** — accumulate in f64, store in f32 (WGSL supports f64 if device supports it)
3. **Reduce accumulation error** — tile the matmul so each thread accumulates fewer terms
4. **Higher clamp threshold** — catch pre-NaN magnitudes and clamp

## Scaling Results (validated dimensions)

| Dim | Bands | Val Loss | Backend | Time/iter | Status |
|---|---|---|---|---|---|
| 128 | 64 | 2.20 | CPU | 67ms | PASS (baseline) |
| 256 | 128 | 2.12 | CPU | 155ms | PASS |
| 384 | 192 | 2.13 | CPU | 340ms | PASS |
| 512 | 256 | 2.13 | CPU | 850ms | PASS |
| 768 | 384 | converging | CPU | 1.6s | PASS (lr=1e-4, rk4=16) |
| 768 | 384 | NaN | GPU | 1.2s | FAIL (GPU numerical issue) |

## Buffer Pool Gains (measured before NaN fix needed)

| Dim | Before pool | After pool | Speedup |
|---|---|---|---|
| 768 | 1.7s | 0.87s | 2.0x |
| 896 | 3.5s | 1.3s | 2.7x |

## Block-Change Weight Caching (implemented)

Weights cached by source pointer — flattened + uploaded once per iteration,
reused for all dispatches. Only re-uploaded after optimizer step invalidates
cache. Eliminates ~40MB redundant weight uploads at 768-dim (320 dispatches
× 128KB per weight matrix = 40MB wasted bandwidth per iteration).
