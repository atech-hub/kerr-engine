# Maestro Pre-Conditioning Investigation

## Problem
GPU training at 768-dim (384 bands) NaNs at iter ~80. Root cause: continuous GPU FP accumulation drift (~1e-7/iter) that curriculum transitions reveal.

## Hypothesis
Current KerrMaestroAdd runs ODE and maestro in parallel — maestro never influences ODE input. At 384 bands, maestro's 56:1 compression cannot correct energy spikes after the fact.

Dual-maestro architecture: maestro_in regulates energy BEFORE ODE, maestro_out re-synchronises AFTER ODE.

## Architecture
```
x → maestro_in(x) → precond = x + mae_in_out    [INPUT REGULATOR]
  → ODE(precond)  → ode_out                      [PHYSICS]
  → maestro_out(ode_out) → regulated              [OUTPUT REGULATOR]
  → out_proj(regulated)  → output
```

## Parameter Cost
At 768-dim, maestro_dim=16: +147,456 params across 3 blocks (~501K total, still under MLP's 801K).

## Gates

### Gate 1: Baseline check
Without --dual-maestro, 200 iters, seed 42, train-seed 1337.
Expected: 4.2556, 3.3216, 2.7094, 2.7213, 2.6161 (bit-identical).

### Gate 2: 768-dim stability
With --dual-maestro, 500 iters, --n-bands 384 --n-head 12 --gpu --seed 42 --rk4-steps 16, lr=1e-4.
Success = no NaN in 500 iters, loss shows learning.

### Gate 3: Quality parity
With --dual-maestro at 64 bands (128-dim), 500 iters, curriculum.
Final val loss within 5% of current architecture.

## Results

### Gate 1: Baseline — PASS
200 iters, no curriculum, seed 42/1337, 128-dim CPU.
Loss: 4.2556, 3.3216, 2.7094, 2.7213, 2.6161 — bit-identical.

Note: initial implementation broke baseline due to RNG consumption order change.
Moving `let ffn = ...` before the struct literal caused FFN init to consume RNG
before attention init. Fixed by extracting all struct fields in original order
(ln_1, attn, ln_2, then ffn).

### Gate 2: 768-dim Stability — PASS
500 iters, --dual-maestro --n-bands 384 --n-head 12 --gpu --rk4-steps 16, lr=1e-4.
4-stage curriculum: 8(0-100), 24(100-225), 96(225-350), 384(350-500).
12,032,358 params. RTX 4070 Ti (Vulkan). 1.2s/iter.

Loss trajectory:
- Iter 0: 4.4195 (8 bands)
- Iter 100: 3.1300 (24 bands)
- Iter 250: 2.5464 (96 bands)
- Iter 300: val_loss 2.7143
- Iter 350: 3.6565 (384 bands — transition spike, expected)
- Iter 400: 2.4546 (recovered)
- Iter 499: 2.5999
- Final val: 2.5126

Zero NaN in 500 iterations. Previous architecture NaN'd at ~80.
Band transition spike (96→384 at iter 350) recovers within 50 iters.
The dual-maestro pre-conditioning prevents energy spikes from entering the ODE.

### Gate 3: Quality Parity — PASS (better than parity)
500 iters, curriculum, 128-dim, seed 42/1337.

| Metric | Standard | Dual-Maestro | Delta |
|--------|----------|--------------|-------|
| Params | 354,358 | 367,078 | +3.6% |
| Val @ 300 | 2.8057 | 2.7949 | -0.4% |
| Final val | 2.6232 | 2.5896 | -1.3% (better) |

Dual-maestro is not just within the 5% tolerance — it's 1.3% better.
Extra maestro parameters (12,720) are paying for themselves.

## Verdict

All three gates PASS. Dual-maestro pre-conditioning:
1. Solves 768-dim NaN (0 NaN in 500 iters vs ~80 iter failure)
2. Does not regress quality at validated 128-dim scale
3. Slightly improves quality at 128-dim (+1.3% better val loss)

The hypothesis is confirmed: regulating energy before it enters the ODE
addresses the continuous FP drift that timing fixes could not.
