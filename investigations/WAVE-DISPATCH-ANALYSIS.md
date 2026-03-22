# Wave Packet Dispatch — Dependency Analysis

## The spec's overlap claim

The MVP spec proposes overlapping FFN/ODE (GPU, ~8ms) with attention softmax (CPU, ~10ms)
within the same block. This **does not work** due to data dependencies:

```
attention(QKV) → attn_out
h1 = hidden + attn_out           ← depends on attention
normed_2 = LN2(h1)               ← depends on h1
FFN(normed_2) → ffn_out          ← depends on normed_2
output = h1 + ffn_out
```

FFN depends on attention through the residual connection. They cannot run in parallel.

Cross-block overlap also fails: block N+1 needs `hidden = h1 + ffn_out` from block N.

## What CAN overlap (within wgpu)

### Option A: Submit-before-readback (~1-2ms gain)

Between FFN submit and FFN readback, do cache construction:
```
submit FFN encoder (GPU starts, async)
build AttnCache, BlockCache (CPU, ~1-2ms)
readback FFN result
```
~1-2ms per block = ~8ms total. Negligible (<1% of 1.2s iter).

### Option B: Parallel Attention + FFN (GPT-J formulation)

Change the block computation from sequential to parallel:
```
Sequential (current): x = x + attn(LN1(x));  x = x + FFN(LN2(x))
Parallel (GPT-J):     x = x + attn(LN(x)) + FFN(LN(x))
```

In parallel form, attention and FFN both take `normed = LN(x)` as input.
They share NO data during computation. GPU runs FFN while CPU runs attention.

This IS the wave packet dispatch — two independent oscillators (GPU ODE, CPU attention)
operating on the same input in phase, with their outputs summing at the boundary.

**The catch:** This is a MODEL ARCHITECTURE change, not just a dispatch optimization.
- Breaks baseline (different computation graph)
- Changes training dynamics (single LN vs two LNs, simultaneous vs sequential)
- Needs its own validation (quality parity test)
- GPT-J, PaLM, and some modern architectures use this and report equivalent quality

### Option C: Backward overlap

The backward is 315ms/block (13x forward). The backward has:
- FFN backward recompute (GPU calls) ~50ms
- Attention backward (CPU per-sequence) ~100ms
- LN backward, gradient accumulation ~165ms

The backward recompute's GPU calls and the attention backward's CPU work
could potentially overlap — but the backward has the same dependency issue
(FFN backward needs attention backward's gradients for the chain rule).

## Recommendation

Option B (parallel attention + FFN) is the only one that gives meaningful overlap.
It's an architecture change. Test it as a new investigation:
1. Implement parallel block variant (single LN, attention + FFN share input)
2. Train at 128-dim, compare val loss to sequential
3. If quality parity: enable overlap, measure GPU%
4. If quality loss: document and move on

This is ~50 lines of code change in the forward block, not a dispatch restructure.
