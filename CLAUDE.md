# CLAUDE.md — Kerr Engine Internal Reference

> Internal developer docs for Claude Code. Terse, factual, no narrative.
> Last updated: 2026-03-12 (end of Chat 5)

---

## What this is

Pure Rust inference and training engine for the Kerr-ODE architecture from the Wave Coherence project. No Python, no PyTorch, no CUDA toolkit. Dependencies: wgpu, bytemuck, pollster.

Parent project: github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive (public, MIT).
This repo: private. Contains implementation IP, not research findings.

---

## Architecture constants

| Constant | Value | Why |
|---|---|---|
| N_BANDS | 64 | Harmonic frequency bands. Sweet spot from frequency budget curve. |
| N_EMBD | 128 | = N_BANDS × 2 (real + imaginary per band) |
| N_HEAD | 4 | Attention heads |
| HEAD_DIM | 32 | = N_EMBD / N_HEAD |
| BLOCK_SIZE | 256 | Max sequence length |
| MAESTRO_DIM | 16 | Maestro bottleneck width. dim=16 optimal (5d finding). |
| RK4_N_STEPS | 8 | ODE integration steps per layer |
| RK4_DT | 0.125 | = 1.0 / RK4_N_STEPS |
| N_LAYERS | 4 | Transformer blocks |

## Model structure

- **Block 0:** Attention + PerBandLinear (analytical L0 — initial projection, 2×2 matrix per band)
- **Blocks 1-3:** Attention + KerrMaestroAdd (Kerr-ODE with maestro sync + additive residual)
- **Embeddings:** Frozen harmonic table. cos(n·θ)/sin(n·θ) where θ = token × 2π/vocab. Not trainable.
- **Positional:** Frozen sinusoidal encoding (standard transformer style). Not trainable.
- **FFN path (blocks 1-3):** Kerr-ODE output + Maestro output → add → out_proj linear
- **Maestro:** Global sync bottleneck. Squeeze (128→16) → GELU → Process (16→128). Additive, not multiplicative.

---

## Module map

| Module | Lines | Role |
|---|---|---|
| `backward.rs` | 541 | Gradient primitives: linear, layer_norm, GELU, softplus, cross-entropy, Kerr-ODE derivative backward, RK4 step backward, full Kerr-ODE backward, attention backward, maestro backward. All hand-derived, no autograd. |
| `pipeline.rs` | 526 | Cached forward pass (saves activations for backward) + full backward chain that orchestrates primitives from backward.rs. Cache structs (ForwardCache, BlockCache, AttnCache). GradAccum struct. The contract: what forward saves, backward consumes. |
| `model.rs` | 498 | Weight structs, architecture constants, inference-only forward pass, Kerr-ODE forward, maestro forward, attention, per-band linear, RK4 integration, harmonic table construction. Source of truth for what the computation is. |
| `optim.rs` | 252 | Adam optimizer with bias correction. Parameter flatten/unflatten (structured weights ↔ flat f32 vec). Gradient flattening. Gradient norm clipping. `count_params`. |
| `backend.rs` | 228 | `ComputeBackend` trait (8 methods: linear, linear_no_bias, layer_norm, kerr_ode, maestro, attention, per_band_linear, kerr_maestro_add). `CpuBackend` implementation. Trait exists for future GPU backend — not speculative, the three tiers (CPU/hybrid/full-GPU) are documented requirements. |
| `gpu.rs` | 209 | WGPU device setup, buffer allocation, compute shader dispatch for Kerr-ODE step. Currently used only for Stage 1 validation. Will become `GpuBackend` when GPU training is implemented. |
| `main.rs` | 192 | CLI dispatch only. Four subcommands (see below). Stage 1/2 validation logic lives here temporarily. |
| `grad_test.rs` | 184 | Stage 3 gradient validation harness. Loads PyTorch reference gradients, compares against Rust analytical gradients. |
| `weights.rs` | 175 | Binary weight loader. Reads model.bin exported from Python. Loads test vectors for Stage 2 validation. |
| `train.rs` | 144 | Training loop orchestration. Batch iteration, loss logging, gradient accumulation across batch, Adam step, periodic text generation. Minimal — delegates to pipeline.rs for forward/backward, optim.rs for stepping. |
| `init.rs` | 110 | Weight initialization from scratch. Matches PyTorch defaults (uniform ±1/√fan_in). Kerr params: γ_raw=softplus⁻¹(0.1), ω=k/N linearly spaced, α=β=0.1. |
| `data.rs` | 53 | Character-level dataset. Loads text file, builds char↔index maps, samples random batches. |
| `rng.rs` | 35 | Deterministic xorshift64 PRNG. Seeded, reproducible. |

**Shaders:**
| File | Role |
|---|---|
| `shaders/kerr_step.wgsl` | GPU compute kernel for one Kerr-ODE derivative evaluation. Reads neighbour bands via [1,1,0,1,1] stencil. One thread per band. Workgroup size 64. |

---

## CLI commands

```
kerr-engine gpu-test                                    # Stage 1: single Euler step, CPU vs GPU
kerr-engine validate <model.bin>                        # Stage 2: full forward pass, Rust vs Python
kerr-engine grad-test [gradient_test.bin]               # Stage 3: analytical gradients vs PyTorch autograd
kerr-engine train [data_path] [iters] [batch] [seq] [lr]  # Stage 4: train from scratch
```

Default training: `kerr-engine train data/input.txt 3000 4 64 3e-4`

---

## Validation gates (all passed)

| Stage | What | Max diff | Tolerance | Status |
|---|---|---|---|---|
| 1 | GPU kernel (single Euler step) | 4.66e-10 | 1e-5 | PASSED |
| 2 | Full forward pass (Rust vs Python logits) | 4.15e-4 | 1e-3 | PASSED |
| 3 | Analytical gradients (Rust vs PyTorch autograd) | 7.63e-6 | 1e-3 | PASSED |
| 4 | Training convergence (loss trajectory match) | Loss 2.21 @ 1000 iters | Python reaches 2.0-2.2 | PASSED |

## Refactor validation baseline

**Every structural change must produce identical output on a 200-iteration training run with seed 42, batch=4, seq=64, lr=3e-4 on Shakespeare (data/input.txt).**

| Iter | Expected loss |
|---|---|
| 0 | 4.2390 |
| 50 | 3.1393 |
| 100 | 2.7872 |
| 150 | 2.5558 |
| 199 | 2.5814 |

Generated text must be character-identical. If numbers drift, the refactor broke something.

---

## Known technical debt

1. **Three-copy duplication.** `model.rs`, `pipeline.rs`, and `backend.rs` each implement some of the same operations (attention, kerr_ode, maestro, linear). `model.rs` has the inference forward. `pipeline.rs` has the cached training forward. `backend.rs` CpuBackend wraps the same logic again. When GPU backend arrives, unify: inference and pipeline should both call through `ComputeBackend`. Not done yet because model.rs forward is the validated Stage 2 reference — don't want to change it before GPU work forces it.

2. **flatten/unflatten fragility.** `optim.rs` has flatten_params, unflatten_params, and flatten_grads that must stay in lockstep. Adding any new parameter type requires updating all three in the same order. No automatic derivation. Test after any architecture change.

3. **Attention backward per-position.** `attention_backward_single` in backward.rs processes one query position at a time. Works correctly but O(T²) with high constant. Batching the backward across positions would help when sequence length grows.

4. **Stage 1/2 validation code in main.rs.** Should move to a validation module. Low priority.

---

## Growth plan

| Feature | Target module | Est. lines | Priority |
|---|---|---|---|
| Batch parallelism (rayon/threads) | train.rs | +30-50 | High — 4x speedup |
| Progressive curriculum (band scheduling) | train.rs | +40 | High — needed for Phase C parity |
| Checkpoint save/load | train.rs | +50 | Medium |
| Eval loop (validation loss) | train.rs | +30 | Medium |
| Word-level tokenizer | data.rs | +80 | Medium — needed for real corpora |
| GPU training backend | gpu.rs + backend.rs | +200-300 | Future — when CPU speed isn't enough |
| Corpus data loading (multi-corpus) | data.rs | +50 | Future — for corpus texture experiments |

**Module growth targets (do not exceed without splitting):**

No module over 550 lines. If approaching, split by responsibility.

---

## Kerr-ODE derivative (the core computation)

```
For each band k:
    mag_sq[k] = r[k]² + s[k]²
    ns[k] = mag_sq[k-2] + mag_sq[k-1] + mag_sq[k+1] + mag_sq[k+2]   (conv1d [1,1,0,1,1])
    φ[k] = ω[k] + α·mag_sq[k] + β·ns[k]
    dr[k] = -γ[k]·r[k] - φ[k]·s[k]
    ds[k] = -γ[k]·s[k] + φ[k]·r[k]
```

Parameters: γ (damping, via softplus of γ_raw), ω (natural frequency), α (self-phase modulation), β (cross-phase modulation via neighbours). Physics from coupled nonlinear optical resonators.

RK4 integrates this 8 times per layer at dt=0.125. Maestro adds a global synchronisation signal. The combination runs at 98.1% of MLP performance with 44% of parameters.

---

## Three-way collaboration model

- **Marco:** Direction, pattern recognition, go/no-go decisions. Sees form, not equations.
- **Claude Desktop (Opus):** Theory, analysis, documentation, math derivations, investigation design.
- **Claude Code:** Implementation, testing, repo management, reality checks on Desktop overclaims.

Rules: Honesty over flattery. Every null documented. Nothing publishes without test results. When Code corrects an interpretation, accept it. Marco's infrastructure instincts drive architecture decisions — don't second-guess them.

---

## Do NOT re-litigate

These are closed decisions from the parent project. Do not revisit:

φ_eff formula (CLOSED), Jessop citation (REMOVED), dispersive coupling (NULL), harmonic attention Q/K (HURTS), per-band alpha/beta (NEGLIGIBLE), band routing (HURTS), multiplicative fusion (HURTS), 9-band+curriculum (DON'T STACK), curriculum at <48 bands (HURTS), two-stage without curriculum (NEGLIGIBLE), wave transduction (PARKED — safety), slow bands=identity (NULL), higher freq=smarter (NULL).
