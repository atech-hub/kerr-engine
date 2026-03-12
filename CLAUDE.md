# CLAUDE.md — Kerr Engine Internal Reference

> Internal developer docs for Claude Code. Terse, factual, no narrative.
> Last updated: 2026-03-12 (end of Chat 6)

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
| `gpu_backend.rs` | 884 | `GpuBackend` implementing `ComputeBackend` (per-call pattern). All 8 trait methods. Validation harness + per-primitive benchmark (CPU vs GPU timing). Per-call pattern: ~500us dispatch overhead per operation. |
| `gpu_persistent.rs` | 720 | `GpuPersistent` — persistent buffer pipeline. Weights uploaded once, scratch buffers pre-allocated, single command encoder submit, single readback. Fused RK4 Kerr-ODE (8 dispatches vs 32). Benchmark: 296x faster than per-call GPU. Still 11-18x slower than CPU at 128-dim. Crossover: ~700-800 dim. |
| `pipeline.rs` | 540 | Cached forward pass (saves activations for backward) + full backward chain that orchestrates primitives from backward.rs. Cache structs (ForwardCache, BlockCache, AttnCache). GradAccum struct. The contract: what forward saves, backward consumes. |
| `backward.rs` | 530 | Gradient primitives: linear, layer_norm, GELU, softplus, cross-entropy, Kerr-ODE derivative backward, RK4 step backward, full Kerr-ODE backward, attention backward, maestro backward. All hand-derived, no autograd. |
| `model.rs` | 498 | Weight structs, architecture constants, inference-only forward pass, Kerr-ODE forward, maestro forward, attention, per-band linear, RK4 integration, harmonic table construction. Source of truth for what the computation is. |
| `train.rs` | 361 | Training loop orchestration. CurriculumSchedule (configurable band scheduling). TrainConfig. Parallel batch forward/backward via std::thread::scope. Loss logging, periodic eval (val loss + text generation), JSON training summary, checkpoint save at intervals. |
| `optim.rs` | 299 | Adam optimizer with bias correction. Parameter flatten/unflatten (structured weights ↔ flat f32 vec). Gradient flattening. Gradient norm clipping. `count_params`. Checkpoint support (from_checkpoint, checkpoint_state). |
| `backend.rs` | 228 | `ComputeBackend` trait (8 methods: linear, linear_no_bias, layer_norm, kerr_ode, maestro, attention, per_band_linear, kerr_maestro_add). `CpuBackend` implementation. |
| `main.rs` | 225 | CLI dispatch. Five subcommands (see below). Stage 1/2 validation logic lives here temporarily. |
| `gpu.rs` | 209 | WGPU device setup for Stage 1 validation (single Euler step). Retained for backward compatibility. |
| `data.rs` | 203 | Dataset with train/val split (90/10). Character-level and word-level tokenization. Word tokenizer: whitespace split, lowercase, punctuation separation, min_count threshold. |
| `grad_test.rs` | 184 | Stage 3 gradient validation harness. Loads PyTorch reference gradients, compares against Rust analytical gradients. |
| `weights.rs` | 175 | Binary weight loader. Reads model.bin exported from Python. Loads test vectors for Stage 2 validation. |
| `checkpoint.rs` | 141 | Binary checkpoint format (magic KCHK v1). Saves/loads full training state: model weights + Adam m/v/t + RNG state + iteration + lr. Bit-perfect resume validated. |
| `init.rs` | 110 | Weight initialization from scratch. Matches PyTorch defaults (uniform ±1/√fan_in). Kerr params: γ_raw=softplus⁻¹(0.1), ω=k/N linearly spaced, α=β=0.1. |
| `rng.rs` | 45 | Deterministic xorshift64 PRNG. Seeded, reproducible. state()/from_state() for checkpointing. |

**Total: 5,361 lines across 16 modules.**

**Shaders:**
| File | Role |
|---|---|
| `shaders/matvec.wgsl` | Matrix-vector multiply: y = W @ x + b. One thread per output row, workgroup size 64. Uniform flag for bias/no-bias. |
| `shaders/layer_norm.wgsl` | Layer normalization via parallel tree reduction in shared memory. Single workgroup of 128 threads. Two-pass: mean+variance reduction, then normalize. |
| `shaders/kerr_step.wgsl` | One Kerr-ODE derivative evaluation. Neighbour coupling via [1,1,0,1,1] stencil. One thread per band. Workgroup size 64. |
| `shaders/kerr_rk4_step.wgsl` | Fused RK4 step — all 4 derivative evaluations + combination in a single dispatch. Uses workgroup shared memory for neighbour synchronisation between sub-steps. 32 round-trips → 1 dispatch. |
| `shaders/gelu.wgsl` | Element-wise GELU activation. One thread per element. |
| `shaders/vec_add.wgsl` | Element-wise vector addition. One thread per element. |

---

## CLI commands

```
kerr-engine gpu-test                                    # Stage 1: single Euler step, CPU vs GPU
kerr-engine gpu-backend-test                            # GPU backend: validate all primitives vs CPU
kerr-engine gpu-bench                                   # Per-primitive CPU vs GPU timing (per-call)
kerr-engine gpu-persistent-bench                        # Persistent GPU pipeline vs per-call vs CPU
kerr-engine validate <model.bin>                        # Stage 2: full forward pass, Rust vs Python
kerr-engine grad-test [gradient_test.bin]               # Stage 3: analytical gradients vs PyTorch autograd
kerr-engine train [data] [iters] [batch] [seq] [lr] [--no-curriculum] [--word] [--resume FILE]
```

Default training: `kerr-engine train data/input.txt 3000 4 64 3e-4`
Word-level: `kerr-engine train data/input.txt 3000 4 64 3e-4 --word`
Resume: `kerr-engine train data/input.txt 3000 4 64 3e-4 --resume checkpoint_iter500.bin`

---

## Validation gates (all passed)

| Stage | What | Max diff | Tolerance | Status |
|---|---|---|---|---|
| 1 | GPU kernel (single Euler step) | 4.66e-10 | 1e-5 | PASSED |
| 2 | Full forward pass (Rust vs Python logits) | 4.15e-4 | 1e-3 | PASSED |
| 3 | Analytical gradients (Rust vs PyTorch autograd) | 7.63e-6 | 1e-3 | PASSED |
| 4 | Training convergence (loss trajectory match) | Loss 2.21 @ 1000 iters | Python reaches 2.0-2.2 | PASSED |
| GPU | GpuBackend linear vs CpuBackend | 1.53e-5 | 1e-4 | PASSED |
| GPU | GpuBackend linear_no_bias vs CpuBackend | 5.72e-6 | 1e-4 | PASSED |
| GPU | GpuBackend layer_norm vs CpuBackend | 2.38e-7 | 1e-4 | PASSED |
| GPU | GpuBackend kerr_ode (RK4) vs CpuBackend | 2.98e-8 | 1e-4 | PASSED |

## Refactor validation baseline

**Every structural change must produce identical output on a 200-iteration training run with seed 42, batch=4, seq=64, lr=3e-4 on Shakespeare (data/input.txt).**

| Iter | Expected loss |
|---|---|
| 0 | 4.2556 |
| 50 | 3.3216 |
| 100 | 2.7094 |
| 150 | 2.7213 |
| 199 | 2.6161 |

Note: baseline shifted from original (4.2390→4.2556 at iter 0) due to train/val split (90/10). Training now samples from 90% of data. This is correct.

Seeds: model init = 42, training RNG = 1337. Both must match for character-identical output. If numbers drift, the refactor broke something.

## Training benchmark: Rust CPU vs Python+PyTorch+CUDA

**Config:** 3000 iters, batch=4, seq=64, lr=3e-4, no curriculum, Shakespeare (1.1M chars, 65 vocab, 354K params).

| | Python (PyTorch + RTX 4070 Ti) | Rust (CPU, 4 threads) |
|---|---|---|
| Total time | ~10 min | 4 min 40 sec |
| Median iter | ~200 ms | ~71 ms |
| Final train loss | ~2.0 | 1.97 |
| Final val loss | ~2.1 | 2.12 |
| Hardware | RTX 4070 Ti (CUDA) | i7-14700K (4 of 28 threads) |

**2.1x faster on CPU only.** Same convergence, same final loss. Framework overhead (Python interpreter, autograd tape, CUDA kernel launch, tensor metadata) exceeds the GPU compute advantage at this scale.

## GPU benchmark: per-call vs persistent vs CPU (128-dim)

| Pattern | FFN chain | Kerr-ODE | Why |
|---|---|---|---|
| CPU | 7 us | 11 us | Direct computation, no dispatch overhead |
| Per-call GPU | 2,247 us (321x) | 36,458 us (3,314x) | Buffer creation + readback per operation |
| Persistent GPU | 125 us (18x) | 123 us (11x) | Weights uploaded once, single submit, single readback |

Persistent is 296x faster than per-call for Kerr-ODE. CPU still wins at 128-dim. Estimated crossover: ~700-800 dim (where O(n^2) matmul overtakes fixed dispatch cost).

---

## Known technical debt

1. **Three-copy duplication.** `model.rs`, `pipeline.rs`, and `backend.rs`/`gpu_backend.rs` each implement some of the same operations (attention, kerr_ode, maestro, linear). Now that GpuBackend exists and is validated, the next step is to wire inference (model.rs) and training (pipeline.rs) to dispatch through `ComputeBackend` rather than calling their own copies. This unification is the gate to GPU-accelerated training.

2. **flatten/unflatten fragility.** `optim.rs` has flatten_params, unflatten_params, and flatten_grads that must stay in lockstep. Adding any new parameter type requires updating all three in the same order. No automatic derivation. Test after any architecture change.

3. **Attention backward per-position.** `attention_backward_single` in backward.rs processes one query position at a time. Works correctly but O(T²) with high constant. Batching the backward across positions would help when sequence length grows.

4. **Stage 1/2 validation code in main.rs.** Should move to a validation module. Low priority.

---

## Growth plan

| Feature | Target module | Status |
|---|---|---|
| Batch parallelism (std::thread::scope) | train.rs | DONE — 3.4x speedup |
| Progressive curriculum (band scheduling) | train.rs | DONE — configurable CurriculumSchedule |
| Checkpoint save/load | checkpoint.rs | DONE — binary KCHK format, bit-perfect resume |
| Eval loop (validation loss) | train.rs | DONE — fixed-seed val loss, text generation |
| Word-level tokenizer | data.rs | DONE — whitespace split, min_count, <unk> |
| Training summary (JSON) | train.rs | DONE — config + loss curves + timing |
| GPU backend (hybrid tier) | gpu_backend.rs | DONE — 3 shaders, all 8 trait methods, validated |
| Wire GpuBackend into inference/training | pipeline.rs + train.rs | Next — replace CPU dispatch with GPU |
| Full-GPU attention shader | shaders/ | Future — fused QKV + softmax in WGSL |
| Corpus data loading (multi-corpus) | data.rs | Future — for corpus texture experiments |

## Compute tiers

| Tier | Implementation | Status |
|---|---|---|
| CPU | `CpuBackend` in backend.rs | Operational. All 8 methods. Used by training loop. |
| Hybrid | `GpuBackend` in gpu_backend.rs | Operational. Linear/layer_norm/kerr_ode on GPU, attention scores + GELU + small ops on CPU. Validated against CpuBackend. |
| Full-GPU | Not yet | Would need fused attention kernel + GELU shader. Not needed until sequence length grows. |

Next step: wire `GpuBackend` into the inference forward pass (model.rs) or training pipeline (pipeline.rs) so it actually runs during training, not just validation.

**Module growth targets (do not exceed without splitting):**

No module over 900 lines. gpu_backend.rs is at 884 (includes benchmark code). If it grows further, split benchmark into gpu_bench.rs.

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
