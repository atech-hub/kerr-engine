# CLAUDE.md — Kerr Engine Developer Reference

> Developer documentation for the Kerr Engine codebase. Module map, shader inventory, validation gates, benchmarks, and architectural decisions.
> Last updated: 2026-03-14

---

## What this is

Pure Rust inference and training engine for the Kerr-ODE architecture from the Wave Coherence project. No Python, no PyTorch, no CUDA toolkit. Dependencies: wgpu, bytemuck, pollster, mimalloc.

Parent project: github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive (public, MIT).
This repo: public (Apache 2.0). Released 2026-03-13.

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
| `gpu_backend.rs` | 10 | Thin re-export module. Imports and re-exports from gpu_pipelines, gpu_dispatch, gpu_validate. Backward-compatible — no other module needs to change imports. |
| `gpu_pipelines.rs` | 1060 | `GpuBackend` struct, all 14 shader pipeline compilations (9 forward, 5 backward), param structs, constructors (`new()`, `with_device_index()`), dispatch helpers (incl. batched: gpu_matvec_batch, gpu_layer_norm_batch, gpu_kerr_derivative_batch, gpu_kerr_ode_batch), GPU adapter enumeration (`list_gpus`). |
| `gpu_dispatch.rs` | 529 | `impl ComputeBackend for GpuBackend`. All forward + backward trait method dispatches. Overrides linear_batch, linear_no_bias_batch, layer_norm_batch for batched GPU dispatch. Batched kerr_maestro_add and per_band_linear. The file to edit for new GPU operations. |
| `gpu_validate.rs` | 271 | GPU validation harness (all primitives vs CPU) + per-primitive benchmark. Isolated from hot path — changes here never affect training. |
| `gpu_persistent.rs` | 720 | `GpuPersistent` — persistent buffer pipeline. Weights uploaded once, scratch buffers pre-allocated, single command encoder submit, single readback. Fused RK4 Kerr-ODE (8 dispatches vs 32). Benchmark: 296x faster than per-call GPU. Still 11-18x slower than CPU at 128-dim. Crossover: ~700-800 dim. |
| `pipeline.rs` | 581 | Cached forward pass routing through `ComputeBackend`. Full backward chain: all operations batched across positions through trait. FFN backward: out_proj via batched linear_backward_dx_batch + outer_product_accum, maestro via batched linears + GELU, Kerr-ODE backward per-position (CPU analytical). Timing instrumentation (TIMING const, disabled by default). Cache structs (ForwardCache, BlockCache, AttnCache). GradAccum struct. |
| `backward.rs` | 554 | Gradient primitives: linear, layer_norm, GELU, softplus, cross-entropy, Kerr-ODE derivative backward, RK4 step backward, full Kerr-ODE backward, attention backward, maestro backward. All hand-derived, no autograd. linear_backward_dx_only for trait dispatch. |
| `model.rs` | 573 | Weight structs, ModelConfig (runtime-configurable architecture), inference-only forward pass, Kerr-ODE forward, maestro forward, attention, per-band linear, RK4 integration, harmonic table construction. All forward ops derive dimensions from config/data (no hardcoded constants). Iterator-based linear for autovectorisation. |
| `train.rs` | 408 | Training loop orchestration. Creates backend at startup via `backend::auto_select(n_embd)`. CurriculumSchedule (configurable band scheduling). TrainConfig with all CLI-configurable fields including ModelConfig overrides. Parallel batch forward/backward via std::thread::scope. generate() routes through ComputeBackend. Loss logging, periodic eval (val loss + text generation), JSON training summary, checkpoint save at intervals. Vocab resize on cross-corpus resume. |
| `optim.rs` | 305 | Adam optimizer with bias correction. Parameter flatten/unflatten (structured weights ↔ flat f32 vec). Gradient flattening. Gradient norm clipping. `count_params`. Checkpoint support (from_checkpoint, checkpoint_state). Extend for vocab resize. |
| `backend.rs` | 476 | `ComputeBackend` trait (8 forward + 3 batch forward + 6 backward methods). `CpuBackend` implementation. Auto-select logic (CPU below 768-dim, GPU above). Backward methods: linear_backward_dx, linear_backward_dx_batch, layer_norm_backward, gelu_backward, outer_product_accum, attention_backward. |
| `main.rs` | 271 | CLI dispatch with full flag parsing (--seed, --train-seed, --threads, --cpu, --gpu, --resume, --word, --no-curriculum). Stage 1/2 validation logic lives here temporarily. mimalloc global allocator. |
| `gpu.rs` | 209 | WGPU device setup for Stage 1 validation (single Euler step). Retained for backward compatibility. |
| `data.rs` | 203 | Dataset with train/val split (90/10). Character-level and word-level tokenization. Word tokenizer: whitespace split, lowercase, punctuation separation, min_count threshold. |
| `grad_test.rs` | 184 | Stage 3 gradient validation harness. Loads PyTorch reference gradients, compares against Rust analytical gradients. |
| `weights.rs` | 175 | Binary weight loader. Reads model.bin exported from Python. Loads test vectors for Stage 2 validation. |
| `checkpoint.rs` | 141 | Binary checkpoint format (magic KCHK v1). Saves/loads full training state: model weights + Adam m/v/t + RNG state + iteration + lr. Bit-perfect resume validated. |
| `init.rs` | 110 | Weight initialization from scratch. Matches PyTorch defaults (uniform ±1/√fan_in). Kerr params: γ_raw=softplus⁻¹(0.1), ω=k/N linearly spaced, α=β=0.1. |
| `rng.rs` | 45 | Deterministic xorshift64 PRNG. Seeded, reproducible. state()/from_state() for checkpointing. |

**Total: ~7,100 lines across 19 modules.**

**Shaders (16 WGSL compute shaders):**
| File | Role |
|---|---|
| `shaders/matvec.wgsl` | Matrix-vector multiply: y = W @ x + b. One thread per output row, workgroup size 64. Uniform flag for bias/no-bias. |
| `shaders/layer_norm.wgsl` | Layer normalization via strided parallel reduction. Workgroup size 256, supports up to 2048-dim. |
| `shaders/kerr_step.wgsl` | One Kerr-ODE derivative evaluation. Neighbour coupling via [1,1,0,1,1] stencil. One thread per band. Workgroup size 64. |
| `shaders/kerr_rk4_step.wgsl` | Fused RK4 step — all 4 derivative evaluations + combination in a single dispatch. Workgroup size 256, supports up to 256 bands (512-dim). Falls back to kerr_step.wgsl for larger. |
| `shaders/gelu.wgsl` | Element-wise GELU activation. One thread per element. |
| `shaders/vec_add.wgsl` | Element-wise vector addition. One thread per element. |
| `shaders/matvec_backward.wgsl` | d_x = W^T @ d_y (single position). One thread per input element, workgroup size 64. |
| `shaders/matvec_backward_batch.wgsl` | Batched d_x[pos] = W^T @ d_y[pos] for all positions. One thread per (pos, j). Eliminates per-position dispatch overhead — **28x speedup** over per-position dispatch at 768-dim. |
| `shaders/layer_norm_backward.wgsl` | Full layer norm backward: d_x, d_weight, d_bias. Strided workgroup reduction matching forward pattern. |
| `shaders/gelu_backward.wgsl` | Element-wise GELU gradient. Ready for FFN backward restructuring. |
| `shaders/attn_backward_scores.wgsl` | Attention backward Phase 1: d_score (softmax backward) + d_q. One workgroup per (pos, head). Shared memory reduction for softmax Jacobian. |
| `shaders/attn_backward_dkv.wgsl` | Attention backward Phase 2: d_k + d_v from d_score. One thread per (ki, d_global). No race conditions — each thread writes unique element. |
| `shaders/outer_product.wgsl` | Batched outer product: d_w[i][j] = sum_pos d_y[pos][i] * x[pos][j]. One workgroup per row. Also computes d_b. Replaces CPU weight gradient accumulation. |
| `shaders/matvec_batch.wgsl` | Batched forward linear: y[pos] = W @ x[pos] + b for all positions. One thread per (pos, out_dim). use_bias flag. Replaces N separate matvec dispatches. QKV: 260ms→23ms (11x). |
| `shaders/layer_norm_batch.wgsl` | Batched forward layer norm: one workgroup (256 threads) per position. Strided reduction for mean/variance. LN1+LN2: 118ms→3ms (39x). |
| `shaders/kerr_step_batch.wgsl` | Batched Kerr-ODE derivative: all positions in one dispatch. One thread per (pos, band). Shared gamma/omega/alpha/beta. 2,048 dispatches/block→32. FFN: 2,020ms→84ms (24x). |

---

## CLI quick reference

Full CLI documentation with examples is in README.md. This is the quick reference for Claude Code sessions.

```
kerr-engine gpu-test                   # Stage 1 validation
kerr-engine gpu-backend-test           # GPU primitives vs CPU
kerr-engine gpu-bench                  # Per-primitive timing
kerr-engine gpu-persistent-bench       # Persistent pipeline timing
kerr-engine validate <model.bin>       # Stage 2 validation
kerr-engine grad-test [gradient.bin]   # Stage 3 validation
kerr-engine train [data] [iters] [batch] [seq] [lr] [flags]
```

**Training flags:** `--seed N`, `--train-seed N`, `--threads N`, `--cpu`, `--gpu`, `--gpu-device N`, `--no-curriculum`, `--word`, `--resume FILE`, `--n-bands N`, `--n-head N`, `--n-layers N`, `--maestro-dim N`, `--block-size N`, `--rk4-steps N`

**Other commands:** `list-gpus` — enumerate available GPU adapters

All defaults are sensible. Nothing hardcoded. See README.md for full descriptions and examples.

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

## Refactor validation baselines

**Every structural change must produce identical output. Two baselines depending on curriculum setting.**

Seeds: model init = 42, training RNG = 1337. Batch=4, seq=64, lr=3e-4, data/input.txt.

**Baseline A (no curriculum) — `--no-curriculum`:**

| Iter | Expected loss |
|---|---|
| 0 | 4.2556 |
| 50 | 3.3216 |
| 100 | 2.7094 |
| 150 | 2.7213 |
| 199 | 2.6161 |

**Baseline B (default, with curriculum):**

| Iter | Expected loss |
|---|---|
| 0 | 4.4437 |
| 50 | 3.3695 |
| 100 | 2.9763 |
| 150 | 2.9444 |
| 199 | 2.7439 |

Baseline B has higher early loss because curriculum starts with 8 bands (restricted capacity), then opens to 24, then 64. Baseline A uses all 64 bands from step 0.

If numbers drift on either baseline, the refactor broke something.

## Training benchmark: Rust CPU vs Python+PyTorch+CUDA

**Config:** 3000 iters, batch=4, seq=64, lr=3e-4, no curriculum, Shakespeare (1.1M chars, 65 vocab, 354K params).

| | Python (PyTorch + RTX 4070 Ti) | Rust (CPU, 4 threads) |
|---|---|---|
| Total time | ~10 min | ~3 min 50 sec (est. from 200-iter scaling + eval overhead) |
| Median iter | ~200 ms | ~62 ms |
| Final train loss | ~2.0 | 1.97 |
| Final val loss | ~2.1 | 2.12 |
| Hardware | RTX 4070 Ti (CUDA) | i7-14700K (4 of 28 threads) |

**~2.6x faster on CPU only after optimisation (was 2.1x before mimalloc + iterators + clone elimination).** Same convergence, same final loss. Framework overhead (Python interpreter, autograd tape, CUDA kernel launch, tensor metadata) exceeds the GPU compute advantage at this scale. Total optimisation: 20% over unoptimised Rust baseline (15.3s → 12.2s for 200 iters). All gains bit-identical.

## Training benchmark: 768-dim (12M params)

**Config:** batch=4, seq=64, lr=3e-4, no curriculum, Shakespeare, 384 bands, 4 threads.

| Metric | Value |
|---|---|
| Iteration time | 1.72s |
| GPU utilisation | 38% at 49°C |
| VRAM | 1.2 / 12.0 GB |
| Forward pass | ~500ms (down from 8,060ms before batching, 16x) |
| Backward pass | ~970ms (down from 4,400ms before batching, 4.5x) |
| Full speedup | 7.6x (13s → 1.72s) |

Competitive with PyTorch at 768-dim. Faster than PyTorch at 128-dim. On a third of the GPU, at half the temperature.

## GPU benchmark: per-call vs persistent vs CPU (128-dim)

| Pattern | FFN chain | Kerr-ODE | Why |
|---|---|---|---|
| CPU | 7 us | 11 us | Direct computation, no dispatch overhead |
| Per-call GPU | 2,247 us (321x) | 36,458 us (3,314x) | Buffer creation + readback per operation |
| Persistent GPU | 125 us (18x) | 123 us (11x) | Weights uploaded once, single submit, single readback |

Persistent is 296x faster than per-call for Kerr-ODE. CPU still wins at 128-dim. Estimated crossover: ~700-800 dim (where O(n^2) matmul overtakes fixed dispatch cost).

---

## Known technical debt

1. ~~**Three-copy duplication**~~ **RESOLVED for training.** pipeline.rs now routes through `ComputeBackend` trait. Inline math helpers removed. The remaining separate implementation in model.rs is the inference-only forward pass — intentionally kept independent as the Stage 2 validation reference. Do not unify model.rs; it's a safety net, not debt.

2. **flatten/unflatten fragility.** `optim.rs` has flatten_params, unflatten_params, and flatten_grads that must stay in lockstep. Adding any new parameter type requires updating all three in the same order. No automatic derivation. Test after any architecture change.

3. ~~**Attention backward per-position.**~~ **RESOLVED.** GPU attention backward shader (2-dispatch) processes all positions in parallel. 4ms at 768-dim. CpuBackend still uses per-position fallback.

4. ~~**FFN backward per-position.**~~ **PARTIALLY RESOLVED.** Out_proj backward (linear_backward_dx_batch + outer_product_accum) and maestro backward (batched squeeze/process linears + GELU) now batch across all positions in one dispatch. Kerr-ODE backward remains per-position CPU — measured at 768-dim: ~277ms/block (down from ~340ms), with ~250ms of that being Kerr-ODE analytical backward (RK4 × 32 derivative evals × 64 positions). The Kerr backward is now the bottleneck, not the linears.

5. ~~**gpu_backend.rs size.**~~ **RESOLVED.** Split into gpu_pipelines.rs (755), gpu_dispatch.rs (483), gpu_validate.rs (271), with gpu_backend.rs as a 10-line re-export. Bit-identical through refactor.

6. **Stage 1/2 validation code in main.rs.** Should move to a validation module. Low priority.

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
| GPU backend (forward) | gpu_backend.rs | DONE — 6 forward shaders, all 8 trait methods, validated |
| GPU backward shaders | gpu_backend.rs + shaders/ | DONE — 5 backward shaders: matvec_backward, matvec_backward_batch, layer_norm_backward, attn_backward (2-dispatch), outer_product, gelu_backward. Attention backward 4ms on GPU. Batched matvec 28x speedup. |
| Batched forward dispatch | gpu_dispatch.rs + shaders/ | DONE — 3 new shaders: matvec_batch, layer_norm_batch, kerr_step_batch. All forward operations batch positions into single GPU dispatches. Forward pass 8,060ms→516ms (16x). Full iteration 13s→1.72s (7.6x). |
| ModelConfig refactor | model.rs, pipeline.rs | DONE — runtime-configurable n_bands, n_head, n_layers, maestro_dim, block_size, rk4_steps. All forward ops derive from config/data. |
| Multi-GPU selection | gpu_backend.rs, main.rs | DONE — `list-gpus` command, `--gpu-device N` flag, `with_device_index(idx)` constructor |
| generate() GPU routing | train.rs | DONE — inference routes through ComputeBackend via forward_with_cache() |
| Wire GpuBackend into training pipeline | pipeline.rs + train.rs | DONE — auto_select at startup, ComputeBackend dispatch |
| Wire GpuBackend into inference (model.rs) | model.rs | Deferred — model.rs kept as independent validation reference |
| Batch parallelism (std::thread::scope) | train.rs | DONE — 3.4x speedup |
| Progressive curriculum (band scheduling) | train.rs | DONE — configurable CurriculumSchedule |
| Checkpoint save/load | checkpoint.rs | DONE — binary KCHK format, bit-perfect resume |
| Eval loop (validation loss) | train.rs | DONE — fixed-seed val loss, text generation |
| Word-level tokenizer | data.rs | DONE — whitespace split, min_count, <unk> |
| Training summary (JSON) | train.rs | DONE — config + loss curves + timing |
| Sub-batch CPU parallelism (per-head, per-position) | pipeline.rs | NOT VIABLE at 128-dim — thread spawn overhead (~10μs) exceeds per-op compute (~1-10μs). Revisit at larger dims. |
| mimalloc allocator | Cargo.toml | DONE — ~10% gain from reduced 4-thread Windows heap contention |
| Clone elimination (4 redundant copies) | pipeline.rs | DONE — ~5% from removing output, ffn_input, pre_proj, pre_ln_f copies never read by backward |
| Iterator-based dot products | pipeline.rs, backend.rs | DONE — ~3-5% from bounds check elimination + better LLVM autovectorisation |
| Explicit +avx2,+fma target features | .cargo/config.toml | DONE — small gain, workaround for rust-lang/rust#147176 (target-cpu=native detection bug) |
| Pre-allocated attention scratch | pipeline.rs | DONE — ~1% from eliminating 1024 Vec allocations per batch element |
| Batched linear operations | backend.rs | NO IMPACT at 128-dim — matrices already fit in L1 cache |
| FFN backward GPU batching (Tier 1) | pipeline.rs | DONE — out_proj + maestro backward batched across positions via existing GPU shaders (linear_backward_dx_batch, outer_product_accum, gelu_backward). Kerr-ODE backward stays per-position CPU. Bit-identical verified. |
| Corpus data loading (multi-corpus) | data.rs | Future — for corpus texture experiments |

## Compute tiers

| Tier | Implementation | Status |
|---|---|---|
| CPU | `CpuBackend` in backend.rs | Operational. All forward + backward methods. Bit-identical baseline validated. |
| GPU | `GpuBackend` in gpu_backend.rs | Operational. Forward: all operations batched across positions — linear_batch, layer_norm_batch, kerr_ode_batch, per_band_linear, kerr_maestro_add all dispatch once for all positions. Backward: attention backward (2-dispatch), batched linear_backward_dx, outer_product_accum, layer_norm_backward, gelu_backward all on GPU. FFN backward: out_proj + maestro batched on GPU, Kerr-ODE per-position CPU. Activates at N_EMBD >= 768. 16 shaders total (9 forward, 5 backward, 2 attention backward). |

Training pipeline dispatches through `ComputeBackend`. auto_select picks backend at startup based on N_EMBD. Inference routes through backend via `forward_with_cache()`. model.rs forward kept independent as Stage 2 validation reference.

**GPU profiling (768-dim, 12M params, RTX 4070 Ti):**

Backward pass:

| Operation | Before batching | After batching | Speedup |
|---|---|---|---|
| c_proj backward | ~200ms | ~7ms | 28x |
| c_attn backward | ~400ms | ~14ms | 28x |
| attention backward | CPU ~400ms (est.) | GPU 4ms | ~100x |
| FFN+LN2 backward | ~340ms | ~277ms (measured) | 1.2x |
| Total backward | ~4.4s | ~968ms (measured) | 4.5x |

Forward pass:

| Operation | Before batching | After batching | Speedup |
|---|---|---|---|
| LN1 (per block) | 55ms | 1.4ms | 39x |
| QKV projection | 260ms | 23ms | 11x |
| Out projection | 107ms | 8ms | 13x |
| Residual+LN2 | 63ms | 1.7ms | 37x |
| FFN KerrMaestro | 2,020ms | 84ms | 24x |
| FFN PerBand | 104ms | 8ms | 13x |
| Total forward (4 blocks) | ~8,060ms | ~516ms | 16x |
| **Full iteration** | **~13s** | **1.72s** | **7.6x** |

Forward bottleneck was per-position GPU dispatch overhead: 2,048 dispatches per Kerr block (64 positions × 32 RK4 derivative evals). Three new batched shaders reduced this to 32 dispatches per block. Remaining forward time dominated by attention head loop (28ms, CPU — trivial).

Backward remaining bottleneck: Kerr-ODE analytical backward (RK4 forward recompute + 32 derivative evaluations × 64 positions) at ~250ms/block. Same batching pattern applicable but separate session.

**Module growth targets:** Keep files under ~550 lines. Current largest: gpu_pipelines.rs (755) — contains 11 shader compilations, stable.

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

## Collaboration model

- **Marco Da Cunha:** Direction, pattern recognition, architecture decisions
- **Claude Desktop (Opus):** Theory, analysis, documentation, mathematical derivations
- **Claude Code:** Implementation, testing, validation, repository management

Every null result is documented alongside positives. Nothing publishes without test results. See README.md for full credits.

---

## Closed investigations

These architectural decisions have been tested and resolved in the parent project. See the parent repo's investigation docs for full experimental results.

φ_eff formula (closed), dispersive coupling (null), harmonic attention Q/K (hurts performance), per-band alpha/beta (negligible), band routing (hurts performance), multiplicative fusion (hurts performance), 9-band+curriculum (don't stack), curriculum below 48 bands (hurts), two-stage without curriculum (negligible), slow bands as identity carriers (null), higher frequency equals higher intelligence (null).
