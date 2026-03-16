# Kerr Engine

Pure Rust training and inference engine for the Kerr-ODE transformer architecture. No Python. No PyTorch. No CUDA toolkit.

**3x faster than PyTorch+CUDA on CPU alone.** Same convergence, GPU sitting idle.

---

## What is this?

A specialised engine for training and running [Wave Coherence](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) transformer models. The Kerr-ODE replaces dense MLP layers with a physics-inspired wave propagation step — achieving 98.1% of MLP performance at 44% of the parameters.

This engine implements the full training pipeline in Rust with hand-derived analytical gradients. No automatic differentiation, no computation graph, no framework overhead.

### Key numbers

| | PyTorch + RTX 4070 Ti | Kerr Engine (CPU, 4 threads) |
|---|---|---|
| 3000-iter training (curriculum) | ~10 min | 3 min 21 sec |
| Median iteration | ~200 ms | ~49 ms |
| Final train loss | ~2.0 | 1.91 |
| Final val loss | ~2.1 | 2.20 |
| Dependencies | Python, PyTorch, CUDA | wgpu, bytemuck, pollster, mimalloc |
| Hardware required | NVIDIA GPU | Any CPU (GPU optional) |

---

## Quick start

```bash
# Build
cargo build --release

# See all commands
cargo run --release -- --help

# Train on Shakespeare (default settings)
cargo run --release -- train data/input.txt

# See all training options
cargo run --release -- train --help

# Train with explicit configuration
cargo run --release -- train data/input.txt 3000 4 64 3e-4 --seed 42

# Train with BPE tokenizer (real language models)
cargo run --release -- train data/corpus.txt 3000 --bpe tokenizer.json
```

The engine auto-detects your hardware and selects the optimal backend. At 128-dim, CPU is faster. At 768+ dim, it switches to GPU via WGPU — no CUDA required, works on NVIDIA, AMD, Intel, and Apple Silicon.

**Honest scale note:** All headline benchmarks (3x faster, 98.1% MLP performance, 354K params) are at **128-dim on CPU**. This is the validated, production-tested configuration. GPU training at 768-dim has been benchmarked (1.72s/iter, 50 iterations) but not stress-tested in full training runs. Dimensions between 128 and 768 are untested. If you're scaling up, expect to be an early tester — please file issues.

---

## Installation

Requires Rust nightly (tested on nightly-2025-11-13). A `rust-toolchain.toml` is included.

```bash
git clone https://github.com/atech-hub/kerr-engine.git
cd kerr-engine
cargo build --release
```

No Python. No pip. No conda. No CUDA toolkit. Just Rust.

---

## Training

```
kerr-engine train [data] [iters] [batch] [seq] [lr] [flags]
```

### Positional arguments

| Argument | Default | Description |
|---|---|---|
| data | `data/input.txt` | Path to training corpus (text file) |
| iters | 3000 | Total training iterations |
| batch | 4 | Batch size |
| seq | 64 | Sequence length per training example |
| lr | 3e-4 | Learning rate (Adam optimizer) |

### Flags

| Flag | Default | Description |
|---|---|---|
| `--seed N` | 42 | Model weight initialisation seed |
| `--train-seed N` | seed + 1295 | Training data sampling seed (independent from model seed) |
| `--threads N` | auto-detect | Thread count for batch parallelism |
| `--cpu` | auto | Force CPU backend |
| `--gpu` | auto | Force GPU backend (falls back to CPU if unavailable) |
| `--no-curriculum` | curriculum on | Disable progressive band curriculum |
| `--word` | character-level | Use word-level tokenizer |
| `--bpe FILE` | none | Use BPE tokenizer from HuggingFace tokenizer.json (Qwen, Llama, GPT-2) |
| `--resume FILE` | fresh init | Resume from checkpoint (handles vocab resize for cross-corpus training) |
| `--n-bands N` | 64 | Harmonic frequency bands |
| `--n-head N` | 4 | Attention heads |
| `--n-layers N` | 4 | Transformer blocks |
| `--maestro-dim N` | 16 | Maestro bottleneck width |
| `--block-size N` | 256 | Max sequence length |
| `--rk4-steps N` | 8 | ODE integration steps per layer |

### Examples

```bash
# Default: Shakespeare, curriculum, auto threads, auto backend
cargo run --release -- train data/input.txt 3000 4 64 3e-4

# Explicit seeds for reproducibility
cargo run --release -- train data/input.txt 3000 4 64 3e-4 --seed 42 --train-seed 9999

# No curriculum, single-threaded, force GPU
cargo run --release -- train data/input.txt 3000 4 64 3e-4 --no-curriculum --threads 1 --gpu

# Sequential corpus training (corpus curriculum)
cargo run --release -- train data/children.txt 3000 4 64 3e-4 --seed 42
cargo run --release -- train data/input.txt 6000 4 64 3e-4 --seed 42 --resume checkpoint_final.bin

# Word-level tokenization
cargo run --release -- train data/input.txt 3000 4 64 3e-4 --word

# BPE tokenizer (download any HuggingFace tokenizer.json)
cargo run --release -- train data/corpus.txt 10000 --bpe tokenizer.json

# Larger model (768-dim)
cargo run --release -- train data/input.txt 3000 --n-bands 384 --n-head 12 --gpu
```

Nothing is hardcoded. Every training parameter is a CLI switch with a sensible default. Run `kerr-engine train --help` for full details.

---

## Validation & benchmarks

```bash
cargo run --release -- gpu-test              # GPU kernel validation
cargo run --release -- gpu-backend-test      # All GPU primitives vs CPU
cargo run --release -- gpu-bench             # CPU vs GPU timing
cargo run --release -- gpu-persistent-bench  # Persistent GPU pipeline benchmark
cargo run --release -- validate model.bin    # Forward pass: Rust vs Python
cargo run --release -- grad-test             # Analytical gradients vs PyTorch autograd
```

### Validation gates (all passed)

| Stage | Test | Max difference | Status |
|---|---|---|---|
| 1 | GPU kernel (single step) | 4.66e-10 | PASS |
| 2 | Full forward pass (Rust vs Python) | 4.15e-4 | PASS |
| 3 | Analytical gradients (vs PyTorch autograd) | 7.63e-6 | PASS |
| 4 | Training convergence (loss trajectory) | Matches Python | PASS |

Every gradient is hand-derived and verified against PyTorch's automatic differentiation. No autograd, no computation graph, no tape.

### Performance results

**128-dim (354K params, Shakespeare, 3000 iters, i7-14700K + RTX 4070 Ti):**

| Metric | PyTorch + CUDA | Kerr Engine (CPU) | Improvement |
|---|---|---|---|
| Total training time | ~10 min | 3 min 21 sec | **3x faster** |
| Median iteration | ~200 ms | ~49 ms | 4x |
| GPU utilisation | 100% at ~70°C | 0% (off) | No GPU needed |
| CPU utilisation | ~100% (1 core, Python GIL) | ~14% (4 of 28 threads) | 86% headroom |
| Final train loss | ~2.0 | 1.91 | Better convergence |
| Final val loss | ~2.1 | 2.20 | Same range |

**768-dim (12M params, Shakespeare, RTX 4070 Ti full GPU backend) — BENCHMARK ONLY:**

These numbers are from a **50-iteration benchmark**, not a complete training run. GPU training at this scale has not been stress-tested with full training (10K+ iterations), checkpoint saves, curriculum, or eval passes. The shaders are validated against CPU (bit-identical) but production GPU training at 768-dim is untested.

| Metric | Value |
|---|---|
| GPU utilisation | 16% at 49°C (dispatch overhead dominated) |
| VRAM used | 1.2 / 12.0 GB |
| Iteration time | 1.72s (measured, 20-iter avg, down from 13s before batching) |
| Loss trajectory | 4.28 → 3.07 (50 iters) — correct convergence |
| Attention backward (GPU) | 4 ms |
| c_attn weight gradient | 14 ms (28x faster than per-position dispatch) |
| c_proj weight gradient | 7 ms (28x faster than per-position dispatch) |

**GPU backward shader impact (768-dim, per block):**

| Operation | Before batching | After batching | Speedup |
|---|---|---|---|
| c_proj backward | ~200 ms | ~7 ms | 28x |
| c_attn backward | ~400 ms | ~14 ms | 28x |
| Attention backward | CPU ~400 ms (est.) | GPU 4 ms | ~100x |
| FFN backward | ~340 ms | ~61 ms | 5.6x |
| Total backward pass | ~4.4 s | ~968 ms | 4.5x |

**Forward pass batching impact (768-dim):**

| Component | Before batching | After batching | Speedup |
|---|---|---|---|
| Kerr-ODE forward (3 blocks) | ~6,060 ms | ~150 ms (est.) | ~40x |
| QKV + output projections | ~1,465 ms | ~50 ms (est.) | ~30x |
| Layer norms | ~418 ms | ~30 ms (est.) | ~14x |
| Total forward pass | ~8,060 ms | ~500 ms | **16x** |
| **Total iteration** | **~13 s** | **1.72 s** (measured, 20-iter avg) | **7.6x** |

**Optimisation history (128-dim, 200 iters):**

| Optimisation | Iter time | Cumulative gain |
|---|---|---|
| Initial (single-threaded) | 270 ms | baseline |
| Batch parallelism (4 threads) | 78 ms | 3.5x |
| mimalloc allocator | 65 ms | 4.2x |
| Clone elimination | 62 ms | 4.4x |
| Iterator dot products + AVX2/FMA | 50 ms | 5.4x |

All optimisations are bit-identical — same loss trajectory, same generated text. The speed gains come entirely from removing software overhead, not from changing the maths.

---

## Architecture

The Kerr-ODE architecture replaces dense feed-forward layers with a physics-inspired wave propagation step based on coupled nonlinear optical resonators.

**Model structure (4 blocks):**
- Block 0: Causal self-attention + PerBandLinear (2×2 per-band projection)
- Blocks 1-3: Causal self-attention + Kerr-ODE with Maestro sync

**Kerr-ODE derivative (the core computation):**
```
For each frequency band k:
    φ[k] = ω[k] + α·|Z[k]|² + β·neighbours[k]
    dZ[k]/dt = -γ[k]·Z[k] + i·φ[k]·Z[k]
```

Where neighbours are coupled via a [1,1,0,1,1] convolution kernel — each band interacts with its two nearest neighbours on each side. This is a stencil operation, not a dense matrix multiply. It scales linearly with band count where MLP scales quadratically with hidden dimension.

**Maestro:** A global synchronisation bottleneck (128→16→128 with GELU) that coordinates across all bands. Additive, not multiplicative.

**Embeddings:** Frozen harmonic table using cos(n·θ)/sin(n·θ). Not trainable — the harmonic structure is fixed by design.

Integration: 8 RK4 steps per layer at dt=0.125.

---

## GPU backend

The engine includes a WGPU compute shader backend that runs on any GPU — NVIDIA, AMD, Intel, Apple Silicon. No CUDA dependency.

**Two tiers:**
- **CPU** — Used automatically below 768-dim. Fastest at small scale due to zero dispatch overhead.
- **Full GPU** — Forward and backward on GPU. All operations batched across positions. Attention backward via two-dispatch shader (4ms at 768-dim). Batched linear backward (28x speedup). Batched Kerr-ODE forward and backward. Weight gradient accumulation via outer product shader. Auto-selects at 768+ dim. Remaining bottleneck: CPU dispatch overhead (16% GPU utilisation at 768-dim). Unified memory architectures (Apple Silicon) would eliminate this entirely.

**Benchmarked at 128-dim (RTX 4070 Ti):**

| Backend | Kerr-ODE time | vs CPU |
|---|---|---|
| CPU | 11 μs | baseline |
| GPU per-call | 36,458 μs | 3,314x slower |
| GPU persistent | 123 μs | 11x slower |

At 128-dim, CPU wins. The GPU persistent pipeline eliminates 99.7% of dispatch overhead vs per-call, but the remaining ~120μs fixed cost exceeds CPU compute time. The crossover is ~768-dim, where O(n²) matmul compute overtakes fixed dispatch cost.

---

## Compute shaders

| Shader | Purpose |
|---|---|
| `matvec.wgsl` | Matrix-vector multiply (powers all linear projections) |
| `layer_norm.wgsl` | Parallel tree reduction layer normalisation (up to 2048-dim) |
| `kerr_step.wgsl` | Single Kerr-ODE derivative evaluation |
| `kerr_rk4_step.wgsl` | Fused RK4 (4 derivatives + combination in one dispatch) |
| `gelu.wgsl` | Element-wise GELU activation |
| `vec_add.wgsl` | Element-wise vector addition |
| `matvec_backward.wgsl` | Single-position gradient through linear layer |
| `matvec_backward_batch.wgsl` | Batched gradient across all positions (28x speedup) |
| `layer_norm_backward.wgsl` | Full layer norm backward (d_x, d_weight, d_bias) |
| `gelu_backward.wgsl` | Element-wise GELU gradient |
| `attn_backward_scores.wgsl` | Attention backward Phase 1: softmax backward + d_q |
| `attn_backward_dkv.wgsl` | Attention backward Phase 2: d_k + d_v |
| `outer_product.wgsl` | Batched weight gradient accumulation (d_W = D_Y^T @ X) |
| `matvec_batch.wgsl` | Batched forward linear: all positions in one dispatch |
| `layer_norm_batch.wgsl` | Batched forward layer norm: all positions in one dispatch |
| `kerr_step_batch.wgsl` | Batched Kerr-ODE derivative: all positions in one dispatch |
| `kerr_backward_batch.wgsl` | Batched Kerr-ODE derivative backward: all positions in one dispatch |

---

## Model compatibility

The Kerr-ODE is a novel architecture — it is NOT a standard transformer and does NOT work with existing inference clients out of the box.

**Validated at 128-dim (CPU):**
- **Train and generate** — full training pipeline with curriculum, checkpointing, eval. 3000+ iteration runs tested, loss trajectories match Python reference. This is production-ready.
- **Serve via [Kerr Server](https://github.com/atech-hub/kerr-server)** — OpenAI-compatible API server (~1,900 lines Rust, SSE streaming, bearer token auth, wave memory). CPU inference at 128-dim is instant. Verified with LM Studio 0.4.6.
- **BPE tokenizer support** — load any HuggingFace `tokenizer.json` (Qwen, Llama, GPT-2) via `--bpe` flag

**Benchmarked but not production-tested (GPU, 768-dim):**
- GPU forward and backward shaders validated against CPU (bit-identical)
- 50-iteration benchmark at 768-dim: 1.72s/iter, correct convergence
- **Full training runs (10K+ iters) at 768-dim have NOT been tested**
- **Server inference is CPU-only** — at 768-dim, inference is ~1.7s per token (impractical for chat). GPU inference for the server is not implemented.
- **Dimensions between 128 and 768 are untested**

**What won't work natively:**
- LM Studio, Ollama, llama.cpp — these expect dense MLP feed-forward layers. The Kerr-ODE uses RK4 integration with neighbour coupling, which no existing runtime supports natively. Use the [Kerr Server](https://github.com/atech-hub/kerr-server) as the bridge.
- Hugging Face Transformers — no Kerr-ODE model class exists in the library
- GGUF/GGML export — the format has no representation for ODE integration steps or stencil coupling

**What's needed for deeper ecosystem integration:**
- A GGUF exporter with custom Kerr-ODE operators and a corresponding llama.cpp fork
- A Hugging Face model class implementing the Kerr-ODE forward pass

These are documented as defensive publications in the parent project's [ENGINE-PATTERNS.md](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive/blob/main/ENGINE-PATTERNS.md) (Pattern 68) to prevent patent enclosure. Anyone can build them.

---

## Project structure

```
kerr-engine/
├── src/
│   ├── main.rs          CLI dispatch
│   ├── model.rs         Weight structs, inference forward pass
│   ├── pipeline.rs      Cached forward + backward chain (training)
│   ├── backward.rs      Hand-derived gradient primitives
│   ├── backend.rs       ComputeBackend trait + CpuBackend + auto-select
│   ├── gpu_backend.rs   Re-export module (thin wrapper)
│   ├── gpu_pipelines.rs Shader pipeline compilation + GpuBackend struct
│   ├── gpu_dispatch.rs  ComputeBackend impl for GpuBackend
│   ├── gpu_validate.rs  GPU validation + benchmarking
│   ├── gpu_persistent.rs Persistent GPU pipeline (fused RK4)
│   ├── gpu.rs           Stage 1 GPU validation
│   ├── train.rs         Training loop, curriculum, eval
│   ├── optim.rs         Adam optimizer, parameter flatten/unflatten
│   ├── data.rs          Dataset, char/word tokenizers
│   ├── init.rs          Weight initialisation
│   ├── checkpoint.rs    Binary checkpoint save/load (bit-perfect resume)
│   ├── bpe.rs           BPE tokenizer (HuggingFace tokenizer.json)
│   ├── lib.rs           Library interface for external crates
│   ├── weights.rs       Python weight loader (for validation)
│   ├── grad_test.rs     Gradient validation harness
│   └── rng.rs           Deterministic xorshift64 PRNG
├── shaders/             WGSL compute shaders
├── data/                Training corpora
├── reference/           Python export scripts for validation
├── CLAUDE.md            Internal developer reference
└── LICENSE              Apache 2.0
```

~7,500 lines of Rust. 21 modules. 17 compute shaders. 4 dependencies.

---

## Security

WGSL compute shaders are compiled and executed by the system's GPU driver stack (Vulkan, Metal, DirectX). While the engine's shipped shaders are simple compute operations (stencil ops, matvec, reductions), GPU driver shader compilers have known vulnerability classes — see [DarthShader (CCS 2024)](https://github.com/wgslfuzz/darthshader) for documented examples across Chrome, Firefox, and Safari.

**For users:** The shipped shaders are validated, correct, and structurally simple. They do not contain patterns known to trigger driver bugs (no self-referential types, no infinite loops, no deeply nested structures). Risk is minimal.

**For contributors:** Shader files (`.wgsl`) are security-sensitive. Changes to shaders must pass the 200-iteration bit-identical validation baseline and will be reviewed for both correctness and structural safety. Unnecessarily complex shader patterns are a red flag.

**Design principle:** The engine does not accept user-supplied shaders at runtime. All shaders are compiled into the binary at build time. This eliminates the runtime attack surface that browser-based WebGPU faces.

Report security concerns via [GitHub Security Advisories](https://github.com/atech-hub/kerr-engine/security/advisories) or open an issue with the `security` label.

---

## Contributing

The maintainer (Marco Da Cunha) is an IT systems administrator, not a programmer. The engine was built through a three-way collaboration with AI (Claude Desktop for theory, Claude Code for implementation). This is stated openly.

What this means for contributions:

- **Main branch is protected.** All changes go through pull requests.
- **Fork and branch.** Want to optimise a shader, add a feature, fix a bug? Fork the repo, create a branch, do your work, submit a PR.
- **The validation gate is the review.** Every PR that touches engine internals must pass the 200-iteration bit-identical baseline (seed 42, batch=4, seq=64, lr=3e-4, `--no-curriculum`). Expected: 4.2556, 3.3216, 2.7094, 2.7213, 2.6161. If the numbers change, something broke.
- **Shader PRs get extra scrutiny.** WGSL files are security-sensitive (see Security section). Keep shaders simple and structurally clean.
- **The maintainer merges based on validation results and description, not code review.** Be clear about what you changed and why.

**Known optimisation targets for contributors:**
- Persistent buffer pool for GPU dispatches (320 backward dispatches each create ~13 buffers — GPU at 16% utilisation because CPU dispatch overhead dominates. Reusing buffers would eliminate the bottleneck. The gpu_persistent.rs pattern already solves this for inference — extending it to training is the biggest remaining win)
- Fused attention forward shader (head loop is 28ms, minor but could eliminate last CPU component)
- SIMD inner loops (viable at 512+ dim, not at 128-dim)

Every gradient is mathematically verified. Every validation gate passes independently. The code is the code — it either works or it doesn't.

## Credits

- **Marco Da Cunha** — Direction, architecture decisions, pattern recognition
- **Claude Desktop (Opus)** — Theory, analysis, documentation, mathematical derivations
- **Claude Code** — Implementation, testing, validation, reality checks

---

## Related

- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT license)
- [Kerr Server](https://github.com/atech-hub/kerr-server) — OpenAI-compatible inference server for this engine (public, Apache 2.0)
- DOI: [10.5281/zenodo.18607190](https://doi.org/10.5281/zenodo.18607190) (concept DOI — always resolves to latest version)

---

## License

Apache 2.0. See [LICENSE](LICENSE).
