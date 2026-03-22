# Kerr Engine

> **⚠️ This repository is the original prototype implementation and is no longer under active development.**
>
> The project has evolved into a new architecture with significant improvements. The production training engine is now **[wave-engine](https://github.com/atech-hub/wave-engine)**, which features:
>
> - **Parallel block architecture** (GPT-J formulation) replacing the sequential block design
> - **Frozen harmonic coherence attention** — phase-based scoring replaces dot-product attention, no attention training needed
> - **Four training tiers** — CPU, cross-platform GPU (wgpu, any GPU), GPU fast mode (2.8x speedup), and NVIDIA CUDA (Candle)
> - **OFDM-inspired FFT ODE acceleration** — stencil coupling as frequency-domain convolution
> - **Pipeline monitor** — always-on per-section timing showing exactly where every millisecond goes
> - **Full train → save → serve → chat pipeline** with [wave-server](https://github.com/atech-hub/wave-server) (OpenAI-compatible API with KV-cache)
>
> The kerr-engine remains available as a historical reference. Its validated findings (98.1% MLP performance at 44% parameters, scaling to 1280-dim, dual-maestro architecture) are carried forward into wave-engine. The 70 defensive patterns documented in [ENGINE-PATTERNS.md](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive/blob/main/ENGINE-PATTERNS.md) remain protected.
>
> **New repos:**
> - **[wave-engine](https://github.com/atech-hub/wave-engine)** — Production training engine (Apache 2.0)
> - **[wave-server](https://github.com/atech-hub/wave-server)** — OpenAI-compatible inference server with KV-cache (Apache 2.0)
> - **[kerr-memory](https://github.com/atech-hub/kerr-memory)** — Wave memory state management (Apache 2.0, model-agnostic, works with both engines)

---

Pure Rust training and inference engine for the Kerr-ODE transformer architecture. No Python. No PyTorch. No CUDA toolkit.

**3x faster than PyTorch+CUDA on CPU alone.** Same convergence, GPU sitting idle.

---

## What is this?

A specialised engine for training and running [Wave Coherence](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) transformer models. The Kerr-ODE replaces dense MLP layers with a physics-inspired wave propagation step — achieving 98.1% of MLP performance at 44% of the parameters.

This engine implements the full training pipeline in Rust with hand-derived analytical gradients. No automatic differentiation, no computation graph, no framework overhead.

### Why we moved on

The kerr-engine proved the core concept — coupled harmonic oscillators can replace MLP layers with dramatically fewer parameters. But the architecture had limitations that became clear during scaling:

- **Sequential block design** — attention had to complete before FFN could start. The wave-engine's parallel block (GPT-J) formulation allows attention and FFN to run from the same input simultaneously.
- **Standard dot-product attention** — trained attention weights consumed parameters and compute. Wave-engine's frozen harmonic coherence attention eliminates attention training entirely.
- **Single maestro** — the pre-ODE coordination bottleneck worked but was insufficient at scale. Wave-engine uses dual-maestro (pre-ODE and post-ODE) validated to prevent NaN at 768+ dimensions.
- **GPU precision challenges** — kerr-engine's fused GPU pipeline worked (20% utilisation, correct training) but the wave-engine's ping-pong buffer pattern and per-section monitors provided better diagnostic visibility and the data to optimise the right operations.

The validated findings from kerr-engine — maestro dim=16 as a universal constant, curriculum training (+1.46pp), stochastic resonance (α=0.05, -8.8% perplexity), implicit regularisation, the ComputeBackend trait pattern — all transfer directly to wave-engine.

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

**Honest scale note:** All headline benchmarks (3x faster, 98.1% MLP performance, 354K params) are at **128-dim on CPU**. This is the validated, production-tested configuration. GPU training at 768-dim has been benchmarked (1.72s/iter, 50 iterations) but not stress-tested in full training runs. Dimensions between 128 and 768 are untested. If you're scaling up, consider using [wave-engine](https://github.com/atech-hub/wave-engine) which has been validated across the full range.

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

## Validated findings (carried forward to wave-engine)

These findings were discovered and validated in kerr-engine and are directly used in wave-engine:

- **98.1% of MLP performance at 44% parameters** (354K vs 801K, Phase C)
- **Maestro dim=16 is a universal constant** — tested at dim 16/96/128/160 on 768-dim, all within 0.028 loss
- **Curriculum training** — +1.46 percentage points, starting at fewer bands and unlocking progressively
- **Stochastic resonance** — α=0.05 noise on ODE initial conditions gives -8.8% perplexity improvement
- **Implicit regularisation** — Kerr-ODE stable where MLP overfits at 128 bands
- **Dual-maestro** — pre-ODE and post-ODE coordination, prevents NaN at 768+ dimensions
- **ComputeBackend trait** — CPU/GPU abstraction that routes all operations through the same device, ensuring forward/backward consistency
- **Scaling to 1280-dim** — validated across 128 to 1280-dim (640 bands), Shakespeare ceiling at val loss 2.47
- **lr=1e-4 and rk4-steps=16 required at 512+ bands** — hyperparameter boundary, not architectural limit

---

## Related

- **[wave-engine](https://github.com/atech-hub/wave-engine)** — Production training engine (successor to this repo)
- **[wave-server](https://github.com/atech-hub/wave-server)** — OpenAI-compatible inference server with KV-cache
- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT license)
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Persistent wave memory state (public, Apache 2.0)
- [Kerr Server](https://github.com/atech-hub/kerr-server) — OpenAI-compatible inference server for this engine (historical)

---

## Contributing

This repository is no longer under active development. Bug fixes and documentation improvements are welcome. For new feature development, please contribute to [wave-engine](https://github.com/atech-hub/wave-engine) or [wave-server](https://github.com/atech-hub/wave-server).

The maintainer (Marco Da Cunha) is an IT systems administrator, not a programmer. The engine was built through collaboration with AI (Claude Desktop for theory, Claude Code for implementation). This is stated openly.

## Credits

- **Marco Da Cunha** — Direction, architecture decisions, pattern recognition
- **Claude Desktop (Opus)** — Theory, analysis, documentation, mathematical derivations
- **Claude Code** — Implementation, testing, validation, reality checks

---

## License

Apache 2.0. See [LICENSE](LICENSE).
