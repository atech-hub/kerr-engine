# kerr-engine

Wave-native compute engine for the Kerr-ODE architecture.

Rust + rust-gpu + WGPU. No Python, no PyTorch, no CUDA dependency.

## Status

Stage 1 — Proof of concept (in progress)

## Architecture

```
kernels/kerr-ode/    -- GPU compute kernel (rust-gpu, compiles to SPIR-V)
src/                 -- Host-side orchestration (WGPU device, buffers, dispatch)
reference/           -- Python weight export scripts
tests/               -- Validation against Python reference outputs
```

## Stages

1. One Kerr-ODE layer, 64 bands, forward pass only. Validate against Python output.
2. Full forward pass (4 layers, positional encoding, maestro sync). Load Python-trained weights.
3. Analytical gradients (adjoint method). Training without autograd.
4. Full training loop. Python-free pipeline.

## Validation Gate

Stage N does not begin until Stage N-1 produces output matching Python
to within floating point tolerance. Numbers match or we don't move forward.
