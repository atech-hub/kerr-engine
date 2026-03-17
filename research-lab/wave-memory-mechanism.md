# Wave Memory Mechanism for Kerr-ODE Architectures

> Status: DRAFT — research-lab only, not for publication.
> Date: 2026-03-15
> Sources: Claude Desktop (theoretical motivation) + Claude Code (implementation design)

---

## 1. Core Concept

A persistent wave state file that accumulates experience across conversations.
The model weights never change. The memory file modifies where the Kerr-ODE
starts from — different initial conditions, different trajectory, different output.

The model reads and writes memory in the same medium it thinks in — harmonic
band oscillator states. No translation layer, no lossy conversion.

---

## 2. Why This Only Works in Kerr-ODE

Standard transformers have no equivalent mechanism:
- No ODE initial conditions to modify
- No nonlinear coupling that naturally amplifies/damps
- No band structure that enables selective memory per frequency
- No implicit regularisation that prevents accumulated state from diverging
- No native wave format to store memory in

Every other memory system (RAG, fine-tuning, vector databases) operates in a
TRANSLATED medium. The model thinks in hidden states, but remembers in text,
in frozen vectors, or in weight updates. There is always a translation
boundary that loses information.

Wave memory operates in the NATIVE medium. The model thinks in band oscillator
states. It remembers in band oscillator states. The Kerr dynamics that process
the current input are the SAME dynamics that process the memory.

---

## 3. Natural Accentuation — The Physics Does the Work

The Kerr-ODE's nonlinear dynamics naturally amplify meaningful memories and
damp meaningless ones. No separate attention mechanism needed.

### 3.1 Self-Phase Modulation as Selective Recall

The Kerr nonlinearity: `phi_k = omega_k + alpha * |Z_k|^2 + beta * neighbours_k`

The term `alpha * |Z_k|^2` means: bands with high magnitude get additional phase
rotation. If the memory state has high energy in band 17 (because previous
conversations consistently activated it), and the current input ALSO activates
band 17, the combined magnitude `|Z_17|^2` is larger than either alone. The phase
rotation is stronger. The band resonates.

If the memory has energy in band 17 but the current input doesn't activate it,
`|Z_17|^2` is small (just the memory contribution). The phase rotation is weak.
The band doesn't resonate. The memory fades naturally for this context.

This is selective recall without any mechanism for selective recall. The physics
does it.

### 3.2 Cross-Phase Modulation as Associative Recall

The neighbour coupling: `beta * sum(|Z_{k+-1,2}|^2)`

If the memory state has a cluster of high-energy bands (say bands 15-19 are all
active from previous conversations), and the current input excites band 17, the
neighbour coupling amplifies bands 15-16 and 18-19 as well. The memory "spreads"
through the local band neighbourhood.

This is associative recall. Activating one band pulls related bands from memory.
The stencil coupling IS the association mechanism.

### 3.3 Damping as Forgetting

The damping: `dZ_k/dt = -gamma_k * Z_k + ...`

Bands that are not reinforced by current input get damped toward zero. Over many
conversations, memory bands that never resonate with new input gradually decay.
The model naturally forgets what's no longer relevant.

The rate of forgetting is controlled by `gamma_k` — the same parameter that
controls stability during training. High-damping bands forget fast. Low-damping
bands remember long. The model already learned which bands should have high vs
low damping during training — that learning now doubles as a memory retention
schedule.

---

## 4. The Mechanism

### 4.1 Where Memory Injects (model.rs:454-460)

The Kerr-ODE deinterleaves the hidden state into `r[k]` and `s[k]` (real and
imaginary parts per band) before RK4 integration. This is the injection point.

```
Current code:
    for k in 0..n_bands {
        r[k] = x[k * 2];
        s[k] = x[k * 2 + 1];
    }

With wave memory:
    for k in 0..n_bands {
        r[k] = x[k * 2]     + alpha_mem * memory_r[layer][k];
        s[k] = x[k * 2 + 1] + alpha_mem * memory_s[layer][k];
    }
```

The injection is ADDITIVE — it shifts the starting point, it doesn't replace it.
This mirrors the maestro pattern (Pattern 61): additive fusion works,
multiplicative fusion destroys structure.

`alpha_mem` bounds how much memory can influence the initial conditions.
Small alpha (0.01-0.1) = gentle nudge. The ODE dynamics amplify the nudge
through the nonlinear coupling — the memory's influence grows through
integration, not through a large initial perturbation.

**Safety:** The Kerr-ODE has implicit regularisation (Pattern 62). The damping
term `-gamma_k * Z_k` pulls all states toward zero. If the memory pushes a band
to an unusual starting point, the damping counteracts it. The dynamics are
self-correcting — the ODE cannot be destabilised by bad initial conditions.

### 4.2 Where Memory Extracts (model.rs:462-467)

After RK4 integration completes, the final `r[k]` and `s[k]` represent the
ODE's evolved state. This is the extraction point — the inference IS the writing.

```
After RK4, for each token:
    for k in 0..n_bands {
        accum_r[layer][k] = decay * accum_r[layer][k] + (1.0 - decay) * r[k];
        accum_s[layer][k] = decay * accum_s[layer][k] + (1.0 - decay) * s[k];
    }
```

No separate extraction step. The forward pass writes the memory as a byproduct
of computation. The model doesn't "decide" to remember — it processes tokens,
and the processing leaves a trace.

### 4.3 Accumulation Strategy

**Within a conversation:** Running exponential moving average (EMA) across all
tokens. Every token contributes proportional to its ODE energy. High-energy
bands (strong signals) naturally dominate. Low-energy bands (noise) get washed
out over many tokens.

```
accum[k] = decay * accum[k] + (1.0 - decay) * current[k]
```

- `decay = 0.99`: slow accumulation, recent tokens weighted slightly more
- Bands consistently active across the conversation dominate
- Bands that spiked once and faded contribute minimally

**Optional enhancement — energy-gated writing:** Only update bands above an
energy threshold (the wave packet principle, Pattern 45):

```
energy_k = r_k^2 + s_k^2
if energy_k > threshold:
    accum[k] = (1-decay) * accum[k] + decay * current[k]
```

The threshold can be set from the model's training statistics — mean band energy.

### 4.4 Cross-Conversation Persistence

At conversation end, the accumulated state merges into the persistent memory file:

```
memory[k] = beta * memory[k] + (1.0 - beta) * accum[k]
```

`beta` controls how fast old memories fade vs new ones take hold:
- `beta = 0.95`: slow adaptation, 20 conversations to build strong memory
- `beta = 0.8`: fast adaptation, recent conversations dominate
- `beta = 1.0`: frozen, never updates (useful as control baseline)

### 4.5 Per-Layer Memory

Memory is stored per-layer (blocks 1-3 each have their own Kerr-ODE). The
frequency-depth investigation showed different layers do different things — L1
may accumulate different band patterns than L3. Each layer gets its own memory
vector.

Block 0 (PerBandLinear) has no ODE — no memory injection there.

---

## 5. The Complete Cycle

```
CONVERSATION START
|
+-- Load memory file M[layer][band] (or zeros if fresh)
|
+-- For each input token:
|   +-- Embedding + positional encoding -> h
|   +-- Block 0: Attention + PerBandLinear (no memory)
|   +-- Blocks 1-3: Attention + KerrMaestroAdd
|   |   +-- LayerNorm -> x
|   |   +-- Z_k = (x[2k], x[2k+1]) + alpha_mem * M[layer][k]  <- READ
|   |   +-- RK4 integration (8 steps)
|   |   |   Kerr dynamics: damping + dispersion + self-phase + coupling
|   |   |   (memory naturally amplified or damped by dynamics)
|   |   +-- Maestro (global coordination)
|   |   +-- Output projection
|   +-- After final Kerr layer: S_t = final ODE state
|   +-- accum[k] = decay * accum[k] + (1-decay) * S_t[k]       <- WRITE
|
+-- After all tokens:
|   +-- memory[k] = beta * memory[k] + (1-beta) * accum[k]
|   +-- Save to disk
|
CONVERSATION END
```

Properties:
- Read and write happen IN the forward pass, not before/after
- No separate memory model, no extraction, no translation
- The Kerr dynamics ARE the memory processing mechanism
- Damping provides natural forgetting
- Nonlinear coupling provides natural association
- Energy gating provides natural importance weighting
- Delete the file = fresh start, zero residual effects

---

## 6. Memory File Format (KWMF)

```
Magic:    "KWMF" (4 bytes)  -- Kerr Wave Memory File
Version:  1 (u32)
n_layers: u32               -- ODE layers (typically 3: blocks 1-3)
n_bands:  u32               -- bands per layer (typically 64)
alpha:    f32               -- injection strength used
decay:    f32               -- within-conversation EMA decay
beta:     f32               -- cross-conversation merge rate
n_convos: u32               -- conversations accumulated

For each layer (n_layers):
    r[n_bands]: f32[]       -- real part of memory state
    s[n_bands]: f32[]       -- imaginary part of memory state
```

Total: 32 bytes header + 3 x 64 x 2 x 4 = **1,568 bytes** at 128-dim.
At 768-dim (384 bands): 32 + 3 x 384 x 2 x 4 = **9,248 bytes**. Still tiny.

---

## 7. Test Plan

### Test 1: Injection Sanity Check

**Goal:** Verify that modifying initial conditions changes output without
breaking the model.

```
1. Train a model on Shakespeare (existing checkpoint)
2. Generate 200 tokens with no memory (baseline)
3. Create synthetic memory states:
   a. Zero memory -> output must match baseline exactly
   b. Random small memory (alpha=0.01) -> output differs but stays coherent
   c. Random large memory (alpha=1.0) -> output degrades (too much perturbation)
4. Measure: perplexity vs alpha, character-level coherence vs alpha
```

**Expected:** Sweet spot where small alpha produces slightly different but
coherent text. Proves the injection point works and the model is sensitive
to initial conditions.

### Test 2: Accumulation Stability

**Goal:** Verify that the EMA accumulator converges to a stable state.

```
1. Load trained model
2. Run 10 conversations (10 x 200-token generations)
3. After each conversation, save the accumulated memory state
4. Run harmonic census on each saved state
5. Plot: per-band energy across 10 conversations
```

**Expected:** Early conversations show high variance. Later conversations
converge to stable band energies. The implicit regularisation prevents runaway
growth. If any band grows without bound -> decay is too low.

### Test 3: Memory Carries Semantic Content

**Goal:** Show that accumulated memory produces contextually different output.

```
1. Train model on Shakespeare
2. Generate 5 conversations about "love" -> save love_memory.kwmf
3. Generate 5 conversations about "war" -> save war_memory.kwmf
4. Generate from a NEUTRAL prompt with:
   a. No memory (baseline)
   b. love_memory
   c. war_memory
5. Measure: word frequency overlap with love/war vocabulary,
   harmonic census difference between love/war memories
```

**Expected:** Love-memory produces love-themed text, war-memory produces
war-themed text, from the same neutral prompt. The harmonic census shows
different band distributions. Love and war live in different frequencies.

### Test 4: Memory Reset

**Goal:** Verify that deleting the memory file restores baseline behaviour.

```
1. Build up 20 conversations of memory
2. Generate with memory -> output_with_memory
3. Delete memory file (or load zero-state)
4. Generate with same prompt -> output_without_memory
5. Compare to original baseline from Test 1
```

**Expected:** Output without memory is statistically identical to baseline.
Complete separation. No residual effects.

### Test 5: Harmonic Census Inspection

**Goal:** Demonstrate that memory is inspectable and anomalies detectable.

```
1. Accumulate memory from 20 Shakespeare conversations
2. Run harmonic census:
   - Per-band energy (r^2+s^2 per band per layer)
   - Cross-layer correlation
   - Band stability (variance across layers)
3. Inject adversarial pattern (spike band 32 to 10x normal)
4. Run census again -> detect the anomaly
```

**Expected:** Normal memory has a smooth energy profile. Adversarial spike
is immediately visible. A simple threshold detector catches it.

---

## 8. Implementation Order

1. Add `WaveMemory` struct to model.rs (read-only, no accumulation yet)
2. Modify `kerr_ode_forward` to accept optional memory injection
3. Run Test 1 (injection sanity check) — validate mechanism works at all
4. Add `MemoryAccumulator` for extraction during generation
5. Add `wave_memory.rs` module (save/load KWMF format)
6. Run Test 2 (stability) — validate accumulation doesn't diverge
7. Run Test 3 (semantic memory) — validate memory carries meaning
8. Run Tests 4-5 (safety/inspection) — validate reset and auditability
9. Integrate into kerr-server (--memory flag)

Each step independently testable. Stop at any null result.

---

## 9. Hyperparameters to Sweep

| Parameter | Range | Purpose |
|-----------|-------|---------|
| alpha_mem (injection strength) | 0.001 - 0.5 | How much memory influences initial conditions |
| decay (within-conversation EMA) | 0.9 - 0.999 | How fast tokens accumulate within a conversation |
| beta (cross-conversation merge) | 0.8 - 0.99 | How fast old memories fade |
| layer selection | all layers vs last only | Whether all ODE layers have separate memory |
| band masking | all bands vs top-K energy | Whether to zero low-energy bands before saving |

Start with: alpha_mem=0.05, decay=0.99, beta=0.95, all layers, all bands.
Sweep one parameter at a time from this baseline.

---

## 10. Open Questions

1. **Should memory inject into Block 0?** Block 0 uses PerBandLinear, not
   Kerr-ODE. It does impedance matching. Memory injection there might help
   or hurt. Test it.

2. **Per-position vs aggregate?** The draft uses aggregate (one memory vector
   per layer). Alternative: per-position memory (what was said at position 5
   matters). 256x more data but might capture sequence-level patterns. Start
   with aggregate.

3. **Phase vs magnitude priority?** The spherical coherence investigation
   showed phase carries semantics and magnitude amplifies. The EMA naturally
   handles this — phase relationships that persist survive averaging,
   magnitude spikes don't. But explicitly separating them may help.

4. **Training awareness?** Current design is inference-only — the model was
   never trained with memory injection. Future step: train with random memory
   injections so the model learns to USE the signal. But start without this.

5. **The gamma connection.** The model's trained gamma values encode a
   retention schedule. High-damping bands forget fast, low-damping bands
   remember long. Can we use gamma to weight the memory directly?
   `alpha_mem_k = base_alpha / gamma_k` — bands the model learned to
   preserve get stronger memory influence.

---

## 11. Code Changes Required

### model.rs

```rust
pub struct WaveMemory {
    pub r: Vec<Vec<f32>>,    // [n_layers][n_bands]
    pub s: Vec<Vec<f32>>,    // [n_layers][n_bands]
    pub alpha: f32,          // injection strength
}

// Modify kerr_ode_forward signature:
pub fn kerr_ode_forward(
    &self,
    weights: &KerrWeights,
    x: &[f32],
    memory: Option<(&WaveMemory, usize)>,  // (memory, layer_idx)
) -> Vec<f32> {
    // ... deinterleave r, s from x ...

    // Inject memory
    if let Some((mem, layer)) = memory {
        for k in 0..n_bands {
            r[k] += mem.alpha * mem.r[layer][k];
            s[k] += mem.alpha * mem.s[layer][k];
        }
    }

    // ... RK4 integration (unchanged) ...
}
```

### New module: wave_memory.rs

```rust
pub struct WaveMemoryFile {
    pub layers: Vec<MemoryLayer>,  // [n_layers]
    pub alpha: f32,
    pub decay: f32,
    pub beta: f32,
    pub n_convos: u32,
}

pub struct MemoryLayer {
    pub r: Vec<f32>,  // [n_bands]
    pub s: Vec<f32>,  // [n_bands]
}

pub struct MemoryAccumulator {
    pub layers: Vec<MemoryLayer>,
    pub decay: f32,
}

impl WaveMemoryFile {
    pub fn save(&self, path: &str) -> io::Result<()> { ... }
    pub fn load(path: &str) -> io::Result<Self> { ... }
    pub fn zeros(n_layers: usize, n_bands: usize) -> Self { ... }
    pub fn merge(&mut self, accum: &MemoryAccumulator) { ... }
    pub fn to_wave_memory(&self) -> WaveMemory { ... }
}

impl MemoryAccumulator {
    pub fn zeros(n_layers: usize, n_bands: usize, decay: f32) -> Self { ... }
    pub fn update(&mut self, layer: usize, r: &[f32], s: &[f32]) { ... }
}
```

### pipeline.rs

Modify `forward_with_cache` to thread `Option<&WaveMemory>` through block
processing. Only ODE layers (blocks 1+) receive the memory parameter.

### kerr-server main.rs

Add `--memory FILE` flag. Load KWMF at startup, pass to inference, save
updated state after each conversation.

---

*This document merges theoretical motivation (why wave memory works in the*
*Kerr-ODE's physics) with implementation design (where to inject, how to*
*test, what to build). The physics sections come from Desktop's analysis.*
*The test plan and implementation come from Code's engineering draft.*
