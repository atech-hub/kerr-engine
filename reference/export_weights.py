"""
Export trained Phase C model weights to binary format for Rust loader.

Format: flat binary file, f32 little-endian.
Header: [magic=0x4B455252 ('KERR'), version=1, vocab_size, n_layers=4]
Then weights in fixed order (see WEIGHT_ORDER below).

Usage:
    python export_weights.py <model_checkpoint.pt> <output.bin>
    python export_weights.py --train-fresh <output.bin>
"""

import sys, os, struct
import torch
import numpy as np

# Add experiments dir
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'GitHub',
                                'Wave-Coherence-as-a-Computational-Primitive', 'experiments'))

from phaseC_integrated import (
    GPT, Dataset, build_harmonic_table, build_positional_table,
    N_BANDS, N_EMBD, N_HEAD, BLOCK_SIZE, DEVICE, MAESTRO_DIM,
    PROG_STAGES, MAX_ITERS, LEARNING_RATE, estimate_loss,
)

MAGIC = 0x4B455252  # 'KERR'
VERSION = 1


def write_f32(f, data):
    """Write a flat array of f32 values."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().float().numpy()
    if isinstance(data, np.ndarray):
        data = data.flatten()
    f.write(struct.pack(f'<{len(data)}f', *data))


def write_linear(f, module):
    """Write Linear layer: weight [out, in] then bias [out]."""
    write_f32(f, module.weight)
    write_f32(f, module.bias)


def write_layernorm(f, module):
    """Write LayerNorm: weight [dim] then bias [dim]."""
    write_f32(f, module.weight)
    write_f32(f, module.bias)


def export_model(model, output_path):
    """Export full model to binary."""
    vocab_size = model.wte_phase.shape[0]

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', len(model.blocks)))

        # Frozen embeddings: wte_phase [vocab_size, N_EMBD]
        write_f32(f, model.wte_phase)

        # Frozen positional: wpe [BLOCK_SIZE, N_EMBD]
        write_f32(f, model.wpe)

        # Blocks
        for i, block in enumerate(model.blocks):
            # LayerNorm 1
            write_layernorm(f, block.ln_1)

            # Attention
            write_linear(f, block.attn.c_attn)
            write_linear(f, block.attn.c_proj)

            # LayerNorm 2
            write_layernorm(f, block.ln_2)

            # FFN
            if i == 0:
                # PerBandLinear
                ffn = block.ffn
                write_f32(f, ffn.band_w)   # [N_BANDS, 2, 2]
                write_f32(f, ffn.band_b)   # [N_BANDS, 2]
                write_linear(f, ffn.out_proj)
            else:
                # KerrMaestroAdd
                ffn = block.ffn

                # Kerr weights
                write_f32(f, ffn.kerr._gamma_raw)  # [N_BANDS]
                write_f32(f, ffn.kerr.omega)        # [N_BANDS]
                write_f32(f, ffn.kerr.alpha)        # scalar
                write_f32(f, ffn.kerr.beta)         # scalar

                # Maestro weights
                write_linear(f, ffn.maestro.squeeze)      # [MAESTRO_DIM, N_EMBD]
                write_linear(f, ffn.maestro.process[1])   # [N_EMBD, MAESTRO_DIM] (index 1 = Linear after GELU)

                # Out projection
                write_linear(f, ffn.out_proj)

        # Final layer norm
        write_layernorm(f, model.ln_f)

        # LM head: weight [vocab_size, N_EMBD] (no bias)
        write_f32(f, model.lm_head.weight)

    file_size = os.path.getsize(output_path)
    print(f"Exported to {output_path} ({file_size:,} bytes)")


def train_and_export(output_path):
    """Train a fresh Shakespeare model and export it."""
    # Load Shakespeare
    data_path = os.path.join(_here, '..', '..', 'GitHub',
                             'Wave-Coherence-as-a-Computational-Primitive',
                             'experiments', 'data', 'shakespeare.txt')
    if not os.path.exists(data_path):
        import urllib.request
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            data_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    dataset = Dataset(text)
    print(f"Dataset: {len(text):,} chars, vocab={dataset.vocab_size}")

    # Train
    torch.manual_seed(42)
    model = GPT(dataset.vocab_size, mode="kerr", use_maestro=True, use_mag=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for i in range(MAX_ITERS):
        for step_thresh, nb in PROG_STAGES:
            if i >= step_thresh:
                model.n_bands_active = nb

        if i % 500 == 0:
            losses = estimate_loss(model, dataset)
            print(f"  step {i:>5} | val {losses['val']:.4f}")

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.n_bands_active = N_BANDS
    final = estimate_loss(model, dataset)
    print(f"  Final val: {final['val']:.4f}")

    # Also export test vectors for Rust validation
    export_test_vectors(model, dataset, output_path)

    # Export weights
    model.eval()
    export_model(model, output_path)


def export_test_vectors(model, dataset, weights_path):
    """Export input/output pairs for Rust validation."""
    model.eval()
    test_path = weights_path.replace('.bin', '_test.bin')

    with torch.no_grad():
        # Use first 32 tokens from validation set as test input
        test_tokens = dataset.val_data[:32].tolist()
        x = torch.tensor([test_tokens], dtype=torch.long).to(DEVICE)
        logits, _ = model(x)
        logits = logits[0].cpu().numpy()  # [32, vocab_size]

    with open(test_path, 'wb') as f:
        # Header
        n_tokens = len(test_tokens)
        vocab_size = logits.shape[1]
        f.write(struct.pack('<I', n_tokens))
        f.write(struct.pack('<I', vocab_size))

        # Input tokens
        f.write(struct.pack(f'<{n_tokens}I', *test_tokens))

        # Expected logits [n_tokens, vocab_size]
        write_f32(f, logits)

    print(f"Test vectors exported to {test_path}")
    print(f"  {n_tokens} tokens, vocab={vocab_size}")
    print(f"  First token: {test_tokens[0]}, logits range: [{logits[0].min():.4f}, {logits[0].max():.4f}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_weights.py --train-fresh <output.bin>")
        sys.exit(1)

    if sys.argv[1] == "--train-fresh":
        output = sys.argv[2] if len(sys.argv) > 2 else "model.bin"
        train_and_export(output)
    else:
        print("Checkpoint loading not yet implemented. Use --train-fresh.")
