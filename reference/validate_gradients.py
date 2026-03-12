"""
Gradient validation: compute gradients in PyTorch for comparison with Rust.

Exports gradient test vectors:
1. Kerr-ODE derivative backward (single derivative call)
2. Full RK4 step backward
3. Full Kerr-ODE backward (8 steps)

For each: save input, parameters, upstream gradient, and expected output gradients.
"""

import sys, os, struct
import torch
import torch.nn as nn
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'GitHub',
                                'Wave-Coherence-as-a-Computational-Primitive', 'experiments'))

from phaseC_integrated import N_BANDS, N_EMBD

torch.manual_seed(42)


def write_f32(f, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().float().numpy()
    if isinstance(data, np.ndarray):
        data = data.flatten()
    f.write(struct.pack(f'<{len(data)}f', *data))


def test_kerr_derivative_backward():
    """Test backward through a single Kerr-ODE derivative evaluation."""
    print("=== Kerr derivative backward ===")

    # Create inputs that require grad
    r = torch.randn(1, N_BANDS, requires_grad=True)
    s = torch.randn(1, N_BANDS, requires_grad=True)

    # Parameters
    gamma_raw = torch.randn(N_BANDS, requires_grad=True)
    gamma = torch.nn.functional.softplus(gamma_raw)
    omega = torch.arange(1, N_BANDS + 1, dtype=torch.float32) / N_BANDS
    omega = omega.clone().requires_grad_(True)
    alpha = torch.tensor(0.1, requires_grad=True)
    beta = torch.tensor(0.1, requires_grad=True)

    # Forward: compute derivative
    mag_sq = r * r + s * s
    kernel = torch.tensor([[[1., 1., 0., 1., 1.]]])
    ns = torch.nn.functional.conv1d(mag_sq.unsqueeze(1), kernel, padding=2).squeeze(1)
    phi = omega + alpha * mag_sq + beta * ns
    dr = -gamma * r - phi * s
    ds_out = -gamma * s + phi * r

    # Create upstream gradient
    d_dr = torch.randn_like(dr)
    d_ds = torch.randn_like(ds_out)

    # Backward
    loss = (dr * d_dr + ds_out * d_ds).sum()
    loss.backward()

    print(f"  d_r max: {r.grad.abs().max().item():.6f}")
    print(f"  d_s max: {s.grad.abs().max().item():.6f}")
    print(f"  d_gamma_raw max: {gamma_raw.grad.abs().max().item():.6f}")
    print(f"  d_omega max: {omega.grad.abs().max().item():.6f}")
    print(f"  d_alpha: {alpha.grad.item():.6f}")
    print(f"  d_beta: {beta.grad.item():.6f}")

    return {
        'r': r.detach(), 's': s.detach(),
        'gamma_raw': gamma_raw.detach(), 'omega': omega.detach(),
        'alpha': alpha.detach(), 'beta': beta.detach(),
        'd_dr': d_dr.detach(), 'd_ds': d_ds.detach(),
        'd_r': r.grad.detach(), 'd_s': s.grad.detach(),
        'd_gamma_raw': gamma_raw.grad.detach(),
        'd_omega': omega.grad.detach(),
        'd_alpha': alpha.grad.detach(),
        'd_beta': beta.grad.detach(),
    }


def test_kerr_ode_backward():
    """Test backward through full Kerr-ODE (8 RK4 steps)."""
    print("\n=== Full Kerr-ODE backward (8 RK4 steps) ===")

    # Input
    x_flat = torch.randn(1, N_BANDS, 2, requires_grad=True)
    r = x_flat[:, :, 0]
    s = x_flat[:, :, 1]

    # Parameters (mimicking KerrODE layer init)
    gamma_raw = nn.Parameter(torch.full((N_BANDS,), np.log(np.exp(0.1) - 1)))
    omega = nn.Parameter(torch.arange(1, N_BANDS + 1, dtype=torch.float32) / N_BANDS)
    alpha_param = nn.Parameter(torch.tensor(0.1))
    beta_param = nn.Parameter(torch.tensor(0.1))
    kernel = torch.tensor([[[1., 1., 0., 1., 1.]]])

    n_steps = 8
    dt = 1.0 / n_steps

    # Forward: 8 RK4 steps
    r_curr = r
    s_curr = s
    for _ in range(n_steps):
        gamma = torch.nn.functional.softplus(gamma_raw)

        def deriv(r_in, s_in):
            msq = r_in * r_in + s_in * s_in
            ns = torch.nn.functional.conv1d(msq.unsqueeze(1), kernel, padding=2).squeeze(1)
            phi = omega + alpha_param * msq + beta_param * ns
            return -gamma * r_in - phi * s_in, -gamma * s_in + phi * r_in

        k1r, k1s = deriv(r_curr, s_curr)
        k2r, k2s = deriv(r_curr + 0.5 * dt * k1r, s_curr + 0.5 * dt * k1s)
        k3r, k3s = deriv(r_curr + 0.5 * dt * k2r, s_curr + 0.5 * dt * k2s)
        k4r, k4s = deriv(r_curr + dt * k3r, s_curr + dt * k3s)

        r_curr = r_curr + (dt / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        s_curr = s_curr + (dt / 6) * (k1s + 2 * k2s + 2 * k3s + k4s)

    # Output
    output = torch.stack([r_curr, s_curr], dim=2)  # [1, N_BANDS, 2]

    # Upstream gradient
    d_output = torch.randn_like(output)

    # Backward
    loss = (output * d_output).sum()
    loss.backward()

    print(f"  d_input max: {x_flat.grad.abs().max().item():.6f}")
    print(f"  d_gamma_raw max: {gamma_raw.grad.abs().max().item():.6f}")
    print(f"  d_omega max: {omega.grad.abs().max().item():.6f}")
    print(f"  d_alpha: {alpha_param.grad.item():.6f}")
    print(f"  d_beta: {beta_param.grad.item():.6f}")

    return {
        'input': x_flat.detach().reshape(-1),
        'gamma_raw': gamma_raw.detach(),
        'omega': omega.detach(),
        'alpha': alpha_param.detach(),
        'beta': beta_param.detach(),
        'd_output': d_output.detach().reshape(-1),
        'd_input': x_flat.grad.detach().reshape(-1),
        'd_gamma_raw': gamma_raw.grad.detach(),
        'd_omega': omega.grad.detach(),
        'd_alpha': alpha_param.grad.detach(),
        'd_beta': beta_param.grad.detach(),
    }


def export_gradient_tests(output_path):
    """Export gradient test vectors to binary file."""
    deriv_test = test_kerr_derivative_backward()
    ode_test = test_kerr_ode_backward()

    with open(output_path, 'wb') as f:
        # Magic + version
        f.write(struct.pack('<I', 0x47524144))  # 'GRAD'
        f.write(struct.pack('<I', 1))

        # Test 1: Kerr derivative backward
        f.write(struct.pack('<I', 1))  # test id
        for key in ['r', 's', 'gamma_raw', 'omega', 'alpha', 'beta',
                     'd_dr', 'd_ds', 'd_r', 'd_s', 'd_gamma_raw',
                     'd_omega', 'd_alpha', 'd_beta']:
            write_f32(f, deriv_test[key])

        # Test 2: Full Kerr-ODE backward
        f.write(struct.pack('<I', 2))  # test id
        for key in ['input', 'gamma_raw', 'omega', 'alpha', 'beta',
                     'd_output', 'd_input', 'd_gamma_raw', 'd_omega',
                     'd_alpha', 'd_beta']:
            write_f32(f, ode_test[key])

    print(f"\nExported to {output_path} ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "reference/gradient_test.bin"
    export_gradient_tests(output)
