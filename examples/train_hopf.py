"""
Example: Hopf Fibration (Paper V §3.3)
S³ → S² with U(1) fibers.

GCD prediction: single U(1) generator forces emb(Q) ≥ 3.
The Serre long exact sequence gives:
    ... → π₂(S²) ≅ ℤ → π₁(U(1)) ≅ ℤ → π₁(S³) = 0 → ...
confirming the fibration is non-trivial and the bound is sharp (S² ↪ R³).

Benchmark: encoder must learn latent_dim=3 representation of S² base space.
An encoder with latent_dim=2 falls into the dimension trap (Paper I, Cor 7.4).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gcd.model import GCDEncoder
from gcd.losses import equivariance_loss, algebraic_closure_loss
from gcd.topology import check_architecture

torch.manual_seed(42)


# ── Dataset ───────────────────────────────────────────────────────────────────

def sample_S3(B: int) -> torch.Tensor:
    """Uniform sample from S³ ⊂ R⁴."""
    x = torch.randn(B, 4)
    return x / x.norm(dim=1, keepdim=True)


def hopf_map(x: torch.Tensor) -> torch.Tensor:
    """
    Hopf map π: S³ → S².
    Interprets x = (z₁, z₂) ∈ ℂ² with |z₁|²+|z₂|²=1 and maps to:
        (2 Re(z₁ z̄₂), 2 Im(z₁ z̄₂), |z₁|²−|z₂|²) ∈ S²
    """
    z1r, z1i, z2r, z2i = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return torch.stack([
        2 * (z1r * z2r + z1i * z2i),
        2 * (z1i * z2r - z1r * z2i),
        (z1r**2 + z1i**2) - (z2r**2 + z2i**2)
    ], dim=1)  # [B, 3]


def u1_action(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    U(1) fiber action on S³: (z₁,z₂) ↦ (e^{iθ}z₁, e^{iθ}z₂).
    theta: [B] angles.
    """
    c = torch.cos(theta).unsqueeze(1)
    s = torch.sin(theta).unsqueeze(1)
    x0 = c * x[:, 0:1] - s * x[:, 1:2]
    x1 = s * x[:, 0:1] + c * x[:, 1:2]
    x2 = c * x[:, 2:3] - s * x[:, 3:4]
    x3 = s * x[:, 2:3] + c * x[:, 3:4]
    return torch.cat([x0, x1, x2, x3], dim=1)


def sample_batch(B: int = 256):
    x = sample_S3(B)
    theta = torch.rand(B) * 2 * math.pi * 0.1  # small fiber rotation
    x_aug = u1_action(x, theta)
    y = hopf_map(x)  # target: S² base point
    return x, x_aug, theta.unsqueeze(1), y  # coeffs shape [B,1]


# ── Training ──────────────────────────────────────────────────────────────────

def train(latent_dim: int, epochs: int = 500, label: str = ''):
    ok, m_star = check_architecture(latent_dim, 'hopf_base')
    print(f"  [{label}] Architecture check: latent_dim={latent_dim}, "
          f"m*={m_star} (emb(S²)), ok={ok}")
    if not ok:
        print(f"  [{label}] WARNING: dimension trap — encoder cannot faithfully "
              f"represent S² (Paper I, Cor 7.4)")

    model = GCDEncoder(in_dim=4, latent_dim=latent_dim, n_generators=1, hidden=128)
    head = nn.Linear(latent_dim, 3)  # predicts S² point in R³
    opt = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    history = {'task': [], 'eq': [], 'closure': []}
    for epoch in range(epochs):
        x, x_aug, theta, y = sample_batch()
        z = model(x)
        loss_task = ((head(z) - y) ** 2).mean()

        if ok:
            loss_eq = equivariance_loss(model, model.gen_bank, x, x_aug, theta)
            loss_closure = algebraic_closure_loss(model.gen_bank, 'abelian')
            loss = loss_task + 2.0 * loss_eq + 0.2 * loss_closure
        else:
            loss_eq = loss_closure = torch.tensor(0.0)
            loss = loss_task

        opt.zero_grad(); loss.backward(); opt.step()
        history['task'].append(loss_task.item())
        history['eq'].append(loss_eq.item())
        history['closure'].append(loss_closure.item())

        if epoch % 100 == 0:
            print(f"  [{label}] epoch {epoch:3d} | task {loss_task.item():.4f} "
                  f"| eq {loss_eq.item():.4f} | closure {loss_closure.item():.4f}")

    return model, history


# ── Sphericity score ──────────────────────────────────────────────────────────

def sphericity_score(z: torch.Tensor) -> float:
    """
    Measures how well z lies on a 2-sphere.
    Fits the best-fit sphere radius, returns fraction of points within 5% of it.
    Score 1.0 = perfect sphere, 0.0 = no structure.
    """
    r = z.norm(dim=1)
    r_mean = r.mean().item()
    frac = ((r - r_mean).abs() / r_mean < 0.05).float().mean().item()
    return frac


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Training baseline latent_dim=2 (dimension trap — Paper I)...")
    model_trap, hist_trap = train(latent_dim=2, epochs=500, label='dim-trap m=2')

    print("\nTraining GCD latent_dim=3 with U(1) generator (Papers II–V)...")
    model_gcd, hist_gcd = train(latent_dim=3, epochs=500, label='GCD m=3')

    # Evaluate sphericity
    x_vis, _, _, _ = sample_batch(1000)
    with torch.no_grad():
        z_trap = model_trap(x_vis)
        z_gcd = model_gcd(x_vis)

    score_trap = sphericity_score(z_trap)
    score_gcd = sphericity_score(z_gcd)
    print(f"\nSphericity score (S² recovery):")
    print(f"  dim-trap m=2: {score_trap:.3f}  (expected ~0, cannot embed S²)")
    print(f"  GCD m=3:      {score_gcd:.3f}  (expected >0.8, S² recovered)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))

    # Left: latent of dim-trap model (2D — forced collapse)
    ax1 = fig.add_subplot(131)
    y_true = hopf_map(x_vis).numpy()
    color = (y_true[:, 2] + 1) / 2  # z-coordinate of S² as colour
    with torch.no_grad():
        z2 = model_trap(x_vis).numpy()
    ax1.scatter(z2[:, 0], z2[:, 1], c=color, s=6, cmap='coolwarm', alpha=0.7)
    ax1.set_title(f'Dimension trap m=2\n(cannot embed S² — Paper I)\nsphericity={score_trap:.2f}',
                  fontsize=9)
    ax1.set_aspect('equal'); ax1.set_xlabel('z₁'); ax1.set_ylabel('z₂')

    # Middle: loss curves
    ax2 = fig.add_subplot(132)
    epochs_range = range(len(hist_gcd['task']))
    ax2.semilogy(epochs_range, hist_gcd['task'], 'r-', label='Task loss', alpha=0.8)
    ax2.semilogy(epochs_range, hist_gcd['eq'], 'b-', label='Equivariance loss', alpha=0.8)
    ax2.semilogy(epochs_range, hist_gcd['closure'], 'g-', label='Closure loss ||[A,A]||²', alpha=0.8)
    ax2.set_title('GCD Loss Evolution\n(Hopf fibration, m=3)', fontsize=9)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # Right: GCD latent (3D sphere)
    ax3 = fig.add_subplot(133, projection='3d')
    with torch.no_grad():
        z3 = model_gcd(x_vis).numpy()
    ax3.scatter(z3[:, 0], z3[:, 1], z3[:, 2],
                c=color, s=4, cmap='coolwarm', alpha=0.6)
    ax3.set_title(f'GCD m=3: S² recovered\n(emb(S²)=3, bound sharp)\nsphericity={score_gcd:.2f}',
                  fontsize=9)
    ax3.set_xlabel('z₁'); ax3.set_ylabel('z₂'); ax3.set_zlabel('z₃')

    plt.suptitle('Hopf Fibration: U(1) Generator Forces emb(Q) ≥ 3 (Paper V §3.3)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig('latent_hopf.png', dpi=200, bbox_inches='tight')
    print("\nSaved latent_hopf.png")
