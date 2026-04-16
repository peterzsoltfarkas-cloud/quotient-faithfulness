"""
Example: Phase Cylinder (Paper I–V running example)
X = [cosθ, sinθ] + noise, θ ∈ [0,2π)
Task: predict θ (regression) — task-sufficient but not quotient-faithful with m=1.

We compare:
- baseline m=1 (traversal trap, Paper I): task loss alone, no generator
- GCD m=2 with U(1) generator (Papers II–V): recovers circle topology

The output figure latent_gcd.png shows both cases side-by-side:
left panel: collapsed line (traversal trap)
right panel: recovered circle ring (GCD)
"""
import torch, math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gcd.model import GCDEncoder
from gcd.losses import equivariance_loss, algebraic_closure_loss
from gcd.topology import check_architecture

torch.manual_seed(0)


def sample_batch(B=256, noise=0.05):
    theta = torch.rand(B) * 2 * math.pi
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    x = x + noise * torch.randn_like(x)
    # augmentation: rotate by delta (U(1) action in observation space)
    delta = (torch.rand(B) - 0.5) * 0.5
    x_aug = torch.stack([
        torch.cos(theta + delta), torch.sin(theta + delta)
    ], dim=1) + noise * torch.randn_like(x)
    y = theta.unsqueeze(1)
    # coeffs shape [B,1] for single U(1) generator
    return x, x_aug, delta.unsqueeze(1), y


# ── Baseline: m=1, task loss only (traversal trap) ───────────────────────────

def train_baseline(epochs=300):
    """Pure task loss, latent_dim=1. Falls into traversal trap (Paper I)."""
    enc = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    )
    head = nn.Linear(1, 1)
    opt = optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)

    for epoch in range(epochs):
        x, _, _, y = sample_batch()
        z = enc(x)
        loss = ((head(z) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 100 == 0:
            print(f"  [baseline] epoch {epoch:3d} | task loss {loss.item():.4f}")
    return enc


# ── GCD: m=2, U(1) generator (recovers circle) ───────────────────────────────

def train_gcd(latent_dim=2, epochs=300):
    ok, m_star = check_architecture(latent_dim, 'S1')
    print(f"  [GCD] Architecture check: latent_dim={latent_dim}, m*={m_star}, ok={ok}")
    model = GCDEncoder(in_dim=2, latent_dim=latent_dim, n_generators=1)
    head = nn.Linear(latent_dim, 1)
    opt = optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=1e-3
    )

    for epoch in range(epochs):
        x, x_aug, delta, y = sample_batch()
        z = model(x)
        loss_task = ((head(z) - y) ** 2).mean()
        loss_eq = equivariance_loss(model, model.gen_bank, x, x_aug, delta)
        loss_closure = algebraic_closure_loss(model.gen_bank, 'abelian')
        loss = loss_task + 1.0 * loss_eq + 0.1 * loss_closure
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 100 == 0:
            print(f"  [GCD]      epoch {epoch:3d} | task {loss_task.item():.4f} "
                  f"| eq {loss_eq.item():.4f} | closure {loss_closure.item():.4f}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training baseline m=1 (traversal trap, Paper I)...")
    baseline = train_baseline(epochs=300)

    print("\nTraining GCD m=2 with U(1) generator (Papers II–V)...")
    gcd_model = train_gcd(latent_dim=2, epochs=300)

    # Sample for visualisation
    x_vis, _, _, _ = sample_batch(1000, noise=0.02)
    theta_vis = torch.atan2(x_vis[:, 1], x_vis[:, 0]).numpy()

    with torch.no_grad():
        z_base = baseline(x_vis).cpu().numpy()      # [N,1]
        z_gcd  = gcd_model(x_vis).cpu().numpy()     # [N,2]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: traversal trap — all points collapsed onto a line
    axes[0].scatter(z_base[:, 0], [0] * len(z_base),
                    c=theta_vis, s=8, cmap='hsv', alpha=0.7)
    axes[0].set_title('Baseline m=1\n(traversal trap — Paper I)', fontsize=11)
    axes[0].set_xlabel('z₁'); axes[0].set_yticks([])
    axes[0].set_xlim(-3, 3)

    # Right: GCD — circle topology recovered
    axes[1].scatter(z_gcd[:, 0], z_gcd[:, 1],
                    c=theta_vis, s=8, cmap='hsv', alpha=0.7)
    axes[1].set_aspect('equal')
    axes[1].set_title('GCD m=2, U(1) generator\n(circle recovered — Papers II–V)', fontsize=11)
    axes[1].set_xlabel('z₁'); axes[1].set_ylabel('z₂')

    plt.suptitle('Phase Cylinder Running Example — Quotient Faithfulness Series',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('latent_gcd.png', dpi=200, bbox_inches='tight')
    print("\nSaved latent_gcd.png — left: traversal trap, right: circle recovered")
