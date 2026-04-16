"""
Paper V §8.1 — Generator Matrix Parametrisation
Implements differentiable Lie algebra generators.
We learn skew-symmetric matrices A_i ∈ so(d) for compact groups.
For U(1)/SO(2) in 2D, a single generator suffices: A = [[0,-a],[a,0]].
For T² = S¹×S¹ (double pendulum), two commuting U(1) generators are needed.
"""
import torch
import torch.nn as nn


class LieGeneratorBank(nn.Module):
    def __init__(self, latent_dim: int, n_generators: int = 1, init_scale: float = 0.1):
        super().__init__()
        self.d = latent_dim
        self.k = n_generators
        self.W = nn.Parameter(torch.randn(n_generators, latent_dim, latent_dim) * init_scale)

    def generators(self):
        """Return skew-symmetric matrices A_i ∈ so(d)."""
        W = self.W
        return 0.5 * (W - W.transpose(-1, -2))  # [k, d, d]

    def action(self, z: torch.Tensor, coeffs: torch.Tensor):
        """
        Apply group action exp(Σ c_i A_i) to latent z.
        z: [B, d], coeffs: [B, k]
        """
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0).expand(z.size(0), -1)
        if coeffs.shape[-1] == 1 and self.k == 1:
            coeffs = coeffs.expand(z.size(0), 1)
        assert coeffs.shape == (z.size(0), self.k), (
            f"coeffs shape {tuple(coeffs.shape)} does not match "
            f"(batch={z.size(0)}, n_generators={self.k}). "
            f"For a single U(1) generator pass coeffs of shape [B,1]."
        )
        A = self.generators()  # [k, d, d]
        M = torch.einsum('bk,kij->bij', coeffs, A)  # [B, d, d]
        R = torch.matrix_exp(M)
        return torch.einsum('bij,bj->bi', R, z)

    def commutator_closure(self):
        """Compute [A_i, A_j] for all pairs i < j."""
        A = self.generators()
        comms = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                C = A[i] @ A[j] - A[j] @ A[i]
                comms.append(C)
        if not comms:
            return torch.zeros(0, self.d, self.d, device=A.device)
        return torch.stack(comms)  # [k(k-1)/2, d, d]
