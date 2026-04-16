"""
Paper V §8.2–8.3 — Equivariance and Algebraic Closure Losses
"""
import torch
import torch.nn.functional as F


def equivariance_loss(encoder, gen_bank, x, x_aug, coeff):
    """
    §8.2: L_eq = E[ || f(g·x) - exp(c A) f(x) ||² ]
    x_aug is g·x in observation space.
    coeff is the Lie algebra coordinate for that transformation, shape [B, k].
    """
    z = encoder(x)
    z_aug = encoder(x_aug)
    z_pred = gen_bank.action(z, coeff)
    return F.mse_loss(z_aug, z_pred)


def algebraic_closure_loss(gen_bank, target_structure='abelian'):
    """
    §8.3: L_closure = sum_{i<j} || [A_i, A_j] - sum_k c_{ijk} A_k ||²
    For abelian targets (U(1), T²=U(1)×U(1)), enforce [A_i,A_j] ≈ 0.
    """
    comms = gen_bank.commutator_closure()
    if comms.numel() == 0:
        return torch.tensor(0.0, device=gen_bank.W.device)
    if target_structure == 'abelian':
        return (comms ** 2).mean()
    else:
        raise NotImplementedError(
            "Non-abelian structure constant learning is not yet implemented. "
            "See Paper V §8.3 and the 'next steps' section of the README."
        )
