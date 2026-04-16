"""
GCDEncoder — encoder with attached generator bank.
Paper V §9.1–9.4
"""
import torch.nn as nn
from .generators import LieGeneratorBank


class GCDEncoder(nn.Module):
    def __init__(self, in_dim=2, latent_dim=2, n_generators=1, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        self.gen_bank = LieGeneratorBank(latent_dim, n_generators)
        self.latent_dim = latent_dim

    def forward(self, x):
        return self.encoder(x)
