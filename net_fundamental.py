

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from net_autoencoder import load_autoencoder


class FundamentalFeaturesExtractor(BaseFeaturesExtractor):
    """Per-stock autoencoder + FFNN, potem konkatenacja z sector_weights."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        autoencoder_path: str,
        autoencoder_hidden: tuple,
        autoencoder_latent: int,
        ffnn_hidden: tuple = (128, 64),
        dropout: float = 0.15,
    ):
        # Sprawdz ksztalty
        fund_space = observation_space.spaces["fundamental"]
        sec_space = observation_space.spaces["sector_weights"]
        assert len(fund_space.shape) == 3, f"oczekiwany [N, L, F], jest {fund_space.shape}"
        n_stocks, lookback, n_features = fund_space.shape
        n_sectors = sec_space.shape[0]
        per_stock_latent = 32

        # KRYTYCZNE: super().__init__() PRZED przypisaniem nn.Module
        super().__init__(observation_space,
                          features_dim=n_stocks * per_stock_latent + n_sectors)

        # Encoder (frozen)
        self.encoder = load_autoencoder(
            autoencoder_path, hidden=list(autoencoder_hidden),
            input_dim=n_features, latent_dim=autoencoder_latent, dropout=0.0,
        )

        # Per-stock FFNN: dostaje sklejony lookback*latent -> latent space
        ffnn_input = lookback * autoencoder_latent
        layers = []
        prev = ffnn_input
        for h in ffnn_hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, per_stock_latent)]
        self.ffnn = nn.Sequential(*layers)

        self.n_stocks = n_stocks
        self.lookback = lookback
        self.n_features = n_features
        self.latent_dim = autoencoder_latent
        self.per_stock_latent = per_stock_latent
        self.n_sectors = n_sectors

    def forward(self, obs: dict) -> torch.Tensor:
        fund = obs["fundamental"]               # [B, N, L, F]
        sec_w = obs["sector_weights"]            # [B, n_sectors]
        B = fund.shape[0]
        N, L, F = self.n_stocks, self.lookback, self.n_features

        # Encoder per (spolka, dzien)
        flat = fund.reshape(B * N * L, F)
        with torch.no_grad():
            z = self.encoder.encode(flat)        # [B*N*L, latent]
        # [B*N, L*latent]
        z = z.reshape(B * N, L * self.latent_dim)
        # FFNN per stock
        per_stock = self.ffnn(z)                  # [B*N, per_stock_latent]
        per_stock = per_stock.reshape(B, N * self.per_stock_latent)

        # Konkatenacja: per_stock + sector_weights
        return torch.cat([per_stock, sec_w], dim=-1)
