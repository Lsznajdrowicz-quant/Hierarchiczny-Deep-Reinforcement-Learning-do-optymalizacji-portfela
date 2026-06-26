
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _PerStockEncoder(nn.Module):
    """CNN1D + LSTM stosowane do KAZDEJ spolki osobno (params shared)."""

    def __init__(self, n_features: int, lookback: int,
                  cnn_channels: tuple, cnn_kernel: int,
                  lstm_hidden: int, dropout: float):
        super().__init__()
        layers = []
        prev = n_features
        for c in cnn_channels:
            layers += [
                nn.Conv1d(prev, c, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = c
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(prev, lstm_hidden, batch_first=True)
        self.lstm_hidden = lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [BN, L, F]
        x = x.transpose(1, 2)         # [BN, F, L]
        x = self.cnn(x)                # [BN, C_out, L]
        x = x.transpose(1, 2)         # [BN, L, C_out]
        _, (h, _) = self.lstm(x)
        return h[-1]                   # [BN, lstm_hidden]


class TechFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_channels: tuple = (32, 48),
        cnn_kernel: int = 3,
        lstm_hidden: int = 64,
        head_hidden: tuple = (96, 48),
        dropout: float = 0.15,
    ):
        tech_space = observation_space.spaces["technical"]
        assert len(tech_space.shape) == 3, f"oczekiwany [N, L, F], jest {tech_space.shape}"
        n_stocks, lookback, n_features = tech_space.shape
        per_stock_out = 8

        total_dim = n_stocks * per_stock_out + 3 * n_stocks  # + target, current, prev_tilts
        # KRYTYCZNE: super().__init__() PRZED przypisaniem nn.Module
        super().__init__(observation_space, features_dim=total_dim)

        # Per-stock encoder (shared params)
        self.encoder = _PerStockEncoder(
            n_features=n_features, lookback=lookback,
            cnn_channels=cnn_channels, cnn_kernel=cnn_kernel,
            lstm_hidden=lstm_hidden, dropout=dropout,
        )

        # Head MLP per stock (shared) -> 8 wymiarow na spolke
        head_layers = []
        prev = lstm_hidden
        for h in head_hidden:
            head_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        head_layers += [nn.Linear(prev, per_stock_out)]
        self.head = nn.Sequential(*head_layers)

        self.n_stocks = n_stocks
        self.lookback = lookback
        self.n_features = n_features
        self.per_stock_out = per_stock_out

    def forward(self, obs: dict) -> torch.Tensor:
        tech = obs["technical"]                # [B, N, L, F]
        target = obs["target_weights"]          # [B, N]
        current = obs["current_weights"]        # [B, N]
        prev_tilt = obs["previous_tilts"]       # [B, N]
        B, N, L, F = tech.shape

        # Reshape do per-stock processing — kazda spolka jako osobna probka
        x = tech.reshape(B * N, L, F)
        per_stock_latent = self.encoder(x)      # [B*N, lstm_hidden]
        per_stock = self.head(per_stock_latent) # [B*N, per_stock_out]
        per_stock = per_stock.reshape(B, N * self.per_stock_out)

        # Konkatenacja per-stock features + kontekst
        return torch.cat([per_stock, target, current, prev_tilt], dim=-1)
