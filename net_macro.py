# file: net_macro.py


from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from net_autoencoder import load_autoencoder


class MacroFeaturesExtractor(BaseFeaturesExtractor):
    """Encoder-frozen + LSTM."""

    def __init__(
        self,
        observation_space: gym.Space,
        autoencoder_path: str,
        autoencoder_hidden: tuple,
        autoencoder_latent: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        head_hidden: tuple = (128, 64),
        dropout: float = 0.15,
    ):
        # observation_space: Box [lookback, F_macro]
        assert len(observation_space.shape) == 2, f"oczekiwany [L, F], jest {observation_space.shape}"
        lookback, n_features = observation_space.shape

        # features_dim = ostatni wymiar head (lub lstm_hidden jesli head pusty)
        out_dim = head_hidden[-1] if len(head_hidden) > 0 else lstm_hidden

        # KRYTYCZNE: super().__init__() PRZED przypisaniem jakichkolwiek nn.Module
        super().__init__(observation_space, features_dim=out_dim)

        # Pre-trained encoder (frozen)
        self.encoder = load_autoencoder(
            autoencoder_path, hidden=list(autoencoder_hidden),
            input_dim=n_features, latent_dim=autoencoder_latent,
            dropout=0.0,
        )

        # LSTM nad sekwencja zakodowana
        self.lstm = nn.LSTM(autoencoder_latent, lstm_hidden, num_layers=lstm_layers,
                             batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)

        # Head MLP
        head_layers = []
        prev = lstm_hidden
        for h in head_hidden:
            head_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.head = nn.Sequential(*head_layers)

        self.lookback = lookback
        self.n_features = n_features
        self.latent_dim = autoencoder_latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        L, F = self.lookback, self.n_features
        # [B, L, F] -> [B*L, F] -> encoder -> [B*L, latent]
        flat = obs.reshape(B * L, F)
        with torch.no_grad():
            z = self.encoder.encode(flat)
        z = z.reshape(B, L, self.latent_dim)
        # LSTM
        out, (h, _) = self.lstm(z)
        last = h[-1]    # [B, lstm_hidden]
        return self.head(last)
