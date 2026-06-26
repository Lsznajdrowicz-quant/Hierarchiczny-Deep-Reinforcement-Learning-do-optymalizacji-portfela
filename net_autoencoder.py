# file: net_autoencoder.py


from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """Prosty FFNN autoencoder z bottleneckiem."""

    def __init__(self, input_dim: int, hidden: Iterable[int], latent_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        hidden = list(hidden)

        # Encoder
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*layers)

        # Decoder (mirror)
        layers = []
        prev = latent_dim
        for h in reversed(hidden):
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*layers)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_autoencoder(
    data: np.ndarray,
    input_dim: int,
    hidden: Iterable[int],
    latent_dim: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
) -> Autoencoder:
    """Trenuj autoencoder na surowych danych [N, F].

    data: numpy [N_samples, input_dim]. NaN/Inf zostana wyczyszczone.
    """
    arr = np.nan_to_num(np.asarray(data, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)
    assert arr.shape[1] == input_dim, f"Oczekiwany F={input_dim}, jest {arr.shape[1]}"

    model = Autoencoder(input_dim, hidden, latent_dim, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(arr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose and (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{epochs}  recon_loss={total_loss/max(1,n_batches):.5f}")

    model.eval()
    return model


def save_autoencoder(model: Autoencoder, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "latent_dim": model.latent_dim,
    }, path)


def load_autoencoder(path: str, hidden: Iterable[int],
                      input_dim: int, latent_dim: int,
                      dropout: float = 0.0,
                      device: str = "cpu") -> Autoencoder:
    """Wczytaj zapisany autoencoder. Hidden musi odpowiadac konfigowi treningu."""
    model = Autoencoder(input_dim, hidden, latent_dim, dropout=dropout)
    # weights_only=False: to nasz wlasny plik (PyTorch 2.6+ domyslnie True).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model