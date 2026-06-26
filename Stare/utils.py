# ===== file: utils.py =====
"""
Wspolne narzedzia: seedy, urzadzenia, logger, sanity-checks, sciezki runow,
helpery predykcji (z obsluga RecurrentPPO LSTM state).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(requested: str = "cuda") -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_logger(name: str = "hrl", outputs_dir: str = "outputs") -> logging.Logger:
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    # Wyczysc poprzednie handlery (rozne runy = rozne pliki logow)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(Path(outputs_dir) / "training.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def safe_divide(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full_like(a, default, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[mask] = a[mask] / b[mask]
    return out


def assert_finite(x: Any, name: str = "tensor") -> None:
    arr = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: znaleziono NaN/Inf (shape={arr.shape})")


def assert_shape(x: np.ndarray, expected: tuple, name: str = "tensor") -> None:
    if tuple(x.shape) != tuple(expected):
        raise ValueError(f"{name}: oczekiwany {expected}, dostalem {tuple(x.shape)}")


def long_only_normalize(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s < eps:
        return np.ones_like(w) / len(w)
    return w / s


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# SCIEZKI RUNOW / FOLDOW
# ============================================================================
def configure_run_paths(cfg, experiment_mode: str = "single",
                        run_name: str = "default",
                        fold_name: Optional[str] = None):
    """Ustawia outputs_dir / tensorboard_dir / autoencoder_dir per run/fold.

    single:        outputs/single/{run_name}
    walk_forward:  outputs/walk_forward/{run_name}/{fold_name}
    """
    if experiment_mode == "walk_forward":
        if not fold_name:
            raise ValueError("walk_forward wymaga fold_name")
        base = Path("outputs") / "walk_forward" / run_name / fold_name
    else:
        base = Path("outputs") / "single" / run_name

    base.mkdir(parents=True, exist_ok=True)
    cfg.paths.outputs_dir = str(base)
    cfg.paths.tensorboard_dir = str(base / "tensorboard")
    cfg.paths.autoencoder_dir = str(base / "autoencoders")
    cfg.experiment.experiment_mode = experiment_mode
    cfg.experiment.run_name = run_name
    cfg.experiment.fold_name = fold_name
    return cfg


def apply_date_overrides(cfg, train_end=None, val_end=None, test_end=None):
    if train_end:
        cfg.data.train_end = train_end
    if val_end:
        cfg.data.val_end = val_end
    if test_end:
        cfg.data.test_end = test_end
    return cfg


# ============================================================================
# PREDYKCJA (z obsluga RecurrentPPO LSTM state)
# ============================================================================
def predict_action(model, obs, algo: str, deterministic: bool = True,
                   lstm_states=None, episode_starts=None):
    """Jednolity interfejs predykcji. Dla recurrent_ppo przekazuje i zwraca
    lstm_states; dla pozostalych ignoruje state.

    Returns: (action, lstm_states)
    """
    if algo == "recurrent_ppo":
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=deterministic,
        )
        return action, lstm_states
    action, _ = model.predict(obs, deterministic=deterministic)
    return action, lstm_states


# ============================================================================
# Hierarchical helpers
# ============================================================================
def generate_sector_weights_by_t(macro_agent, macro_tensor, rebalance_freq: int,
                                  n_sectors: int, sector_max_weight: float = 0.6,
                                  algo: str = "td3") -> np.ndarray:
    """Roll wytrenowanego macro_agent przez wszystkie kroki i zwroc trajektorie
    wag sektorow [T, n_sectors] z freezingiem co rebalance_freq dni.

    Obsluguje RecurrentPPO przez predict_action (utrzymuje LSTM state).
    """
    T = macro_tensor.shape[0]
    weights = np.zeros((T, n_sectors), dtype=np.float32)
    current = np.ones(n_sectors, dtype=np.float64) / n_sectors

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    for t in range(T):
        if t % rebalance_freq == 0:
            obs = macro_tensor[t].astype(np.float32)
            action, lstm_states = predict_action(
                macro_agent, obs, algo, deterministic=True,
                lstm_states=lstm_states, episode_starts=episode_starts,
            )
            action = np.asarray(action, dtype=np.float64).flatten()
            new_w = softmax_np(action)
            new_w = np.minimum(new_w, sector_max_weight)
            new_w = new_w / max(new_w.sum(), 1e-12)
            current = new_w
            episode_starts = np.zeros((1,), dtype=bool)
        weights[t] = current.astype(np.float32)

    return weights
