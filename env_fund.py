# file: env_fund.py


from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import ProjectConfig
from state_builder import LookbackTensors
from rewards import SortinoReward


def masked_softmax_within_sectors(
    logits: np.ndarray, sector_ids: np.ndarray, sector_weights: np.ndarray,
) -> np.ndarray:
    """Per sektor: softmax logitow tylko jego spolek, skalowane do wagi sektora.
    Wynik: wagi spolek sumujace sie do 1."""
    n_stocks = len(logits)
    n_sectors = len(sector_weights)
    out = np.zeros(n_stocks, dtype=np.float64)
    for s in range(n_sectors):
        mask = sector_ids == s
        if not mask.any():
            continue
        sub = logits[mask]
        sub = sub - sub.max()
        e = np.exp(sub)
        e = e / e.sum()
        out[mask] = e * sector_weights[s]
    total = out.sum()
    if total > 1e-12:
        out = out / total
    return out


class FundEnv(gym.Env):
    """Srodowisko dla agenta fundamentalnego (poziom 2)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tensors: LookbackTensors,
        stock_returns: np.ndarray,                       # [T, N]
        sector_ids: np.ndarray,                          # [N]
        cfg: ProjectConfig,
        sector_weights_by_t: Optional[np.ndarray] = None,  # [T, n_sectors] lub None
    ):
        super().__init__()
        self.tensors = tensors
        self.stock_returns = np.asarray(stock_returns, dtype=np.float64)
        self.sector_ids = np.asarray(sector_ids, dtype=np.int64)
        self.cfg = cfg

        self.T = tensors.fundamental.shape[0]
        _, self.n_stocks, self.lookback, self.n_fund = tensors.fundamental.shape
        self.n_sectors = int(sector_ids.max()) + 1
        self.rebalance_freq = cfg.hierarchy.rebalance_freq
        self.cost_rate = (cfg.execution.transaction_cost_bps
                          + cfg.execution.spread_cost_bps) / 10_000.0

        # Mode wag sektorowych
        if sector_weights_by_t is not None:
            sw = np.asarray(sector_weights_by_t, dtype=np.float32)
            if sw.shape != (self.T, self.n_sectors):
                raise ValueError(f"sector_weights_by_t shape {sw.shape}, oczekiwany ({self.T},{self.n_sectors})")
            self.sector_weights_by_t = sw
            self.sector_mode = "macro"
        else:
            equal = np.ones(self.n_sectors, dtype=np.float32) / self.n_sectors
            self.sector_weights_by_t = np.tile(equal, (self.T, 1))
            self.sector_mode = "equal"

        self.observation_space = spaces.Dict({
            "fundamental": spaces.Box(low=-np.inf, high=np.inf,
                shape=(self.n_stocks, self.lookback, self.n_fund), dtype=np.float32),
            "sector_weights": spaces.Box(low=0.0, high=1.0,
                shape=(self.n_sectors,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-cfg.execution.action_clip, high=cfg.execution.action_clip,
            shape=(self.n_stocks,), dtype=np.float32,
        )

        self.reward_fn = SortinoReward(cfg.reward)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        # Startowe wagi: rowne wewnatrz sektorow przy aktualnych wagach sektorow
        self.current_stock_weights = self._equal_within_sectors(self.sector_weights_by_t[0])
        self.reward_fn.reset()

        self.history_returns_gross = []
        self.history_returns_net = []
        self.history_turnover = []
        self.history_costs = []
        self.history_stock_weights = []
        return self._get_obs(), {}

    def _equal_within_sectors(self, sector_weights: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_stocks, dtype=np.float64)
        for s in range(self.n_sectors):
            mask = self.sector_ids == s
            n = int(mask.sum())
            if n > 0:
                out[mask] = sector_weights[s] / n
        return out

    def _get_obs(self) -> dict:
        return {
            "fundamental": self.tensors.fundamental[self.t].astype(np.float32),
            "sector_weights": self.sector_weights_by_t[self.t].astype(np.float32),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action, -self.cfg.execution.action_clip, self.cfg.execution.action_clip)

        rebalance_today = (self.t % self.rebalance_freq == 0)
        turnover = 0.0
        cost = 0.0

        if rebalance_today:
            sw = self.sector_weights_by_t[self.t].astype(np.float64)
            new_w = masked_softmax_within_sectors(action, self.sector_ids, sw)
            turnover = float(np.abs(new_w - self.current_stock_weights).sum())
            cost = self.cost_rate * turnover
            self.current_stock_weights = new_w

        day_returns = self.stock_returns[self.t]
        gross_return = float(np.dot(self.current_stock_weights, day_returns))
        net_return = gross_return - cost

        reward, reward_info = self.reward_fn.compute(net_return)

        self.history_returns_gross.append(gross_return)
        self.history_returns_net.append(net_return)
        self.history_turnover.append(turnover)
        self.history_costs.append(cost)
        self.history_stock_weights.append(self.current_stock_weights.copy())

        self.t += 1
        terminated = self.t >= self.T - 1
        obs = self._get_obs() if not terminated else {
            "fundamental": self.tensors.fundamental[-1].astype(np.float32),
            "sector_weights": self.sector_weights_by_t[-1].astype(np.float32),
        }
        info = {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost": cost,
            "rebalance": rebalance_today,
            **reward_info,
        }
        return obs, reward, terminated, False, info
