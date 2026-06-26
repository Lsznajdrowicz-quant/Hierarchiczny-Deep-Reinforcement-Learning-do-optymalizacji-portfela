

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import ProjectConfig
from state_builder import LookbackTensors
from rewards import SortinoReward
from utils import softmax_np


class MacroEnv(gym.Env):
    """Srodowisko dla agenta makroekonomicznego (poziom 1) z kosztami."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tensors: LookbackTensors,
        sector_returns: np.ndarray,     # [T, n_sectors]
        sector_ids: np.ndarray,         # [N] per spolka
        cfg: ProjectConfig,
    ):
        super().__init__()
        self.tensors = tensors
        self.sector_returns = np.asarray(sector_returns, dtype=np.float64)
        self.sector_ids = np.asarray(sector_ids, dtype=np.int64)
        self.cfg = cfg

        self.T = tensors.macro.shape[0]
        self.lookback, self.n_macro = tensors.macro.shape[1], tensors.macro.shape[2]
        self.n_sectors = sector_returns.shape[1]
        self.rebalance_freq = cfg.hierarchy.rebalance_freq
        self.cost_rate = (cfg.execution.transaction_cost_bps
                          + cfg.execution.spread_cost_bps) / 10_000.0

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_macro), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-cfg.execution.action_clip, high=cfg.execution.action_clip,
            shape=(self.n_sectors,), dtype=np.float32,
        )

        self.reward_fn = SortinoReward(cfg.reward)

        if self.T < self.rebalance_freq * 2:
            raise ValueError(f"Za malo krokow w env: T={self.T}")

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.current_sector_weights = np.ones(self.n_sectors) / self.n_sectors
        self.reward_fn.reset()

        self.history_returns_gross = []
        self.history_returns_net = []
        self.history_turnover = []
        self.history_costs = []
        self.history_sector_weights = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return self.tensors.macro[self.t].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action, -self.cfg.execution.action_clip, self.cfg.execution.action_clip)

        rebalance_today = (self.t % self.rebalance_freq == 0)
        turnover = 0.0
        cost = 0.0

        if rebalance_today:
            new_w = softmax_np(action)
            new_w = np.minimum(new_w, self.cfg.hierarchy.sector_max_weight)
            new_w = new_w / new_w.sum()
            turnover = float(np.abs(new_w - self.current_sector_weights).sum())
            cost = self.cost_rate * turnover
            self.current_sector_weights = new_w

        day_sector_returns = self.sector_returns[self.t]
        gross_return = float(np.dot(self.current_sector_weights, day_sector_returns))
        net_return = gross_return - cost

        reward, reward_info = self.reward_fn.compute(net_return)

        self.history_returns_gross.append(gross_return)
        self.history_returns_net.append(net_return)
        self.history_turnover.append(turnover)
        self.history_costs.append(cost)
        self.history_sector_weights.append(self.current_sector_weights.copy())

        self.t += 1
        terminated = self.t >= self.T - 1
        obs = self._get_obs() if not terminated else self.tensors.macro[-1].astype(np.float32)
        info = {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost": cost,
            "rebalance": rebalance_today,
            **reward_info,
        }
        return obs, reward, terminated, False, info
