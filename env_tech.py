
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import ProjectConfig
from state_builder import LookbackTensors
from rewards import TechnicalOverlayReward
from env_fund import masked_softmax_within_sectors
from utils import softmax_np, long_only_normalize


_LOG_FLOOR = -0.9999


class TechEnv(gym.Env):
    """Srodowisko dla agenta technicznego (poziom 3) — overlay na macro+fund."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        tensors: LookbackTensors,
        stock_returns: np.ndarray,
        sector_returns: np.ndarray,
        sector_ids: np.ndarray,
        macro_agent,
        fund_agent,
        cfg: ProjectConfig,
    ):
        super().__init__()
        self.tensors = tensors
        self.stock_returns = np.asarray(stock_returns, dtype=np.float64)
        self.sector_returns = np.asarray(sector_returns, dtype=np.float64)
        self.sector_ids = np.asarray(sector_ids, dtype=np.int64)
        self.macro_agent = macro_agent
        self.fund_agent = fund_agent
        self.cfg = cfg

        self.T = tensors.technical.shape[0]
        _, self.n_stocks, self.lookback, self.n_tech = tensors.technical.shape
        self.n_macro = tensors.macro.shape[2]
        self.n_fund = tensors.fundamental.shape[3]
        self.n_sectors = int(sector_ids.max()) + 1
        self.rebalance_freq = cfg.hierarchy.rebalance_freq

        self.observation_space = spaces.Dict({
            "technical": spaces.Box(low=-np.inf, high=np.inf,
                shape=(self.n_stocks, self.lookback, self.n_tech), dtype=np.float32),
            "target_weights": spaces.Box(low=0.0, high=1.0,
                shape=(self.n_stocks,), dtype=np.float32),
            "current_weights": spaces.Box(low=0.0, high=1.0,
                shape=(self.n_stocks,), dtype=np.float32),
            "previous_tilts": spaces.Box(low=cfg.hierarchy.tilt_low, high=cfg.hierarchy.tilt_high,
                shape=(self.n_stocks,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-cfg.execution.action_clip, high=cfg.execution.action_clip,
            shape=(self.n_stocks,), dtype=np.float32,
        )

        self.reward_fn = TechnicalOverlayReward(cfg.reward)
        self.cost_rate = (cfg.execution.transaction_cost_bps + cfg.execution.spread_cost_bps) / 10_000.0

        self.reset()

    def _query_macro_fund(self) -> np.ndarray:
        """Wywoluje zamrozone agenty macro+fund. Zwraca docelowe wagi spolek [N]."""
        macro_obs = self.tensors.macro[self.t].astype(np.float32)
        sector_action, _ = self.macro_agent.predict(macro_obs, deterministic=True)
        sector_w = softmax_np(np.asarray(sector_action, dtype=np.float64))
        sector_w = np.minimum(sector_w, self.cfg.hierarchy.sector_max_weight)
        sector_w = sector_w / sector_w.sum()

        fund_obs = {
            "fundamental": self.tensors.fundamental[self.t].astype(np.float32),
            "sector_weights": sector_w.astype(np.float32),
        }
        stock_action, _ = self.fund_agent.predict(fund_obs, deterministic=True)
        stock_w = masked_softmax_within_sectors(
            np.asarray(stock_action, dtype=np.float64),
            self.sector_ids, sector_w,
        )
        return stock_w

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.target_weights = self._query_macro_fund()
        self.current_weights = self.target_weights.copy()
        self.previous_tilts = np.ones(self.n_stocks, dtype=np.float64)
        self.reward_fn.reset()

        self.history_returns_gross = []
        self.history_returns_net = []
        self.history_baseline_returns = []
        self.history_active_simple_returns = []
        self.history_active_log_returns = []
        self.history_turnover = []
        self.history_costs = []
        self.history_weights = []
        self.history_tilts = []
        self.history_targets = []
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        return {
            "technical": self.tensors.technical[self.t].astype(np.float32),
            "target_weights": self.target_weights.astype(np.float32),
            "current_weights": self.current_weights.astype(np.float32),
            "previous_tilts": self.previous_tilts.astype(np.float32),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        action = np.clip(action, -self.cfg.execution.action_clip, self.cfg.execution.action_clip)

        if self.t > 0 and self.t % self.rebalance_freq == 0:
            self.target_weights = self._query_macro_fund()

        # Tilty: tanh -> srodek=1, polowa=(tilt_high-tilt_low)/2
        tilt_half = (self.cfg.hierarchy.tilt_high - self.cfg.hierarchy.tilt_low) / 2.0
        tilts = 1.0 + tilt_half * np.tanh(action)
        tilts = np.clip(tilts, self.cfg.hierarchy.tilt_low, self.cfg.hierarchy.tilt_high)

        # Aplikuj tilty -> raw -> normalizuj
        raw = self.target_weights * tilts
        new_weights = long_only_normalize(raw)

        turnover = float(np.abs(new_weights - self.current_weights).sum())
        cost = self.cost_rate * turnover
        tilt_turnover = float(np.abs(tilts - self.previous_tilts).mean())

        day_returns = self.stock_returns[self.t]
        baseline_return = float(np.dot(self.target_weights, day_returns))
        gross_return = float(np.dot(new_weights, day_returns))
        net_return = gross_return - cost

        # Aktywne zwroty
        safe_net = max(net_return, _LOG_FLOOR)
        safe_base = max(baseline_return, _LOG_FLOOR)
        active_log = math.log1p(safe_net) - math.log1p(safe_base)
        active_simple = net_return - baseline_return

        reward, reward_info = self.reward_fn.compute(net_return, baseline_return, tilt_turnover)

        self.history_returns_gross.append(gross_return)
        self.history_returns_net.append(net_return)
        self.history_baseline_returns.append(baseline_return)
        self.history_active_simple_returns.append(active_simple)
        self.history_active_log_returns.append(active_log)
        self.history_turnover.append(turnover)
        self.history_costs.append(cost)
        self.history_weights.append(new_weights.copy())
        self.history_tilts.append(tilts.copy())
        self.history_targets.append(self.target_weights.copy())

        self.current_weights = new_weights
        self.previous_tilts = tilts.copy()

        self.t += 1
        terminated = self.t >= self.T - 1
        obs = self._get_obs() if not terminated else {
            "technical": self.tensors.technical[-1].astype(np.float32),
            "target_weights": self.target_weights.astype(np.float32),
            "current_weights": self.current_weights.astype(np.float32),
            "previous_tilts": self.previous_tilts.astype(np.float32),
        }
        info = {
            "gross_return": gross_return,
            "net_return": net_return,
            "baseline_return": baseline_return,
            "active_simple_return": active_simple,
            "active_log_return": active_log,
            "turnover": turnover,
            "cost": cost,
            "tilt_turnover": tilt_turnover,
            **reward_info,
        }
        return obs, reward, terminated, False, info
