# file: rewards.py


from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from config import RewardConfig


_LOG_FLOOR = -0.9999     # ochrona przed log1p przy zwrotach blisko -1


@dataclass
class RollingStats:
    window: int
    buffer: deque

    @classmethod
    def make(cls, window: int) -> "RollingStats":
        return cls(window=window, buffer=deque(maxlen=window))

    def push(self, x: float) -> None:
        self.buffer.append(float(x))

    def values(self) -> np.ndarray:
        return np.asarray(self.buffer, dtype=np.float64)

    def reset(self) -> None:
        self.buffer.clear()


def sortino_ratio(returns: np.ndarray, annualization: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mean = float(r.mean())
    downside = r[r < 0]
    if len(downside) < 1:
        return 5.0 if mean > 0 else 0.0
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float(np.abs(downside).mean())
    if downside_std < 1e-12:
        return 5.0 if mean > 0 else 0.0
    return mean / downside_std * math.sqrt(annualization)


class SortinoReward:
    """Reward dla macro+fund: rolling Sortino + warmup."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.rolling = RollingStats.make(cfg.sortino_window)
        self.step_count = 0

    def reset(self) -> None:
        self.rolling.reset()
        self.step_count = 0

    def compute(self, portfolio_return: float) -> tuple[float, dict]:
        self.rolling.push(portfolio_return)
        self.step_count += 1
        if len(self.rolling.buffer) < max(5, self.cfg.sortino_window // 4):
            return 0.0, {"sortino": 0.0, "n": len(self.rolling.buffer)}
        s = sortino_ratio(self.rolling.values(), annualization=self.cfg.sortino_annualization)
        s_clipped = max(-self.cfg.sortino_clip, min(self.cfg.sortino_clip, s))
        reward = self.cfg.sortino_scale * s_clipped
        return float(reward), {"sortino": float(s), "sortino_clipped": float(s_clipped),
                                "n": len(self.rolling.buffer)}


class TechnicalOverlayReward:
    """Reward agenta technicznego jako overlay wzgledem baseline macro+fund.

    r = active_scale * active_log_return
        + profit_scale * net_return
        - beta_risk * sigma2(active_log_returns)
        - xi_turnover * |tilt_turnover|

    Gdzie:
      active_log_return = log1p(net_return) - log1p(baseline_return)
      sigma2 = rolling variance aktywnych log-returnow

    Glowny sygnal sily — czy techniczny overlay dodaje wartosc vs portfel
    macro+fund. Maly skladnik absolutny zapobiega "satisfaction with losing
    less than baseline" gdy oba portfele traca.
    """

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.active_log_rolling = RollingStats.make(cfg.technical_vol_window)
        self.step_count = 0

    def reset(self) -> None:
        self.active_log_rolling.reset()
        self.step_count = 0

    def compute(
        self,
        portfolio_return_net: float,
        baseline_return: float,
        tilt_turnover: float,
    ) -> tuple[float, dict]:
        # Aktywne zwroty — log i simple
        safe_net = max(float(portfolio_return_net), _LOG_FLOOR)
        safe_base = max(float(baseline_return), _LOG_FLOOR)
        active_log = math.log1p(safe_net) - math.log1p(safe_base)
        active_simple = float(portfolio_return_net) - float(baseline_return)

        # Rolling sigma2 aktywnych log-returnow
        self.active_log_rolling.push(active_log)
        self.step_count += 1
        if len(self.active_log_rolling.buffer) >= 5:
            sigma2 = float(np.var(self.active_log_rolling.values(), ddof=1))
        else:
            sigma2 = 0.0

        active_term = self.cfg.technical_active_scale * active_log
        profit_term = self.cfg.technical_profit_scale * float(portfolio_return_net)
        risk_term = self.cfg.technical_beta_risk * sigma2
        cost_term = self.cfg.technical_xi_turnover * float(tilt_turnover)

        reward = active_term + profit_term - risk_term - cost_term
        reward = max(-self.cfg.technical_max_clip,
                     min(self.cfg.technical_max_clip, reward))

        return float(reward), {
            "active_log_return": float(active_log),
            "active_simple_return": float(active_simple),
            "active_term": float(active_term),
            "profit_term": float(profit_term),
            "risk_term": float(risk_term),
            "cost_term": float(cost_term),
            "sigma2_active": float(sigma2),
        }
