# file: metrics.py


from __future__ import annotations

import math

import numpy as np


# ============================================================================
# PODSTAWOWE METRYKI
# ============================================================================
def cumulative_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    return float(np.prod(1.0 + r) - 1.0)


def annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return 0.0
    eq = float(np.prod(1.0 + r))
    if eq <= 0:
        return -1.0
    return eq ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float((r.mean() - rf / periods_per_year) / r.std(ddof=1) * math.sqrt(periods_per_year))


def sortino_full(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) < 2:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    ann = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(ann / abs(mdd))


def avg_turnover(turnover_series: np.ndarray) -> float:
    t = np.asarray(turnover_series, dtype=np.float64)
    return float(t.mean()) if len(t) > 0 else 0.0


# ============================================================================
# METRYKI VS BENCHMARK
# ============================================================================
def tracking_error(returns: np.ndarray, benchmark_returns: np.ndarray,
                    periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)[:len(r)]
    active = r - b
    if len(active) < 2:
        return 0.0
    return float(active.std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                       periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)[:len(r)]
    active = r - b
    if len(active) < 2 or active.std(ddof=1) < 1e-12:
        return 0.0
    return float(active.mean() / active.std(ddof=1) * math.sqrt(periods_per_year))


def active_return_total(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Kumulatywny aktywny zwrot = cumret(strategy) - cumret(benchmark)."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)[:len(r)]
    return cumulative_return(r) - cumulative_return(b)


def annualized_active_return(returns: np.ndarray, benchmark_returns: np.ndarray,
                              periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)[:len(r)]
    return annualized_return(r, periods_per_year) - annualized_return(b, periods_per_year)


def summarize_returns(returns: np.ndarray, costs: np.ndarray = None,
                       turnover: np.ndarray = None) -> dict:
    """Tylko metryki strategii (bez porownan)."""
    r = np.asarray(returns, dtype=np.float64)
    out = {
        "n_steps": int(len(r)),
        "cumulative_return": cumulative_return(r),
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_full(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r),
        "avg_daily_return": float(r.mean()) if len(r) > 0 else 0.0,
    }
    if turnover is not None:
        out["avg_turnover"] = avg_turnover(turnover)
    if costs is not None:
        out["avg_cost"] = float(np.asarray(costs).mean()) if len(costs) > 0 else 0.0
        out["total_cost"] = float(np.asarray(costs).sum())
    return out


def summarize_vs_benchmark(returns: np.ndarray, benchmark_returns: np.ndarray) -> dict:
    """Metryki porownawcze strategy vs benchmark."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)[:len(r)]
    out = {
        "strategy_cumulative_return": cumulative_return(r),
        "benchmark_cumulative_return": cumulative_return(b),
        "alpha_total": active_return_total(r, b),
        "annualized_alpha": annualized_active_return(r, b),
        "tracking_error": tracking_error(r, b),
        "information_ratio": information_ratio(r, b),
        "win_rate_vs_benchmark": float((r > b).mean()) if len(r) > 0 else 0.0,
    }
    if r.std() > 1e-12 and b.std() > 1e-12:
        out["beta_vs_benchmark"] = float(np.cov(r, b)[0, 1] / np.var(b))
        out["correlation_vs_benchmark"] = float(np.corrcoef(r, b)[0, 1])
    else:
        out["beta_vs_benchmark"] = 0.0
        out["correlation_vs_benchmark"] = 0.0
    return out


def equal_weight_benchmark(stock_returns: np.ndarray) -> np.ndarray:
    """Equal-weight benchmark z macierzy zwrotow [T, N]."""
    return np.nanmean(np.asarray(stock_returns, dtype=np.float64), axis=1)


# ============================================================================
# MARKOWITZ — STATIC WEIGHTS, NO LEAKAGE
# ============================================================================
def _safe_long_only_normalize(w: np.ndarray, n_assets: int) -> np.ndarray:
    """Long-only + sum=1 + fallback do equal-weight przy zlych wagach."""
    w = np.asarray(w, dtype=np.float64).flatten()
    if not np.all(np.isfinite(w)) or w.size != n_assets:
        return np.ones(n_assets) / n_assets
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s < 1e-12:
        return np.ones(n_assets) / n_assets
    return w / s


def markowitz_min_variance_weights(train_returns: np.ndarray,
                                    ridge: float = 1e-4) -> np.ndarray:
    """Long-only min-variance approximation: w = inv(cov + ridge*I) @ 1 -> max(0,.) / sum.

    Bez data leakage: 'train_returns' powinno zawierac TYLKO dane wczesniejsze
    niz okno backtestu.
    """
    R = np.asarray(train_returns, dtype=np.float64)
    R = R[np.all(np.isfinite(R), axis=1)]
    n_assets = R.shape[1]
    if R.shape[0] < n_assets + 5:
        return np.ones(n_assets) / n_assets
    cov = np.cov(R, rowvar=False, ddof=1)
    cov_reg = cov + ridge * np.eye(n_assets)
    try:
        inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return np.ones(n_assets) / n_assets
    raw = inv @ np.ones(n_assets)
    return _safe_long_only_normalize(raw, n_assets)


def markowitz_max_sharpe_weights(train_returns: np.ndarray,
                                  ridge: float = 1e-4) -> np.ndarray:
    """Long-only max-Sharpe approximation: w = inv(cov + ridge*I) @ mu -> max(0,.) / sum."""
    R = np.asarray(train_returns, dtype=np.float64)
    R = R[np.all(np.isfinite(R), axis=1)]
    n_assets = R.shape[1]
    if R.shape[0] < n_assets + 5:
        return np.ones(n_assets) / n_assets
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False, ddof=1)
    cov_reg = cov + ridge * np.eye(n_assets)
    try:
        inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return np.ones(n_assets) / n_assets
    raw = inv @ mu
    return _safe_long_only_normalize(raw, n_assets)


def portfolio_returns_from_weights(stock_returns: np.ndarray,
                                    weights: np.ndarray) -> np.ndarray:
    """Statyczne wagi: portfolio_return[t] = dot(weights, stock_returns[t]).

    Traktuje portfel jakby byl rebalansowany do target co dzien (bez kosztow).
    Standardowy benchmark dla porownan.
    """
    R = np.asarray(stock_returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ret = R @ w
    return np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# HELPER DLA WALK-FORWARD SUMMARY
# ============================================================================
def flatten_metrics_for_summary(metrics: dict) -> dict:
    """Wyciaga plaska strukture z metrics.json do jednego wiersza CSV summary."""
    s = metrics.get("strategy_net", {})
    base = metrics.get("strategy_baseline", {})
    bench = metrics.get("benchmarks", {})
    vs = metrics.get("strategy_vs_benchmarks", {})

    def g(d, k, default=0.0):
        return d.get(k, default) if isinstance(d, dict) else default

    return {
        "n_steps": metrics.get("n_steps", 0),
        "strategy_cumulative_return": g(s, "cumulative_return"),
        "strategy_annualized_return": g(s, "annualized_return"),
        "strategy_volatility": g(s, "annualized_volatility"),
        "strategy_sharpe": g(s, "sharpe"),
        "strategy_sortino": g(s, "sortino"),
        "strategy_max_drawdown": g(s, "max_drawdown"),
        "strategy_calmar": g(s, "calmar"),
        "avg_turnover": g(s, "avg_turnover"),
        "total_cost": g(s, "total_cost"),
        "baseline_cumulative_return": g(base, "cumulative_return"),
        "equal_weight_cumulative_return": g(bench.get("equal_weight", {}), "cumulative_return"),
        "markowitz_minvar_cumulative_return": g(bench.get("markowitz_min_variance", {}), "cumulative_return"),
        "markowitz_maxsharpe_cumulative_return": g(bench.get("markowitz_max_sharpe", {}), "cumulative_return"),
        "alpha_vs_equal_weight": g(vs.get("vs_equal_weight", {}), "alpha_total"),
        "alpha_vs_baseline_macro_fund": g(vs.get("vs_baseline_macro_fund", {}), "alpha_total"),
        "alpha_vs_markowitz_max_sharpe": g(vs.get("vs_markowitz_max_sharpe", {}), "alpha_total"),
        "ir_vs_equal_weight": g(vs.get("vs_equal_weight", {}), "information_ratio"),
        "ir_vs_baseline_macro_fund": g(vs.get("vs_baseline_macro_fund", {}), "information_ratio"),
        "ir_vs_markowitz_max_sharpe": g(vs.get("vs_markowitz_max_sharpe", {}), "information_ratio"),
        "win_rate_vs_baseline": g(vs.get("vs_baseline_macro_fund", {}), "win_rate_vs_benchmark"),
    }
