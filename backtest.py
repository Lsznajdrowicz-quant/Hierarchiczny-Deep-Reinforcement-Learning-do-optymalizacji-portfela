

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_config, SECTOR_COLUMNS
from utils import (
    set_global_seed,
    resolve_device,
    setup_logger,
    configure_run_paths,
    apply_date_overrides,
    predict_action,
)
from data_loader import load_all
from preprocessing import preprocess
from state_builder import (
    build_lookback_tensors_with_context,
    concat_splits,
)
from env_tech import TechEnv
from algorithms import load_model
from metrics import (
    summarize_returns,
    summarize_vs_benchmark,
    equal_weight_benchmark,
    markowitz_min_variance_weights,
    markowitz_max_sharpe_weights,
    portfolio_returns_from_weights,
)


def _resolve_agent_path(base: Path, algo: str, name: str, prefer: str) -> Path:
    """Znajdz model agenta w folderze models/{agent}/{algo}/."""
    best = base / "best_reward" / "best_model.zip"
    final = base / f"{name}_{algo}.zip"

    if prefer == "best_reward":
        order = [best, final]
    elif prefer == "final":
        order = [final, best]
    else:
        raise ValueError(f"Nieznany prefer-model: {prefer}")

    for p in order:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Brak {name}_agent w {base}\n"
        f"Sprawdzane: {[str(p) for p in order]}\n"
        f"Uruchom najpierw train_{name}.py"
    )


def _sector_returns_from_stocks(
    stock_rets: np.ndarray,
    sector_ids: np.ndarray,
    n_sectors: int,
) -> np.ndarray:
    """[T, N] -> [T, n_sectors] jako equal-weight return w sektorze."""
    T = stock_rets.shape[0]
    out = np.zeros((T, n_sectors), dtype=np.float64)

    for s in range(n_sectors):
        mask = sector_ids == s
        if mask.any():
            out[:, s] = stock_rets[:, mask].mean(axis=1)

    return out


def _build_tensors_for_split(dataset, split_name: str, cfg, logger):
    """Buduje LookbackTensors dla val/test z historycznym kontekstem.

    val:
        context = train
        target = val

    test:
        context = train + val
        target = test
    """
    if split_name == "val":
        context_split = dataset.train
        target_split = dataset.val
        markowitz_fit_returns = dataset.train.stock_returns
        logger.info("[context] val: lookback context = train")
        logger.info(f"[markowitz] fit na train: T={markowitz_fit_returns.shape[0]}")

    elif split_name == "test":
        context_split = concat_splits(dataset.train, dataset.val)
        target_split = dataset.test
        markowitz_fit_returns = context_split.stock_returns
        logger.info("[context] test: lookback context = train+val")
        logger.info(f"[markowitz] fit na train+val: T={markowitz_fit_returns.shape[0]}")

    else:
        raise ValueError(f"Nieznany split: {split_name}")

    tensors = build_lookback_tensors_with_context(
        context_split=context_split,
        target_split=target_split,
        macro_lookback=cfg.data.macro_lookback,
        fund_lookback=cfg.data.fund_lookback,
        tech_lookback=cfg.data.tech_lookback,
    )

    return tensors, markowitz_fit_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["td3", "sac", "tqc", "recurrent_ppo"])
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--experiment-mode", type=str, default="single",
                        choices=["single", "walk_forward"])
    parser.add_argument("--fold-name", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--prefer-model", type=str, default="best_reward",
                        choices=["best_reward", "final"])
    args = parser.parse_args()

    cfg = get_config(algo=args.algo)
    cfg = apply_date_overrides(cfg, args.train_end, args.val_end, args.test_end)
    cfg = configure_run_paths(cfg, args.experiment_mode, args.run_name, args.fold_name)

    set_global_seed(cfg.train.seed)
    cfg.train.device = resolve_device(cfg.train.device)

    logger = setup_logger("backtest", cfg.paths.outputs_dir)
    logger.info(
        f"=== BACKTEST  algo={cfg.train.algo}  run={args.run_name} "
        f"fold={args.fold_name}  split={args.split}  prefer={args.prefer_model} ==="
    )
    logger.info(
        f"[lookback] macro={cfg.data.macro_lookback}  "
        f"fund={cfg.data.fund_lookback}  tech={cfg.data.tech_lookback}"
    )

    raw = load_all(cfg)
    dataset = preprocess(raw, cfg)

    tensors, markowitz_fit_returns = _build_tensors_for_split(
        dataset=dataset,
        split_name=args.split,
        cfg=cfg,
        logger=logger,
    )

    logger.info(
        f"split T={tensors.technical.shape[0]}  "
        f"dates {tensors.dates[0].date()} -> {tensors.dates[-1].date()}"
    )

    sector_ids = np.array([raw["sector_map"][t] for t in raw["tickers"]], dtype=np.int64)
    n_sectors = len(SECTOR_COLUMNS)

    sec_rets = _sector_returns_from_stocks(
        stock_rets=tensors.returns,
        sector_ids=sector_ids,
        n_sectors=n_sectors,
    )

    base = Path(cfg.paths.outputs_dir) / "models"

    macro_path = _resolve_agent_path(
        base / "macro" / cfg.train.algo,
        cfg.train.algo,
        "macro",
        args.prefer_model,
    )
    fund_path = _resolve_agent_path(
        base / "fund" / cfg.train.algo,
        cfg.train.algo,
        "fund",
        args.prefer_model,
    )
    tech_path = _resolve_agent_path(
        base / "tech" / cfg.train.algo,
        cfg.train.algo,
        "tech",
        args.prefer_model,
    )

    logger.info(f"[load] macro: {macro_path}")
    logger.info(f"[load] fund:  {fund_path}")
    logger.info(f"[load] tech:  {tech_path}")

    macro_agent = load_model(str(macro_path), cfg.train.algo, device=cfg.train.device)
    fund_agent = load_model(str(fund_path), cfg.train.algo, device=cfg.train.device)
    tech_agent = load_model(str(tech_path), cfg.train.algo, device=cfg.train.device)

    # Rollout — TechEnv wola macro/fund wewnetrznie; tutaj sterujemy tech_agent.
    env = TechEnv(
        tensors,
        tensors.returns,
        sec_rets,
        sector_ids,
        macro_agent,
        fund_agent,
        cfg,
    )

    obs, _ = env.reset()
    done = False
    step = 0

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while not done:
        action, lstm_states = predict_action(
            tech_agent,
            obs,
            cfg.train.algo,
            deterministic=True,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_starts = np.array([done], dtype=bool)
        step += 1

    logger.info(f"[run] {step} krokow ukonczone")

    # ------------------------------------------------------------------
    # Historie z TechEnv
    # ------------------------------------------------------------------
    weights = np.array(env.history_weights)
    tilts = np.array(env.history_tilts)
    targets = np.array(env.history_targets)

    gross = np.array(env.history_returns_gross)
    net = np.array(env.history_returns_net)
    baseline = np.array(env.history_baseline_returns)
    active_simple = np.array(env.history_active_simple_returns)
    active_log = np.array(env.history_active_log_returns)
    turnover = np.array(env.history_turnover)
    costs = np.array(env.history_costs)

    n_steps = len(net)
    if n_steps == 0:
        raise RuntimeError("Backtest zakonczyl sie bez zadnego kroku")

    dates = pd.DatetimeIndex(env.tensors.dates[:n_steps])

    # ------------------------------------------------------------------
    # Agregacje wag sektorowych i fund within-sector
    # ------------------------------------------------------------------
    sector_alloc = np.zeros((len(weights), n_sectors), dtype=np.float64)

    for s in range(n_sectors):
        mask = sector_ids == s
        sector_alloc[:, s] = weights[:, mask].sum(axis=1)

    fund_within = np.zeros_like(targets)

    for s in range(n_sectors):
        mask = sector_ids == s
        sec_target = targets[:, mask].sum(axis=1, keepdims=True)
        sec_target = np.where(sec_target > 1e-12, sec_target, 1.0)
        fund_within[:, mask] = targets[:, mask] / sec_target

    # ------------------------------------------------------------------
    # Benchmarki — Markowitz bez data leakage
    # ------------------------------------------------------------------
    w_minvar = markowitz_min_variance_weights(markowitz_fit_returns, ridge=1e-4)
    w_maxsh = markowitz_max_sharpe_weights(markowitz_fit_returns, ridge=1e-4)

    bench_ew = equal_weight_benchmark(tensors.returns[:n_steps])
    bench_minvar = portfolio_returns_from_weights(tensors.returns[:n_steps], w_minvar)
    bench_maxsh = portfolio_returns_from_weights(tensors.returns[:n_steps], w_maxsh)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    out_dir = Path(cfg.paths.outputs_dir) / "backtest" / cfg.train.algo / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(weights, index=dates, columns=raw["tickers"]).to_csv(
        out_dir / "daily_weights.csv"
    )

    pd.DataFrame(targets, index=dates, columns=raw["tickers"]).to_csv(
        out_dir / "fund_target_weights.csv"
    )

    pd.DataFrame(fund_within, index=dates, columns=raw["tickers"]).to_csv(
        out_dir / "fund_within_sector.csv"
    )

    pd.DataFrame(tilts, index=dates, columns=raw["tickers"]).to_csv(
        out_dir / "tech_multipliers.csv"
    )

    pd.DataFrame(sector_alloc, index=dates, columns=SECTOR_COLUMNS).to_csv(
        out_dir / "sector_weights.csv"
    )

    pd.DataFrame({
        "date": dates,
        "gross_return": gross,
        "net_return": net,
        "baseline_return": baseline,
        "active_simple_return": active_simple,
        "active_log_return": active_log,
        "turnover": turnover,
        "cost": costs,
        "equity_gross": np.cumprod(1.0 + gross),
        "equity_net": np.cumprod(1.0 + net),
        "equity_baseline": np.cumprod(1.0 + baseline),
        "active_cumulative_simple": np.cumsum(active_simple),
        "active_cumulative_log": np.cumsum(active_log),
    }).to_csv(out_dir / "returns.csv", index=False)

    pd.DataFrame({
        "date": dates,
        "equal_weight_return": bench_ew,
        "equal_weight_equity": np.cumprod(1.0 + bench_ew),
        "markowitz_minvar_return": bench_minvar,
        "markowitz_minvar_equity": np.cumprod(1.0 + bench_minvar),
        "markowitz_maxsharpe_return": bench_maxsh,
        "markowitz_maxsharpe_equity": np.cumprod(1.0 + bench_maxsh),
    }).to_csv(out_dir / "benchmark_markowitz.csv", index=False)

    pd.DataFrame({
        "ticker": raw["tickers"],
        "minvar_weight": w_minvar,
        "maxsharpe_weight": w_maxsh,
    }).to_csv(out_dir / "markowitz_weights.csv", index=False)

    metrics = {
        "split": args.split,
        "algo": cfg.train.algo,
        "run_name": args.run_name,
        "fold_name": args.fold_name,
        "n_steps": int(n_steps),
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "lookbacks": {
            "macro_lookback": int(cfg.data.macro_lookback),
            "fund_lookback": int(cfg.data.fund_lookback),
            "tech_lookback": int(cfg.data.tech_lookback),
        },
        "strategy_net": summarize_returns(net, costs=costs, turnover=turnover),
        "strategy_gross": summarize_returns(gross),
        "strategy_baseline": summarize_returns(baseline),
        "benchmarks": {
            "equal_weight": summarize_returns(bench_ew),
            "markowitz_min_variance": summarize_returns(bench_minvar),
            "markowitz_max_sharpe": summarize_returns(bench_maxsh),
        },
        "strategy_vs_benchmarks": {
            "vs_equal_weight": summarize_vs_benchmark(net, bench_ew),
            "vs_markowitz_min_variance": summarize_vs_benchmark(net, bench_minvar),
            "vs_markowitz_max_sharpe": summarize_vs_benchmark(net, bench_maxsh),
            "vs_baseline_macro_fund": summarize_vs_benchmark(net, baseline),
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Log summary
    # ------------------------------------------------------------------
    s = metrics["strategy_net"]

    logger.info("=== STRATEGY NET ===")
    for k in [
        "cumulative_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "avg_turnover",
    ]:
        if k in s:
            logger.info(f"  {k:24s} {s[k]:>10.4f}")

    logger.info("=== VS BENCHMARKS ===")
    for name, comp in metrics["strategy_vs_benchmarks"].items():
        logger.info(
            f"  {name}: alpha={comp['alpha_total']:+.4f} "
            f"IR={comp['information_ratio']:+.3f} "
            f"win={comp['win_rate_vs_benchmark']:.3f}"
        )

    logger.info(f"\nZapisano do: {out_dir}")


if __name__ == "__main__":
    main()
