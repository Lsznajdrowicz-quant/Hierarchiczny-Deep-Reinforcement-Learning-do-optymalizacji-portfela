# ===== file: train_tech.py =====


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import get_config, SECTOR_COLUMNS
from utils import (set_global_seed, resolve_device, setup_logger,
                   configure_run_paths, apply_date_overrides)
from data_loader import load_all
from preprocessing import preprocess
from state_builder import build_lookback_tensors
from env_tech import TechEnv
from algorithms import build_model, load_model


def _resolve_agent_path(base: Path, algo: str, name: str) -> Path:
    best = base / "best_reward" / "best_model.zip"
    if best.exists():
        return best
    final = base / f"{name}_{algo}.zip"
    if final.exists():
        return final
    raise FileNotFoundError(
        f"Brak {name}_agent w {base}\n"
        f"Oczekiwane: {best} ALBO {final}\n"
        f"Uruchom najpierw: python train_{name}.py --algo {algo}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["td3", "sac", "tqc", "recurrent_ppo"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--experiment-mode", type=str, default="single",
                        choices=["single", "walk_forward"])
    parser.add_argument("--fold-name", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--force-retrain-ae", action="store_true")
    parser.add_argument("--macro-algo", type=str, default=None)
    parser.add_argument("--fund-algo", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(algo=args.algo)
    cfg = apply_date_overrides(cfg, args.train_end, args.val_end, args.test_end)
    cfg = configure_run_paths(cfg, args.experiment_mode, args.run_name, args.fold_name)
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps

    set_global_seed(cfg.train.seed)
    cfg.train.device = resolve_device(cfg.train.device)
    logger = setup_logger("train_tech", cfg.paths.outputs_dir)
    logger.info(f"=== TRAIN TECH  algo={cfg.train.algo}  run={args.run_name} "
                f"fold={args.fold_name}  device={cfg.train.device} ===")

    raw = load_all(cfg)
    dataset = preprocess(raw, cfg)
    train_tensors = build_lookback_tensors(
    dataset.train,
    macro_lookback=cfg.data.macro_lookback,
    fund_lookback=cfg.data.fund_lookback,
    tech_lookback=cfg.data.tech_lookback,
    )

    val_tensors = build_lookback_tensors(
        dataset.val,
        macro_lookback=cfg.data.macro_lookback,
        fund_lookback=cfg.data.fund_lookback,
        tech_lookback=cfg.data.tech_lookback,
    )
    logger.info(f"train T={train_tensors.technical.shape[0]}  val T={val_tensors.technical.shape[0]}")

    sector_ids = np.array([raw["sector_map"][t] for t in raw["tickers"]], dtype=np.int64)
    n_sectors = len(SECTOR_COLUMNS)

    def sector_returns_from_stocks(stock_rets: np.ndarray) -> np.ndarray:
        T = stock_rets.shape[0]
        out = np.zeros((T, n_sectors), dtype=np.float64)
        for s in range(n_sectors):
            mask = sector_ids == s
            if mask.any():
                out[:, s] = stock_rets[:, mask].mean(axis=1)
        return out

    train_sec_rets = sector_returns_from_stocks(train_tensors.returns)
    val_sec_rets = sector_returns_from_stocks(val_tensors.returns)

    macro_algo = args.macro_algo or cfg.train.algo
    fund_algo = args.fund_algo or cfg.train.algo
    macro_dir = Path(cfg.paths.outputs_dir) / "models" / "macro" / macro_algo
    fund_dir = Path(cfg.paths.outputs_dir) / "models" / "fund" / fund_algo

    macro_path = _resolve_agent_path(macro_dir, macro_algo, "macro")
    fund_path = _resolve_agent_path(fund_dir, fund_algo, "fund")
    logger.info(f"[load] macro: {macro_path}")
    logger.info(f"[load] fund:  {fund_path}")
    macro_agent = load_model(str(macro_path), macro_algo, device=cfg.train.device)
    fund_agent = load_model(str(fund_path), fund_algo, device=cfg.train.device)

    def make_train_env():
        return TechEnv(train_tensors, train_tensors.returns, train_sec_rets,
                        sector_ids, macro_agent, fund_agent, cfg)

    def make_val_env():
        return TechEnv(val_tensors, val_tensors.returns, val_sec_rets,
                        sector_ids, macro_agent, fund_agent, cfg)

    train_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_train_env())]))
    val_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_val_env())]))

    autoencoder_paths = {
        "macro": str(Path(cfg.paths.autoencoder_dir) / "macro_ae.pt"),
        "fund": str(Path(cfg.paths.autoencoder_dir) / "fund_ae.pt"),
    }
    model = build_model(train_env, cfg, agent_type="tech",
                         autoencoder_paths=autoencoder_paths,
                         tensorboard_log=cfg.paths.tensorboard_dir)

    save_dir = Path(cfg.paths.outputs_dir) / "models" / "tech" / cfg.train.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        val_env, best_model_save_path=str(save_dir / "best_reward"),
        log_path=str(save_dir / "eval_reward"),
        eval_freq=cfg.eval.eval_freq,
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True, verbose=1,
    )

    logger.info(f"[train] start {cfg.train.total_timesteps} krokow")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=eval_cb, progress_bar=True)

    final_path = save_dir / f"tech_{cfg.train.algo}.zip"
    model.save(str(final_path))
    logger.info(f"[done] tech_agent zapisany: {final_path}")


if __name__ == "__main__":
    main()
