# ===== file: train_macro.py =====
"""
Krok 1 kaskady: pre-train autoencodera makro + trening macro_agent.

Modele: {outputs_dir}/models/macro/{algo}/  (outputs_dir zalezy od run/fold)

Uruchomienie:
  python train_macro.py --algo td3 --run-name td3_default --timesteps 75000
"""

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
from net_autoencoder import train_autoencoder, save_autoencoder
from env_macro import MacroEnv
from algorithms import build_model


def add_common_cli(parser):
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


def main():
    parser = argparse.ArgumentParser()
    add_common_cli(parser)
    args = parser.parse_args()

    cfg = get_config(algo=args.algo)
    cfg = apply_date_overrides(cfg, args.train_end, args.val_end, args.test_end)
    cfg = configure_run_paths(cfg, args.experiment_mode, args.run_name, args.fold_name)
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps

    set_global_seed(cfg.train.seed)
    cfg.train.device = resolve_device(cfg.train.device)
    logger = setup_logger("train_macro", cfg.paths.outputs_dir)
    logger.info(f"=== TRAIN MACRO  algo={cfg.train.algo}  run={args.run_name} "
                f"fold={args.fold_name}  device={cfg.train.device} ===")

    raw = load_all(cfg)
    dataset = preprocess(raw, cfg)
    train_tensors = build_lookback_tensors(dataset.train, cfg.data.lookback)
    val_tensors = build_lookback_tensors(dataset.val, cfg.data.lookback)
    logger.info(f"train T={train_tensors.macro.shape[0]}  val T={val_tensors.macro.shape[0]}")

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

    # Autoencoder
    ae_path = Path(cfg.paths.autoencoder_dir) / "macro_ae.pt"
    if ae_path.exists() and not args.force_retrain_ae:
        logger.info(f"[ae] autoencoder makro juz istnieje: {ae_path}")
    else:
        logger.info("[ae] trenuje autoencoder makro...")
        flat_train = dataset.train.macro.astype(np.float32)
        ae = train_autoencoder(
            flat_train, input_dim=flat_train.shape[1],
            hidden=list(cfg.networks.macro_autoencoder_hidden),
            latent_dim=cfg.networks.macro_autoencoder_latent,
            epochs=cfg.networks.autoencoder_epochs,
            batch_size=cfg.networks.autoencoder_batch_size,
            lr=cfg.networks.autoencoder_lr,
            dropout=cfg.networks.autoencoder_dropout,
            device=cfg.train.device,
        )
        save_autoencoder(ae, str(ae_path))
        logger.info(f"[ae] zapisany: {ae_path}")

    def make_train_env():
        return MacroEnv(train_tensors, train_sec_rets, sector_ids, cfg)

    def make_val_env():
        return MacroEnv(val_tensors, val_sec_rets, sector_ids, cfg)

    train_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_train_env())]))
    val_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_val_env())]))

    autoencoder_paths = {"macro": str(ae_path), "fund": ""}
    model = build_model(train_env, cfg, agent_type="macro",
                         autoencoder_paths=autoencoder_paths,
                         tensorboard_log=cfg.paths.tensorboard_dir)

    save_dir = Path(cfg.paths.outputs_dir) / "models" / "macro" / cfg.train.algo
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

    final_path = save_dir / f"macro_{cfg.train.algo}.zip"
    model.save(str(final_path))
    logger.info(f"[done] macro_agent zapisany: {final_path}")


if __name__ == "__main__":
    main()
