# ===== file: train_fund.py =====


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import get_config, SECTOR_COLUMNS
from utils import (set_global_seed, resolve_device, setup_logger,
                   configure_run_paths, apply_date_overrides,
                   generate_sector_weights_by_t,
                   WarmupEvalCallback)
from data_loader import load_all
from preprocessing import preprocess
from state_builder import build_lookback_tensors
from net_autoencoder import train_autoencoder, save_autoencoder
from env_fund import FundEnv
from algorithms import build_model, load_model


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
    parser.add_argument("--sector-mode", type=str, default="macro",
                        choices=["macro", "equal"])
    parser.add_argument("--macro-algo", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(algo=args.algo)
    cfg = apply_date_overrides(cfg, args.train_end, args.val_end, args.test_end)
    cfg = configure_run_paths(cfg, args.experiment_mode, args.run_name, args.fold_name)
    if args.timesteps:
        cfg.train.total_timesteps = args.timesteps

    set_global_seed(cfg.train.seed)
    cfg.train.device = resolve_device(cfg.train.device)
    logger = setup_logger("train_fund", cfg.paths.outputs_dir)
    logger.info(f"=== TRAIN FUND  algo={cfg.train.algo}  run={args.run_name} "
                f"fold={args.fold_name}  sector_mode={args.sector_mode} ===")

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
    logger.info(f"train T={train_tensors.fundamental.shape[0]}  val T={val_tensors.fundamental.shape[0]}")

    sector_ids = np.array([raw["sector_map"][t] for t in raw["tickers"]], dtype=np.int64)
    n_sectors = len(SECTOR_COLUMNS)

    # Autoencoder fundamentalny
    ae_path = Path(cfg.paths.autoencoder_dir) / "fund_ae.pt"
    if ae_path.exists() and not args.force_retrain_ae:
        logger.info(f"[ae] autoencoder fund juz istnieje: {ae_path}")
    else:
        logger.info("[ae] trenuje autoencoder fundamentalny...")
        flat = [dataset.train.fundamental_per_ticker[tk] for tk in raw["tickers"]]
        flat_train = np.vstack(flat).astype(np.float32)
        ae = train_autoencoder(
            flat_train, input_dim=flat_train.shape[1],
            hidden=list(cfg.networks.fund_autoencoder_hidden),
            latent_dim=cfg.networks.fund_autoencoder_latent,
            epochs=cfg.networks.autoencoder_epochs,
            batch_size=cfg.networks.autoencoder_batch_size,
            lr=cfg.networks.autoencoder_lr,
            dropout=cfg.networks.autoencoder_dropout,
            device=cfg.train.device,
        )
        save_autoencoder(ae, str(ae_path))
        logger.info(f"[ae] zapisany: {ae_path}")

    # sector_weights_by_t (tryb macro)
    train_sw, val_sw = None, None
    if args.sector_mode == "macro":
        macro_algo = args.macro_algo or cfg.train.algo
        macro_dir = Path(cfg.paths.outputs_dir) / "models" / "macro" / macro_algo
        macro_path = macro_dir / "best_reward" / "best_model.zip"
        if not macro_path.exists():
            macro_path = macro_dir / f"macro_{macro_algo}.zip"
        if not macro_path.exists():
            raise FileNotFoundError(
                f"Brak macro_agent dla --sector-mode macro: {macro_path}\n"
                f"Uruchom najpierw: python train_macro.py --algo {macro_algo} "
                f"--run-name {args.run_name}\n"
                f"Albo uzyj --sector-mode equal (ablation)"
            )
        logger.info(f"[load] macro_agent z {macro_path}")
        macro_agent = load_model(str(macro_path), macro_algo, device=cfg.train.device)
        logger.info("[macro] generuje sector_weights_by_t dla train/val...")
        train_sw = generate_sector_weights_by_t(
            macro_agent, train_tensors.macro, cfg.hierarchy.rebalance_freq,
            n_sectors, cfg.hierarchy.sector_max_weight, algo=macro_algo,
        )
        val_sw = generate_sector_weights_by_t(
            macro_agent, val_tensors.macro, cfg.hierarchy.rebalance_freq,
            n_sectors, cfg.hierarchy.sector_max_weight, algo=macro_algo,
        )
        logger.info(f"[macro] train_sw mean={train_sw.mean(axis=0).round(3).tolist()}")
    else:
        logger.info("[mode] equal: wagi 1/n_sectors")

    def make_train_env():
        return FundEnv(train_tensors, train_tensors.returns, sector_ids, cfg,
                        sector_weights_by_t=train_sw)

    def make_val_env():
        return FundEnv(val_tensors, val_tensors.returns, sector_ids, cfg,
                        sector_weights_by_t=val_sw)

    train_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_train_env())]))
    val_env = VecMonitor(DummyVecEnv([lambda: Monitor(make_val_env())]))

    autoencoder_paths = {
        "macro": str(Path(cfg.paths.autoencoder_dir) / "macro_ae.pt"),
        "fund": str(ae_path),
    }
    model = build_model(train_env, cfg, agent_type="fund",
                         autoencoder_paths=autoencoder_paths,
                         tensorboard_log=cfg.paths.tensorboard_dir)

    save_dir = Path(cfg.paths.outputs_dir) / "models" / "fund" / cfg.train.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    eval_cb = WarmupEvalCallback(
        val_env, best_model_save_path=str(save_dir / "best_reward"),
        log_path=str(save_dir / "eval_reward"),
        eval_freq=cfg.eval.eval_freq,
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=True, verbose=1,
        min_timesteps_before_best=cfg.train.learning_starts,
    )

    logger.info(f"[train] start {cfg.train.total_timesteps} krokow")
    model.learn(total_timesteps=cfg.train.total_timesteps, callback=eval_cb, progress_bar=True)

    final_path = save_dir / f"fund_{cfg.train.algo}.zip"
    model.save(str(final_path))
    logger.info(f"[done] fund_agent zapisany: {final_path}")


if __name__ == "__main__":
    main()
