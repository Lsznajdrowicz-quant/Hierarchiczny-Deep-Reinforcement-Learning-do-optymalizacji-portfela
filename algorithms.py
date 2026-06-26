# file: algorithms.py


from __future__ import annotations

from typing import Optional

import numpy as np
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import TQC, RecurrentPPO

from config import ProjectConfig
from net_macro import MacroFeaturesExtractor
from net_fundamental import FundamentalFeaturesExtractor
from net_technical import TechFeaturesExtractor


def _compute_buffer_size(env, requested: int, max_gb: float = 5.0) -> int:
    """Dobiera buffer_size tak, by replay buffer zmiescil sie w max_gb RAM.

    DictReplayBuffer trzyma obs + next_obs (2x rozmiar obserwacji). Dla agenta
    fund/tech obserwacja [N, L, F] jest ogromna, wiec buffer trzeba zmniejszyc.
    """
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        obs_floats = sum(int(np.prod(s.shape)) for s in obs_space.spaces.values())
    else:
        obs_floats = int(np.prod(obs_space.shape))
    bytes_per_obs = obs_floats * 4              # float32
    bytes_per_transition = bytes_per_obs * 2.2  # obs + next_obs + narzut
    max_transitions = int(max_gb * 1e9 / max(1, bytes_per_transition))
    chosen = min(requested, max(1000, max_transitions))
    if chosen < requested:
        print(f"[buffer] obniżam buffer_size {requested} -> {chosen} "
              f"(obs={bytes_per_obs/1e3:.0f}KB, budzet={max_gb}GB)")
    return chosen


def _common_kwargs(env, cfg: ProjectConfig) -> dict:
    """Wspolne kwargs dla off-policy algorithms."""
    return {
        "env": env,
        "learning_rate": cfg.train.learning_rate,
        "buffer_size": _compute_buffer_size(env, cfg.train.buffer_size, cfg.train.buffer_max_gb),
        "batch_size": cfg.train.batch_size,
        "tau": cfg.train.tau,
        "gamma": cfg.train.gamma,
        "learning_starts": cfg.train.learning_starts,
        "gradient_steps": cfg.train.gradient_steps,
        "train_freq": cfg.train.train_freq,
        "seed": cfg.train.seed,
        "device": cfg.train.device,
        "verbose": 1,
    }


def _features_extractor_for(agent_type: str, cfg: ProjectConfig,
                              autoencoder_paths: dict) -> tuple[type, dict]:
    """Zwroc (klasa, kwargs) dla danego typu agenta."""
    nets = cfg.networks
    if agent_type == "macro":
        return MacroFeaturesExtractor, {
            "autoencoder_path": autoencoder_paths["macro"],
            "autoencoder_hidden": nets.macro_autoencoder_hidden,
            "autoencoder_latent": nets.macro_autoencoder_latent,
            "lstm_hidden": nets.macro_lstm_hidden,
            "lstm_layers": nets.macro_lstm_layers,
            "head_hidden": nets.macro_head_hidden,
            "dropout": nets.macro_dropout,
        }
    if agent_type == "fund":
        return FundamentalFeaturesExtractor, {
            "autoencoder_path": autoencoder_paths["fund"],
            "autoencoder_hidden": nets.fund_autoencoder_hidden,
            "autoencoder_latent": nets.fund_autoencoder_latent,
            "ffnn_hidden": nets.fund_ffnn_hidden,
            "dropout": nets.fund_dropout,
        }
    if agent_type == "tech":
        return TechFeaturesExtractor, {
            "cnn_channels": nets.tech_cnn_channels,
            "cnn_kernel": nets.tech_cnn_kernel,
            "lstm_hidden": nets.tech_lstm_hidden,
            "head_hidden": nets.tech_head_hidden,
            "dropout": nets.tech_dropout,
        }
    raise ValueError(f"Unknown agent_type: {agent_type}")


def _policy_kwargs(agent_type: str, cfg: ProjectConfig, autoencoder_paths: dict) -> dict:
    fe_cls, fe_kwargs = _features_extractor_for(agent_type, cfg, autoencoder_paths)
    return {
        "features_extractor_class": fe_cls,
        "features_extractor_kwargs": fe_kwargs,
        "net_arch": list(cfg.networks.critic_hidden),
    }


def build_model(
    env,
    cfg: ProjectConfig,
    agent_type: str,                     # "macro" | "fund" | "tech"
    autoencoder_paths: dict,             # {"macro": "...", "fund": "..."}
    tensorboard_log: Optional[str] = None,
):
    """Buduje model SB3 odpowiedniego algorytmu z customowym extractorem.

    agent_type: ktora siec features uzyc
    cfg.train.algo: ktory algorithm (td3/sac/tqc/recurrent_ppo)
    """
    algo = cfg.train.algo
    policy_kwargs = _policy_kwargs(agent_type, cfg, autoencoder_paths)
    common = _common_kwargs(env, cfg)
    if tensorboard_log:
        common["tensorboard_log"] = tensorboard_log

    # Policy class dependent na obs space
    is_dict_obs = isinstance(env.observation_space, type(env.observation_space)) and \
                   hasattr(env.observation_space, 'spaces')

    if algo == "td3":
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.train.td3_action_noise_sigma * np.ones(n_actions),
        )
        policy = "MultiInputPolicy" if is_dict_obs else "MlpPolicy"
        return TD3(
            policy, **common,
            policy_kwargs=policy_kwargs,
            action_noise=action_noise,
            policy_delay=cfg.train.td3_policy_delay,
            target_policy_noise=cfg.train.td3_target_policy_noise,
            target_noise_clip=cfg.train.td3_target_noise_clip,
        )

    if algo == "sac":
        policy = "MultiInputPolicy" if is_dict_obs else "MlpPolicy"
        return SAC(
            policy, **common,
            policy_kwargs=policy_kwargs,
            ent_coef=cfg.train.ent_coef,
        )

    if algo == "tqc":
        policy = "MultiInputPolicy" if is_dict_obs else "MlpPolicy"
        return TQC(
            policy, **common,
            policy_kwargs={
                **policy_kwargs,
                "n_quantiles": cfg.train.n_quantiles,
                "n_critics": cfg.train.n_critics,
            },
            ent_coef=cfg.train.ent_coef,
            top_quantiles_to_drop_per_net=cfg.train.top_quantiles_to_drop,
        )

    if algo == "recurrent_ppo":
        policy = "MultiInputLstmPolicy" if is_dict_obs else "MlpLstmPolicy"
        ppo_kwargs = {
            "env": env,
            "learning_rate": cfg.train.learning_rate,
            "n_steps": cfg.train.rppo_n_steps,
            "batch_size": cfg.train.batch_size,
            "n_epochs": cfg.train.rppo_n_epochs,
            "gamma": cfg.train.gamma,
            "clip_range": cfg.train.rppo_clip_range,
            "seed": cfg.train.seed,
            "device": cfg.train.device,
            "verbose": 1,
            "policy_kwargs": {
                **policy_kwargs,
                "lstm_hidden_size": cfg.train.rppo_lstm_hidden,
            },
        }
        if tensorboard_log:
            ppo_kwargs["tensorboard_log"] = tensorboard_log
        return RecurrentPPO(policy, **ppo_kwargs)

    raise ValueError(f"Unknown algo: {algo}")


def load_model(path: str, algo: str, env=None, device: str = "cpu"):
    """Wczytaj zapisany model. Algo musi odpowiadac temu z treningu."""
    if algo == "td3":
        return TD3.load(path, env=env, device=device)
    if algo == "sac":
        return SAC.load(path, env=env, device=device)
    if algo == "tqc":
        return TQC.load(path, env=env, device=device)
    if algo == "recurrent_ppo":
        return RecurrentPPO.load(path, env=env, device=device)
    raise ValueError(f"Unknown algo: {algo}")
