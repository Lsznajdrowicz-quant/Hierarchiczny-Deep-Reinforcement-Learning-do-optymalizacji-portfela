# ===== file: config.py =====

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


AlgoName = Literal["td3", "sac", "tqc", "recurrent_ppo"]


# =========================================================================
# SCIEZKI
# =========================================================================
@dataclass
class PathsConfig:
    technical_dir: str = r"Tu wstaw sciezke do folderu z danymi technicznymi"
    macro_file: str = r"Tu wstaw sciezke do pliku z danymi makroekonomicznymi"
    fundamental_dir: str = r"Tu wstaw sciezke do folderu z danymi fundamentalnymi"
    stock_returns_csv: str = r"Tu wstaw sciezke do pliku z zwrotami akcji"
    sector_returns_csv: str = r"Tu wstaw sciezke do pliku z zwrotami sektorow"
    sector_map_csv: str = r"Tu wstaw sciezke do pliku z mapowaniem sektorow"
    outputs_dir: str = "outputs"
    tensorboard_dir: str = "outputs/tensorboard"
    autoencoder_dir: str = "outputs/autoencoders"

#Tu wstaw nazwy sektorow w kolejnosci alfabetycznej, zgodnie z mapowaniem sektorow w pliku sector_map_csv
SECTOR_COLUMNS = [
    "Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate",
    "Technology", "Utilities",
]

# Kolumny meta — wykluczone ze wszystkich strumieni cech
META_COLUMNS = {"date", "ticker", "symbol", "sector", "sector_id", "return", "returns",
                 "ret", "target", "label", "id"}

# Kolumny fundamentalne wymagajace standaryzacji sektorowej (cross-sectional per date)
FUNDAMENTAL_SECTOR_STD_COLS = [
    "pe_close_ttm", "price_to_sales", "price_to_book", "price_to_earnings",
    "ev_to_ebitda", "price_to_fcf", "ebitda_margin", "net_margin", "debt_to_ebitda",
    "eps_surprise_pct", "revenue_qoq", "revenue_yoy", "EBITDA_qoq", "EBITDA_yoy",
    "netIncome_qoq", "netIncome_yoy", "totalDebt_qoq", "totalDebt_yoy", "freeCashflow_qoq",
    "freeCashflow_yoy", "reportedEPS_qoq", "reportedEPS_yoy",
]


@dataclass
class DataConfig:
    date_col: str = "date"
    macro_lookback: int = 50
    fund_lookback: int = 50
    tech_lookback: int = 21
    start_date: str = "2014-01-02"        # pierwsza data decyzyjna

    train_end: str = "2023-06-30"
    val_end: str = "2024-06-30"
    test_end: str = "2025-12-29"

    technical_scaler: Literal["robust", "standard"] = "robust"
    macro_scaler: Literal["robust", "standard"] = "robust"
    fundamental_scaler: Literal["robust", "standard"] = "robust"

    technical_per_ticker_scaling: bool = True
    fundamental_sector_zscore: bool = True
    fundamental_sector_zscore_clip: float = 5.0

    expected_n_stocks: int = 100


@dataclass
class HierarchyConfig:
    rebalance_freq: int = 21                # makro+fund co 21 dni
    tilt_low: float = 0.7
    tilt_high: float = 1.3
    sector_min_weight: float = 0.0
    sector_max_weight: float = 0.6


@dataclass
class ExecutionConfig:
    transaction_cost_bps: float = 5.0
    spread_cost_bps: float = 10.0
    turnover_epsilon: float = 0.0
    rebalance_eta: float = 1.0
    initial_portfolio_value: float = 1.0
    action_clip: float = 5.0


@dataclass
class RewardConfig:
    # === NIE ZMIENIAC — rewardy zamrozone metodologicznie ===
    macro_reward_weight: float = 1.0
    fundamental_reward_weight: float = 1.0
    technical_reward_weight: float = 1.0

    # Sortino na 21-dniowym oknie (makro + fund)
    sortino_window: int = 21
    sortino_annualization: int = 252
    sortino_scale: float = 1.0
    sortino_clip: float = 10.0

    # Reward agenta technicznego jako OVERLAY:
    # r = active_scale * active_log_return + profit_scale * net_return
    #     - beta_risk * sigma2(active_log) - xi_turnover * |tilt_turnover|
    technical_active_scale: float = 100.0
    technical_profit_scale: float = 10.0
    technical_beta_risk: float = 5.0
    technical_vol_window: int = 21
    technical_xi_turnover: float = 1.0
    technical_max_clip: float = 5.0


@dataclass
class NetworksConfig:
    macro_autoencoder_latent: int = 24
    macro_autoencoder_hidden: tuple[int, ...] = (128, 64)
    fund_autoencoder_latent: int = 24
    fund_autoencoder_hidden: tuple[int, ...] = (128, 64)
    autoencoder_epochs: int = 50
    autoencoder_lr: float = 1e-3
    autoencoder_batch_size: int = 256
    autoencoder_dropout: float = 0.1

    # LSTM makro
    macro_lstm_hidden: int = 64
    macro_lstm_layers: int = 2
    macro_head_hidden: tuple[int, ...] = (96, 32)
    macro_dropout: float = 0.15

    # FFNN fundamentalny
    fund_ffnn_hidden: tuple[int, ...] = (192, 96)
    fund_dropout: float = 0.15

    # CNN+LSTM techniczny (per-ticker, shared params)
    tech_cnn_channels: tuple[int, ...] = (32, 48)
    tech_cnn_kernel: int = 3
    tech_lstm_hidden: int = 64
    tech_head_hidden: tuple[int, ...] = (96, 48)
    tech_dropout: float = 0.15

    critic_hidden: tuple[int, ...] = (256, 128)


@dataclass
class TrainConfig:
    algo: AlgoName = "td3"
    seed: int = 42
    device: str = "cuda"
    total_timesteps: int = 100_000
    learning_starts: int = 3000
    buffer_size: int = 30_000
    buffer_max_gb: float = 5.0
    batch_size: int = 128
    learning_rate: float = 1e-4
    tau: float = 0.005
    gamma: float = 0.99
    gradient_steps: int = 1
    train_freq: int = 1

    # TD3
    td3_action_noise_sigma: float = 0.35
    td3_policy_delay: int = 2
    td3_target_policy_noise: float = 0.2
    td3_target_noise_clip: float = 0.5

    # SAC/TQC
    ent_coef: str = "auto"

    # TQC
    n_quantiles: int = 25
    top_quantiles_to_drop: int = 2
    n_critics: int = 2

    # RPPO
    rppo_n_steps: int = 256
    rppo_n_epochs: int = 10
    rppo_clip_range: float = 0.2
    rppo_lstm_hidden: int = 128

    eval_freq: int = 4000
    n_eval_episodes: int = 1


@dataclass
class EvalConfig:
    eval_freq: int = 4000
    n_eval_episodes: int = 1
    use_reward_eval: bool = True
    save_best_reward: bool = True


@dataclass
class ExperimentConfig:
    experiment_mode: Literal["single", "walk_forward"] = "single"
    run_name: str = "default"
    fold_name: Optional[str] = None


@dataclass
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    networks: NetworksConfig = field(default_factory=NetworksConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# =========================================================================
# PRESETY ALGORYTMOW
# =========================================================================
def apply_algo_preset(cfg: ProjectConfig) -> ProjectConfig:
    """Ustawia hyperparams per algorytm. NIE dotyka rewardow ani sieci."""
    algo = cfg.train.algo
    t = cfg.train

    if algo == "td3":
        # Stabilny baseline
        t.learning_rate = 1e-4
        t.buffer_size = 100_000
        t.learning_starts = 5_000
        t.batch_size = 256
        t.gradient_steps = 1
        t.train_freq = 1
        t.tau = 0.005
        t.gamma = 0.99
        t.td3_action_noise_sigma = 0.25
        t.td3_policy_delay = 2
        t.td3_target_policy_noise = 0.2
        t.td3_target_noise_clip = 0.5

    elif algo == "sac":
        t.learning_rate = 3e-4
        t.buffer_size = 100_000
        t.learning_starts = 5_000
        t.batch_size = 256
        t.gradient_steps = 1
        t.train_freq = 1
        t.tau = 0.005
        t.gamma = 0.99
        t.ent_coef = "auto_0.1"

    elif algo == "tqc":
        # Najwazniejszy preset — TQC dzialal slabo
        t.learning_rate = 3e-4
        t.buffer_size = 150_000
        t.learning_starts = 10_000
        t.batch_size = 256
        t.gradient_steps = 2
        t.train_freq = 1
        t.tau = 0.005
        t.gamma = 0.995
        t.ent_coef = "auto_0.1"
        t.n_quantiles = 25
        t.n_critics = 2
        t.top_quantiles_to_drop = 2

    elif algo == "recurrent_ppo":
        t.learning_rate = 3e-4
        t.batch_size = 128
        t.gamma = 0.99
        t.rppo_n_steps = 512
        t.rppo_n_epochs = 5
        t.rppo_clip_range = 0.15
        t.rppo_lstm_hidden = 128

    return cfg


def get_config(algo: Optional[AlgoName] = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if algo is not None:
        cfg.train.algo = algo
    cfg = apply_algo_preset(cfg)
    return cfg
