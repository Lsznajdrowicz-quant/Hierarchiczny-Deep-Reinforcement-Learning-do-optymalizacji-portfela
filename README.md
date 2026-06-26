# Hierarchical Deep Reinforcement Learning for Portfolio Allocation

This project implements a hierarchical deep reinforcement learning framework for dynamic stock portfolio allocation.  
The system combines macroeconomic, fundamental and technical information into a three-level decision process:

1. **Macroeconomic allocation agent** - allocates capital across sectors.
2. **Fundamental stock selection agent** - allocates capital among companies within sector constraints.
3. **Technical overlay agent** - applies bounded technical tilts to the macro-fundamental target portfolio.

The project is designed for research and experimentation in quantitative finance, systematic portfolio construction and deep reinforcement learning.

---

## Key Features

- Hierarchical portfolio allocation: macro → fundamental → technical.
- Custom Gymnasium environments for each agent.
- Support for **TD3**, **SAC**, **TQC** and **Recurrent PPO**.
- Deep feature extractors based on frozen autoencoders, LSTM networks, per-stock feed-forward networks and shared CNN-LSTM encoders.
- Transaction costs, spread costs, turnover and rebalancing logic.
- Backtesting against equal-weight, Markowitz minimum-variance, Markowitz maximum-Sharpe and macro-fundamental baseline portfolios.
- Walk-forward validation across expanding annual folds.
- Outputs saved separately by algorithm, run name and fold.

---

## Project Structure

```text
.
├── config.py                 # Global configuration, paths, lookbacks, rewards and algorithm presets
├── data_loader.py            # Loads macro, fundamental, technical, returns and sector mapping data
├── preprocessing.py          # Aligns calendars, scales features and creates train/val/test splits
├── scalers.py                # Robust/standard scaling and cross-sectional sector normalization
├── state_builder.py          # Builds lookback tensors for macro, fundamental and technical states
│
├── env_macro.py              # Gymnasium environment for the macro sector allocation agent
├── env_fund.py               # Gymnasium environment for the fundamental stock selection agent
├── env_tech.py               # Gymnasium environment for the technical overlay agent
│
├── net_autoencoder.py        # Autoencoder model and training utilities
├── net_macro.py              # Macro feature extractor: frozen autoencoder + LSTM
├── net_fundamental.py        # Fundamental extractor: frozen autoencoder + per-stock FFNN
├── net_technical.py          # Technical extractor: shared CNN-LSTM per stock
│
├── algorithms.py             # Builds TD3, SAC, TQC and Recurrent PPO models
├── rewards.py                # Sortino reward and technical overlay reward
├── metrics.py                # Performance metrics and benchmark calculations
├── utils.py                  # Shared utilities, paths, seeds, Recurrent PPO prediction support
│
├── train_macro.py            # Trains the macro agent
├── train_fund.py             # Trains the fundamental agent
├── train_tech.py             # Trains the technical overlay agent
├── backtest.py               # Runs validation/test backtests and saves metrics
├── run_single.py             # Main single-run orchestration script
└── walk_forward.py           # Walk-forward experiment runner
```

---

## Methodology Overview

### 1. Data Pipeline

The project uses multiple data streams:

- macroeconomic variables,
- company-level fundamental factors,
- technical indicators per stock,
- daily stock returns,
- sector mapping.

The preprocessing stage aligns all sources to a common calendar, fits scalers only on the training period and creates train, validation and test splits. Fundamental features can also be normalized cross-sectionally within sectors and dates.

### 2. State Construction

The `state_builder.py` module converts preprocessed data into lookback tensors:

```text
Macro state:        [T, L_macro, F_macro]
Fundamental state:  [T, N, L_fund, F_fund]
Technical state:    [T, N, L_tech, F_tech]
Returns matrix:     [T, N]
```

For validation and test, the code can use historical context from the previous split to avoid losing the first `max_lookback - 1` evaluation days. This is not data leakage because the context comes only from the past.

### 3. Hierarchical Agents

#### Agent 1 - Macroeconomic Allocation

The macro agent receives a lookback window of macroeconomic variables and outputs continuous sector logits. These logits are converted into long-only sector weights using softmax and sector weight constraints.

```text
Macro window → Autoencoder latent → LSTM → Policy/Critic → Sector weights
```

#### Agent 2 - Fundamental Stock Selection

The fundamental agent receives company fundamentals and sector weights produced by the macro agent. It ranks companies within sectors and converts stock logits into sector-constrained stock weights using masked softmax.

```text
Fundamental factors + Sector weights → Autoencoder + per-stock FFNN → Stock weights
```

#### Agent 3 - Technical Overlay

The technical agent receives technical time series, macro-fundamental target weights, current weights and previous tilts. It does not build a new portfolio from scratch. Instead, it applies bounded multiplicative tilts to the existing macro-fundamental portfolio.

```text
Technical time series + Portfolio context → CNN-LSTM → Bounded tilts → Final weights
```

---

## Reinforcement Learning Algorithms

The project supports the following algorithms:

- **TD3** - deterministic off-policy actor-critic with twin critics and delayed policy updates.
- **SAC** - entropy-regularized off-policy actor-critic.
- **TQC** - distributional actor-critic with truncated quantiles.
- **Recurrent PPO** - on-policy recurrent policy with LSTM state.

The selected algorithm is controlled by the `--algo` argument.

---

## Installation

A typical environment can be prepared with Conda:

```bash
conda create -n mlstack python=3.11
conda activate mlstack
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn torch gymnasium stable-baselines3 sb3-contrib tqdm matplotlib
```

If CUDA is available, install a PyTorch build compatible with your GPU and CUDA version.

---

## Data Configuration

Data paths are configured in `config.py`:

```python
technical_dir = r"...\\Dane techniczne\\csv_selected"
macro_file = r"...\\Makro\\dane_makro.csv"
fundamental_dir = r"...\\Dane fundamentalne_selected"
stock_returns_csv = r"...\\ticker_returns.csv"
sector_returns_csv = r"...\\sector_returns.csv"
sector_map_csv = r"...\\sector_map.csv"
```

Expected input structure:

- one CSV file with macroeconomic features,
- one CSV file with stock returns,
- one CSV file with sector mapping,
- one technical CSV file per ticker,
- one fundamental CSV file per ticker.

Each input file should contain a `date` column.

---

## Running a Single Experiment

Run the full pipeline for SAC:

```bash
python run_single.py --algo sac --run-name sac_test --timesteps 50000
```

Run the full pipeline for TD3:

```bash
python run_single.py --algo td3 --run-name td3_test --timesteps 50000
```

Run the full pipeline for Recurrent PPO:

```bash
python run_single.py --algo recurrent_ppo --run-name recurrent_ppo_test --timesteps 50000
```

---

## Running Selected Stages

Skip macro and fundamental training, train only the technical agent and then run backtests:

```bash
python run_single.py --algo sac --run-name sac_test --timesteps 50000 --skip macro fund
```

Run only validation and test backtests using already saved models:

```bash
python run_single.py --algo sac --run-name sac_test --timesteps 50000 --skip macro fund tech
```

Run only the test backtest directly:

```bash
python backtest.py --algo sac --experiment-mode single --run-name sac_test --prefer-model best_reward --split test
```

---

## Walk-Forward Evaluation

Run walk-forward validation for one algorithm:

```bash
python walk_forward.py --algo sac --experiment-name wf_sac --timesteps 50000
```

Run walk-forward validation for all algorithms:

```bash
python walk_forward.py --algo all --experiment-name wf_all --timesteps 50000
```

The walk-forward script uses expanding annual folds and saves summary files for each algorithm.

---

## Output Structure

Single-run outputs are saved under:

```text
outputs/
└── single/
    └── {run_name}/
        ├── autoencoders/
        ├── models/
        │   ├── macro/{algo}/
        │   ├── fund/{algo}/
        │   └── tech/{algo}/
        ├── backtest/
        │   └── {algo}/
        │       ├── val/
        │       └── test/
        └── training.log
```

Backtest folders contain:

```text
daily_weights.csv
fund_target_weights.csv
fund_within_sector.csv
tech_multipliers.csv
sector_weights.csv
returns.csv
benchmark_markowitz.csv
markowitz_weights.csv
metrics.json
```

Walk-forward outputs are saved under:

```text
outputs/
└── walk_forward/
    └── {experiment_name}/
        ├── {fold_name}/
        ├── walk_forward_summary_{algo}.csv
        ├── walk_forward_summary_{algo}.json
        └── walk_forward_all_algos_summary.csv
```

---

## Evaluation Metrics

The strategy is evaluated using:

- cumulative return,
- annualized return,
- annualized volatility,
- Sharpe ratio,
- Sortino ratio,
- maximum drawdown,
- Calmar ratio,
- turnover,
- transaction costs,
- alpha versus equal-weight,
- alpha versus Markowitz benchmarks,
- alpha versus macro-fundamental baseline,
- information ratio,
- win rate versus benchmark.

---

## Research Notes

This project is not only a return prediction model. It is a controlled allocation framework that combines temporal feature extraction, reinforcement learning, portfolio constraints, transaction costs, benchmark comparison and walk-forward robustness testing.

The technical overlay is evaluated separately against the macro-fundamental baseline, which allows an ablation-style interpretation of whether short-term technical signals add value.

---

## Example Presentation Summary

The code implements a hierarchical deep reinforcement learning system for portfolio allocation. The macro agent allocates capital across sectors, the fundamental agent selects stocks within those sector constraints, and the technical agent applies bounded tilts to the resulting target portfolio.

The deep learning part is implemented through autoencoders, LSTM networks, per-stock feed-forward networks and CNN-LSTM encoders. The final strategy is evaluated through backtesting, benchmark comparison and walk-forward validation.

---

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and should not be treated as a production trading system without further validation, risk controls and execution-layer testing.
