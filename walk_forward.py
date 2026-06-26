# ===== file: walk_forward.py =====


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import flatten_metrics_for_summary


FOLDS_EXPANDING_ANNUAL = [
    {"fold_name": "fold_01_2021", "train_end": "2019-12-31", "val_end": "2020-12-31", "test_end": "2021-12-31"},
    {"fold_name": "fold_02_2022", "train_end": "2020-12-31", "val_end": "2021-12-31", "test_end": "2022-12-31"},
    {"fold_name": "fold_03_2023", "train_end": "2021-12-31", "val_end": "2022-12-31", "test_end": "2023-12-31"},
    {"fold_name": "fold_04_2024", "train_end": "2022-12-31", "val_end": "2023-12-31", "test_end": "2024-12-31"},
    {"fold_name": "fold_05_2025", "train_end": "2023-06-30", "val_end": "2024-06-30", "test_end": "2025-12-29"},
]

ALL_ALGOS = ["td3", "sac", "tqc", "recurrent_ppo"]


def run_step(script: str, cli_args: list[str]) -> int:
    cmd = [sys.executable, script] + cli_args
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def run_fold(algo: str, exp_name: str, fold: dict, timesteps: int,
             sector_mode: str, prefer_model: str, force_ae: bool,
             skip_existing: bool) -> bool:
    """Wykonaj caly pipeline dla jednego algo+fold. Zwraca True jesli OK."""
    fold_name = fold["fold_name"]
    test_metrics = (Path("outputs") / "walk_forward" / exp_name / fold_name /
                    "backtest" / algo / "test" / "metrics.json")
    if skip_existing and test_metrics.exists():
        print(f"[skip] {algo}/{fold_name} juz istnieje")
        return True

    common = ["--algo", algo,
              "--experiment-mode", "walk_forward",
              "--run-name", exp_name,
              "--fold-name", fold_name,
              "--train-end", fold["train_end"],
              "--val-end", fold["val_end"],
              "--test-end", fold["test_end"],
              "--timesteps", str(timesteps)]
    if force_ae:
        common += ["--force-retrain-ae"]

    if run_step("train_macro.py", common) != 0:
        return False
    if run_step("train_fund.py", common + ["--sector-mode", sector_mode]) != 0:
        return False
    if run_step("train_tech.py", common) != 0:
        return False

    bt = ["--algo", algo, "--experiment-mode", "walk_forward",
          "--run-name", exp_name, "--fold-name", fold_name,
          "--train-end", fold["train_end"], "--val-end", fold["val_end"],
          "--test-end", fold["test_end"], "--prefer-model", prefer_model]
    run_step("backtest.py", bt + ["--split", "val"])
    if run_step("backtest.py", bt + ["--split", "test"]) != 0:
        return False
    return True


def collect_summary(algo: str, exp_name: str) -> pd.DataFrame:
    """Zbierz metrics.json (test) ze wszystkich foldow do DataFrame."""
    rows = []
    for fold in FOLDS_EXPANDING_ANNUAL:
        fn = fold["fold_name"]
        mpath = (Path("outputs") / "walk_forward" / exp_name / fn /
                 "backtest" / algo / "test" / "metrics.json")
        if not mpath.exists():
            print(f"[warn] brak {mpath}")
            continue
        with open(mpath, encoding="utf-8") as f:
            metrics = json.load(f)
        row = {"algo": algo, "fold_name": fn,
               "train_end": fold["train_end"], "val_end": fold["val_end"],
               "test_end": fold["test_end"]}
        row.update(flatten_metrics_for_summary(metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_json(df: pd.DataFrame, algo: str) -> dict:
    if df.empty:
        return {"algo": algo, "n_folds": 0}
    cr = df["strategy_cumulative_return"]
    return {
        "algo": algo,
        "n_folds": int(len(df)),
        "mean_cumulative_return": float(cr.mean()),
        "median_cumulative_return": float(cr.median()),
        "std_cumulative_return": float(cr.std(ddof=0)),
        "mean_sharpe": float(df["strategy_sharpe"].mean()),
        "mean_sortino": float(df["strategy_sortino"].mean()),
        "mean_max_drawdown": float(df["strategy_max_drawdown"].mean()),
        "mean_alpha_vs_equal_weight": float(df["alpha_vs_equal_weight"].mean()),
        "mean_alpha_vs_baseline": float(df["alpha_vs_baseline_macro_fund"].mean()),
        "positive_alpha_vs_baseline_folds": int((df["alpha_vs_baseline_macro_fund"] > 0).sum()),
        "positive_alpha_vs_equal_weight_folds": int((df["alpha_vs_equal_weight"] > 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["td3", "sac", "tqc", "recurrent_ppo", "all"])
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--folds-preset", type=str, default="expanding_annual",
                        choices=["expanding_annual"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sector-mode", type=str, default="macro", choices=["macro", "equal"])
    parser.add_argument("--prefer-model", type=str, default="best_reward",
                        choices=["best_reward", "final"])
    parser.add_argument("--force-retrain-ae", action="store_true")
    args = parser.parse_args()

    algos = ALL_ALGOS if args.algo == "all" else [args.algo]
    exp_dir = Path("outputs") / "walk_forward" / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}\n# WALK-FORWARD  algos={algos}  exp={args.experiment_name}\n{'#'*70}")

    all_summaries = []
    for algo in algos:
        print(f"\n{'='*70}\n=== ALGO: {algo} ===\n{'='*70}")
        for fold in FOLDS_EXPANDING_ANNUAL:
            ok = run_fold(algo, args.experiment_name, fold, args.timesteps,
                          args.sector_mode, args.prefer_model,
                          args.force_retrain_ae, args.skip_existing)
            if not ok:
                print(f"[error] {algo}/{fold['fold_name']} nie ukonczony")

        df = collect_summary(algo, args.experiment_name)
        if not df.empty:
            csv_path = exp_dir / f"walk_forward_summary_{algo}.csv"
            df.to_csv(csv_path, index=False)
            agg = aggregate_json(df, algo)
            with open(exp_dir / f"walk_forward_summary_{algo}.json", "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, ensure_ascii=False)
            print(f"\n[summary] {algo}: {csv_path}")
            print(f"  mean_cum_ret={agg['mean_cumulative_return']:+.4f}  "
                  f"mean_sharpe={agg['mean_sharpe']:.3f}  "
                  f"pos_alpha_vs_baseline={agg['positive_alpha_vs_baseline_folds']}/{agg['n_folds']}")
            df["algo"] = algo
            all_summaries.append(df)

    # Zbiorczy dla --algo all
    if len(algos) > 1 and all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_path = exp_dir / "walk_forward_all_algos_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n[summary] all algos: {combined_path}")

    print(f"\n{'#'*70}\n# WALK-FORWARD UKONCZONY\n{'#'*70}")


if __name__ == "__main__":
    main()
