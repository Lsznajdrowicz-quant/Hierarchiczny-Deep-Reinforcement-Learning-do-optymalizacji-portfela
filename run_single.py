# ===== file: run_single.py =====

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script: str, cli_args: list[str]) -> None:
    cmd = [sys.executable, script] + cli_args
    print(f"\n{'='*70}\n>>> {' '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script} blad ({result.returncode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3",
                        choices=["td3", "sac", "tqc", "recurrent_ppo"])
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["macro", "fund", "tech", "backtest_val", "backtest_test"])
    parser.add_argument("--sector-mode", type=str, default="macro", choices=["macro", "equal"])
    parser.add_argument("--prefer-model", type=str, default="best_reward",
                        choices=["best_reward", "final"])
    parser.add_argument("--force-retrain-ae", action="store_true")
    args = parser.parse_args()

    # Wspolne argumenty przekazywane do kazdego skryptu
    common = ["--algo", args.algo,
              "--experiment-mode", "single",
              "--run-name", args.run_name]
    if args.timesteps:
        common += ["--timesteps", str(args.timesteps)]
    if args.train_end:
        common += ["--train-end", args.train_end]
    if args.val_end:
        common += ["--val-end", args.val_end]
    if args.test_end:
        common += ["--test-end", args.test_end]
    if args.force_retrain_ae:
        common += ["--force-retrain-ae"]

    print(f"\n{'#'*70}\n# RUN SINGLE  algo={args.algo}  run={args.run_name}\n{'#'*70}")

    if "macro" not in args.skip:
        run_step("train_macro.py", common)
    if "fund" not in args.skip:
        run_step("train_fund.py", common + ["--sector-mode", args.sector_mode])
    if "tech" not in args.skip:
        run_step("train_tech.py", common)

    # Backtest nie przyjmuje --timesteps/--force-retrain-ae/--sector-mode
    bt_common = ["--algo", args.algo,
                 "--experiment-mode", "single",
                 "--run-name", args.run_name,
                 "--prefer-model", args.prefer_model]
    if args.train_end:
        bt_common += ["--train-end", args.train_end]
    if args.val_end:
        bt_common += ["--val-end", args.val_end]
    if args.test_end:
        bt_common += ["--test-end", args.test_end]

    if "backtest_val" not in args.skip:
        run_step("backtest.py", bt_common + ["--split", "val"])
    if "backtest_test" not in args.skip:
        run_step("backtest.py", bt_common + ["--split", "test"])

    base = Path("outputs") / "single" / args.run_name / "backtest" / args.algo
    print(f"\n{'#'*70}\n# RUN SINGLE UKONCZONY\n{'#'*70}")
    print(f"  val:  {base / 'val' / 'metrics.json'}")
    print(f"  test: {base / 'test' / 'metrics.json'}")


if __name__ == "__main__":
    main()
