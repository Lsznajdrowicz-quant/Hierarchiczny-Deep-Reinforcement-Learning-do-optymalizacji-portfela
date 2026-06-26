# file: preprocessing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import ProjectConfig
from data_loader import get_common_features
from scalers import (
    FittedScalers, fit_all_scalers,
    transform_macro, transform_technical_per_ticker, transform_fundamental,
)


@dataclass
class SplitData:
    """Pojedynczy split (train/val/test) z gotowymi tensorami."""
    dates: pd.DatetimeIndex
    macro: np.ndarray                          # [T, F_macro] scaled
    technical_per_ticker: Dict[str, np.ndarray]  # ticker -> [T, F_tech] scaled
    fundamental_per_ticker: Dict[str, np.ndarray]  # ticker -> [T, F_fund] scaled
    stock_returns: np.ndarray                  # [T, N] (nie skalowane — uzywane do reward)
    tickers: list[str]


@dataclass
class PreprocessedDataset:
    train: SplitData
    val: SplitData
    test: SplitData
    fitted_scalers: FittedScalers
    feature_cols: dict      # {macro: [...], technical: [...], fundamental: [...]}


def build_common_calendar(
    macro: pd.DataFrame,
    technical_panels: Dict[str, pd.DataFrame],
    fundamental_panels: Dict[str, pd.DataFrame],
    stock_returns: pd.DataFrame,
    start_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Wspolny indeks dat: intersection wszystkich zrodel, od start_date."""
    common = set(macro.index)
    common &= set(stock_returns.index)
    for df in technical_panels.values():
        common &= set(df.index)
    for df in fundamental_panels.values():
        common &= set(df.index)
    common_idx = pd.DatetimeIndex(sorted(d for d in common if d >= start_date))
    return common_idx


def slice_split(idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return idx[(idx >= start) & (idx <= end)]


def preprocess(raw: dict, cfg: ProjectConfig) -> PreprocessedDataset:
    """Pelen pipeline: alignment + fit scalers + transform per split."""
    # 1. Wspolny kalendarz dat
    start_date = pd.Timestamp(cfg.data.start_date)
    full_idx = build_common_calendar(
        raw["macro"], raw["technical_panels"], raw["fundamental_panels"],
        raw["stock_returns"], start_date,
    )
    max_lookback = max(
    cfg.data.macro_lookback,
    cfg.data.fund_lookback,
    cfg.data.tech_lookback,
    )

    if len(full_idx) < max_lookback + 30:
        raise ValueError(f"Za malo wspolnych dat: {len(full_idx)}")



    print(f"[preproc] wspolny kalendarz: {len(full_idx)} dni od {full_idx[0].date()} do {full_idx[-1].date()}")

    # 2. Granice splitow
    train_end = pd.Timestamp(cfg.data.train_end)
    val_end = pd.Timestamp(cfg.data.val_end)
    test_end = pd.Timestamp(cfg.data.test_end)

    # 3. Dopasuj wszystkie skalery (TYLKO na danych train)
    fitted, cols = fit_all_scalers(raw, cfg)

    # 4. Transform — pelen indeks, potem slice na splity
    macro_scaled = transform_macro(raw["macro"].reindex(full_idx), fitted.macro_scaler)
    tech_scaled = transform_technical_per_ticker(
        raw["technical_panels"], fitted.technical_per_ticker,
        cols["technical"], full_idx,
    )
    fund_scaled = transform_fundamental(
        raw["fundamental_panels"], fitted.fundamental_scaler,
        raw["sector_map"],
        cols["fundamental"], full_idx,
        sector_zscore=cfg.data.fundamental_sector_zscore,
        clip=cfg.data.fundamental_sector_zscore_clip,
    )
    rets = raw["stock_returns"].reindex(full_idx)[raw["tickers"]].to_numpy(dtype=np.float64)
    rets = np.nan_to_num(rets, nan=0.0)

    # 5. Slice na splity
    def build_split(start: pd.Timestamp, end: pd.Timestamp) -> SplitData:
        mask = (full_idx >= start) & (full_idx <= end)
        sub_idx = full_idx[mask]
        if len(sub_idx) == 0:
            raise ValueError(f"Pusty split {start} -> {end}")
        return SplitData(
            dates=sub_idx,
            macro=macro_scaled[mask],
            technical_per_ticker={tk: arr[mask] for tk, arr in tech_scaled.items()},
            fundamental_per_ticker={tk: arr[mask] for tk, arr in fund_scaled.items()},
            stock_returns=rets[mask],
            tickers=raw["tickers"],
        )

    train = build_split(start_date, train_end)
    val = build_split(train_end + pd.Timedelta(days=1), val_end)
    test = build_split(val_end + pd.Timedelta(days=1), test_end)

    print(f"[preproc] train={len(train.dates)}  val={len(val.dates)}  test={len(test.dates)}")
    return PreprocessedDataset(
        train=train, val=val, test=test,
        fitted_scalers=fitted, feature_cols=cols,
    )
