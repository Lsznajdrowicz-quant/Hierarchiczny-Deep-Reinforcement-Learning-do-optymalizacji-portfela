

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import ProjectConfig, META_COLUMNS


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"{path}: brak kolumny 'date'")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).drop_duplicates(date_col).set_index(date_col)


def load_sector_map(path: str) -> dict[str, int]:
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "sector_id" not in df.columns:
        raise ValueError(f"{path}: oczekiwane kolumny ticker, sector_id")
    return dict(zip(df["ticker"].astype(str), df["sector_id"].astype(int)))


def load_macro(path: str) -> pd.DataFrame:
    """Wczytaj makro CSV — zwraca DataFrame [T, F_macro] z dateindex."""
    df = _read_csv(path)
    # odrzuc kolumny meta i nieliczbowe
    keep = [c for c in df.columns
            if c.lower() not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    return df[keep]


def load_stock_returns(path: str, tickers: list[str]) -> pd.DataFrame:
    """Zwroty per spolka — DataFrame [T, N]."""
    df = _read_csv(path)
    avail = [t for t in tickers if t in df.columns]
    if len(avail) < len(tickers):
        missing = set(tickers) - set(avail)
        print(f"[WARN] brak zwrotow dla: {missing}")
    return df[avail]


def load_sector_returns(path: str) -> Optional[pd.DataFrame]:
    """Zwroty sektorowe (opcjonalne)."""
    try:
        return _read_csv(path)
    except Exception as e:
        print(f"[INFO] sector_returns niewczytane: {e}")
        return None


def load_ticker_panel(folder: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Wczytaj CSV-y per spolka z folderu. Klucz = ticker."""
    base = Path(folder)
    out = {}
    for t in tickers:
        p = base / f"{t}.csv"
        if not p.exists():
            print(f"[WARN] brak {p}")
            continue
        df = _read_csv(str(p))
        # Wybierz tylko liczbowe, bez meta
        keep = [c for c in df.columns
                if c.lower() not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
        out[t] = df[keep]
    if not out:
        raise FileNotFoundError(f"Brak plikow w {base}")
    return out


def get_common_features(ticker_dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Wspolne kolumny w plikach (intersekcja). Posortowane alfabetycznie."""
    sets = [set(df.columns) for df in ticker_dfs.values()]
    common = sorted(set.intersection(*sets))
    return common


def get_universe(cfg: ProjectConfig) -> tuple[list[str], dict[str, int]]:
    """Lista spolek do treningu (intersekcja: technical + fundamental + zwroty + sector_map)."""
    sec_map = load_sector_map(cfg.paths.sector_map_csv)
    tickers_in_map = sorted(sec_map.keys())

    # Sprawdz dostepnosc plikow
    tech_dir = Path(cfg.paths.technical_dir)
    fund_dir = Path(cfg.paths.fundamental_dir)
    rets_df = _read_csv(cfg.paths.stock_returns_csv)

    available = []
    for t in tickers_in_map:
        if not (tech_dir / f"{t}.csv").exists():
            continue
        if not (fund_dir / f"{t}.csv").exists():
            continue
        if t not in rets_df.columns:
            continue
        available.append(t)

    if cfg.data.expected_n_stocks and len(available) > cfg.data.expected_n_stocks:
        available = available[:cfg.data.expected_n_stocks]
    return available, sec_map


def load_all(cfg: ProjectConfig) -> dict[str, object]:
    """Wczytaj wszystko: makro, tech, fund, zwroty, sektory. Zwroc slownik."""
    tickers, sec_map = get_universe(cfg)
    print(f"[data] universe: {len(tickers)} spolek")

    macro = load_macro(cfg.paths.macro_file)
    print(f"[data] macro: {macro.shape}")

    tech_panels = load_ticker_panel(cfg.paths.technical_dir, tickers)
    fund_panels = load_ticker_panel(cfg.paths.fundamental_dir, tickers)
    print(f"[data] technical: {len(tech_panels)} spolek")
    print(f"[data] fundamental: {len(fund_panels)} spolek")

    returns = load_stock_returns(cfg.paths.stock_returns_csv, tickers)
    print(f"[data] returns: {returns.shape}")

    sector_returns = load_sector_returns(cfg.paths.sector_returns_csv)

    return {
        "tickers": tickers,
        "sector_map": sec_map,
        "macro": macro,
        "technical_panels": tech_panels,
        "fundamental_panels": fund_panels,
        "stock_returns": returns,
        "sector_returns": sector_returns,
    }
