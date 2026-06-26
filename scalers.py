# file: scalers.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from config import FUNDAMENTAL_SECTOR_STD_COLS


def _make_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaler: {kind}")


@dataclass
class FittedScalers:
    """Kontener na dopasowane skalery (TYLKO globalne — sektorowe juz nie sa
    historyczne, bo liczymy je per-date w transform)."""
    macro_scaler: Optional[object] = None
    technical_per_ticker: Dict[str, object] = field(default_factory=dict)
    fundamental_scaler: Optional[object] = None


def fit_macro_scaler(macro_train: pd.DataFrame, kind: str = "robust"):
    sc = _make_scaler(kind)
    arr = macro_train.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    sc.fit(arr)
    return sc


def transform_macro(macro_df: pd.DataFrame, scaler, clip: float = 10.0) -> np.ndarray:
    arr = macro_df.to_numpy(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    out = scaler.transform(arr)
    out = np.clip(out, -clip, clip)
    return out.astype(np.float32)


def fit_technical_per_ticker(
    panels: Dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    common_cols: list[str],
    kind: str = "robust",
) -> Dict[str, object]:
    out = {}
    for tk, df in panels.items():
        train_slice = df.loc[df.index <= train_end, common_cols]
        if len(train_slice) < 30:
            out[tk] = None
            continue
        sc = _make_scaler(kind)
        arr = train_slice.to_numpy(dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        sc.fit(arr)
        out[tk] = sc
    return out


def transform_technical_per_ticker(
    panels: Dict[str, pd.DataFrame],
    scalers: Dict[str, object],
    common_cols: list[str],
    dates_index: pd.DatetimeIndex,
    clip: float = 10.0,
) -> Dict[str, np.ndarray]:
    out = {}
    for tk, df in panels.items():
        aligned = df.reindex(dates_index)[common_cols].apply(pd.to_numeric, errors="coerce")
        arr = aligned.to_numpy(dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        sc = scalers.get(tk)
        if sc is not None:
            arr = sc.transform(arr)
        arr = np.clip(arr, -clip, clip)
        out[tk] = arr.astype(np.float32)
    return out


def fit_fundamental_scaler(
    panels: Dict[str, pd.DataFrame],
    train_end: pd.Timestamp,
    common_cols: list[str],
    kind: str = "robust",
) -> object:
    """Fit GLOBAL scaler na wszystkich (ticker, date) z train. Nic sektorowego."""
    stacked = []
    for tk, df in panels.items():
        sub = df.loc[df.index <= train_end, common_cols]
        stacked.append(sub.to_numpy(dtype=np.float64))
    full = np.vstack(stacked)
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
    sc = _make_scaler(kind)
    sc.fit(full)
    return sc


def transform_fundamental(
    panels: Dict[str, pd.DataFrame],
    scaler,
    sector_map: Dict[str, int],
    common_cols: list[str],
    dates_index: pd.DatetimeIndex,
    sector_zscore: bool = True,
    clip: float = 5.0,
    min_sector_size: int = 3,
    iqr_eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Per ticker: globalny scaler + CROSS-SECTIONAL SECTOR Z-SCORE PER DATE.

    Dla kazdej daty t i sektora s liczymy mediane i IQR z wartosci wybranych
    kolumn (FUNDAMENTAL_SECTOR_STD_COLS) zebranych od WSZYSTKICH spolek w
    sektorze s wlasnie w dacie t. Potem przeliczamy z = (x - med) / IQR i
    klipujemy do [-clip, clip].

    Fallback: jezeli sektor ma w danej dacie < min_sector_size poprawnych wartosci
    lub IQR jest blizniaczy zera, dla danej kolumny w tej dacie zostawia wynik
    globalnego scalera (NIE NaN/Inf).

    Returns: {ticker: [T, F_fund]} jak poprzednio.
    """
    tickers = list(panels.keys())
    T = len(dates_index)
    N = len(tickers)
    F = len(common_cols)

    # 1. Zbierz panel [T, N, F] surowych wartosci (przed jakimkolwiek skalerem)
    raw_panel = np.zeros((T, N, F), dtype=np.float64)
    for j, tk in enumerate(tickers):
        df = panels[tk]
        aligned = df.reindex(dates_index)[common_cols].apply(pd.to_numeric, errors="coerce")
        arr = aligned.to_numpy(dtype=np.float64)
        raw_panel[:, j, :] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Globalny scaler aplikowany do calego panelu (jako baza)
    flat = raw_panel.reshape(T * N, F)
    scaled = scaler.transform(flat).reshape(T, N, F)
    scaled = np.clip(scaled, -clip * 2, clip * 2)   # twarda granica

    # 3. Indeksy sektorow per ticker
    sector_ids = np.array([sector_map.get(tk, -1) for tk in tickers], dtype=np.int64)
    unique_sectors = sorted(set(sector_ids.tolist()) - {-1})

    # 4. Indeksy kolumn ktore podlegaja sektorowemu z-score
    std_col_idx = [i for i, c in enumerate(common_cols) if c in FUNDAMENTAL_SECTOR_STD_COLS]

    if sector_zscore and std_col_idx:
        # Dla kazdej daty t, dla kazdego sektora s, dla kazdej kolumny std:
        # licz mediane i IQR z RAW wartosci tej kolumny po spolkach w sektorze
        for s in unique_sectors:
            stock_mask = (sector_ids == s)
            n_in_sector = int(stock_mask.sum())
            if n_in_sector < min_sector_size:
                continue   # fallback do globalnego scalera dla calego sektora
            # raw[T, n_s, len(std_cols)] dla wybranych spolek i kolumn
            sub_raw = raw_panel[:, stock_mask, :][:, :, std_col_idx]  # [T, n_s, n_std]

            # Mediana po osi spolek (axis=1)
            med = np.nanmedian(sub_raw, axis=1)              # [T, n_std]
            q75 = np.nanpercentile(sub_raw, 75, axis=1)
            q25 = np.nanpercentile(sub_raw, 25, axis=1)
            iqr = q75 - q25                                   # [T, n_std]

            # Maska dat gdzie IQR jest sensowny dla danej kolumny
            iqr_ok = iqr > iqr_eps                            # [T, n_std]
            # Bezpieczny IQR (1.0 gdzie zly, nie wplynie bo zaraz zamaskujemy)
            iqr_safe = np.where(iqr_ok, iqr, 1.0)

            # z = (raw - med) / iqr_safe dla wybranych spolek i kolumn
            # broadcast: med [T, 1, n_std], iqr_safe [T, 1, n_std]
            z = (sub_raw - med[:, None, :]) / iqr_safe[:, None, :]
            z = np.clip(z, -clip, clip)
            z = np.where(np.isfinite(z), z, 0.0)

            # Nadpisz tylko tam gdzie IQR byl sensowny (broadcast po spolkach)
            for k, col_idx in enumerate(std_col_idx):
                date_mask = iqr_ok[:, k]                       # [T]
                if not date_mask.any():
                    continue
                # scaled[date_mask, stock_mask, col_idx] = z[date_mask, :, k]
                # Wymaga ostrozniejszego indeksowania:
                rows = np.where(date_mask)[0]
                cols = np.where(stock_mask)[0]
                for ridx in rows:
                    scaled[ridx, cols, col_idx] = z[ridx, :, k]

    # 5. Rozbij z powrotem na slownik per-ticker
    out = {}
    for j, tk in enumerate(tickers):
        out[tk] = scaled[:, j, :].astype(np.float32)
    return out


# ============================================================================
# PIPELINE WYSOKOPOZIOMOWY
# ============================================================================
def fit_all_scalers(raw: dict, cfg) -> tuple[FittedScalers, dict]:
    """Dopasuj wszystkie skalery i zwroc + listy wspolnych kolumn.

    Globalne skalery sa fitowane na train. Sektorowy z-score liczymy w transform
    cross-sectionally per date — nic do dopasowania, wiec brak go w FittedScalers.
    """
    from data_loader import get_common_features

    train_end = pd.Timestamp(cfg.data.train_end)

    macro_train = raw["macro"].loc[raw["macro"].index <= train_end]
    macro_scaler = fit_macro_scaler(macro_train, kind=cfg.data.macro_scaler)

    tech_cols = get_common_features(raw["technical_panels"])
    tech_scalers = fit_technical_per_ticker(
        raw["technical_panels"], train_end, tech_cols, kind=cfg.data.technical_scaler,
    )

    fund_cols = get_common_features(raw["fundamental_panels"])
    fund_scaler = fit_fundamental_scaler(
        raw["fundamental_panels"], train_end, fund_cols, kind=cfg.data.fundamental_scaler,
    )

    fitted = FittedScalers(
        macro_scaler=macro_scaler,
        technical_per_ticker=tech_scalers,
        fundamental_scaler=fund_scaler,
    )
    cols = {"technical": tech_cols, "fundamental": fund_cols,
            "macro": list(raw["macro"].columns)}
    return fitted, cols
