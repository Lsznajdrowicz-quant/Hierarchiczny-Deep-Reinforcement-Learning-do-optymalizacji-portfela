# file: state_builder.py
"""
Konwersja SplitData na tensory lookbackow uzywane przez 3 srodowiska RL.

  Macro state:        [T, L, F_macro]
  Fundamental state:  [T, N, L, F_fund]
  Technical state:    [T, N, L, F_tech]
  Returns matrix:     [T, N]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocessing import SplitData


@dataclass
class LookbackTensors:
    """Tensory gotowe do podania srodowiskom."""
    macro: np.ndarray              # [T, L, F_macro]
    fundamental: np.ndarray        # [T, N, L, F_fund]
    technical: np.ndarray          # [T, N, L, F_tech]
    returns: np.ndarray            # [T, N]
    dates: object                  # pd.DatetimeIndex (od L-1 do konca)
    tickers: list[str]


def build_macro_tensor(split: SplitData, lookback: int) -> np.ndarray:
    """[T_full, F] -> [T_full - L + 1, L, F]"""
    arr = split.macro
    T_full, F = arr.shape
    T_out = T_full - lookback + 1
    out = np.zeros((T_out, lookback, F), dtype=np.float32)
    for t in range(T_out):
        out[t] = arr[t:t + lookback]
    return out


def build_per_ticker_tensor(
    per_ticker_dict: dict[str, np.ndarray],
    tickers: list[str],
    lookback: int,
) -> np.ndarray:
    """[T_full, F] per ticker -> [T_full - L + 1, N, L, F]"""
    # Wymiary z pierwszej spolki
    sample = per_ticker_dict[tickers[0]]
    T_full, F = sample.shape
    N = len(tickers)
    T_out = T_full - lookback + 1
    out = np.zeros((T_out, N, lookback, F), dtype=np.float32)
    for j, tk in enumerate(tickers):
        arr = per_ticker_dict[tk]
        for t in range(T_out):
            out[t, j] = arr[t:t + lookback]
    return out


def build_lookback_tensors(
    split: SplitData,
    macro_lookback: int,
    fund_lookback: int,
    tech_lookback: int,
) -> LookbackTensors:
    """Zbuduj tensory lookbackowe z osobnym lookbackiem dla macro/fund/tech.

    Wszystkie tensory są wyrównane do tej samej daty końcowej okna.
    Przykład:
        macro_lookback = 50 -> macro widzi t-49 ... t
        fund_lookback = 21  -> fund widzi t-20 ... t
        tech_lookback = 21  -> tech widzi t-20 ... t

    Pierwsza wspólna próbka zaczyna się od max_lookback - 1.
    """

    max_lb = max(macro_lookback, fund_lookback, tech_lookback)

    macro_full = build_macro_tensor(split, macro_lookback)
    fund_full = build_per_ticker_tensor(
        split.fundamental_per_ticker,
        split.tickers,
        fund_lookback,
    )
    tech_full = build_per_ticker_tensor(
        split.technical_per_ticker,
        split.tickers,
        tech_lookback,
    )

    # Ile początkowych próbek trzeba wyrzucić, żeby wszystkie okna kończyły się na tej samej dacie.
    macro_drop = max_lb - macro_lookback
    fund_drop = max_lb - fund_lookback
    tech_drop = max_lb - tech_lookback

    macro = macro_full[macro_drop:]
    fund = fund_full[fund_drop:]
    tech = tech_full[tech_drop:]

    # Zwroty i daty też zaczynają się od końca najdłuższego okna.
    rets = split.stock_returns[max_lb - 1:]
    dates = split.dates[max_lb - 1:]

    assert macro.shape[0] == fund.shape[0] == tech.shape[0] == rets.shape[0], (
        f"Niezgodne długości tensorów: "
        f"macro={macro.shape[0]}, fund={fund.shape[0]}, "
        f"tech={tech.shape[0]}, returns={rets.shape[0]}"
    )

    return LookbackTensors(
        macro=macro,
        fundamental=fund,
        technical=tech,
        returns=rets,
        dates=dates,
        tickers=split.tickers,
    )
