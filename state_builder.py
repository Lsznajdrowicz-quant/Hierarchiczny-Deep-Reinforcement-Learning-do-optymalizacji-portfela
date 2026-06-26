# ===== file: state_builder.py =====


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from preprocessing import SplitData


@dataclass
class LookbackTensors:
    """Tensory gotowe do podania srodowiskom."""
    macro: np.ndarray              # [T, L_macro, F_macro]
    fundamental: np.ndarray        # [T, N, L_fund, F_fund]
    technical: np.ndarray          # [T, N, L_tech, F_tech]
    returns: np.ndarray            # [T, N]
    dates: object                  # pd.DatetimeIndex odpowiadajacy returns
    tickers: list[str]


def _validate_lookback(T_full: int, lookback: int, name: str) -> None:
    if lookback <= 0:
        raise ValueError(f"{name}: lookback musi byc dodatni, dostalem {lookback}")
    if T_full < lookback:
        raise ValueError(
            f"{name}: za malo obserwacji do zbudowania lookbacku. "
            f"T_full={T_full}, lookback={lookback}"
        )


def build_macro_tensor(split: SplitData, lookback: int) -> np.ndarray:
    """[T_full, F] -> [T_full - L + 1, L, F]."""
    arr = np.asarray(split.macro, dtype=np.float32)
    T_full, F = arr.shape
    _validate_lookback(T_full, lookback, "macro")

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
    """[T_full, F] per ticker -> [T_full - L + 1, N, L, F]."""
    if not tickers:
        raise ValueError("Brak tickerow")

    if tickers[0] not in per_ticker_dict:
        raise KeyError(f"Brak danych dla pierwszego tickera: {tickers[0]}")

    sample = np.asarray(per_ticker_dict[tickers[0]], dtype=np.float32)
    T_full, F = sample.shape
    _validate_lookback(T_full, lookback, "per_ticker")

    N = len(tickers)
    T_out = T_full - lookback + 1
    out = np.zeros((T_out, N, lookback, F), dtype=np.float32)

    for j, tk in enumerate(tickers):
        if tk not in per_ticker_dict:
            raise KeyError(f"Brak danych dla tickera: {tk}")

        arr = np.asarray(per_ticker_dict[tk], dtype=np.float32)

        if arr.shape[0] != T_full:
            raise ValueError(
                f"Niezgodna liczba dat dla {tk}: {arr.shape[0]} vs {T_full}"
            )
        if arr.shape[1] != F:
            raise ValueError(
                f"Niezgodna liczba cech dla {tk}: {arr.shape[1]} vs {F}"
            )

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

    Wszystkie tensory sa wyrownane do tej samej daty koncowej okna.

    Przyklad:
        macro_lookback = 50 -> macro widzi t-49 ... t
        fund_lookback = 21  -> fund widzi t-20 ... t
        tech_lookback = 21  -> tech widzi t-20 ... t

    Pierwsza wspolna probka zaczyna sie od max_lookback - 1.
    Dla train jest to normalne. Dla val/test lepiej uzyc
    build_lookback_tensors_with_context().
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

    macro_drop = max_lb - macro_lookback
    fund_drop = max_lb - fund_lookback
    tech_drop = max_lb - tech_lookback

    macro = macro_full[macro_drop:]
    fundamental = fund_full[fund_drop:]
    technical = tech_full[tech_drop:]

    returns = split.stock_returns[max_lb - 1:]
    dates = split.dates[max_lb - 1:]

    if not (
        macro.shape[0]
        == fundamental.shape[0]
        == technical.shape[0]
        == returns.shape[0]
        == len(dates)
    ):
        raise AssertionError(
            f"Niezgodne dlugosci tensorow: "
            f"macro={macro.shape[0]}, fund={fundamental.shape[0]}, "
            f"tech={technical.shape[0]}, returns={returns.shape[0]}, dates={len(dates)}"
        )

    return LookbackTensors(
        macro=macro,
        fundamental=fundamental,
        technical=technical,
        returns=returns,
        dates=dates,
        tickers=split.tickers,
    )


def _slice_split_tail(split: SplitData, n: int) -> SplitData:
    """Zwroc ostatnie n obserwacji splitu jako SplitData."""
    if n <= 0:
        raise ValueError(f"n musi byc dodatnie, dostalem {n}")

    n = min(n, len(split.dates))

    return SplitData(
        dates=split.dates[-n:],
        macro=split.macro[-n:],
        technical_per_ticker={
            tk: arr[-n:]
            for tk, arr in split.technical_per_ticker.items()
        },
        fundamental_per_ticker={
            tk: arr[-n:]
            for tk, arr in split.fundamental_per_ticker.items()
        },
        stock_returns=split.stock_returns[-n:],
        tickers=split.tickers,
    )


def concat_splits(*splits: SplitData) -> SplitData:
    """Sklej kilka splitow czasowo.

    Uzywane glownie do zbudowania historycznego kontekstu lookbacku.
    Nie oznacza mieszania train/val/test do treningu modelu.

    Przyklad:
        train_val_context = concat_splits(dataset.train, dataset.val)
    """
    if len(splits) == 0:
        raise ValueError("concat_splits wymaga co najmniej jednego splitu")

    tickers = splits[0].tickers

    for s in splits:
        if s.tickers != tickers:
            raise ValueError("Nie mozna skleic splitow z roznymi tickerami")

    dates = pd.DatetimeIndex(np.concatenate([s.dates.values for s in splits]))
    macro = np.concatenate([s.macro for s in splits], axis=0)
    stock_returns = np.concatenate([s.stock_returns for s in splits], axis=0)

    technical_per_ticker = {
        tk: np.concatenate([s.technical_per_ticker[tk] for s in splits], axis=0)
        for tk in tickers
    }

    fundamental_per_ticker = {
        tk: np.concatenate([s.fundamental_per_ticker[tk] for s in splits], axis=0)
        for tk in tickers
    }

    return SplitData(
        dates=dates,
        macro=macro,
        technical_per_ticker=technical_per_ticker,
        fundamental_per_ticker=fundamental_per_ticker,
        stock_returns=stock_returns,
        tickers=tickers,
    )


def build_lookback_tensors_with_context(
    context_split: Optional[SplitData],
    target_split: SplitData,
    macro_lookback: int,
    fund_lookback: int,
    tech_lookback: int,
) -> LookbackTensors:
    """Zbuduj tensory dla target_split z historycznym kontekstem.

    Cel:
        Nie tracic pierwszych max_lookback - 1 dni walidacji/testu.

    Przyklad:
        validation:
            context_split = train
            target_split = val

        test:
            context_split = train+val
            target_split = test

    Dzialanie:
        1. Bierze ostatnie max_lookback - 1 obserwacji z context_split.
        2. Dokleja target_split.
        3. Buduje lookbacki na polaczonym zbiorze.
        4. Zwraca tylko probki odpowiadajace datom target_split.

    To nie jest data leakage, bo context_split jest historycznie wczesniejszy.
    """
    max_lb = max(macro_lookback, fund_lookback, tech_lookback)
    needed_context = max_lb - 1

    if context_split is None or needed_context <= 0:
        return build_lookback_tensors(
            target_split,
            macro_lookback=macro_lookback,
            fund_lookback=fund_lookback,
            tech_lookback=tech_lookback,
        )

    ctx_len = min(len(context_split.dates), needed_context)

    if ctx_len <= 0:
        return build_lookback_tensors(
            target_split,
            macro_lookback=macro_lookback,
            fund_lookback=fund_lookback,
            tech_lookback=tech_lookback,
        )

    context_tail = _slice_split_tail(context_split, ctx_len)
    combined = concat_splits(context_tail, target_split)

    tensors = build_lookback_tensors(
        combined,
        macro_lookback=macro_lookback,
        fund_lookback=fund_lookback,
        tech_lookback=tech_lookback,
    )

    expected_len = len(target_split.dates)

    # Przy pelnym kontekscie len(tensors) == len(target_split).
    # Przy krotszym kontekscie moze byc mniej probek.
    if tensors.returns.shape[0] > expected_len:
        start = tensors.returns.shape[0] - expected_len
        tensors = LookbackTensors(
            macro=tensors.macro[start:],
            fundamental=tensors.fundamental[start:],
            technical=tensors.technical[start:],
            returns=tensors.returns[start:],
            dates=tensors.dates[start:],
            tickers=tensors.tickers,
        )

    if len(tensors.dates) == 0:
        raise ValueError("Po zbudowaniu lookback_tensors_with_context brak dat")

    if tensors.dates[-1] != target_split.dates[-1]:
        raise ValueError(
            f"Niepoprawne wyrownanie dat: ostatnia data tensorow "
            f"{tensors.dates[-1]} != ostatnia data target_split {target_split.dates[-1]}"
        )

    return tensors
