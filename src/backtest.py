"""rolling rebalance backtest of the screener.

we step through history in chunks of ``rebalance_days`` trading days.
at each step we look back ``lookback`` days, score every stock on the
gpu in parallel, buy an equal weight basket of the top n, hold for the
next chunk, and record the return. an spy buy and hold curve is tracked
alongside as a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .gpu_utils import to_numpy, to_xp, xp
from .strategies import compute_scores, rank_top_n


@dataclass
class BacktestResult:
    dates: List[pd.Timestamp]
    equity_curve: np.ndarray
    benchmark_curve: np.ndarray
    top_picks_history: List[List[str]]
    picks_scores_history: List[np.ndarray]
    metrics: dict = field(default_factory=dict)


def run_backtest(
    prices: pd.DataFrame,
    strategy: str,
    lookback: int,
    top_n: int,
    rebalance_days: int,
    benchmark_ticker: str = "SPY",
) -> BacktestResult:
    tickers_all = list(prices.columns)
    selectable = [t for t in tickers_all if t != benchmark_ticker]
    if not selectable:
        raise ValueError("no selectable tickers after removing benchmark")

    prices_sel = prices[selectable]
    price_array = to_xp(prices_sel.values.T.astype(np.float64))
    log_prices = xp.log(price_array + 1e-12)

    dates = prices_sel.index
    num_days = len(dates)

    rebalance_indices = list(range(lookback, num_days - 1, rebalance_days))
    if not rebalance_indices:
        raise ValueError(
            f"not enough history: need > {lookback + rebalance_days} days, "
            f"got {num_days}"
        )

    strat_equity = [1.0]
    bench_equity = [1.0]
    curve_dates = [dates[rebalance_indices[0]]]
    top_picks_history: List[List[str]] = []
    picks_scores_history: List[np.ndarray] = []

    bench_prices = None
    if benchmark_ticker in prices.columns:
        bench_prices = prices[benchmark_ticker].values

    for idx in rebalance_indices:
        window_prices = price_array[:, idx - lookback : idx]
        window_log = log_prices[:, idx - lookback : idx]
        window_returns = xp.diff(window_log, axis=1)

        scores = compute_scores(window_returns, window_prices, strategy)
        top_idx = rank_top_n(scores, top_n)
        top_idx_cpu = to_numpy(top_idx).astype(int)
        top_scores_cpu = to_numpy(scores[top_idx])

        top_picks_history.append([selectable[i] for i in top_idx_cpu])
        picks_scores_history.append(top_scores_cpu)

        end_idx = min(idx + rebalance_days, num_days - 1)
        period_start = price_array[top_idx, idx]
        period_end = price_array[top_idx, end_idx]
        period_returns = period_end / period_start - 1.0
        portfolio_return = float(to_numpy(xp.mean(period_returns)))
        strat_equity.append(strat_equity[-1] * (1.0 + portfolio_return))

        if bench_prices is not None:
            bench_ret = bench_prices[end_idx] / bench_prices[idx] - 1.0
            bench_equity.append(bench_equity[-1] * (1.0 + float(bench_ret)))
        else:
            bench_equity.append(bench_equity[-1])

        curve_dates.append(dates[end_idx])

    metrics = _compute_metrics(strat_equity, bench_equity, rebalance_days)

    return BacktestResult(
        dates=curve_dates,
        equity_curve=np.array(strat_equity),
        benchmark_curve=np.array(bench_equity),
        top_picks_history=top_picks_history,
        picks_scores_history=picks_scores_history,
        metrics=metrics,
    )


def _compute_metrics(strat_equity, bench_equity, rebalance_days) -> dict:
    strat = np.array(strat_equity)
    bench = np.array(bench_equity)

    periods = max(len(strat) - 1, 1)
    years = max(periods * rebalance_days / 252.0, 1e-6)

    cagr = strat[-1] ** (1.0 / years) - 1.0
    bench_cagr = bench[-1] ** (1.0 / years) - 1.0

    strat_returns = np.diff(strat) / strat[:-1]
    if strat_returns.std() > 0:
        periods_per_year = 252.0 / rebalance_days
        sharpe = float(strat_returns.mean() / strat_returns.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    return {
        "total_return": float(strat[-1] - 1.0),
        "benchmark_total_return": float(bench[-1] - 1.0),
        "cagr": float(cagr),
        "benchmark_cagr": float(bench_cagr),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(strat),
        "benchmark_max_drawdown": _max_drawdown(bench),
        "num_rebalances": int(periods),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    return float(drawdowns.min())
