"""main cli entry point for the gpu stock screener."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import download_prices, load_tickers_with_sectors
from .gpu_utils import backend_name, gpu_available, to_numpy, to_xp, xp
from .plots import (
    plot_equity_curve,
    plot_leaderboard_gif,
    plot_price_grid,
    plot_risk_return_scatter,
    plot_sector_heatmap,
)
from .strategies import (
    STRATEGIES,
    TRADING_DAYS_PER_YEAR,
    compute_scores,
    rank_top_n,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="gpu stock screener with rolling backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", type=Path, default=Path("data/tickers/sp100.txt"),
                        help="text file with TICKER,SECTOR lines")
    parser.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="sharpe",
                        help="scoring strategy, 'all' runs every strategy")
    parser.add_argument("--lookback", type=int, default=60,
                        help="lookback window in trading days")
    parser.add_argument("--top", type=int, default=20,
                        help="number of top stocks to report")
    parser.add_argument("--start", type=str, default="2019-01-01",
                        help="history start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="history end date YYYY-MM-DD")
    parser.add_argument("--benchmark", type=str, default="SPY",
                        help="benchmark ticker, excluded from scoring")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="directory for csv and json outputs")
    parser.add_argument("--artefacts", type=Path, default=Path("execution_artefacts"),
                        help="directory for png and gif plots")
    parser.add_argument("--backtest", action="store_true",
                        help="run the rolling backtest")
    parser.add_argument("--rebalance", type=int, default=21,
                        help="rebalance period in trading days for the backtest")
    parser.add_argument("--no-gif", action="store_true",
                        help="skip animated leaderboard gif")
    parser.add_argument("--force-download", action="store_true",
                        help="ignore price cache and re-download")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def _current_scores(
    prices: pd.DataFrame,
    strategy: str,
    lookback: int,
    tickers_selectable: List[str],
):
    prices_sel = prices[tickers_selectable]
    price_array = to_xp(prices_sel.values.T.astype(np.float64))
    log_prices = xp.log(price_array + 1e-12)
    window_prices = price_array[:, -lookback:]
    window_log = log_prices[:, -lookback:]
    window_returns = xp.diff(window_log, axis=1)
    scores = compute_scores(window_returns, window_prices, strategy)
    return scores, window_returns, window_prices


def run_strategy(
    args: argparse.Namespace,
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    tickers_selectable: List[str],
    strategy: str,
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    args.artefacts.mkdir(parents=True, exist_ok=True)

    _log(f"[{strategy}] scoring on {backend_name()}", args.verbose)
    start = time.time()
    scores, window_returns, _ = _current_scores(
        prices, strategy, args.lookback, tickers_selectable
    )
    top_idx = rank_top_n(scores, args.top)
    scores_cpu = to_numpy(scores)
    top_idx_cpu = to_numpy(top_idx).astype(int)
    mean_r = to_numpy(xp.mean(window_returns, axis=1)) * TRADING_DAYS_PER_YEAR
    vol_r = to_numpy(xp.std(window_returns, axis=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    elapsed = time.time() - start
    _log(
        f"[{strategy}] scored {len(tickers_selectable)} tickers in {elapsed*1000:.1f} ms",
        args.verbose,
    )

    leaderboard = pd.DataFrame(
        {
            "rank": range(1, len(top_idx_cpu) + 1),
            "ticker": [tickers_selectable[i] for i in top_idx_cpu],
            "sector": [sectors.get(tickers_selectable[i], "Unknown") for i in top_idx_cpu],
            "score": scores_cpu[top_idx_cpu],
            "annualized_return": mean_r[top_idx_cpu],
            "annualized_volatility": vol_r[top_idx_cpu],
        }
    )
    csv_path = args.output / f"leaderboard_{strategy}.csv"
    leaderboard.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[{strategy}] wrote leaderboard to {csv_path}")
    if args.verbose:
        print(leaderboard.to_string(index=False))

    plot_risk_return_scatter(
        tickers_selectable, mean_r, vol_r, top_idx_cpu, strategy,
        args.artefacts / f"scatter_{strategy}.png",
    )
    plot_sector_heatmap(
        tickers_selectable, sectors, scores_cpu, strategy,
        args.artefacts / f"sector_heatmap_{strategy}.png",
    )
    plot_price_grid(
        prices, [tickers_selectable[i] for i in top_idx_cpu], strategy,
        args.artefacts / f"price_grid_{strategy}.png",
    )
    _log(f"[{strategy}] wrote scatter, sector heatmap, price grid", args.verbose)

    if args.backtest:
        _log(
            f"[{strategy}] running backtest (lookback={args.lookback}, "
            f"rebalance={args.rebalance})",
            args.verbose,
        )
        start = time.time()
        result = run_backtest(
            prices=prices,
            strategy=strategy,
            lookback=args.lookback,
            top_n=args.top,
            rebalance_days=args.rebalance,
            benchmark_ticker=args.benchmark,
        )
        elapsed = time.time() - start
        _log(
            f"[{strategy}] backtest done in {elapsed:.2f}s "
            f"({result.metrics['num_rebalances']} rebalances)",
            args.verbose,
        )

        equity_df = pd.DataFrame(
            {
                "date": result.dates,
                "strategy_equity": result.equity_curve,
                "benchmark_equity": result.benchmark_curve,
            }
        )
        equity_path = args.output / f"equity_curve_{strategy}.csv"
        equity_df.to_csv(equity_path, index=False, float_format="%.6f")

        metrics_path = args.output / f"metrics_{strategy}.json"
        with open(metrics_path, "w") as fh:
            json.dump(result.metrics, fh, indent=2, default=float)

        plot_equity_curve(
            result.dates, result.equity_curve, result.benchmark_curve, strategy,
            result.metrics, args.artefacts / f"equity_curve_{strategy}.png",
        )

        if not args.no_gif and len(result.top_picks_history) >= 2:
            _log(f"[{strategy}] rendering leaderboard gif", args.verbose)
            plot_leaderboard_gif(
                result.top_picks_history, result.picks_scores_history, result.dates,
                strategy, args.artefacts / f"leaderboard_{strategy}.gif",
                top_n=min(10, args.top),
            )

        print(f"\n[{strategy}] backtest metrics:")
        for key, val in result.metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"gpu stock screener, backend: {backend_name()}")
    if not gpu_available():
        print("  (no cuda gpu detected, running on cpu via numpy)")

    tickers, sectors = load_tickers_with_sectors(args.tickers)
    print(f"loaded {len(tickers)} tickers from {args.tickers}")

    all_needed = sorted(set(tickers + [args.benchmark]))
    prices = download_prices(
        all_needed, args.start, args.end, force=args.force_download
    )
    print(f"price matrix shape: {prices.shape}")

    tickers_selectable = [
        t for t in tickers if t in prices.columns and t != args.benchmark
    ]
    if args.benchmark not in prices.columns:
        print(
            f"warning: benchmark {args.benchmark} not in price data, "
            "backtest comparison will be flat"
        )

    strategies_to_run = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    for strat in strategies_to_run:
        run_strategy(args, prices, sectors, tickers_selectable, strat)

    print("\ndone.")


if __name__ == "__main__":
    main()
