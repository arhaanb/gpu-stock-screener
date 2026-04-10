"""matplotlib visualizations for the screener and backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_risk_return_scatter(
    tickers: Sequence[str],
    mean_returns_annual: np.ndarray,
    vol_annual: np.ndarray,
    top_indices: np.ndarray,
    strategy: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.ones(len(tickers), dtype=bool)
    mask[top_indices] = False

    ax.scatter(
        vol_annual[mask],
        mean_returns_annual[mask],
        s=30, c="#888", alpha=0.6, label="all stocks",
    )
    ax.scatter(
        vol_annual[top_indices],
        mean_returns_annual[top_indices],
        s=80, c="#e24a4a", edgecolors="black", linewidths=0.6,
        label=f"top {len(top_indices)}",
    )
    for i in top_indices[: min(10, len(top_indices))]:
        ax.annotate(
            tickers[i],
            (vol_annual[i], mean_returns_annual[i]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized return")
    ax.set_title(f"risk vs return, strategy: {strategy}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sector_heatmap(
    tickers: Sequence[str],
    sectors: Dict[str, str],
    scores: np.ndarray,
    strategy: str,
    out_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "ticker": list(tickers),
            "sector": [sectors.get(t, "Unknown") for t in tickers],
            "score": np.asarray(scores),
        }
    )
    agg = df.groupby("sector")["score"].agg(["mean", "count"]).sort_values("mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    span = agg["mean"].max() - agg["mean"].min() + 1e-9
    colors = plt.cm.RdYlGn((agg["mean"] - agg["mean"].min()) / span)
    bars = ax.barh(agg.index, agg["mean"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  n={int(row['count'])}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("mean score")
    ax.set_title(f"sector breakdown, strategy: {strategy}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_equity_curve(
    dates: Sequence,
    strat_curve: np.ndarray,
    bench_curve: np.ndarray,
    strategy: str,
    metrics: dict,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates, strat_curve, label=f"screener ({strategy})", linewidth=2, color="#2c7bb6")
    ax.plot(dates, bench_curve, label="SPY benchmark", linewidth=2, color="#d7191c", linestyle="--")
    ax.set_ylabel("portfolio value (start = 1.0)")
    ax.set_title(
        f"backtest equity curve, {strategy}\n"
        f"strategy cagr {metrics['cagr']*100:.1f}%, "
        f"benchmark cagr {metrics['benchmark_cagr']*100:.1f}%, "
        f"sharpe {metrics['sharpe']:.2f}, "
        f"max dd {metrics['max_drawdown']*100:.1f}%"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_price_grid(
    prices: pd.DataFrame,
    top_tickers: Sequence[str],
    strategy: str,
    out_path: Path,
) -> None:
    n = len(top_tickers)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for ax, ticker in zip(axes, top_tickers):
        if ticker not in prices.columns:
            ax.axis("off")
            continue
        series = prices[ticker].dropna()
        ax.plot(series.index, series.values, color="#2c7bb6", linewidth=1.2)
        ax.set_title(ticker, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[len(top_tickers):]:
        ax.axis("off")
    fig.suptitle(f"top {n} picks, strategy: {strategy}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_leaderboard_gif(
    history: List[List[str]],
    scores_history: List[np.ndarray],
    dates: Sequence,
    strategy: str,
    out_path: Path,
    top_n: int = 10,
    fps: int = 2,
) -> None:
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("matplotlib animation unavailable, skipping leaderboard gif")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    def draw(frame_idx: int):
        ax.clear()
        frame_tickers = history[frame_idx][:top_n]
        frame_scores = scores_history[frame_idx][:top_n]
        order = np.argsort(frame_scores)
        tickers_sorted = [frame_tickers[i] for i in order]
        scores_sorted = frame_scores[order]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tickers_sorted)))
        ax.barh(tickers_sorted, scores_sorted, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("score")
        frame_date = pd.Timestamp(dates[frame_idx + 1]).date()
        ax.set_title(f"top {top_n} picks, {strategy}, {frame_date}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(history), interval=1000 / fps)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
