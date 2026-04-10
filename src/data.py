"""ticker loading and yfinance price download with on-disk caching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/cache")


def load_tickers_with_sectors(path: Path) -> Tuple[List[str], Dict[str, str]]:
    tickers: List[str] = []
    sectors: Dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                ticker, sector = line.split(",", 1)
                ticker, sector = ticker.strip().upper(), sector.strip()
            else:
                ticker, sector = line.upper(), "Unknown"
            if ticker in sectors:
                continue
            tickers.append(ticker)
            sectors[ticker] = sector
    return tickers, sectors


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prices_{start}_{end}.csv"

    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            return cached[tickers]
        print(f"cache hit is incomplete ({len(missing)} missing), re-downloading full set")

    import yfinance as yf

    print(f"downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=-1).copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="all")
    prices = prices.dropna(axis=0, how="any")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    prices.to_csv(cache_path)
    print(f"cached to {cache_path}  (shape={prices.shape})")
    return prices


def prices_to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()
