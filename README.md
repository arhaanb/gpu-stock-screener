# GPU Stock Screener

A CUDA-accelerated stock screener with a rolling backtest. You give it a list of tickers. It downloads price history, scores every stock in parallel on the GPU using a chosen strategy, writes a ranked leaderboard, and can backtest the strategy month by month against SPY to show whether it actually works.

Built as the capstone project for the Coursera GPU Specialization.

## What it does

1. Loads a universe of tickers (default is about 100 large cap US stocks).
2. Downloads daily adjusted close prices from Yahoo Finance and caches them locally.
3. Scores every stock on the GPU using one of five simple strategies.
4. Writes a ranked leaderboard CSV and three plots: a risk vs return scatter, a sector score heatmap, and a price grid of the top picks.
5. Optionally runs a rolling rebalance backtest: at every rebalance date it scores the universe, equal weights the top N, holds until the next rebalance, and compares the resulting equity curve to SPY buy and hold.
6. Renders an animated bar chart GIF showing how the top picks shift over time.

## Why GPU

The scoring step is the classic embarrassingly parallel workload. For every stock you take a lookback window of daily returns and reduce it to a single number (mean, stddev, ratio). Using CuPy, a single line like `xp.mean(returns, axis=1)` dispatches one GPU thread per stock and reduces them all in parallel.

The speedup grows when you backtest. A five year backtest with a 21 day rebalance window runs the screener around 60 times, which is 60 rebalances times 100 stocks times several reductions each. On a Colab T4 this finishes in well under a second. On CPU with NumPy it is clearly slower. The code has a CuPy or NumPy fallback so the same file runs on either backend.

## Strategies

| name              | idea                                             | score formula                          |
|-------------------|--------------------------------------------------|----------------------------------------|
| `sharpe`          | steady winners                                   | annualised mean return / annualised volatility |
| `momentum`        | biggest recent gainers                           | sum of log returns over the lookback   |
| `low_vol`         | boring, steady stocks                            | 1 / annualised volatility              |
| `mean_reversion`  | bet biggest recent losers bounce back            | negative sum of log returns            |
| `breakout`        | stocks sitting near a recent high                | (latest price - window high) / window high |

## Project structure

```
.
├── src/
│   ├── screener.py      main cli entry point
│   ├── gpu_utils.py     cupy or numpy backend shim
│   ├── data.py          ticker loader and yfinance cache
│   ├── strategies.py    gpu scoring functions
│   ├── backtest.py      rolling rebalance backtest
│   └── plots.py         scatter, heatmap, equity curve, price grid, gif
├── data/
│   ├── tickers/sp100.txt  ticker,sector universe
│   └── cache/             yfinance csv cache (gitignored)
├── output/                leaderboards, metrics, equity curves
├── execution_artefacts/   png plots and leaderboard gif
├── notebooks/
│   └── gpu_screener_colab.ipynb   colab runner
├── Makefile
├── run.sh
├── requirements.txt
└── README.md
```

## Quick start

### Local

```bash
pip install -r requirements.txt
./run.sh
```

The first run downloads price history from Yahoo Finance and caches it under `data/cache/`. Subsequent runs use the cache.

### Google Colab

Open `notebooks/gpu_screener_colab.ipynb` in Colab with a GPU runtime selected. The first cell clones or pulls the repo, the second cell cleans old outputs, and the following cells install dependencies and run the screener.

## CLI

```bash
python -m src.screener --help
```

Key flags:

| flag | default | description |
|------|---------|-------------|
| `--tickers` | `data/tickers/sp100.txt` | ticker,sector file |
| `--strategy` | `sharpe` | one of `sharpe`, `momentum`, `low_vol`, `mean_reversion`, `breakout`, or `all` |
| `--lookback` | `60` | trading days used to score each stock |
| `--top` | `20` | leaderboard size |
| `--start`, `--end` | 2019-01-01, 2024-12-31 | history date range |
| `--benchmark` | `SPY` | ticker used as the buy and hold benchmark |
| `--backtest` | off | run the rolling rebalance backtest |
| `--rebalance` | `21` | backtest rebalance period in trading days |
| `--no-gif` | off | skip the animated leaderboard gif |
| `--force-download` | off | ignore the price cache and re-download |
| `-v`, `--verbose` | off | print per step timings and the leaderboard table |

Examples:

```bash
# single strategy, no backtest
python -m src.screener --strategy momentum --lookback 90 --top 15 -v

# every strategy with the rolling backtest
python -m src.screener --strategy all --backtest --rebalance 21 -v

# different time window and lookback
python -m src.screener --strategy sharpe --start 2020-01-01 --end 2023-12-31 --lookback 120 --backtest -v
```

## Make targets

```bash
make install         # install python deps
make run             # run one strategy with no backtest
make backtest        # run one strategy with the rolling backtest
make all-strategies  # run every strategy with the backtest
make clean           # remove outputs and plots
make reset           # clean plus drop the price cache
```

Override variables from the command line:

```bash
make run STRATEGY=momentum LOOKBACK=90 TOP=10
make backtest STRATEGY=low_vol REBALANCE=63
```

## Outputs

`output/` contains:

- `leaderboard_<strategy>.csv` the ranked top N with scores, annualised return, annualised volatility, sector.
- `equity_curve_<strategy>.csv` the strategy vs benchmark portfolio value over time.
- `metrics_<strategy>.json` CAGR, Sharpe, total return, max drawdown for strategy and benchmark.

`execution_artefacts/` contains:

- `scatter_<strategy>.png` every stock plotted as risk vs return, top picks highlighted.
- `sector_heatmap_<strategy>.png` mean score per sector.
- `price_grid_<strategy>.png` small price charts of every top pick.
- `equity_curve_<strategy>.png` strategy vs SPY equity curve with metrics in the title.
- `leaderboard_<strategy>.gif` animated bar chart race of the top picks over time.

## Lessons learned

- For a screener this small, the GPU scoring itself is almost instant. The interesting wall clock is the backtest loop, which runs the scoring dozens of times.
- CuPy is the right level of abstraction for a beginner GPU project. You get automatic parallelism from basic NumPy style calls without writing a raw CUDA kernel. The `xp = cupy or numpy` pattern means the same code runs locally for debugging and on Colab for speed.
- yfinance is fast enough for this project size but rate limits if you hit it too often, so aggressive caching matters.
- Filling the first day of returns with NaN and then dropping that row is simpler than handling the edge case inside the GPU kernel.
- The rolling backtest is what turns a toy calculator into a research tool. Without it the leaderboard is just a snapshot and you cannot tell if the strategy works.

## Next steps

- Pull in volume and earnings data for a breadth of signals.
- Add a proper portfolio model with transaction costs and position sizing.
- Try a composite strategy that blends several scores with user weights.
- Replace yfinance with a paid feed for intraday data.
- Wrap the whole thing in a tiny Streamlit app for interactive demos.
