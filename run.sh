#!/usr/bin/env bash
# one command install plus full backtest run
set -euo pipefail

cd "$(dirname "$0")"

python3 -m pip install -q -r requirements.txt

python3 -m src.screener \
    --tickers data/tickers/sp100.txt \
    --strategy all \
    --lookback 60 \
    --top 20 \
    --start 2019-01-01 \
    --end 2024-12-31 \
    --backtest \
    --rebalance 21 \
    --verbose
