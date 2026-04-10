"""gpu scoring strategies.

each strategy takes a (num_stocks, lookback_days) returns matrix and a
(num_stocks, lookback_days) prices matrix and returns a 1d score array
with one value per stock. every call runs on cupy when a gpu is present
and on numpy otherwise, via the shared ``xp`` alias.
"""

from typing import Dict

from .gpu_utils import xp

STRATEGIES = ("sharpe", "momentum", "low_vol", "mean_reversion", "breakout")
TRADING_DAYS_PER_YEAR = 252


def _annualize_return(mean_daily):
    return mean_daily * TRADING_DAYS_PER_YEAR


def _annualize_vol(std_daily):
    return std_daily * xp.sqrt(TRADING_DAYS_PER_YEAR)


def score_sharpe(returns, prices):
    mean_r = xp.mean(returns, axis=1)
    std_r = xp.std(returns, axis=1) + 1e-9
    return _annualize_return(mean_r) / _annualize_vol(std_r)


def score_momentum(returns, prices):
    return xp.sum(returns, axis=1)


def score_low_vol(returns, prices):
    std_r = xp.std(returns, axis=1) + 1e-9
    return 1.0 / _annualize_vol(std_r)


def score_mean_reversion(returns, prices):
    return -xp.sum(returns, axis=1)


def score_breakout(returns, prices):
    latest = prices[:, -1]
    window_high = xp.max(prices, axis=1) + 1e-9
    return (latest - window_high) / window_high


_STRATEGY_FUNCS: Dict[str, object] = {
    "sharpe": score_sharpe,
    "momentum": score_momentum,
    "low_vol": score_low_vol,
    "mean_reversion": score_mean_reversion,
    "breakout": score_breakout,
}


def compute_scores(returns, prices, strategy: str):
    if strategy not in _STRATEGY_FUNCS:
        raise ValueError(
            f"unknown strategy '{strategy}'. available: {sorted(_STRATEGY_FUNCS)}"
        )
    return _STRATEGY_FUNCS[strategy](returns, prices)


def rank_top_n(scores, n: int):
    order = xp.argsort(-scores)
    return order[:n]
