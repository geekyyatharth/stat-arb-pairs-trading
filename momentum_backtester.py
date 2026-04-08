# momentum_backtester.py
# Responsibility: Cross-sectional momentum strategy on Nifty IT universe.
# Signal: 12-1 momentum (12-month return, skip last month)
# Portfolio: Top 3 long / Bottom 3 short, equal weight within each leg
# Rebalancing: Monthly

import numpy as np
import pandas as pd
import logging
from config import INITIAL_CAPITAL, TOTAL_COST_PER_SIDE

logger = logging.getLogger(__name__)

# --- Strategy Parameters ---
LOOKBACK_DAYS  = 252   # 12 months of trading days
SKIP_DAYS      = 21    # Skip last 1 month (short-term reversal effect)
N_LONG         = 3     # Number of stocks in long book
N_SHORT        = 3     # Number of stocks in short book


def calculate_momentum_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-1 momentum score for each stock on each day.

    Signal = (Price[t - SKIP_DAYS]) / (Price[t - LOOKBACK_DAYS]) - 1

    We use price from 21 days ago (not today) as the numerator
    to avoid the short-term reversal effect.

    Returns
    -------
    pd.DataFrame : same shape as prices, values are momentum scores
    """
    # Price 21 days ago (numerator — skip last month)
    price_recent = prices.shift(SKIP_DAYS)

    # Price 252 days ago (denominator — 12 months back)
    price_past   = prices.shift(LOOKBACK_DAYS)

    # Momentum score = return over the formation period
    momentum = (price_recent / price_past) - 1

    return momentum


def generate_monthly_positions(
    prices: pd.DataFrame,
    momentum: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate daily position matrix based on monthly rebalancing.

    On the first trading day of each month:
        - Rank all stocks by momentum score
        - Long top N_LONG, Short bottom N_SHORT
        - Equal weight within each leg: weight = 1/N

    Position values:
        +1/N_LONG  = long position (fraction of long book capital)
        -1/N_SHORT = short position (fraction of short book capital)
         0         = no position

    Returns
    -------
    pd.DataFrame : same shape as prices, values are position weights
    """
    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Identify first trading day of each month
    month_starts = prices.resample("MS").first().index
    # Filter to dates that actually exist in our price data
    rebalance_dates = [d for d in month_starts if d in prices.index]

    current_position = pd.Series(0.0, index=prices.columns)

    for i, date in enumerate(prices.index):
        # Check if today is a rebalance day
        if date in rebalance_dates:
            scores = momentum.loc[date].dropna()

            # Need at least N_LONG + N_SHORT stocks with valid scores
            if len(scores) >= N_LONG + N_SHORT:
                ranked = scores.sort_values(ascending=False)

                new_position = pd.Series(0.0, index=prices.columns)

                # Long leg: equal weight across top N_LONG
                for ticker in ranked.index[:N_LONG]:
                    new_position[ticker] = 1.0 / N_LONG

                # Short leg: equal weight across bottom N_SHORT (negative)
                for ticker in ranked.index[-N_SHORT:]:
                    new_position[ticker] = -1.0 / N_SHORT

                current_position = new_position

        positions.loc[date] = current_position

    return positions


def run_momentum_backtest(
    prices: pd.DataFrame,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full momentum backtest.

    Capital Allocation:
        Long book  = capital / 2  (split equally across N_LONG stocks)
        Short book = capital / 2  (split equally across N_SHORT stocks)

    Daily PnL:
        For each stock i:
            pnl_i = position_weight_i * (capital/2) * daily_return_i

        Long leg pnl  = sum of long positions
        Short leg pnl = sum of short positions (negative weight * return)
        Total pnl     = long pnl + short pnl

    Transaction Costs:
        Applied on rebalance days proportional to turnover.
        Turnover = sum of |new_weight - old_weight| across all stocks.
        Cost = turnover * TOTAL_COST_PER_SIDE * capital
    """

    # --- 1. MOMENTUM SIGNAL ---
    momentum = calculate_momentum_signal(prices)

    # --- 2. POSITION MATRIX ---
    positions = generate_monthly_positions(prices, momentum)

    # --- 3. DAILY RETURNS ---
    daily_returns = prices.pct_change().fillna(0)

    # --- 4. DAILY PnL ---
    # Each position weight × half capital × stock return
    # Long leg uses capital/2, short leg uses capital/2
    # The sign of position handles direction automatically:
    #   +weight × positive return = profit (long, stock goes up)
    #   -weight × positive return = loss  (short, stock goes up against us)

    cap_per_leg = capital / 2

    # Shift positions by 1 day — we trade on close, positions effective next day
    pos_shifted = positions.shift(1).fillna(0)

    # PnL matrix: each cell = position_weight × cap_per_leg × return
    pnl_matrix = pos_shifted * cap_per_leg * daily_returns
    gross_pnl  = pnl_matrix.sum(axis=1)   # Sum across all stocks each day

    # --- 5. TRANSACTION COSTS ---
    # Turnover = absolute change in weights on rebalance days
    pos_diff  = positions.diff().abs().sum(axis=1)
    cost      = pos_diff * TOTAL_COST_PER_SIDE * capital
    net_pnl   = gross_pnl - cost

    # --- 6. EQUITY CURVE ---
    equity = capital + net_pnl.cumsum()

    # --- 7. RESULTS ---
    results = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "cost":      cost,
        "net_pnl":   net_pnl,
        "equity":    equity,
    })

    metrics = calculate_momentum_metrics(results, capital)
    logger.info(
        f"Momentum | Sharpe={metrics['sharpe']:.2f} | "
        f"MDD={metrics['max_drawdown_pct']:.1f}% | "
        f"Return={metrics['total_return_pct']:.1f}%"
    )

    return results, metrics


def calculate_momentum_metrics(
    results: pd.DataFrame,
    capital: float,
) -> dict:
    """Standard performance metrics — same methodology as backtester.py."""

    daily_ret = results["net_pnl"] / capital

    # Sharpe
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    # Sortino
    downside     = daily_ret[daily_ret < 0]
    sortino      = (daily_ret.mean() / downside.std() * np.sqrt(252)
                    if downside.std() > 0 else 0.0)

    # MDD
    equity       = results["equity"]
    peak         = equity.cummax()
    drawdown     = (equity - peak) / peak
    max_dd       = drawdown.min()

    # Total return
    total_return = (equity.iloc[-1] - capital) / capital * 100

    return {
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return, 2),
        "final_equity":     round(equity.iloc[-1], 2),
        "total_net_pnl":    round(results["net_pnl"].sum(), 2),
    }