# backtester.py
# Responsibility: Vectorized backtest of the pairs trading strategy.
# NO plotting here. Returns a clean PnL DataFrame + metrics dict.

import numpy as np
import pandas as pd
import logging
from config import (
    INITIAL_CAPITAL, ENTRY_ZSCORE, EXIT_ZSCORE, STOP_LOSS_Z,
    TOTAL_COST_PER_SIDE, ZSCORE_WINDOW
)

logger = logging.getLogger(__name__)


def run_backtest(
    log_prices: pd.DataFrame,
    s1: str,
    s2: str,
    beta: float,
    alpha: float,
    entry_z: float = ENTRY_ZSCORE,
    exit_z: float  = EXIT_ZSCORE,
    stop_z: float  = STOP_LOSS_Z,
    window: int    = ZSCORE_WINDOW,
    capital: float = INITIAL_CAPITAL,
) -> tuple[pd.DataFrame, dict]:
    """
    Vectorized pairs trading backtest.

    Strategy Logic:
        LONG spread  (buy S1, sell S2): when Z < -entry_z  (spread too low)
        SHORT spread (sell S1, buy S2): when Z > +entry_z  (spread too high)
        EXIT both legs               : when |Z| < exit_z   (spread reverted)
        STOP LOSS                    : when |Z| > stop_z   (spread blew out)

    Capital Allocation:
        Each leg gets capital/2.
        S1 leg: (capital/2) / price_s1  → number of shares
        S2 leg: (capital/2) / price_s2  → number of shares
        This creates approximate dollar-neutrality.

    Returns
    -------
    results_df : pd.DataFrame
        Daily portfolio value, positions, Z-score, spread
    metrics : dict
        Sharpe, Sortino, MDD, Win Rate, total trades
    """

    # --- 1. BUILD SPREAD AND ZSCORE ---
    spread = log_prices[s1] - beta * log_prices[s2] - alpha
    roll_mean = spread.rolling(window).mean()
    roll_std  = spread.rolling(window).std().replace(0, np.nan)
    zscore    = (spread - roll_mean) / roll_std

# --- 2. GENERATE RAW SIGNALS ---
# Signal convention:
#  +1 = LONG spread  (long S1, short S2)
#  -1 = SHORT spread (short S1, long S2)
#   0 = FLAT (no position)

    signal = pd.Series(0, index=zscore.index)

# Entry signals
    signal[zscore < -entry_z] =  1   # Spread below mean → expect rise → LONG
    signal[zscore >  entry_z] = -1   # Spread above mean → expect fall → SHORT

# Exit: force flat when Z crosses back through 0
# We'll handle this properly in the position state machine below
# Stop loss: force flat when Z exceeds stop_z in either direction
    signal[zscore >  stop_z]  =  0
    signal[zscore < -stop_z]  =  0

# --- 3. POSITION STATE MACHINE ---
# Raw signals have a problem: they don't model HOLDING a position.
# Between entry (|Z|>2) and exit (|Z|<0), we must STAY in the trade.
# A state machine solves this correctly.

    position = pd.Series(0, index=zscore.index)
    current_pos = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            position.iloc[i] = 0
            continue

        if current_pos == 0:
            # FLAT: look for entry
            if z < -entry_z:
                current_pos =  1
            elif z > entry_z:
                current_pos = -1

        elif current_pos == 1:
            # LONG spread: exit when Z reverts above 0, OR blows out more negative
            if z > -exit_z:          # normal exit: Z reverted to mean
                current_pos = 0
            elif z < -stop_z:        # stop loss: spread blew out further negative
                current_pos = 0

        elif current_pos == -1:
            # SHORT spread: exit when Z reverts below 0, OR blows out more positive
            if z < exit_z:           # normal exit: Z reverted to mean
                current_pos = 0
            elif z > stop_z:         # stop loss: spread blew out further positive
                current_pos = 0

        position.iloc[i] = current_pos

# --- 4. DETECT TRADE EVENTS (for cost calculation) ---
# A trade occurs when position changes state
    pos_diff = position.diff().fillna(0)

# Entry: position goes from 0 to ±1
    entries = (pos_diff != 0) & (position != 0)
# Exit:  position goes from ±1 to 0
    exits   = (pos_diff != 0) & (position == 0)

# --- 5. CALCULATE DAILY PnL ---
# Use raw prices (not log) for actual PnL calculation
# Log prices were only for spread/signal computation

    raw_s1 = np.exp(log_prices[s1])
    raw_s2 = np.exp(log_prices[s2])

# Dollar returns per day per leg
# Daily return of S1: (P_t - P_{t-1}) / P_{t-1}
    ret_s1 = raw_s1.pct_change().fillna(0)
    ret_s2 = raw_s2.pct_change().fillna(0)

# Capital per leg = half of total capital
    cap_per_leg = capital / 2

# PnL from each leg:
# LONG spread  (+1): long S1 (+ret_s1), short S2 (-ret_s2)
# SHORT spread (-1): short S1 (-ret_s1), long S2 (+ret_s2)
    pnl_s1 =  position.shift(1) * cap_per_leg * ret_s1
    pnl_s2 = -position.shift(1) * cap_per_leg * ret_s2
    gross_pnl = pnl_s1 + pnl_s2

# --- 6. TRANSACTION COSTS ---
# Applied on BOTH legs at ENTRY and EXIT
# Cost = TOTAL_COST_PER_SIDE * capital (affects both legs)
# We apply cost on the day the position changes

    trade_days = entries | exits
# Cost hits both legs → multiply by full capital
    cost = trade_days.astype(float) * TOTAL_COST_PER_SIDE * capital
    net_pnl = gross_pnl - cost

# --- 7. PORTFOLIO EQUITY CURVE ---
    equity = capital + net_pnl.cumsum()
    equity.name = "equity"

# --- 8. ASSEMBLE RESULTS DATAFRAME ---
    results = pd.DataFrame({
        "spread":    spread,
        "zscore":    zscore,
        "position":  position,
        "gross_pnl": gross_pnl,
        "cost":      cost,
        "net_pnl":   net_pnl,
        "equity":    equity,
    })

# --- 9. COMPUTE METRICS ---
    metrics = calculate_metrics(results, capital)
    logger.info(
        f"{s1}/{s2} | Sharpe={metrics['sharpe']:.2f} | "
        f"MDD={metrics['max_drawdown_pct']:.1f}% | "
        f"Trades={metrics['total_trades']}"
    )

    return results, metrics


def calculate_metrics(results: pd.DataFrame, capital: float) -> dict:
    """
    Calculate all performance metrics from the results DataFrame.

    Sharpe Ratio:
        (Mean Daily Return / Std Daily Return) * sqrt(252)
        Annualized. Risk-free rate assumed 0 for simplicity.
        (In production: subtract Nifty overnight rate ~6.5%/252 per day)

    Sortino Ratio:
        (Mean Daily Return / Downside Std) * sqrt(252)
        Downside Std = std of NEGATIVE returns only.
        Penalizes downside volatility more than upside — more relevant
        for strategies with asymmetric return distributions.

    Maximum Drawdown (MDD):
        MDD = max(Peak - Trough) / Peak
        Measures the worst peak-to-trough loss in the equity curve.
        Most important risk metric for interview discussions.

    Win Rate:
        % of trades where net PnL > 0.
        Pairs trading typically has high win rate (60-70%) but
        losses can be large when cointegration breaks down.
    """
    daily_returns = results["net_pnl"] / capital

    # --- Sharpe ---
    mean_ret = daily_returns.mean()
    std_ret  = daily_returns.std()
    sharpe   = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # --- Sortino ---
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std()
    sortino  = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # --- Maximum Drawdown ---
    equity        = results["equity"]
    rolling_peak  = equity.cummax()
    drawdown      = (equity - rolling_peak) / rolling_peak
    max_dd        = drawdown.min()  # Most negative value

    # --- Trade-level Win Rate ---
    # Isolate PnL per trade (between entry and exit)
    position     = results["position"]
    trade_pnls   = []
    trade_start  = None
    running_pnl  = 0.0

    for i in range(len(position)):
        pos = position.iloc[i]
        prev_pos = position.iloc[i-1] if i > 0 else 0

        if prev_pos == 0 and pos != 0:
            # Trade opened
            trade_start = i
            running_pnl = 0.0

        if trade_start is not None:
            running_pnl += results["net_pnl"].iloc[i]

        if prev_pos != 0 and pos == 0:
            # Trade closed
            trade_pnls.append(running_pnl)
            trade_start = None
            running_pnl = 0.0

    total_trades = len(trade_pnls)
    win_rate     = (
        sum(1 for p in trade_pnls if p > 0) / total_trades
        if total_trades > 0 else 0.0
    )

    return {
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct":     round(win_rate * 100, 1),
        "total_trades":     total_trades,
        "total_net_pnl":    round(results["net_pnl"].sum(), 2),
        "final_equity":     round(results["equity"].iloc[-1], 2),
    }