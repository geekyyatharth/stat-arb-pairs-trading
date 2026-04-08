# ============================================================
# FILE: stats_engine.py (v2 — returns window in dict)
# ============================================================

import pandas as pd
import numpy as np

def compute_spread(log_prices, s1, s2, beta, alpha):
    return log_prices[s1] - beta * log_prices[s2] - alpha

def compute_zscore(spread, window):
    roll_mean = spread.rolling(window=window, min_periods=window).mean()
    roll_std  = spread.rolling(window=window, min_periods=window).std()
    return (spread - roll_mean) / roll_std

def get_pair_zscore(log_prices, pairs_df, s1, s2, window_multiplier=2):
    row = pairs_df[
        (pairs_df["stock_1"] == s1) &
        (pairs_df["stock_2"] == s2)
    ].iloc[0]

    beta      = row["beta"]
    alpha     = row["alpha"]
    half_life = row["half_life"]
    window    = max(int(window_multiplier * half_life), 20)

    spread = compute_spread(log_prices, s1, s2, beta, alpha)
    zscore = compute_zscore(spread, window)

    return {
        "spread"   : spread,
        "zscore"   : zscore,
        "beta"     : beta,
        "alpha"    : alpha,
        "half_life": half_life,
        "window"   : window      # ← this was missing
    }

def get_trade_markers(signals, prices, s1, s2):
    pos_change  = signals.diff()
    entry_dates = pos_change[pos_change != 0].index
    entry_pos   = signals[entry_dates]
    exit_dates  = pos_change[
        (pos_change != 0) & (signals == 0)
    ].index

    entries = pd.DataFrame({
        "date"    : entry_dates,
        "price_s1": prices[s1].reindex(entry_dates),
        "signal"  : entry_pos.values
    })
    exits = pd.DataFrame({
        "date"    : exit_dates,
        "price_s1": prices[s1].reindex(exit_dates),
    })
    return entries, exits