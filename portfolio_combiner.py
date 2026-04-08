# portfolio_combiner.py
# Responsibility: Combine stat-arb and momentum strategies into a single portfolio.
# Allocation method: Inverse Volatility Weighting (Risk Parity lite)
# Each strategy contributes equal RISK (variance), not equal CAPITAL.

import numpy as np
import pandas as pd
import logging
from config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)

# Lookback window for volatility estimation (in trading days)
VOL_LOOKBACK = 60  # 3 months — enough to be stable, short enough to be adaptive


def combine_strategies(
    statarb_results: pd.DataFrame,
    momentum_results: pd.DataFrame,
    capital: float = INITIAL_CAPITAL,
    vol_lookback: int = VOL_LOOKBACK,
) -> tuple[pd.DataFrame, dict]:
    """
    Combine two strategy return streams via inverse volatility weighting.

    Inverse Volatility Weighting Logic:
        w_i = (1 / σ_i) / Σ(1 / σ_j)

    Where σ_i = rolling standard deviation of strategy i's daily returns.

    This ensures each strategy contributes equal variance to the portfolio.
    High-volatility strategies (momentum) get lower weight.
    Low-volatility strategies (stat-arb) get higher weight.

    This is NOT Markowitz optimization — it does not require estimating
    the correlation matrix, which is notoriously unstable on short samples.
    Inverse vol weighting is robust, simple, and widely used in practice.

    Parameters
    ----------
    statarb_results  : output DataFrame from backtester.run_backtest()
    momentum_results : output DataFrame from momentum_backtester.run_momentum_backtest()

    Returns
    -------
    combined_results : pd.DataFrame with daily portfolio metrics
    metrics          : dict of performance statistics
    """

    # --- 1. ALIGN BOTH RETURN SERIES ON COMMON DATES ---
    # Both strategies must share the same index for combination
    sa_returns = statarb_results["net_pnl"] / capital
    mo_returns = momentum_results["net_pnl"] / capital

    # Align on common dates — inner join drops any date missing from either
    returns_df = pd.DataFrame({
        "statarb":  sa_returns,
        "momentum": mo_returns,
    }).dropna()

    logger.info(f"Combined period: {returns_df.index[0].date()} → "
                f"{returns_df.index[-1].date()} | {len(returns_df)} days")

    # --- 2. ROLLING INVERSE VOLATILITY WEIGHTS ---
    # Calculate rolling volatility for each strategy
    rolling_vol = returns_df.rolling(window=vol_lookback).std()

    # Inverse volatility — higher vol → lower weight
    inv_vol = 1.0 / rolling_vol

    # Normalize so weights sum to 1.0 each day
    # w_statarb + w_momentum = 1.0 always
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights.columns = ["w_statarb", "w_momentum"]

    # Fill early NaN period (before vol_lookback days) with equal weights
    weights.fillna(0.5, inplace=True)

    # --- 3. COMBINED DAILY RETURNS ---
    # Portfolio return = weighted sum of individual strategy returns
    portfolio_returns = (
        weights["w_statarb"]  * returns_df["statarb"] +
        weights["w_momentum"] * returns_df["momentum"]
    )

    # --- 4. PORTFOLIO EQUITY CURVE ---
    portfolio_equity = capital * (1 + portfolio_returns.cumsum())

    # --- 5. ASSEMBLE RESULTS ---
    combined = pd.DataFrame({
        "statarb_return":  returns_df["statarb"],
        "momentum_return": returns_df["momentum"],
        "w_statarb":       weights["w_statarb"],
        "w_momentum":      weights["w_momentum"],
        "portfolio_return": portfolio_returns,
        "portfolio_equity": portfolio_equity,
    })

    # --- 6. METRICS ---
    metrics = calculate_portfolio_metrics(combined, capital)

    # Log comparison table
    logger.info(f"Stat-Arb alone  | Sharpe={_sharpe(returns_df['statarb']):.2f}")
    logger.info(f"Momentum alone  | Sharpe={_sharpe(returns_df['momentum']):.2f}")
    logger.info(f"Combined        | Sharpe={metrics['sharpe']:.2f}")

    return combined, metrics


def calculate_portfolio_metrics(
    combined: pd.DataFrame,
    capital: float,
) -> dict:
    """Full performance metrics on the combined portfolio."""

    r = combined["portfolio_return"]
    equity = combined["portfolio_equity"]

    # Sharpe
    sharpe = _sharpe(r)

    # Sortino
    downside = r[r < 0]
    sortino  = (r.mean() / downside.std() * np.sqrt(252)
                if downside.std() > 0 else 0.0)

    # MDD
    peak   = equity.cummax()
    mdd    = ((equity - peak) / peak).min()

    # Total return
    total_return = (equity.iloc[-1] - capital) / capital * 100

    # Strategy correlation — KEY metric for diversification claim
    corr = combined["statarb_return"].corr(combined["momentum_return"])

    # Average weights
    avg_w_sa = combined["w_statarb"].mean()
    avg_w_mo = combined["w_momentum"].mean()

    return {
        "sharpe":               round(sharpe, 3),
        "sortino":              round(sortino, 3),
        "max_drawdown_pct":     round(mdd * 100, 2),
        "total_return_pct":     round(total_return, 2),
        "final_equity":         round(equity.iloc[-1], 2),
        "strategy_correlation": round(corr, 3),
        "avg_weight_statarb":   round(avg_w_sa, 3),
        "avg_weight_momentum":  round(avg_w_mo, 3),
    }


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio helper."""
    std = returns.std()
    return float(returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0