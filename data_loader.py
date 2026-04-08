# data_loader.py
# Responsibility: ONE job only — load and validate price data.
# All other modules import from here. Nobody else touches raw files.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Module-level logger — avoids print() in production code
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_prices(
    filepath: str = "nifty_it_prices.csv",
    date_col: str = "DATE",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load daily close prices from a CSV file.

    Expected CSV format:
        date, TCS, INFY, WIPRO, ...
        2020-01-01, 2100.5, 1050.3, ...

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (sorted ascending)
        Columns: Stock tickers
        Values: Adjusted close prices (float64), NO NaNs in output
    """

    path = Path(filepath)
    if not path.exists():
        # Hard fail — nothing downstream can work without data
        raise FileNotFoundError(f"Price file not found: {path.resolve()}")

    logger.info(f"Loading prices from: {path.resolve()}")

    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)

    # --- Validation Block ---
    # 1. Sort index — jugaad-data sometimes returns unsorted dates
    df.sort_index(inplace=True)

    # 2. Remove duplicate dates (data quality issue seen in NSE historical feeds)
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Dropping {n_dupes} duplicate date rows.")
        df = df[~df.index.duplicated(keep="first")]

    # 3. Apply date range filter if specified
    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    # 4. Drop columns that are entirely NaN (dead stocks, bad exports)
    cols_before = set(df.columns)
    df.dropna(axis=1, how="all", inplace=True)
    dropped = cols_before - set(df.columns)
    if dropped:
        logger.warning(f"Dropped all-NaN columns: {dropped}")

    # 5. Forward-fill then back-fill remaining NaNs
    #    Forward-fill = carry last known price over exchange holidays
    #    Back-fill = handles NaN at the very start of a series
    #    This is standard practice for survivorship-complete NSE data
    nan_count_before = df.isna().sum().sum()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    if nan_count_before > 0:
        logger.info(f"Imputed {nan_count_before} NaN cells via ffill/bfill.")

    # 6. Enforce float64 — prevents silent integer arithmetic bugs downstream
    df = df.astype(np.float64)

    logger.info(
        f"Loaded {df.shape[1]} tickers | "
        f"{len(df)} trading days | "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )

    return df


def load_pairs(filepath: str = "cointegrated_pairs.csv") -> pd.DataFrame:
    """
    Load the pre-screened cointegrated pairs list.

    Expected CSV format:
        stock1, stock2, pvalue, hedge_ratio
        TCS, INFY, 0.021, 1.34

    Returns
    -------
    pd.DataFrame with columns: stock1, stock2, pvalue, hedge_ratio
    """

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path.resolve()}")

    pairs_df = pd.read_csv(path)

    # Rename columns to standard internal names
    pairs_df.rename(columns={
        "stock_1":  "stock1",
        "stock_2":  "stock2",
        "p_value":  "pvalue",
        "beta":     "hedge_ratio",
    }, inplace=True)

    required_cols = {"stock1", "stock2", "pvalue", "hedge_ratio"}
    missing = required_cols - set(pairs_df.columns)
    if missing:
        raise ValueError(f"Pairs CSV missing required columns: {missing}")

    # Filter to only statistically significant pairs on load
    # This makes the loader self-defending — bad pairs never enter the pipeline
    from config import COINT_PVALUE_THRESHOLD
    n_before = len(pairs_df)
    pairs_df = pairs_df[pairs_df["pvalue"] < COINT_PVALUE_THRESHOLD].copy()
    logger.info(
        f"Pairs loaded: {n_before} total, "
        f"{len(pairs_df)} significant (p < {COINT_PVALUE_THRESHOLD})"
    )

    return pairs_df.reset_index(drop=True)