# config.py
# Central configuration for the Stat-Arb strategy.
# Changing a parameter here propagates everywhere. No more hunting for hardcoded values.

# --- Data ---
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# --- Statistical Thresholds ---
COINT_PVALUE_THRESHOLD = 0.05   # Engle-Granger p-value cutoff
ADF_PVALUE_THRESHOLD   = 0.05   # ADF stationarity cutoff on the spread
ZSCORE_WINDOW          = 50     # Rolling window (days) for mean/std of spread

# --- Signal Thresholds ---
ENTRY_ZSCORE =  2.0   # Open trade when |Z| crosses this
EXIT_ZSCORE  =  0.0   # Close trade when Z reverts to mean
STOP_LOSS_Z  =  3.0   # Optional hard stop if spread blows out

# --- Indian Market Transaction Costs ---
# These are applied per SIDE of the trade (entry and exit separately)
STT_RATE        = 0.001   # Securities Transaction Tax: 0.1% on sell-side (equity delivery)
BROKERAGE_RATE  = 0.0003  # Flat 0.03% per order (Zerodha/discount broker approximation)
SLIPPAGE_RATE   = 0.0005  # 0.05% market impact assumption for mid-cap NSE stocks

TOTAL_COST_PER_SIDE = STT_RATE + BROKERAGE_RATE + SLIPPAGE_RATE  # ~0.158% per side

# --- Capital ---
INITIAL_CAPITAL = 1_000_000  # ₹10 Lakhs starting capital