\# Statistical Arbitrage — Pairs Trading Engine (NSE Equities)



A production-grade, market-neutral quantitative trading strategy built on

cointegration theory. Identifies mean-reverting stock pairs in the Nifty IT

sector and trades the spread using Z-score signals.



Built as part of a quant research portfolio targeting roles in algorithmic

trading and quantitative research.



\---



\## Strategy Performance — Best Pair (MPHASIS / OFSS)



| Metric | Value |

|--------|-------|

| Total Return | +32.2% |

| Sharpe Ratio | 0.63 |

| Sortino Ratio | 0.97 |

| Maximum Drawdown | -8.86% |

| Win Rate | 60.9% |

| Total Trades | 23 |

| Backtest Period | Apr 2022 – Apr 2026 (991 trading days) |

| Starting Capital | ₹10,00,000 |

## Portfolio Construction — Strategy Combination

| Metric | Stat-Arb | Momentum | Combined |
|--------|----------|----------|----------|
| Sharpe | 0.63 | -0.78 | 0.04 |
| MDD | -8.86% | -46.9% | -14.7% |
| Strategy Correlation | — | — | -0.026 |

**Allocation Method:** Inverse Volatility Weighting (Risk Parity lite).
**Key Finding:** Near-zero strategy correlation (-0.026) confirms genuine
diversification benefit. Combined MDD of -14.7% vs momentum standalone
MDD of -46.9% demonstrates capital protection from portfolio construction.
Momentum underperformance is attributed to single-sector universe — a
stated limitation addressed in production by expanding to Nifty 200.



> Transaction costs included: STT (0.1%), brokerage (0.03%), slippage (0.05%)

> applied on both legs at every entry and exit.



\---



\## What This Project Demonstrates



\- Statistical Rigor: Engle-Granger cointegration test (p < 0.05) + ADF stationarity confirmation on the spread before any trade is taken.

\- Mathematical Depth: Ornstein-Uhlenbeck half-life estimation to calibrate the Z-score rolling window to the actual mean-reversion speed of each pair (not an arbitrary hardcoded value).

\- Production Code Quality: Modular architecture with single-responsibility

&#x20; modules, centralized config, logging, and input validation.

\- Indian Market Realism: All transaction costs specific to NSE equity delivery (STT, SEBI charges, brokerage, slippage) are embedded in the backtest — not added as an afterthought.

\- Honest Results: 4 of 7 tested pairs produced negative Sharpe ratios. Only statistically sound pairs are presented. No curve-fitting.



\---



\## Project Architecture
stat-arb-pairs-trading/
├── app.py              # Streamlit dashboard (interactive Z-score visualizer)
├── backtester.py       # Vectorized PnL engine + performance metrics
├── stats_engine.py     # Cointegration, ADF, hedge ratio, OU half-life
├── data_loader.py      # CSV ingestion, validation, NaN handling
├── config.py           # Single source of truth for all parameters
├── requirements.txt    # Pinned dependencies
└── README.md

---

## Core Methodology

### Step 1 — Pair Selection via Cointegration
Tested all C(11,2) = 55 pairs in the Nifty IT sector using the
Engle-Granger two-step cointegration test. Retained only pairs with
p-value < 0.05, yielding 7 statistically significant pairs.

### Step 2 — Spread Construction
For each cointegrated pair (S1, S2): Spread = log(S1) - β·log(S2) - α

where β (hedge ratio) and α (intercept) are estimated via OLS regression
on log prices. The log transformation ensures price-scale invariance and
improves stationarity of the residuals.

### Step 3 — Mean Reversion Speed (Ornstein-Uhlenbeck Half-Life)
The spread is modeled as an OU process: ΔSpread_t = α + β·Spread_{t-1} + ε

Half-life = -ln(2) / β

MPHASIS/OFSS half-life: **28 trading days** → Z-score window set to 50 days.

### Step 4 — Z-Score Signal Generation
Z = (Spread - Rolling_Mean) / Rolling_Std    [window = 50 days]
Entry Long  : Z < -2.0  →  Buy MPHASIS, Sell OFSS
Entry Short : Z > +2.0  →  Sell MPHASIS, Buy OFSS
Exit        : |Z| < 0.0 →  Close both legs
Stop Loss   : |Z| > 3.0 →  Cut position (cointegration breakdown)

### Step 5 — Dollar-Neutral Position Sizing
Capital is split equally across both legs, creating approximate
dollar-neutrality and reducing directional market exposure.

---

## Key Risk Factors 

1. Cointegration breakdown: The historical relationship can structurally
   break due to M&A activity, sector rotation, or regulatory changes.
   Mitigated here via rolling re-estimation (production enhancement).

2. Look-ahead bias: Hedge ratio and alpha are estimated on the full
   in-sample period. A production system would use expanding-window
   estimation. This is a stated limitation of this research prototype.

3. Crowding risk: Stat-arb strategies are widely deployed. In periods
   of market stress, simultaneous unwinding by multiple funds causes
   spread blowouts that stop losses cannot fully protect against
   (August 2007 quant crisis is the canonical example).

4. Execution risk: Daily-frequency backtest assumes fills at closing
   price. Real execution faces market impact, especially for mid-cap
   names like MPHASIS.

---

## Installation & Usage

```bash
git clone https://github.com/YOUR_USERNAME/stat-arb-pairs-trading
cd stat-arb-pairs-trading
pip install -r requirements.txt
```

Run the backtest:
```bash
python test_backtest.py
```

Launch the dashboard:
```bash
streamlit run app.py
```

---

## Tech Stack

| Purpose | Library |
|---------|---------|
| Data handling | pandas 2.2.3, numpy 1.26.4 |
| Statistical testing | statsmodels 0.14.6 |
| Visualization | plotly 6.6.0, matplotlib |
| Dashboard | streamlit 1.56.0 |
| Data source | NSE historical data via jugaad-data |

---

## Author

Yatharth Bhatt
M.Tech, DA-IICT | NISM VIII & XV Certified | QuantInsti Trained

. https://www.linkedin.com/in/yatharth-bhatt-analyst/

· https://github.com/geekyyatharth