# ============================================================
# FILE: app.py
# RUN:  python -m streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

from data_loader  import load_prices, load_pairs
from stats_engine import get_pair_zscore

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Statistical Arbitrage | Nifty IT",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2a2f3e;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #c9d1d9;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stMetric"] {
    background-color: #161b27;
    border: 1px solid #2a2f3e;
    border-radius: 8px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #e6edf3 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}
[data-testid="stTabs"] button {
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8b949e;
    border: none;
    padding: 10px 20px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
    background: transparent;
}
[data-testid="stDataFrame"] {
    border: 1px solid #2a2f3e;
    border-radius: 8px;
}
hr { border-color: #2a2f3e; }
h1 { color: #e6edf3; font-weight: 700; font-size: 1.6rem; }
h2 { color: #c9d1d9; font-weight: 600; font-size: 1.1rem;
     letter-spacing: 0.03em; }
h3 { color: #8b949e; font-weight: 500; font-size: 0.9rem;
     text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stSelectbox"] > div {
    background-color: #1c2130;
    border: 1px solid #2a2f3e;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter, Segoe UI, sans-serif",
              color="#c9d1d9", size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#161b27",
        bordercolor="#2a2f3e",
        font_size=11
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#2a2f3e",
        borderwidth=1,
        font=dict(size=10)
    )
)

AXIS_STYLE = dict(
    gridcolor="#1e2433",
    linecolor="#2a2f3e",
    tickcolor="#2a2f3e",
    showgrid=True,
    zeroline=False
)

# ── CONSTANTS ────────────────────────────────────────────────
COST_PER_SIDE   = 0.00112
RISK_FREE_DAILY = 0.065 / 252

# ── LOAD DATA ────────────────────────────────────────────────
prices     = load_prices()
pairs_df   = load_pairs()
log_prices = np.log(prices)

# ── SIGNAL + P&L FUNCTIONS ───────────────────────────────────
def generate_signals(zscore, half_life, entry_z, exit_z, stop_z):
    time_stop = int(2.5 * half_life)
    signals   = pd.Series(0, index=zscore.index, dtype=float)
    position  = 0
    days_held = 0
    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue
        if position == 0:
            if z < -entry_z:  position, days_held =  1, 0
            elif z > entry_z: position, days_held = -1, 0
        else:
            days_held += 1
            ex = False
            if position ==  1 and z >= -exit_z: ex = True
            if position == -1 and z <=  exit_z: ex = True
            if days_held >= time_stop:           ex = True
            if position ==  1 and z < -stop_z:  ex = True
            if position == -1 and z >  stop_z:  ex = True
            if ex: position, days_held = 0, 0
        signals.iloc[i] = position
    return signals

def calculate_pnl(s1, s2, beta, signals):
    spread_ret         = log_prices[s1].diff() - beta * log_prices[s2].diff()
    gross              = signals.shift(1) * spread_ret
    pos_change         = signals.diff().abs()
    pos_change.iloc[0] = 0
    cost               = pos_change * 4 * COST_PER_SIDE
    net                = gross - cost
    return pd.DataFrame({
        "gross_pnl": gross,
        "net_pnl"  : net,
        "cum_pnl"  : net.cumsum(),
        "position" : signals,
    })

def compute_metrics(pnl_df):
    active_mask = pnl_df["position"].shift(1) != 0
    net_active  = pnl_df["net_pnl"][active_mask].dropna()
    cum         = pnl_df["cum_pnl"].dropna()
    if len(net_active) == 0:
        return {}
    excess   = net_active - RISK_FREE_DAILY
    sharpe   = (excess.mean() / net_active.std()) * np.sqrt(252) \
               if net_active.std() > 0 else 0
    downside = net_active[net_active < 0]
    sortino  = (excess.mean() / downside.std()) * np.sqrt(252) \
               if len(downside) > 0 else 0
    mdd      = (cum - cum.cummax()).min()
    n_trades = int((pnl_df["position"].diff().abs() > 0).sum() // 2)
    ann_ret  = cum.iloc[-1] / (len(cum) / 252)

    pos, trade_pnl, current = pnl_df["position"], [], []
    for i in range(len(pos)):
        if pos.iloc[i] != 0:
            current.append(pnl_df["net_pnl"].iloc[i])
        else:
            if current:
                trade_pnl.append(sum(current))
                current = []
    if current:
        trade_pnl.append(sum(current))
    win_rate = (np.array(trade_pnl) > 0).mean() if trade_pnl else 0

    return dict(sharpe=sharpe, sortino=sortino, mdd=mdd,
                ann_ret=ann_ret, n_trades=n_trades,
                win_rate=win_rate, total_ret=cum.iloc[-1])

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Strategy Configuration")
    st.markdown("---")

    pair_options  = [f"{r.stock_1} / {r.stock_2}"
                     for _, r in pairs_df.iterrows()]
    selected_pair = st.selectbox("Pair", pair_options)
    s1, s2        = [x.strip() for x in selected_pair.split("/")]

    st.markdown("---")
    st.markdown("## Signal Parameters")
    entry_z = st.slider("Entry Threshold (|Z|)", 1.0, 3.0, 2.0, 0.1)
    exit_z  = st.slider("Exit Threshold (|Z|)",  0.0, 1.5, 0.5, 0.1)
    stop_z  = st.slider("Stop-Loss (|Z|)",        2.5, 5.0, 3.5, 0.1)

    st.markdown("---")
    st.markdown("## Pair Statistics")

    row = pairs_df[
        (pairs_df["stock_1"]==s1) &
        (pairs_df["stock_2"]==s2)
    ].iloc[0]

    st.markdown(f"""
| | |
|---|---|
| **p-value** | `{row.p_value:.4f}` |
| **Half-Life** | `{row.half_life:.1f} days` |
| **Hedge Ratio β** | `{row.beta:.4f}` |
| **Intercept α** | `{row.alpha:.4f}` |
    """)

    st.markdown("---")
    st.caption("Statistical Arbitrage Research Engine v1.0")
    st.caption("Nifty IT Sector  |  NSE  |  2022–2026")

# ── COMPUTE ──────────────────────────────────────────────────
pair_result = get_pair_zscore(log_prices, pairs_df, s1, s2)
spread      = pair_result["spread"]
zscore      = pair_result["zscore"]
half_life   = pair_result["half_life"]
beta        = row["beta"]
window      = pair_result["window"]

signals = generate_signals(zscore, half_life, entry_z, exit_z, stop_z)
pnl_df  = calculate_pnl(s1, s2, beta, signals)
m       = compute_metrics(pnl_df)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("# Statistical Arbitrage Research Dashboard")
st.markdown(
    f"Engle-Granger Cointegration &nbsp;|&nbsp; "
    f"NSE IT Sector &nbsp;|&nbsp; "
    f"4-Year Backtest &nbsp;|&nbsp; "
    f"Active Pair: **{s1} / {s2}**",
    unsafe_allow_html=True
)
st.markdown("---")

# ── METRICS ROW ──────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Sharpe Ratio",     f"{m.get('sharpe', 0):.2f}",
          delta="Active days")
c2.metric("Sortino Ratio",    f"{m.get('sortino', 0):.2f}")
c3.metric("Max Drawdown",     f"{m.get('mdd', 0):.3f}")
c4.metric("Ann. Return",      f"{m.get('ann_ret', 0):.1%}")
c5.metric("Win Rate",         f"{m.get('win_rate', 0):.1%}",
          delta=f"{m.get('n_trades', 0)} trades")
c6.metric("Net Return (log)", f"{m.get('total_ret', 0):.4f}")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Z-SCORE   &   SIGNALS",
    "EQUITY   CURVE",
    "COINTEGRATION   SUMMARY"
])

# ── TAB 1: Z-SCORE ───────────────────────────────────────────
with tab1:
    st.markdown(f"## Spread Z-Score — {s1} / {s2}")
    st.markdown(
        f"Spread: `log({s1}) − {beta:.4f} · log({s2}) − {row.alpha:.4f}`"
        f" &nbsp;|&nbsp; Rolling window: `{window} days`"
        f" &nbsp;|&nbsp; Half-life: `{half_life:.1f} days`",
        unsafe_allow_html=True
    )

    z_plot        = zscore.dropna()
    long_entries  = signals[(signals== 1) & (signals.shift(1)==0)]
    short_entries = signals[(signals==-1) & (signals.shift(1)==0)]
    exits_sig     = signals[(signals== 0) & (signals.shift(1)!=0)]

    fig1 = go.Figure()

    # Z-score line
    fig1.add_trace(go.Scatter(
        x=z_plot.index, y=z_plot,
        name="Z-Score",
        line=dict(color="#58a6ff", width=1.2)
    ))

    # Threshold lines
    thresholds = [
        ( entry_z, "#f85149", f"+{entry_z}σ  Short Entry", "dash"),
        (-entry_z, "#3fb950", f"−{entry_z}σ  Long Entry",  "dash"),
        ( exit_z,  "#8b949e", f"+{exit_z}σ  Exit",         "dot"),
        (-exit_z,  "#8b949e", f"−{exit_z}σ  Exit",         "dot"),
        ( 0,       "#e6edf3", "Mean",                       "solid"),
    ]
    for val, color, name, dash in thresholds:
        fig1.add_hline(
            y=val, line_dash=dash,
            line_color=color, line_width=1,
            annotation_text=name,
            annotation_font_color=color,
            annotation_font_size=9,
            annotation_position="right"
        )

    # Entry / exit markers
    fig1.add_trace(go.Scatter(
        x=long_entries.index,
        y=zscore.reindex(long_entries.index),
        mode="markers",
        marker=dict(symbol="triangle-up", size=10,
                    color="#3fb950",
                    line=dict(color="#0f1117", width=1)),
        name="Long Entry"
    ))
    fig1.add_trace(go.Scatter(
        x=short_entries.index,
        y=zscore.reindex(short_entries.index),
        mode="markers",
        marker=dict(symbol="triangle-down", size=10,
                    color="#f85149",
                    line=dict(color="#0f1117", width=1)),
        name="Short Entry"
    ))
    fig1.add_trace(go.Scatter(
        x=exits_sig.index,
        y=zscore.reindex(exits_sig.index),
        mode="markers",
        marker=dict(symbol="circle", size=6,
                    color="#e3b341",
                    line=dict(color="#0f1117", width=1)),
        name="Exit"
    ))

    fig1.update_layout(
        **PLOTLY_LAYOUT,
        height=460,
        yaxis=dict(**AXIS_STYLE, title="Z-Score", range=[-5, 5]),
        xaxis=dict(**AXIS_STYLE, title="Date"),
    )
    st.plotly_chart(fig1, width="stretch")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Long Entries",  len(long_entries))
    col2.metric("Short Entries", len(short_entries))
    col3.metric("Total Exits",   len(exits_sig))
    col4.metric("Z-Score Range",
                f"[{z_plot.min():.2f}, {z_plot.max():.2f}]")

# ── TAB 2: EQUITY CURVE ──────────────────────────────────────
with tab2:
    st.markdown("## Cumulative P&L — Net of Transaction Costs")
    st.markdown(
        "Costs: STT 0.1% + Brokerage 0.06% + "
        "Exchange fees 0.007% + Slippage 0.06% per round trip"
    )

    cum_plot = pnl_df["cum_pnl"].dropna()
    daily    = pnl_df["net_pnl"].dropna()

    fig2 = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Cumulative Log P&L", "Daily Net P&L"]
    )

    fig2.add_trace(go.Scatter(
        x=cum_plot.index, y=cum_plot,
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="#58a6ff", width=1.5),
        name="Cumulative P&L",
        hovertemplate="%{y:.4f}<extra></extra>"
    ), row=1, col=1)

    roll_max = cum_plot.cummax()
    drawdown = cum_plot - roll_max
    fig2.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.15)",
        line=dict(color="rgba(248,81,73,0.4)", width=0.8),
        name="Drawdown",
        hovertemplate="%{y:.4f}<extra></extra>"
    ), row=1, col=1)

    bar_colors = ["#3fb950" if v >= 0 else "#f85149"
                  for v in daily]
    fig2.add_trace(go.Bar(
        x=daily.index, y=daily,
        marker_color=bar_colors,
        name="Daily Net P&L",
        hovertemplate="%{y:.5f}<extra></extra>"
    ), row=2, col=1)

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=560,
        showlegend=True
    )
    for r in [1, 2]:
        fig2.update_xaxes(**AXIS_STYLE, row=r, col=1)
        fig2.update_yaxes(**AXIS_STYLE, row=r, col=1)

    st.plotly_chart(fig2, width="stretch")

    st.markdown("### Performance Summary")
    perf_data = {
        "Metric": [
            "Annualized Return",
            "Sharpe Ratio (active days)",
            "Sortino Ratio",
            "Maximum Drawdown",
            "Win Rate (trade-level)",
            "Round Trips",
            "Total Net Return (log)"
        ],
        "Value": [
            f"{m.get('ann_ret',    0):.2%}",
            f"{m.get('sharpe',     0):.3f}",
            f"{m.get('sortino',    0):.3f}",
            f"{m.get('mdd',        0):.4f}",
            f"{m.get('win_rate',   0):.1%}",
            f"{m.get('n_trades',   0)}",
            f"{m.get('total_ret',  0):.4f}",
        ],
        "Note": [
            "Log return / years in backtest",
            "Excess over 6.5% RBI proxy, annualized",
            "Downside deviation denominator only",
            "Peak-to-trough, log scale",
            "Profitable trades / total trades",
            "Complete entry + exit cycles",
            "Compounded log P&L over full period"
        ]
    }
    st.dataframe(pd.DataFrame(perf_data), width="stretch",
                 hide_index=True)

# ── TAB 3: COINTEGRATION SUMMARY ─────────────────────────────
with tab3:
    st.markdown("## Cointegration Research — Phase 1 Output")
    st.markdown(
        "**Method:** Engle-Granger 2-step test on 4 years of NSE "
        "daily closing prices (2022–2026). "
        "**Filter:** p-value < 0.05, half-life 5–60 days. "
        "**Universe:** 11 Nifty IT constituents."
    )

    display_df = pairs_df.copy()
    display_df["Strength"] = display_df["p_value"].apply(
        lambda p: "Strong"   if p < 0.005
        else ("Moderate" if p < 0.02 else "Valid")
    )
    display_df["Time-Stop"] = (
        display_df["half_life"] * 2.5
    ).apply(lambda x: f"{x:.0f} days")

    st.dataframe(
        display_df[[
            "stock_1", "stock_2", "beta", "alpha",
            "p_value", "half_life", "Time-Stop", "Strength"
        ]].rename(columns={
            "stock_1"  : "Stock 1",
            "stock_2"  : "Stock 2",
            "beta"     : "Hedge Ratio (β)",
            "alpha"    : "Intercept (α)",
            "p_value"  : "p-value",
            "half_life": "Half-Life (days)",
        }),
        width="stretch",
        hide_index=True
    )

    st.markdown("---")
    st.markdown("### Methodology Notes")
    st.markdown("""
**Step 1 — OLS Regression:**
`log(P₁) = α + β · log(P₂) + ε`
The hedge ratio β is estimated via Ordinary Least Squares.
β represents the number of units of Stock 2 held per unit of Stock 1.

**Step 2 — ADF Test on Residuals:**
The residual series ε is tested for stationarity via the Augmented
Dickey-Fuller test. Rejection of the unit root hypothesis (p < 0.05)
confirms cointegration.

**Half-Life Estimation:**
Fit AR(1) on spread: `Δε = λ · ε(t-1) + η`
Half-life = `−ln(2) / ln(1 + λ)`
Expected days for a spread deviation to revert 50% toward its mean.

**Transaction Costs (Indian Market):**
STT (sell side) 0.10% + Brokerage 0.06% + Exchange fees 0.007% +
Slippage 0.06% = **0.227% per round trip leg.**
    """)
