"""
YFinance Market Intelligence Dashboard
Theme adapted from the supplied Streamlit dashboard:
dark canvas, responsive layout, colored KPI cards, tabs, and Plotly-ready sections.

Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run app.py

Add your own Plotly charts in the marked chart placeholders.
"""

from datetime import date, timedelta
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- LOGO ------------------------------------------
# Put the logo file in the same folder as app.py (rename it to linkfields_logo.svg,
# or keep the original name; both are checked).
from pathlib import Path

_here = Path(__file__).resolve().parent
_logo = next(
    (p for p in [_here / "linkfields_logo.svg", _here / "linkfields_logo 1 1 (2).svg"] if p.exists()),
    None,
)
import base64

_logo_html = ""
if _logo is not None:
    _b64 = base64.b64encode(_logo.read_bytes()).decode("ascii")
    _logo_html = f'<img class="brand-logo" src="data:image/svg+xml;base64,{_b64}" alt="Linkfields logo" />'

# ----------------------------- THEME -----------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0e1117;
        --panel: #161b22;
        --muted: #9aa4b2;
        --text: #f3f6fa;
        --blue: #3498db;
        --green: #28a745;
        --red: #e74c3c;
        --orange: #f39c12;
        --purple: #9b59b6;
        --teal: #1abc9c;
        --ctrl-bg: #1c232d;
        --ctrl-border: #3a4553;
        --ctrl-hover: #263140;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(14,17,23,0); }
    [data-testid="stSidebar"] {
        background: #11161d;
        border-right: 1px solid #29313c;
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* ---- Sidebar buttons (no more white) ---- */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: var(--ctrl-bg);
        color: var(--text);
        border: 1px solid var(--ctrl-border);
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--ctrl-hover);
        border-color: var(--blue);
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] button[kind="primary"] {
        background: var(--blue);
        border: 1px solid var(--blue);
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #2a86c4;
        border-color: #2a86c4;
    }

    /* ---- Sidebar inputs, selects, number steppers ---- */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="base-input"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--ctrl-bg);
        border-color: var(--ctrl-border);
        color: var(--text);
    }
    [data-testid="stSidebar"] input { background: transparent; color: var(--text); }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] {
        background: var(--ctrl-bg);
        color: var(--text);
        border-left: 1px solid var(--ctrl-border);
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover {
        background: var(--ctrl-hover);
        color: #ffffff;
    }
    [data-testid="stSidebar"] svg { fill: var(--text); }
    /* dropdown menus (popovers render outside the sidebar, so style globally) */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] ul,
    ul[role="listbox"],
    li[role="option"] {
        background: var(--ctrl-bg) !important;
        color: var(--text) !important;
    }
    div[data-baseweb="popover"] li:hover,
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: var(--ctrl-hover) !important;
    }
    div[data-baseweb="popover"] * { color: var(--text); }

    /* inputs, selects and buttons everywhere (main area + sidebar) */
    [data-testid="stAppViewContainer"] [data-baseweb="input"],
    [data-testid="stAppViewContainer"] [data-baseweb="base-input"],
    [data-testid="stAppViewContainer"] [data-baseweb="select"] > div {
        background: var(--ctrl-bg) !important;
        border-color: var(--ctrl-border) !important;
        color: var(--text) !important;
    }
    [data-testid="stAppViewContainer"] input {
        background: transparent !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
    }
    [data-testid="stAppViewContainer"] [data-baseweb="select"] div,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] span {
        color: var(--text) !important;
    }
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg { fill: var(--text) !important; }
    [data-testid="stAppViewContainer"] label p { color: var(--muted) !important; }

    [data-testid="stDownloadButton"] button,
    [data-testid="stAppViewContainer"] .stButton > button {
        background: var(--ctrl-bg) !important;
        color: var(--text) !important;
        border: 1px solid var(--ctrl-border) !important;
        border-radius: 8px;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stAppViewContainer"] .stButton > button:hover {
        background: var(--ctrl-hover) !important;
        border-color: var(--blue) !important;
        color: #ffffff !important;
    }
    [data-testid="stDownloadButton"] { margin-top: 1.5rem; }
    [data-testid="stDownloadButton"] button p { color: var(--text) !important; }

    div.block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: clamp(2rem, 6vw, 7rem);
        padding-right: clamp(2rem, 6vw, 7rem);
        max-width: 1800px;
    }
    .main-header {
        font-size: clamp(1.7rem, 3vw, 2.6rem);
        font-weight: 750;
        color: #f7f9fc;
        text-align: left;
        line-height: 1.15;
        margin: 0 0 1.75rem 0;
    }
    .subheader {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }
    .brand-logo { height: 48px; width: auto; display: block; margin: 0 0 2.25rem auto; }
    .ps-block { margin: 1.25rem 0; max-width: 980px; }
    .ps-divider { border-top: 1px solid #4a5563; margin: 0; }
    .ps-label {
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .74rem;
        font-weight: 700;
        margin-bottom: .4rem;
    }
    .ps-problem .ps-label { color: #7f8b99; }
    .ps-problem .ps-text { color: #f7f9fc; font-size: 1.05rem; line-height: 1.65; }
    .ps-solution .ps-label { color: #b9c3d0; }
    .ps-solution .ps-text { color: #c4ccd8; font-size: 1.05rem; line-height: 1.65; }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .13em;
        color: #7f8b99;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .45rem;
    }
    .section-header {
        color: #f7f9fc;
        font-size: 1.3rem;
        font-weight: 700;
        margin: .5rem 0 1rem 0;
    }
    .metric-card {
        min-height: 132px;
        padding: 18px 20px;
        border-radius: 12px;
        background: #161b22;
        border: 1px solid #303945;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 8px;
        box-sizing: border-box;
    }
    .metric-label { font-size: .83rem; color: #a8b2bf; font-weight: 600; }
    .metric-value { font-size: clamp(1.25rem, 2vw, 1.9rem); color: #f7f9fc; font-weight: 750; line-height: 1.15; }
    .metric-note { color: #8b96a4; font-size: .8rem; }
    .accent-blue { border-top: 3px solid var(--blue); }
    .accent-green { border-top: 3px solid var(--green); }
    .accent-red { border-top: 3px solid var(--red); }
    .accent-orange { border-top: 3px solid var(--orange); }
    .accent-purple { border-top: 3px solid var(--purple); }
    .accent-teal { border-top: 3px solid var(--teal); }
    .panel {
        background: #161b22;
        border: 1px solid #29313c;
        border-radius: 12px;
        padding: 16px 18px;
    }
    hr { border-color: #29313c !important; }
    div[data-testid="stTabs"] button { font-weight: 650; }
    div[data-testid="stDataFrame"] { border: 1px solid #29313c; border-radius: 8px; }
    .dark-table-wrap {
        background: #161b22;
        border: 1px solid #29313c;
        border-radius: 8px;
        overflow: auto;
    }
    table.dark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: .86rem;
        color: #e6ebf2;
    }
    table.dark-table th {
        position: sticky;
        top: 0;
        background: #1c232d;
        color: #a8b2bf;
        text-align: left;
        font-weight: 650;
        padding: 9px 12px;
        border-bottom: 1px solid #303945;
        white-space: nowrap;
    }
    table.dark-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #222a35;
        white-space: nowrap;
    }
    table.dark-table tbody tr:nth-child(even) { background: #131921; }
    table.dark-table tbody tr:hover { background: #1f2733; }
    @media (max-width: 768px) {
        div.block-container { padding-left: 1rem; padding-right: 1rem; }
        .metric-card { min-height: 112px; padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def dark_table(frame: pd.DataFrame, max_height: int = 420):
    """Render a DataFrame as a dark HTML table (independent of Streamlit's theme)."""
    html = frame.to_html(
        index=False, border=0, classes="dark-table",
        float_format=lambda x: f"{x:,.4f}", na_rep="",
    )
    st.markdown(
        f'<div class="dark-table-wrap" style="max-height:{max_height}px">{html}</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- SIDEBAR ---------------------------------------
with st.sidebar:
    st.markdown("## Market Controls")
    ticker = st.text_input("Ticker symbol", value="NVDA").strip().upper()
    period = st.selectbox(
        "Historical period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=3,
    )
    interval_options = {
        "1d": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        "1h": ["1d", "5d", "1mo", "3mo", "6mo"],
        "30m": ["1d", "5d", "1mo"],
        "15m": ["1d", "5d", "1mo"],
        "5m": ["1d", "5d", "1mo"],
    }
    interval = st.selectbox("Data interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
    if period not in interval_options[interval]:
        st.caption(f"{period} is not available for {interval} data; use a shorter period.")
    benchmark = st.text_input("Benchmark ticker", value="SPY").strip().upper()
    risk_free_rate = st.number_input(
        "Annual risk-free rate (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.25
    )
    fetch = st.button("Refresh market data", type="primary", use_container_width=True)
    st.caption("Data source: Yahoo Finance via yfinance. Quotes may be delayed or incomplete.")

# ----------------------------- DATA ------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_history(symbol: str, selected_period: str, selected_interval: str):
    data = yf.download(
        symbol,
        period=selected_period,
        interval=selected_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename_axis("Date").reset_index()
    data.columns = [str(c) for c in data.columns]
    return data

@st.cache_data(ttl=900, show_spinner=False)
def download_benchmark(symbol: str, selected_period: str):
    data = yf.download(
        symbol, period=selected_period, interval="1d",
        auto_adjust=True, progress=False, threads=False
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

if fetch:
    download_history.clear()
    download_benchmark.clear()

if _logo_html:
    st.markdown(_logo_html, unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Quantitative Research • Market Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Market Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="ps-divider"></div>'
    '<div class="ps-block ps-problem">'
    '<div class="ps-label">Problem statement</div>'
    '<div class="ps-text">Investors and analysts piece together price behaviour, returns, volatility and '
    'relative performance from several disconnected tools, which is slow and makes consistent, '
    'data-driven decisions harder.</div></div>'
    '<div class="ps-block ps-solution">'
    '<div class="ps-label">Solution</div>'
    '<div class="ps-text">A single market intelligence dashboard that pulls live Yahoo Finance data and brings '
    'price trends, risk analytics and historical data together in one view, with the instrument and '
    'time window controlled from the sidebar.</div></div>'
    '<div class="ps-divider"></div>',
    unsafe_allow_html=True,
)

if not ticker:
    st.warning("Enter a ticker symbol in the sidebar.")
    st.stop()

effective_period = period if period in interval_options[interval] else "6mo"
with st.spinner(f"Downloading {ticker} market data…"):
    try:
        df = download_history(ticker, effective_period, interval)
    except Exception as exc:
        st.error(f"Could not download {ticker}: {exc}")
        st.stop()

if df.empty or "Close" not in df.columns:
    st.error(f"No price data returned for '{ticker}'. Check the symbol, period, and interval.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
df["Daily Return"] = df["Close"].pct_change()
df["Daily Change"] = df["Close"].diff()
df["SMA 20"] = df["Close"].rolling(20).mean()
df["SMA 50"] = df["Close"].rolling(50).mean()
df["Volatility 20D"] = df["Daily Return"].rolling(20).std() * np.sqrt(252) * 100

latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest
price_change = float(latest["Close"] - previous["Close"])
price_change_pct = (price_change / float(previous["Close"]) * 100) if float(previous["Close"]) else 0
first_close = float(df["Close"].iloc[0])
period_return = (float(latest["Close"]) / first_close - 1) * 100 if first_close else 0
annual_vol = float(df["Daily Return"].std() * np.sqrt(252) * 100) if df["Daily Return"].notna().sum() > 1 else 0
high_52 = float(df["High"].tail(252).max()) if "High" in df else float(df["Close"].max())
low_52 = float(df["Low"].tail(252).min()) if "Low" in df else float(df["Close"].min())
avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df else float("nan")
currency_prefix = "$"  # Yahoo Finance price currency can vary by instrument; change this if needed.

# ----------------------------- KPI CARDS ------------------------------------
st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)
kpis = st.columns(5)
metrics = [
    ("Latest Close", f"{currency_prefix}{float(latest['Close']):,.2f}", f"{price_change_pct:+.2f}% vs previous bar", "accent-blue"),
    ("Period Return", f"{period_return:+.2f}%", f"Across selected {effective_period} window", "accent-green" if period_return >= 0 else "accent-red"),
    ("Annualised Volatility", f"{annual_vol:.2f}%", "Std. dev. of returns × √252", "accent-orange"),
    ("52-Period High", f"{currency_prefix}{high_52:,.2f}", "High in loaded observations", "accent-purple"),
    ("Average Volume", f"{avg_volume:,.0f}" if np.isfinite(avg_volume) else "N/A", "Mean over last 20 observations", "accent-teal"),
]
for col, (label, value, note, accent) in zip(kpis, metrics):
    with col:
        st.markdown(
            f'<div class="metric-card {accent}"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.caption(
    f"Instrument: {ticker}  ·  Interval: {interval}  ·  "
    f"Observations: {len(df):,}  ·  Last timestamp: {latest['Date']}"
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Market Overview", "Returns & Risk", "Data Explorer"])

# ----------------------------- TAB 1 -----------------------------------------
with tab1:
    st.markdown('<div class="section-header">Price & Trend Analysis</div>', unsafe_allow_html=True)
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Price history")
        # Replace or extend this chart with your own Plotly figure.
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"], mode="lines", name="Adjusted close",
            line=dict(color="#3498db", width=2),
        ))
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["SMA 20"], mode="lines", name="20-period SMA",
            line=dict(color="#f39c12", width=1.5),
        ))
        fig_price.add_trace(go.Scatter(
            x=df["Date"], y=df["SMA 50"], mode="lines", name="50-period SMA",
            line=dict(color="#9b59b6", width=1.5),
        ))
        fig_price.update_layout(
            template="plotly_dark", height=430, margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title=None, yaxis_title="Price",
        )
        st.plotly_chart(fig_price, use_container_width=True)
    with right:
        st.markdown("#### Price range")
        fig_range = go.Figure(go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker
        )) if all(c in df.columns for c in ["Open", "High", "Low", "Close"]) else go.Figure()
        fig_range.update_layout(
            template="plotly_dark", height=430, margin=dict(l=5, r=5, t=25, b=10),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            xaxis_rangeslider_visible=False, showlegend=False,
        )
        st.plotly_chart(fig_range, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Your Analysis Workspace</div>', unsafe_allow_html=True)
    st.info("Add your own Plotly visualisations in this section. The dark theme and layout are already configured.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Custom chart slot A")
        st.caption("Example: model predictions vs actual prices, technical indicators, or feature importance.")
    with chart_right:
        st.markdown("#### Custom chart slot B")
        st.caption("Example: drawdown, volume analysis, signal confidence, or model comparison.")

# ----------------------------- TAB 2 -----------------------------------------
with tab2:
    st.markdown('<div class="section-header">Returns & Risk Analytics</div>', unsafe_allow_html=True)
    valid_returns = df.dropna(subset=["Daily Return"]).copy()
    if not valid_returns.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Return distribution")
            fig_hist = go.Figure(go.Histogram(
                x=valid_returns["Daily Return"] * 100, nbinsx=40,
                marker_color="#3498db", opacity=0.85,
            ))
            fig_hist.update_layout(
                template="plotly_dark", height=340, paper_bgcolor="#161b22",
                plot_bgcolor="#161b22", margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title="Return (%)", yaxis_title="Frequency",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            st.markdown("#### Rolling volatility")
            fig_vol = go.Figure(go.Scatter(
                x=df["Date"], y=df["Volatility 20D"], mode="lines",
                line=dict(color="#f39c12", width=2), name="20-period volatility",
            ))
            fig_vol.update_layout(
                template="plotly_dark", height=340, paper_bgcolor="#161b22",
                plot_bgcolor="#161b22", margin=dict(l=10, r=10, t=25, b=10),
                xaxis_title=None, yaxis_title="Annualised volatility (%)",
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("#### Return statistics")
        returns = valid_returns["Daily Return"]
        stats = pd.DataFrame({
            "Metric": ["Mean return per bar", "Median return per bar", "Return standard deviation",
                       "Best bar", "Worst bar", "Positive bars", "Negative bars"],
            "Value": [
                f"{returns.mean() * 100:.4f}%",
                f"{returns.median() * 100:.4f}%",
                f"{returns.std() * 100:.4f}%",
                f"{returns.max() * 100:.2f}%",
                f"{returns.min() * 100:.2f}%",
                f"{(returns > 0).sum():,}",
                f"{(returns < 0).sum():,}",
            ],
        })
        dark_table(stats, max_height=400)
    else:
        st.info("More observations are required to calculate returns.")

# ----------------------------- TAB 3 -----------------------------------------
with tab3:
    st.markdown('<div class="section-header">Historical Data Explorer</div>', unsafe_allow_html=True)
    search_col, rows_col = st.columns([2, 1])
    with search_col:
        search_text = st.text_input("Filter columns or values (optional)", value="")
    with rows_col:
        rows_to_show = st.selectbox("Rows to display", [25, 50, 100, 250, 500], index=2)
    view = df.copy()
    if search_text:
        mask = view.astype(str).apply(
            lambda column: column.str.contains(search_text, case=False, na=False)
        ).any(axis=1)
        view = view.loc[mask]
    dark_table(view.tail(rows_to_show), max_height=420)
    st.download_button(
        "Download loaded data as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker.lower()}_market_data.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("For research and educational use only. Market data can be delayed, revised, or unavailable; this dashboard is not investment advice.")