"""
YFinance Market Intelligence Dashboard
Theme adapted from the supplied Streamlit dashboard:
dark canvas, responsive layout, colored KPI cards, tabs, and Plotly-ready sections.

Run:
    pip install streamlit yfinance pandas numpy plotly shap scikit-learn
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

/* Data preview expander matches the Purpose card surface and border */
div[data-testid="stExpander"] {
    border: 1px solid #29313c !important;
    border-radius: 9px !important;
    background: #161b22 !important;
    box-shadow: none !important;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary {
    background: #161b22 !important;
    border: none !important;
}
div[data-testid="stExpander"] summary {
    color: #aeb8c5 !important;
    padding: 12px 15px !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #1c232d !important;
}
div[data-testid="stExpander"] summary svg {
    color: #aeb8c5 !important;
}

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
    /* Hide number-input +/- steppers: keep fields clean and compact. */
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {
        display: none !important;
    }
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        border-radius: 8px !important;
    }

    /* Compact, neutral multiselect chips instead of bright red tags. */
    [data-baseweb="tag"] {
        background: #2a3543 !important;
        border: 1px solid #405064 !important;
        border-radius: 6px !important;
        color: #edf2f7 !important;
    }
    [data-baseweb="tag"] span,
    [data-baseweb="tag"] svg {
        color: #edf2f7 !important;
        fill: #edf2f7 !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        min-height: 44px;
        padding-top: 3px;
        padding-bottom: 3px;
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
        padding-left: clamp(1.25rem, 3.5vw, 4rem);
        padding-right: clamp(1.25rem, 3.5vw, 4rem);
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
    .intro-strip {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 0 0 1.35rem 0;
    }
    .intro-strip span {
        display: block;
        min-height: 54px;
        padding: 12px 15px;
        background: #161b22;
        border: 1px solid #29313c;
        border-radius: 9px;
        color: #aeb8c5;
        font-size: .91rem;
        line-height: 1.45;
        box-sizing: border-box;
    }
    .intro-strip b { color: #f3f6fa; font-weight: 650; }
    .ps-block { margin: 1rem 0; max-width: 980px; }
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
        height: 146px;
        min-height: 146px;
        padding: 16px 17px;
        border-radius: 12px;
        background: linear-gradient(145deg, #172b4d 0%, #14233d 58%, #101c31 100%);
        border: 1px solid #28528a;
        box-shadow: inset 0 1px 0 rgba(110, 180, 255, .08), 0 0 14px rgba(25, 105, 210, .10);
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        gap: 8px;
        box-sizing: border-box;
        overflow: hidden;
    }
    .metric-label {
        min-height: 18px;
        font-size: .81rem;
        color: #a8b4c6;
        font-weight: 600;
        line-height: 1.25;
    }
    .metric-value {
        font-size: clamp(1.12rem, 1.55vw, 1.65rem);
        color: #d4dce8;
        font-weight: 780;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-variant-numeric: tabular-nums;
    }
    .metric-note {
        color: #9aa8bd;
        font-size: .76rem;
        line-height: 1.3;
        min-height: 30px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .metric-value.value-positive { color: #39f2a5; text-shadow: 0 0 12px rgba(57, 242, 165, .18); }
    .metric-value.value-negative { color: #ff626f; text-shadow: 0 0 12px rgba(255, 98, 111, .16); }
    .metric-value.value-neutral { color: #d4dce8; }
    /* Keep KPI cards visually uniform: no colored top accent strip. */
    .metric-card.accent-blue,
    .metric-card.accent-green,
    .metric-card.accent-red,
    .metric-card.accent-orange,
    .metric-card.accent-purple,
    .metric-card.accent-teal {
        border-top: 1px solid #28528a;
    }
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
    @media (max-width: 900px) {
        .intro-strip { grid-template-columns: 1fr; }
        .metric-card { height: 132px; min-height: 132px; padding: 13px; }
        .metric-value { font-size: 1.25rem; }
    }
    @media (max-width: 768px) {
        div.block-container { padding-left: 1rem; padding-right: 1rem; }
        .metric-card { height: 126px; min-height: 126px; padding: 12px; }
        .metric-value { font-size: 1.15rem; }
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
    '<div class="intro-strip">'
    '<span><b>Purpose</b> — Analyse price trends, returns and risk in one view.</span>'
    '<span><b>Data</b> — Yahoo Finance; change the ticker and time window in the sidebar.</span>'
    '</div>',
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

# ----------------------------- DATA PREVIEW ----------------------------------
with st.expander("Data Preview", expanded=False):
    st.caption(f"Showing the most recent 10 rows of {ticker} market data ({len(df):,} observations loaded).")
    preview_columns = [
        column for column in [
            "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "Daily Return", "Daily Change", "SMA 20", "SMA 50", "Volatility 20D"
        ] if column in df.columns
    ]
    preview_df = df[preview_columns].tail(10).sort_values("Date", ascending=False)
    dark_table(preview_df, max_height=360)

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
        if label == "Period Return":
            value_class = "value-positive" if period_return > 0 else ("value-negative" if period_return < 0 else "value-neutral")
        elif label == "Latest Close":
            value_class = "value-positive" if price_change > 0 else ("value-negative" if price_change < 0 else "value-neutral")
        elif label == "Annualised Volatility":
            value_class = "value-negative" if annual_vol >= 35 else "value-neutral"
        else:
            value_class = "value-neutral"
        st.markdown(
            f'<div class="metric-card {accent}"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {value_class}">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.caption(
    f"Instrument: {ticker}  ·  Interval: {interval}  ·  "
    f"Observations: {len(df):,}  ·  Last timestamp: {latest['Date']}"
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Returns & Risk", "Data Explorer", "SHAP Explainability"])


# ----------------------------- SHAP HELPERS ----------------------------------
def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """Technical features computed from information available at each bar."""
    c = d["Close"]
    r = c.pct_change()
    f = pd.DataFrame(index=d.index)
    f["Return 1 bar"] = r
    f["Return 5 bars"] = c.pct_change(5)
    f["Return 10 bars"] = c.pct_change(10)
    f["Price vs SMA20"] = c / c.rolling(20).mean() - 1
    f["Price vs SMA50"] = c / c.rolling(50).mean() - 1
    f["Volatility 20"] = r.rolling(20).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["RSI 14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    if "Volume" in d.columns:
        f["Volume vs avg20"] = d["Volume"] / d["Volume"].rolling(20).mean() - 1
    if {"High", "Low"}.issubset(d.columns):
        f["Range %"] = (d["High"] - d["Low"]) / c
    return f.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=900, show_spinner=False)
def run_shap_model(frame: pd.DataFrame, n_trees: int):
    """Fit a tree model on next-bar return and compute SHAP values on a time-ordered hold-out."""
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    feats = make_features(frame)
    target = frame["Close"].pct_change().shift(-1)
    data = feats.copy()
    data["target"] = target
    train_all = data.dropna()
    if len(train_all) < 120:
        return None

    X = train_all.drop(columns="target")
    y = train_all["target"]
    split = int(len(train_all) * 0.8)  # chronological split, no shuffling
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=n_trees, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)
    pred = model.predict(X_te)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_te))
    latest = feats.iloc[[-1]].fillna(X.median())
    latest_shap = np.asarray(explainer.shap_values(latest))[0]
    base_value = float(np.ravel(explainer.expected_value)[0])

    return {
        "X_test": X_te,
        "shap_values": shap_values,
        "latest": latest,
        "latest_shap": latest_shap,
        "base_value": base_value,
        "latest_pred": float(model.predict(latest)[0]),
        "r2": float(r2_score(y_te, pred)),
        "mae": float(mean_absolute_error(y_te, pred)),
        "dir_acc": float(np.mean(np.sign(pred) == np.sign(y_te))),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


# ----------------------------- STRATEGY HELPERS ------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def run_strategy_model(frame: pd.DataFrame, n_trees: int):
    """Same model and chronological split as the SHAP tab, but returns the hold-out
    predictions so a simple trading rule can be compared with buy-and-hold."""
    from sklearn.ensemble import RandomForestRegressor

    feats = make_features(frame)
    data = feats.copy()
    data["target"] = frame["Close"].pct_change().shift(-1)
    train_all = data.dropna()
    if len(train_all) < 120:
        return None

    X = train_all.drop(columns="target")
    y = train_all["target"]
    split = int(len(train_all) * 0.8)  # chronological split, no shuffling
    model = RandomForestRegressor(
        n_estimators=n_trees, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])

    # A prediction made at bar t is earned over bar t+1, so label each row with the next date.
    realised_dates = frame["Date"].shift(-1).loc[X.index[split:]]
    return pd.DataFrame({
        "Date": realised_dates.to_numpy(),
        "Actual Return": y.iloc[split:].to_numpy(),
        "Predicted Return": pred,
    })


def backtest_long_only(bt: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Hold the asset when the model predicts a positive next-bar return, otherwise stay in cash."""
    out = bt.copy()
    out["Position"] = (out["Predicted Return"] > 0).astype(float)
    trades = out["Position"].diff().abs()
    trades.iloc[0] = out["Position"].iloc[0]  # entering on the first bar counts as a trade
    out["Strategy Return"] = out["Position"] * out["Actual Return"] - trades * cost_bps / 10_000
    out["Model strategy"] = 100 * (1 + out["Strategy Return"]).cumprod()
    out["Buy & hold"] = 100 * (1 + out["Actual Return"]).cumprod()
    return out


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())

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
        st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
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
        st.plotly_chart(fig_range, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown('<div class="section-header">Strategy & Trend Analysis</div>', unsafe_allow_html=True)

    # Put the trading-cost control on its own row so it does not distort chart alignment.
    cost_row, _ = st.columns([1, 2])
    with cost_row:
        cost_bps = st.number_input(
            "Trading cost per trade (bps)",
            min_value=0.0, max_value=100.0, value=5.0, step=1.0, key="strategy_cost_bps",
            help="Applied when the strategy changes position.",
        )

    chart_left, chart_right = st.columns(2)

    # ---- Chart A: model returns vs buy & hold ----
    with chart_left:
        st.markdown("#### Model returns vs buy & hold")
        bt_raw, sklearn_missing = None, False
        bt_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        try:
            bt_raw = run_strategy_model(df[bt_cols].copy(), 150)
        except ImportError:
            sklearn_missing = True

        if sklearn_missing:
            st.info("Install scikit-learn to enable this chart:  `pip install scikit-learn`")
        elif bt_raw is None:
            st.warning("Not enough observations to train the model. Choose a longer period (e.g. 2y or more) on daily data.")
        else:
            bt = backtest_long_only(bt_raw, cost_bps)
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"], y=bt["Buy & hold"], mode="lines", name="Buy & hold",
                line=dict(color="#3498db", width=2),
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"], y=bt["Model strategy"], mode="lines", name="Model strategy",
                line=dict(color="#28a745", width=2),
            ))
            fig_bt.update_layout(
                template="plotly_dark", height=380, margin=dict(l=10, r=10, t=25, b=10),
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title=None, yaxis_title="Growth of 100",
            )
            st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False})

            strat_total = bt["Model strategy"].iloc[-1] / 100 - 1
            hold_total = bt["Buy & hold"].iloc[-1] / 100 - 1
            st.caption(
                f"Test: {bt['Date'].iloc[0]:%Y-%m-%d}–{bt['Date'].iloc[-1]:%Y-%m-%d} · "
                f"Model {strat_total * 100:+.1f}% | Buy & hold {hold_total * 100:+.1f}% · "
                f"Max drawdown {max_drawdown(bt['Model strategy']) * 100:.1f}% | "
                f"Buy & hold {max_drawdown(bt['Buy & hold']) * 100:.1f}%"
            )
            st.caption(
                "Long-only strategy · Chronological 80/20 train-test split · Research use only."
            )

    # ---- Chart B: moving averages over time ----
    with chart_right:
        st.markdown("#### Moving averages")
        # Fixed defaults keep the chart clean: no visible bar-window controls.
        ma_windows = [20, 50, 200]
        ma_type = "SMA"
        usable = [w for w in ma_windows if w < len(df)]
        skipped = [w for w in ma_windows if w >= len(df)]
        ma_palette = ["#f39c12", "#9b59b6", "#1abc9c", "#e74c3c", "#3498db", "#f7dc6f"]

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"], mode="lines", name="Close",
            line=dict(color="#6b7785", width=1.2),
        ))
        mas = {}
        for k, w in enumerate(usable):
            if ma_type == "SMA":
                series = df["Close"].rolling(w).mean()
            else:
                series = df["Close"].ewm(span=w, adjust=False).mean()
                series.iloc[: w - 1] = np.nan  # hide the warm-up period
            mas[w] = series
            fig_ma.add_trace(go.Scatter(
                x=df["Date"], y=series, mode="lines", name=f"{w}-bar {ma_type}",
                line=dict(color=ma_palette[k % len(ma_palette)], width=1.6),
            ))

        if len(usable) >= 2:
            short_w, long_w = usable[0], usable[-1]
            short_ma, long_ma = mas[short_w], mas[long_w]
            valid = short_ma.notna() & long_ma.notna()
            above = short_ma > long_ma
            changed = (
                above.ne(above.shift(fill_value=False)) & valid & valid.shift(fill_value=False)
            )
            for mask, label, symbol, color in [
                (changed & above, f"{short_w}/{long_w} cross up", "triangle-up", "#28a745"),
                (changed & ~above, f"{short_w}/{long_w} cross down", "triangle-down", "#e74c3c"),
            ]:
                if mask.any():
                    fig_ma.add_trace(go.Scatter(
                        x=df.loc[mask, "Date"], y=df.loc[mask, "Close"], mode="markers", name=label,
                        marker=dict(symbol=symbol, size=10, color=color),
                    ))

        fig_ma.update_layout(
            template="plotly_dark", height=380, margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title=None, yaxis_title="Price",
        )
        st.plotly_chart(fig_ma, use_container_width=True, config={"displayModeBar": False})

        if len(usable) >= 2:
            st.caption(
                f"Triangles mark {usable[0]}/{usable[-1]}-period crossovers. Signals can lag or fail in sideways markets."
            )
        elif not usable:
            st.caption("Not enough observations for the selected moving averages.")
        if skipped:
            pass

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
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
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
            st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

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

# ----------------------------- TAB 4 -----------------------------------------
with tab4:
    st.markdown('<div class="section-header">SHAP Model Explainability</div>', unsafe_allow_html=True)
    st.caption(
        "SHAP shows how technical features influence the model's next-bar return prediction. Short-term returns are noisy."
    )

    try:
        import shap  # noqa: F401
        import sklearn  # noqa: F401
        shap_ready = True
    except ImportError:
        shap_ready = False
        st.info("Install the extra packages to enable this tab:  `pip install shap scikit-learn`")

    if shap_ready:
        n_trees = 150  # fixed; no slider
        model_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        with st.spinner("Training model and computing SHAP values…"):
            result = run_shap_model(df[model_cols].copy(), n_trees)

        if result is None:
            st.warning("Not enough observations to train a model. Choose a longer period (e.g. 2y or more) on daily data.")
        else:
            shap_vals = result["shap_values"]
            X_te = result["X_test"]
            feature_names = list(X_te.columns)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            order = np.argsort(mean_abs)  # ascending, so the biggest ends up on top in horizontal plots

            m1, m2, m3, m4 = st.columns(4)
            cards = [
                ("Hold-out R²", f"{result['r2']:.3f}", "Next-bar return, chronological test set", "accent-blue"),
                ("Direction Accuracy", f"{result['dir_acc'] * 100:.1f}%", "Up/down called correctly", "accent-green"),
                ("Mean Abs Error", f"{result['mae'] * 100:.3f}%", "Per-bar return error", "accent-orange"),
                ("Train / Test Bars", f"{result['n_train']:,} / {result['n_test']:,}", "80/20 time-ordered split", "accent-purple"),
            ]
            for col, (label, value, note, accent) in zip([m1, m2, m3, m4], cards):
                with col:
                    if label == "Direction Accuracy":
                        value_class = "value-positive" if result["dir_acc"] >= 0.55 else "value-negative"
                    elif label == "Hold-out R²":
                        value_class = "value-positive" if result["r2"] > 0 else "value-negative"
                    else:
                        value_class = "value-neutral"
                    st.markdown(
                        f'<div class="metric-card {accent}"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value {value_class}">{value}</div><div class="metric-note">{note}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
            g_left, g_right = st.columns(2)

            with g_left:
                st.markdown("#### Global feature importance")
                fig_imp = go.Figure(go.Bar(
                    x=mean_abs[order] * 100, y=[feature_names[i] for i in order],
                    orientation="h", marker_color="#3498db",
                ))
                fig_imp.update_layout(
                    template="plotly_dark", height=420, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    margin=dict(l=10, r=10, t=25, b=10), xaxis_title="Mean |SHAP value| (return, % points)",
                )
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

            with g_right:
                st.markdown("#### SHAP summary (feature effect)")
                rng = np.random.default_rng(42)
                fig_sw = go.Figure()
                for pos, i in enumerate(order):
                    vals = X_te.iloc[:, i].to_numpy(dtype=float)
                    lo, hi = np.nanpercentile(vals, [5, 95])
                    norm = np.clip((vals - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(vals, 0.5)
                    fig_sw.add_trace(go.Scatter(
                        x=shap_vals[:, i] * 100,
                        y=pos + rng.uniform(-0.28, 0.28, size=len(vals)),
                        mode="markers", showlegend=False,
                        marker=dict(
                            size=5, opacity=0.8, color=norm,
                            colorscale=[[0, "#3498db"], [1, "#e74c3c"]], cmin=0, cmax=1,
                            showscale=(pos == len(order) - 1),
                            colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"], len=0.8),
                        ),
                        hovertemplate=f"{feature_names[i]}<br>SHAP: %{{x:.4f}}%<extra></extra>",
                    ))
                fig_sw.update_layout(
                    template="plotly_dark", height=420, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis_title="SHAP value (impact on predicted return, % points)",
                    yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                               ticktext=[feature_names[i] for i in order]),
                )
                st.plotly_chart(fig_sw, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            st.markdown("#### Why the model predicts what it does for the latest bar")
            latest_shap = result["latest_shap"]
            l_order = np.argsort(np.abs(latest_shap))
            fig_local = go.Figure(go.Bar(
                x=latest_shap[l_order] * 100,
                y=[f"{feature_names[i]} = {result['latest'].iloc[0, i]:.3g}" for i in l_order],
                orientation="h",
                marker_color=["#28a745" if v >= 0 else "#e74c3c" for v in latest_shap[l_order]],
            ))
            fig_local.update_layout(
                template="plotly_dark", height=380, paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                margin=dict(l=10, r=10, t=25, b=10), xaxis_title="Contribution to prediction (% points)",
            )
            st.plotly_chart(fig_local, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                f"Baseline {result['base_value'] * 100:+.3f}% · Next-bar prediction {result['latest_pred'] * 100:+.3f}%. "
                "Green raises the prediction; red lowers it."
            )

st.markdown("---")
st.caption("Research use only. Market data may be delayed or incomplete; not investment advice.")