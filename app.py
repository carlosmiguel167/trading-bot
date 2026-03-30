
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta

# ── PAGE CONFIG ────────────────────────────────────
st.set_page_config(
    page_title = "Trading Bot Dashboard",
    page_icon  = "📈",
    layout     = "wide"
)

# ── HELPER FUNCTIONS ───────────────────────────────
def get_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def calculate_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    delta       = df["Close"].diff()
    gain        = delta.clip(lower=0)
    loss        = -delta.clip(upper=0)
    rs          = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"]   = 100 - (100 / (1 + rs))
    exp12       = df["Close"].ewm(span=12, adjust=False).mean()
    exp26       = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = exp12 - exp26
    df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()
    return df

def generate_signal(df, sentiment=0.037):
    latest  = df.iloc[-1]
    score   = 0.0
    reasons = []

    if latest["RSI"] > 75:
        return "HOLD", 0.0, ["⛔ RSI above hard ceiling — HOLD forced"]

    if pd.notna(latest["SMA20"]) and pd.notna(latest["SMA50"]):
        if latest["SMA20"] > latest["SMA50"]:
            score += 0.3
            reasons.append("✓ Uptrend (SMA20 > SMA50)")
        else:
            score -= 0.3
            reasons.append("✗ Downtrend (SMA20 < SMA50)")

    if pd.notna(latest["RSI"]):
        if latest["RSI"] < 40:
            score += 0.2
            reasons.append(f"✓ Oversold RSI {latest['RSI']:.1f}")
        elif latest["RSI"] > 65:
            score -= 0.2
            reasons.append(f"✗ Overbought RSI {latest['RSI']:.1f}")
        else:
            reasons.append(f"→ Neutral RSI {latest['RSI']:.1f}")

    if pd.notna(latest["MACD"]) and pd.notna(latest["Signal_line"]):
        if latest["MACD"] > latest["Signal_line"]:
            score += 0.1
            reasons.append("✓ MACD bullish")
        else:
            score -= 0.1
            reasons.append("✗ MACD bearish")

    score += sentiment * 0.2
    reasons.append(f"→ Sentiment: {sentiment:.3f}")

    signal = "BUY" if score > 0.35 else ("SELL" if score < -0.35 else "HOLD")
    return signal, score, reasons

def load_simulator_state(state_file="paper_trading_state.json"):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return None

def plot_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="#1D9E75",
        decreasing_line_color="#D85A30"
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA20"],
        name="SMA20", line=dict(color="#EF9F27", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SMA50"],
        name="SMA50", line=dict(color="#D85A30", width=1.5)
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"],
        name="BB Upper",
        line=dict(color="gray", width=0.8, dash="dash"),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"],
        name="BB Lower",
        line=dict(color="gray", width=0.8, dash="dash"),
        fill="tonexty", fillcolor="rgba(128,128,128,0.05)",
        showlegend=False
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI", line=dict(color="#7F77DD", width=1.5)
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash="dash",
                  line_color="red",   row=2, col=1)
    fig.add_hline(y=30, line_dash="dash",
                  line_color="green", row=2, col=1)
    fig.add_hline(y=75, line_dash="dot",
                  line_color="darkred",
                  annotation_text="Hard ceiling",
                  row=2, col=1)

    # MACD
    colors = ["#1D9E75" if v >= 0 else "#D85A30"
              for v in (df["MACD"] - df["Signal_line"])]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["MACD"] - df["Signal_line"],
        name="Histogram", marker_color=colors
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#378ADD", width=1.2)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Signal_line"],
        name="Signal", line=dict(color="#EF9F27", width=1.2)
    ), row=3, col=1)

    fig.update_layout(
        title       = f"{ticker} — Technical Dashboard",
        height      = 700,
        xaxis_rangeslider_visible = False,
        template    = "plotly_dark",
        showlegend  = True,
        legend      = dict(orientation="h", y=1.02)
    )
    fig.update_yaxes(title_text="Price (R$)", row=1, col=1)
    fig.update_yaxes(title_text="RSI",        row=2, col=1)
    fig.update_yaxes(title_text="MACD",       row=3, col=1)
    return fig

# ── SIDEBAR ────────────────────────────────────────
st.sidebar.title("📈 Trading Bot")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Select ticker",
    ["PETR4.SA", "VALE3.SA", "ITUB4.SA",
     "WEGE3.SA", "BBDC4.SA", "MGLU3.SA"],
    index=0
)

period = st.sidebar.selectbox(
    "Chart period",
    ["3mo", "6mo", "1y", "2y"],
    index=1
)

sentiment = st.sidebar.slider(
    "News sentiment",
    min_value = -1.0,
    max_value  = 1.0,
    value      = 0.037,
    step       = 0.01
)

st.sidebar.markdown("---")
refresh = st.sidebar.button("🔄 Refresh data", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Select a ticker")
st.sidebar.markdown("2. Read the signal")
st.sidebar.markdown("3. Check the chart")
st.sidebar.markdown("4. Review portfolio")

# ── MAIN CONTENT ───────────────────────────────────
st.title("📈 Algorithmic Trading Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} "
           f"| Ticker: {ticker}")

# Load data
with st.spinner("Fetching live market data..."):
    df = get_data(ticker, period)
    df = calculate_indicators(df)

latest = df.iloc[-1]
prev   = df.iloc[-2]
price  = latest["Close"]
change = ((price - prev["Close"]) / prev["Close"]) * 100

# Generate signal
signal, score, reasons = generate_signal(df, sentiment)

# ── TOP METRICS ROW ────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Price", f"R${price:.2f}",
              f"{change:+.2f}%")
with col2:
    rsi_delta = f"{latest['RSI'] - df['RSI'].iloc[-2]:+.1f}"
    st.metric("RSI", f"{latest['RSI']:.1f}", rsi_delta)
with col3:
    trend = "↑ Up" if latest["SMA20"] > latest["SMA50"] else "↓ Down"
    st.metric("Trend", trend)
with col4:
    st.metric("Score", f"{score:+.3f}")
with col5:
    signal_color = (":green[BUY]"  if signal == "BUY"  else
                    ":red[SELL]"   if signal == "SELL" else
                    ":orange[HOLD]")
    st.metric("Signal", signal)

st.markdown("---")

# ── SIGNAL ANALYSIS ────────────────────────────────
col_signal, col_portfolio = st.columns([1, 1])

with col_signal:
    st.subheader("Signal analysis")

    if signal == "BUY":
        st.success(f"BUY — Score: {score:.3f}")
    elif signal == "SELL":
        st.error(f"SELL — Score: {score:.3f}")
    else:
        st.warning(f"HOLD — Score: {score:.3f}")

    for reason in reasons:
        if reason.startswith("✓"):
            st.success(reason, icon=None)
        elif reason.startswith("✗"):
            st.error(reason, icon=None)
        elif reason.startswith("⛔"):
            st.error(reason, icon=None)
        else:
            st.info(reason, icon=None)

with col_portfolio:
    st.subheader("Portfolio status")

    state = load_simulator_state()
    if state:
        portfolio_value  = state["capital"]
        for t, pos in state["positions"].items():
            p = yf.Ticker(t).history(period="1d")
            if len(p) > 0:
                portfolio_value += pos["shares"] * float(p["Close"].iloc[-1])

        initial  = 10000
        ret      = portfolio_value - initial
        ret_pct  = (ret / initial) * 100

        c1, c2 = st.columns(2)
        c1.metric("Portfolio value",
                  f"R${portfolio_value:,.2f}",
                  f"{ret_pct:+.1f}%")
        c2.metric("Cash",
                  f"R${state['capital']:,.2f}")

        if state["positions"]:
            st.markdown("**Open positions:**")
            for t, pos in state["positions"].items():
                p = yf.Ticker(t).history(period="1d")
                if len(p) > 0:
                    cur_price  = float(p["Close"].iloc[-1])
                    unreal_pl  = (cur_price - pos["entry_price"]) * pos["shares"]
                    unreal_pct = ((cur_price - pos["entry_price"]) /
                                   pos["entry_price"]) * 100
                    st.info(f"{t} — {pos['shares']} shares @ "
                            f"R${pos['entry_price']:.2f} | "
                            f"P&L: R${unreal_pl:+.2f} ({unreal_pct:+.1f}%)")
        else:
            st.info("No open positions — bot is in cash")

        sells = [t for t in state["trade_log"] if t["type"] == "SELL"]
        if sells:
            wins     = [t for t in sells if t["profit"] > 0]
            win_rate = len(wins) / len(sells) * 100
            st.metric("Win rate", f"{win_rate:.1f}%",
                      f"{len(sells)} completed trades")
    else:
        st.info("No simulator state found. "
                "Run the Jupyter bot first to create it.")

st.markdown("---")

# ── CHART ──────────────────────────────────────────
st.subheader(f"{ticker} — Technical chart")
fig = plot_chart(df, ticker)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── MULTI-STOCK SCANNER ────────────────────────────
st.subheader("B3 stock scanner")
st.caption("Quick signal scan across your watchlist")

watchlist = ["PETR4.SA", "VALE3.SA", "ITUB4.SA",
             "WEGE3.SA", "BBDC4.SA", "MGLU3.SA"]

scanner_data = []
with st.spinner("Scanning watchlist..."):
    for t in watchlist:
        try:
            d      = get_data(t, "3mo")
            d      = calculate_indicators(d)
            sig, sc, _ = generate_signal(d, sentiment)
            lat    = d.iloc[-1]
            prev_t = d.iloc[-2]
            chg    = ((lat["Close"] - prev_t["Close"]) /
                       prev_t["Close"] * 100)
            scanner_data.append({
                "Ticker"  : t,
                "Price"   : f"R${lat['Close']:.2f}",
                "Change"  : f"{chg:+.2f}%",
                "RSI"     : f"{lat['RSI']:.1f}",
                "Trend"   : "↑ Up" if lat["SMA20"] > lat["SMA50"]
                             else "↓ Down",
                "Signal"  : sig,
                "Score"   : f"{sc:+.3f}"
            })
        except:
            pass

scanner_df = pd.DataFrame(scanner_data)

def color_signal(val):
    if val == "BUY":
        return "background-color: #1D9E75; color: white"
    elif val == "SELL":
        return "background-color: #D85A30; color: white"
    return "background-color: #BA7517; color: white"

styled = scanner_df.style.applymap(
    color_signal, subset=["Signal"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Studying Machine Trading Bot · "
           "Built with Python · yfinance · Streamlit")
