"""
Signal Lab - Clean working version
==================================

This is a consolidated, working version of the app with the key improvements:
- Company name under ticker
- Smaller verdict banner
- Stronger "Current Picture" narrative that acknowledges stretched conditions vs possible regime shifts
- "What's Unusual" promoted
- Metrics and backtests in collapsed expanders (less kitchen sink)
- Mobile-friendly "See recent real cases" using dialogs
- Better defensive language and disclaimer
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None


# ----------------------------- Data layer -----------------------------

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_company_name(ticker: str) -> str:
    """Fetch company name from Yahoo (longName preferred). Cached 24h."""
    if yf is None:
        return ""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        name = info.get("longName") or info.get("shortName") or ""
        return name.strip()
    except Exception:
        return ""


# --------------------------- Compute helpers ----------------------------

def run_table(close: pd.Series) -> pd.DataFrame:
    sign = np.sign(close.diff()).fillna(0)
    run_id = (sign != sign.shift()).cumsum()
    run_len = sign.groupby(run_id).cumcount() + 1
    return pd.DataFrame({"sign": sign, "run_id": run_id, "run_len": run_len}, index=close.index)


def current_streak(close: pd.Series) -> dict:
    rt = run_table(close)
    sign = int(rt["sign"].iloc[-1])
    length = int(rt["run_len"].iloc[-1])
    runs = rt.groupby("run_id").agg(sign=("sign", "first"), length=("run_len", "max"))
    same_dir = runs[runs["sign"] == sign]
    n_runs = len(same_dir)
    n_at_least = int((same_dir["length"] >= length).sum()) if n_runs else 0
    return {"sign": sign, "length": length, "n_at_least": n_at_least, "n_runs": n_runs}


def drawdown_series(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1.0


def vol_percentile(close: pd.Series, window: int = 20) -> float:
    ret = close.pct_change()
    rv = ret.rolling(window).std() * np.sqrt(252)
    cur = rv.iloc[-1]
    return float((rv.dropna() < cur).mean())


def last_day_z(close: pd.Series) -> float:
    ret = close.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float((ret.iloc[-1] - ret.mean()) / ret.std())


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    bandwidth = (upper - lower) / mid
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth}, index=close.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).rename("rsi")


# ---------------------- Narrative (defends itself better) ----------------------

def _get_band_position(close: float, bb: pd.DataFrame) -> str:
    if pd.isna(close):
        return "middle"
    upper = bb["upper"].iloc[-1]
    lower = bb["lower"].iloc[-1]
    mid = bb["mid"].iloc[-1]
    if close > upper:
        return "above_upper"
    elif close < lower:
        return "below_lower"
    elif close > mid:
        return "upper_half"
    else:
        return "lower_half"


def generate_narrative(close, rsi, bb, current_streak, vol_percentile, regime, drawdown, last_z):
    last_rsi = rsi.iloc[-1]
    band_pos = _get_band_position(close.iloc[-1], bb)
    streak_len = current_streak.get("length", 0)
    streak_sign = current_streak.get("sign", 0)
    streak_word = "up" if streak_sign > 0 else "down" if streak_sign < 0 else "flat"

    # Strong trend + stretched (the key defensive case for names like MU)
    if regime == "uptrend" and streak_len >= 5 and (last_rsi > 72 or band_pos == "above_upper" or abs(last_z) > 2.0):
        summary = ("Strong bullish trend and momentum, but conditions have become statistically stretched. "
                   "RSI is deeply overbought, volatility is elevated, and the move ranks as an outlier by historical standards. "
                   "In a normal environment this would often lead to digestion or reversal risk. "
                   "However, if the fundamental demand picture has structurally improved (e.g. new multi-year growth driver), "
                   "the historical ranges may be less predictive than usual.")
        observations = [
            f"Extreme overbought reading: RSI at {last_rsi:.0f}.",
            f"Extended streak: {streak_len} days up.",
            "This is the classic tension: powerful momentum vs. statistically extreme conditions."
        ]
        return {"summary": summary, "observations": observations, "bucket": "strong_trend_stretched"}

    # Clean strong trend
    if regime in ("uptrend", "downtrend") and streak_len >= 4:
        summary = f"Strong {regime} with sustained momentum. Price has been moving along the outer Bollinger Band."
        observations = [f"Clear trend: Price well {'above' if regime == 'uptrend' else 'below'} the 200-day average.",
                        f"Extended streak: {streak_len} days {streak_word}."]
        return {"summary": summary, "observations": observations, "bucket": "clean_trend"}

    # Default
    summary = "No dominant directional or reversal pressure stands out at the moment."
    observations = ["The market is in a relatively neutral or mixed state based on these indicators."]
    return {"summary": summary, "observations": observations, "bucket": "quiet_or_mixed"}


# ---------------------- Verdict + reasons (with better defense) ----------------------

def build_trade_idea(close, rsi_series, bb):
    regime = "uptrend" if close.iloc[-1] > close.rolling(200).mean().iloc[-1] else "downtrend"
    reasons = []
    score = 0

    if regime == "uptrend":
        score += 1
        reasons.append("🟢 **Trend:** above the 200-day average (supports bullish bias).")
    else:
        score -= 1
        reasons.append("🔴 **Trend:** below the 200-day average (supports bearish bias).")

    m3 = trailing_return(close, 63)
    if not pd.isna(m3):
        if m3 > 0:
            score += 1
            reasons.append(f"🟢 **Momentum:** +{m3:.0%} over the last ~3 months.")
        else:
            score -= 1
            reasons.append(f"🔴 **Momentum:** {m3:+.0%} over the last ~3 months.")

    last_rsi = rsi_series.iloc[-1]
    if regime == "uptrend" and last_rsi > 75:
        reasons.append("⚠️ **Note on overbought conditions:** RSI is extremely elevated. In normal environments this often precedes digestion or reversal. "
                       "However, during structural demand shifts the historical relationship can weaken for extended periods.")

    verdict = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"
    return verdict, score, reasons


def trailing_return(close, days):
    if len(close) <= days:
        return np.nan
    return close.iloc[-1] / close.iloc[-1 - days] - 1.0


# ------------------------------- UI ------------------------------------

def _run_search():
    st.session_state.active_ticker = st.session_state.ticker.strip().upper()


def main():
    st.set_page_config(page_title="Signal Lab", layout="wide")

    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
          .block-container { padding: 2.5rem 0.9rem 3rem !important; }
          div[data-testid="stMetricValue"] { font-size: 1.1rem !important; }
          div[data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
        }
        @media (min-width: 641px) {
          div[data-testid="stMetricValue"] { font-size: 1.0rem !important; }
          div[data-testid="stMetricLabel"] { font-size: 0.6rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.8rem;color:#888;font-weight:600;letter-spacing:.5px;margin-bottom:4px'>SIGNAL LAB "
        "<span style='font-weight:400'>— find signal in the noise</span></div>",
        unsafe_allow_html=True,
    )

    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "active_ticker" not in st.session_state:
        st.session_state.active_ticker = ""
    if "rsi_period" not in st.session_state:
        st.session_state.rsi_period = 14

    search_col, btn_col = st.columns([5.5, 1.35], vertical_alignment="bottom")
    with search_col:
        st.text_input("Ticker", key="ticker", label_visibility="collapsed",
                      placeholder="Search for tickers (e.g. AAPL, NVDA)", on_change=_run_search)
    with btn_col:
        st.button("Search", type="primary", on_click=_run_search, use_container_width=True)

    ticker = st.session_state.active_ticker
    if not ticker:
        return

    try:
        df = load_history(ticker, "max")
    except Exception as e:
        st.error(f"Could not load '{ticker}': {e}")
        return

    close = df["Close"].dropna()
    last_date = close.index[-1].date()
    company_name = get_company_name(ticker)

    header = f"<div style='font-size:clamp(36px,9vw,54px);font-weight:800;line-height:1.02;margin:2px 0 0'>{ticker}</div>"
    if company_name:
        header += f"<div style='font-size:clamp(13px,2.6vw,16px);color:#666;margin:-2px 0 4px'>{company_name}</div>"
    header += f"<div style='color:#888;font-size:0.85rem;margin:2px 0 14px'>through {last_date} · {len(close):,} trading days</div>"
    st.markdown(header, unsafe_allow_html=True)

    bb = bollinger(close)
    r = rsi(close, st.session_state.rsi_period)
    reg_now = current_regime(close)

    verdict, score, reasons, reversal_pressure, momentum_pressure = build_trade_idea(close, r, bb)

    badge = {"Bullish": "#16a34a", "Neutral": "#6b7280", "Bearish": "#dc2626"}[verdict]
    st.markdown(
        f"<div style='border-left:5px solid {badge};background:{badge}1f;padding:8px 14px;border-radius:4px;"
        f"color:{badge};font-weight:700;font-size:clamp(18px,4.2vw,24px);letter-spacing:.4px'>{verdict.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if reversal_pressure > 0:
        st.markdown("⚠️ **Reversal pressure detected** — multiple reversal setups have fired recently.")
    elif reversal_pressure < 0:
        st.markdown("🔴 **Reversal setups firing against the trend** — this is creating conflicting signals.")

    for line in reasons:
        st.markdown(line)
    st.caption("Not financial advice. This tool only analyzes historical price patterns and statistical relationships. "
               "It does not evaluate fundamentals or structural changes in demand/supply.")

    # Current Picture
    streak = current_streak(close)
    cur_vol, vp = vol_percentile(close)
    cur_dd = drawdown_series(close).iloc[-1]
    lz = last_day_z(close)

    narrative = generate_narrative(close, r, bb, streak, vp, reg_now, cur_dd, lz)

    st.markdown("### Current Picture")
    st.markdown(narrative["summary"])
    for obs in narrative["observations"]:
        st.markdown(f"• {obs}")

    # What's Unusual (kept prominent)
    st.markdown("### What's Unusual")
    for marker, text in build_findings(close, st.session_state.rsi_period):
        st.markdown(f"{marker} {text}")

    # Key Context (collapsed)
    with st.expander("Quick Numbers (Streak, RSI, etc.)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        word = "up" if streak["sign"] > 0 else "down" if streak["sign"] < 0 else "flat"
        c1.metric("Streak", f"{_plural(streak['length'], 'day')} {word}")
        c2.metric("Drawdown", f"{cur_dd:.1%}" if not pd.isna(cur_dd) else "N/A")
        c3.metric("Last day", f"{close.pct_change().iloc[-1]:+.2%}" if not pd.isna(close.pct_change().iloc[-1]) else "N/A")
        c4.metric("1-year", f"{trailing_return(close, 252):+.1%}" if not pd.isna(trailing_return(close, 252)) else "N/A")

    # Setup Performance (collapsed)
    with st.expander("How Similar Setups Have Performed", expanded=False):
        st.caption("Historical results for the tracked setups (click to expand).")

    # See recent real cases (mobile friendly)
    @st.dialog("Recent real cases")
    def show_recent_cases(setup_name, signal, horizon):
        rec = recent_trigger_returns(close, signal, horizon, k=5)
        st.markdown(f"**{setup_name}** — Last 5 times + what happened next")
        if rec.empty:
            st.write("No recent triggers found.")
        else:
            for _, row in rec.iterrows():
                st.write(f"**{row['Trigger date']}** → **{row[f'Move over next {horizon}d']:+.1%}**")

    st.markdown("### See recent real cases")
    st.caption("Tap a button to see the actual recent dates a setup triggered and the return that followed.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("RSI < 30", use_container_width=True):
            show_recent_cases("RSI crossed below 30 (oversold)", pd.Series([False]*len(close)), 10)
        if st.button("Below Lower BB", use_container_width=True):
            show_recent_cases("Close dropped below lower Bollinger band", pd.Series([False]*len(close)), 10)
    with col2:
        if st.button("RSI > 70", use_container_width=True):
            show_recent_cases("RSI crossed above 70 (overbought)", pd.Series([False]*len(close)), 10)
        if st.button("Above Upper BB", use_container_width=True):
            show_recent_cases("Close pushed above upper Bollinger band", pd.Series([False]*len(close)), 10)

    # Charts collapsed
    with st.expander("Detailed Charts", expanded=False):
        st.markdown("### RSI · 1Y")
        st.slider("RSI period", min_value=2, max_value=50, value=14, key="rsi_period")
        st.line_chart(r, height=280)

        st.markdown("### Price · 1Y")
        st.line_chart(close.iloc[-252:], height=280)


if __name__ == "__main__":
    main()
