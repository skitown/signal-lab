"""
Signal Lab - Stable Full Version
================================

This is a complete, self-contained working version with:
- Company name under ticker
- Defensive narrative language for regime shifts
- Rich "What's Unusual" section
- Full Setup Performance tables (with short mobile-friendly labels)
- Working "See recent real cases" using dialogs
- Metrics and backtests in collapsed expanders
- Stronger disclaimer
- All functions and constants properly defined at module level to avoid NameErrors
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


# ----------------------------- Constants -----------------------------
MIN_OCCURRENCES = 10
SIGNAL_HORIZON = 10

SETUP_DIRECTION = {
    "RSI crossed below 30 (oversold)": "bullish",
    "Close dropped below lower Bollinger band": "bullish",
    "RSI crossed above 70 (overbought)": "bearish",
    "Close pushed above upper Bollinger band": "bearish",
}


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
    if yf is None:
        return ""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        name = info.get("longName") or info.get("shortName") or ""
        return name.strip()
    except Exception:
        return ""


# --------------------------- Compute layer ----------------------------

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


def days_underwater(close: pd.Series) -> int:
    dd = drawdown_series(close)
    at_peak = dd >= -1e-12
    if at_peak.iloc[-1]:
        return 0
    last_peak = at_peak[at_peak].index.max()
    return int((close.index > last_peak).sum())


def trailing_return(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return np.nan
    return close.iloc[-1] / close.iloc[-1 - days] - 1.0


def ytd_return(close: pd.Series) -> float:
    yr = close.index[-1].year
    this_year = close[close.index.year == yr]
    if len(this_year) < 2:
        return np.nan
    return this_year.iloc[-1] / this_year.iloc[0] - 1.0


def vol_percentile(close: pd.Series, window: int = 20) -> tuple[float, float]:
    ret = close.pct_change()
    rv = ret.rolling(window).std() * np.sqrt(252)
    cur = rv.iloc[-1]
    pct = float((rv.dropna() < cur).mean())
    return cur, pct


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


def upper_band_walk(close: pd.Series, window: int = 20, n_std: float = 2.0) -> int:
    above = close > bollinger(close, window, n_std)["upper"]
    count = 0
    for val in reversed(above.tolist()):
        if val:
            count += 1
        else:
            break
    return count


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).rename("rsi")


def percentile_of(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s <= value).mean()) if len(s) else float("nan")


def current_regime(close: pd.Series, window: int = 200) -> str:
    sma = close.rolling(window).mean()
    if pd.isna(sma.iloc[-1]):
        return "undefined"
    return "uptrend" if close.iloc[-1] > sma.iloc[-1] else "downtrend"


# ---------------------- Setup triggers & backtests ----------------------

def becomes_true(cond: pd.Series) -> pd.Series:
    return cond & ~cond.shift(1, fill_value=False)


def setup_triggers(close: pd.Series, rsi_series: pd.Series, bb: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "RSI crossed below 30 (oversold)": becomes_true(rsi_series < 30),
        "RSI crossed above 70 (overbought)": becomes_true(rsi_series > 70),
        "Close dropped below lower Bollinger band": becomes_true(close < bb["lower"]),
        "Close pushed above upper Bollinger band": becomes_true(close > bb["upper"]),
    }


def forward_returns(close: pd.Series, signal: pd.Series, horizon: int) -> np.ndarray:
    vals = close.to_numpy()
    n = len(vals)
    idx = np.flatnonzero(signal.to_numpy())
    return np.array([vals[i + horizon] / vals[i] - 1 for i in idx if i + horizon < n])


def baseline_forward(close: pd.Series, horizon: int, mask: pd.Series | None = None) -> np.ndarray:
    vals = close.to_numpy()
    n = len(vals)
    if mask is None:
        return vals[horizon:] / vals[:-horizon] - 1
    idx = np.flatnonzero(mask.to_numpy())
    return np.array([vals[i + horizon] / vals[i] - 1 for i in idx if i + horizon < n])


def backtest_setup(close: pd.Series, signal: pd.Series, horizon: int, baseline_mask: pd.Series | None = None) -> dict | None:
    fwd = forward_returns(close, signal, horizon)
    if len(fwd) == 0:
        return None
    base = baseline_forward(close, horizon, baseline_mask)
    base_mean = float(base.mean()) if len(base) else float("nan")
    return {
        "n": int(len(fwd)),
        "hit_rate": float((fwd > 0).mean()),
        "mean": float(fwd.mean()),
        "median": float(np.median(fwd)),
        "base_hit": float((base > 0).mean()) if len(base) else float("nan"),
        "base_mean": base_mean,
        "edge": float(fwd.mean() - base_mean),
    }


def trend_regime(close: pd.Series, window: int = 200) -> tuple[pd.Series, pd.Series]:
    sma = close.rolling(window).mean()
    defined = sma.notna()
    up = (close > sma) & defined
    down = (close <= sma) & defined
    return up, down


def backtest_by_regime(close: pd.Series, signal: pd.Series, horizon: int, window: int = 200) -> dict[str, dict | None]:
    up, down = trend_regime(close, window)
    return {
        "All": backtest_setup(close, signal, horizon),
        "Uptrend (above 200-day)": backtest_setup(close, signal & up, horizon, up),
        "Downtrend (below 200-day)": backtest_setup(close, signal & down, horizon, down),
    }


def recent_trigger_returns(close: pd.Series, signal: pd.Series, horizon: int, k: int = 5) -> pd.DataFrame:
    vals = close.to_numpy()
    n = len(vals)
    idx = np.flatnonzero(signal.to_numpy())
    rows = []
    for i in idx[-k:]:
        fwd = vals[i + horizon] / vals[i] - 1 if i + horizon < n else np.nan
        rows.append({"Trigger date": close.index[i].strftime("%Y-%m-%d"),
                     f"Move over next {horizon}d": fwd})
    return pd.DataFrame(rows)


# ---------------------- Narrative ----------------------

def generate_narrative(close, rsi, bb, current_streak, vol_percentile, regime, drawdown, last_z):
    last_rsi = rsi.iloc[-1]
    streak_len = current_streak.get("length", 0)

    if regime == "uptrend" and streak_len >= 5 and (last_rsi > 72 or abs(last_z) > 2.0):
        summary = ("Strong bullish trend and momentum, but conditions have become statistically stretched. "
                   "RSI is deeply overbought and the move is extended by historical standards. "
                   "In a normal environment this would often lead to digestion or reversal risk. "
                   "However, if the fundamental demand picture has structurally improved, "
                   "the historical ranges may be less predictive than usual.")
        observations = [
            f"Extreme overbought reading: RSI at {last_rsi:.0f}.",
            f"Extended streak: {streak_len} days up.",
            "This is the classic tension: powerful momentum vs. statistically extreme conditions."
        ]
        return {"summary": summary, "observations": observations, "bucket": "strong_trend_stretched"}

    if regime in ("uptrend", "downtrend") and streak_len >= 4:
        summary = f"Strong {regime} with sustained momentum."
        observations = [
            f"Clear trend: Price well {'above' if regime == 'uptrend' else 'below'} the 200-day average.",
            f"Extended streak: {streak_len} days."
        ]
        return {"summary": summary, "observations": observations, "bucket": "clean_trend"}

    summary = "No dominant directional or reversal pressure stands out at the moment."
    observations = ["The market is in a relatively neutral or mixed state based on these indicators."]
    return {"summary": summary, "observations": observations, "bucket": "quiet_or_mixed"}


# ---------------------- Verdict ----------------------

def build_trade_idea(close, rsi_series, bb):
    regime = current_regime(close)
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
        reasons.append(
            "⚠️ **Note on overbought conditions:** RSI is extremely elevated. "
            "In normal environments this often precedes digestion or reversal. "
            "However, during structural demand shifts the historical relationship can weaken for extended periods."
        )

    verdict = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"
    return verdict, score, reasons, 0, 0


# ------------------------- Findings -------------------------

def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def build_findings(close, rsi_period=14):
    findings = []
    dd = drawdown_series(close)
    streak = current_streak(close)
    cur_vol, vp = vol_percentile(close)
    last_rsi = rsi(close, rsi_period).iloc[-1]
    bb = bollinger(close)
    z = last_day_z(close)

    if streak["length"] >= 4 and streak["n_runs"] > 0:
        share = streak["n_at_least"] / streak["n_runs"]
        word = "up" if streak["sign"] > 0 else "down"
        marker = "⚠️" if share <= 0.10 else "•"
        findings.append((marker, f"On a {streak['length']}-day {word} streak — {streak['n_at_least']} of {streak['n_runs']} {word}-runs have reached that length."))

    cur_dd = dd.iloc[-1]
    worse_share = float((dd.dropna() < cur_dd).mean())
    if cur_dd <= -0.05 and worse_share <= 0.10:
        uw = days_underwater(close)
        findings.append(("⚠️", f"Down {abs(cur_dd):.1%} from peak ({_plural(uw, 'trading day')} underwater) — among its deepest stretches."))

    if vp >= 0.9:
        findings.append(("⚠️", f"Volatility ({cur_vol:.0%} annualized) is in the top 10% of its history — stormy regime."))
    elif vp <= 0.1:
        findings.append(("•", f"Volatility is unusually calm (bottom 10%)."))

    if abs(z) >= 2.5:
        findings.append(("⚠️", f"The most recent day was a {z:+.1f}-sigma move — statistical outlier."))

    bw_pct = percentile_of(bb["bandwidth"], bb["bandwidth"].iloc[-1])
    if bw_pct <= 0.1:
        findings.append(("⚠️", f"Bollinger squeeze — bands tighter than 90% of history. Often precedes a sharp move."))

    walk = upper_band_walk(close)
    if walk >= 3:
        findings.append(("•", f"'Walking the upper band' — {walk} days of closes above the upper Bollinger band (strong trend behavior)."))

    if last_rsi >= 70:
        rsi_pct = percentile_of(rsi(close, rsi_period), last_rsi)
        findings.append(("⚠️", f"RSI is {last_rsi:.0f} (overbought) — higher than on {rsi_pct:.0%} of days."))
    elif last_rsi <= 30:
        findings.append(("⚠️", f"RSI is {last_rsi:.0f} (oversold)."))

    win = close.iloc[-252:]
    from_hi = close.iloc[-1] / win.max() - 1
    from_lo = close.iloc[-1] / win.min() - 1
    if from_hi >= -0.005:
        findings.append(("•", "At or near a fresh 52-week high."))
    elif from_lo <= 0.01:
        findings.append(("•", "At or near a 52-week low."))

    if not findings:
        findings.append(("•", "Nothing notably unusual right now."))

    return findings


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
        "<div style='font-size:0.8rem;color:#888;font-weight:600;letter-spacing:.5px;margin-bottom:4px'>"
        "SIGNAL LAB <span style='font-weight:400'>— find signal in the noise</span></div>",
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

    verdict, score, reasons, _, _ = build_trade_idea(close, r, bb)

    badge = {"Bullish": "#16a34a", "Neutral": "#6b7280", "Bearish": "#dc2626"}[verdict]
    st.markdown(
        f"<div style='border-left:5px solid {badge};background:{badge}1f;padding:8px 14px;border-radius:4px;"
        f"color:{badge};font-weight:700;font-size:clamp(18px,4.2vw,24px);letter-spacing:.4px'>{verdict.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

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

    # What's Unusual (rich)
    st.markdown("### What's Unusual")
    for marker, text in build_findings(close, st.session_state.rsi_period):
        st.markdown(f"{marker} {text}")

    # Quick Numbers (collapsed)
    with st.expander("Quick Numbers", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        word = "up" if streak["sign"] > 0 else "down" if streak["sign"] < 0 else "flat"
        c1.metric("Streak", f"{streak['length']} days {word}")
        c2.metric("Drawdown", f"{cur_dd:.1%}" if not pd.isna(cur_dd) else "N/A")
        c3.metric("Last day", f"{close.pct_change().iloc[-1]:+.2%}")
        c4.metric("1-year", f"{trailing_return(close, 252):+.1%}")

    # Setup Performance (full tables inside expander)
    with st.expander("How Similar Setups Have Performed", expanded=False):
        st.caption("Historical results for the tracked setups.")
        if reg_now != "undefined":
            st.markdown(f"**Current regime:** {reg_now.capitalize()}")

        triggers = setup_triggers(close, r, bb)

        def short_setup(name: str) -> str:
            mapping = {
                "RSI crossed below 30 (oversold)": "RSI < 30",
                "Close dropped below lower Bollinger band": "Below Lower BB",
                "RSI crossed above 70 (overbought)": "RSI > 70",
                "Close pushed above upper Bollinger band": "Above Upper BB",
            }
            return mapping.get(name, name)

        def _row(setup_name, regime_label, stats, last_dt):
            last_str = "—" if last_dt is None else last_dt.strftime("%Y-%m-%d")
            short_name = short_setup(setup_name)
            short_reg = "Up" if "Uptrend" in regime_label else "Down"
            if stats is None:
                return {"Setup": short_name, "Regime": short_reg, "Times": 0,
                        "Avg Move": np.nan, "Baseline": np.nan, "Edge": np.nan, "Win %": np.nan}
            return {"Setup": short_name, "Regime": short_reg,
                    "Times": stats["n"], "Avg Move": stats["mean"],
                    "Baseline": stats["base_mean"], "Edge": stats["edge"],
                    "Win %": stats["hit_rate"]}

        cols = ["Setup", "Regime", "Times", "Avg Move", "Baseline", "Edge", "Win %"]
        fmt = {"Times": "{:.0f}", "Avg Move": "{:+.1%}", "Baseline": "{:+.1%}",
               "Edge": "{:+.1%}", "Win %": "{:.0%}"}

        def render_group(title, names):
            st.markdown(f"**{title}**")
            rows = []
            for name in names:
                sig = triggers[name]
                by_reg = backtest_by_regime(close, sig, 10)
                rows.append(_row(name, "Uptrend (above 200-day)", by_reg["Uptrend (above 200-day)"], None))
                rows.append(_row(name, "Downtrend (below 200-day)", by_reg["Downtrend (below 200-day)"], None))
            df_g = pd.DataFrame(rows)[cols]
            st.dataframe(df_g.style.format(fmt, na_rep="—"), use_container_width=True)

        bullish = [n for n in triggers if SETUP_DIRECTION.get(n) == "bullish"]
        bearish = [n for n in triggers if SETUP_DIRECTION.get(n) == "bearish"]
        render_group("Buy-side setups", bullish)
        render_group("Sell-side setups", bearish)

    # See recent real cases (functional)
    @st.dialog("Recent real cases")
    def show_recent_cases(setup_name, signal):
        rec = recent_trigger_returns(close, signal, 10, k=5)
        st.markdown(f"**{setup_name}** — Last 5 times + what happened next")
        if rec.empty:
            st.write("No recent triggers found.")
        else:
            for _, row in rec.iterrows():
                st.write(f"**{row['Trigger date']}** → **{row['Move over next 10d']:+.1%}**")

    st.markdown("### See recent real cases")
    st.caption("Tap a button to see the actual recent dates a setup triggered and the return that followed.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("RSI < 30", use_container_width=True):
            show_recent_cases("RSI crossed below 30 (oversold)", triggers["RSI crossed below 30 (oversold)"])
        if st.button("Below Lower BB", use_container_width=True):
            show_recent_cases("Close dropped below lower Bollinger band", triggers["Close dropped below lower Bollinger band"])
    with col2:
        if st.button("RSI > 70", use_container_width=True):
            show_recent_cases("RSI crossed above 70 (overbought)", triggers["RSI crossed above 70 (overbought)"])
        if st.button("Above Upper BB", use_container_width=True):
            show_recent_cases("Close pushed above upper Bollinger band", triggers["Close pushed above upper Bollinger band"])

    # Charts (collapsed)
    with st.expander("Detailed Charts", expanded=False):
        st.line_chart(close.iloc[-252:], height=280)
        st.line_chart(r.iloc[-252:], height=280)


if __name__ == "__main__":
    main()
