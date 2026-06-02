"""
Signal Lab - Roll Forward v1 (Richer Features Restored + Hourly)
===============================================================

Base: User's last known good "Clean Rollback Version" (2026-06-05)
This version restores the richer, more complete functionality while keeping
the stable structure and all the improvements the user liked.

Restored / Improved:
- Rich, detailed "What's Unusual" section with proper rarity, stats, and context
- Full Setup Performance tables with real backtest data (short mobile-friendly labels)
- Working "See recent real cases" dialogs that show actual recent triggers + returns
- Bollinger Bands chart restored prominently
- All previous defensive language, company name, smaller verdict, stronger disclaimer, collapsed sections
- All functions properly defined at module level (no more NameError / TypeError on load)

NEW in this update:
- Intraday (1h) support via yfinance to surface "oversold on hourly" signals (the exact blind spot
  from the Slack post: "Bought 200 Jan 2027 450 calls. Oversold on hourly Just a trade").
  - load_intraday (60d 1h, graceful), hourly_oversold() reusing rsi/bollinger
  - Narrative + build_findings now detect + call out hourly oversold (esp. when daily RSI >30)
  - Caption + 1h snapshot line in UI for immediate visibility
  - Strong defensive caveats: "noisier, weaker edges, use for timing not conviction"
- Directly answers "I don't think our tool would have shown me this info would it?"

This is a meaningful step forward from the stable baseline.
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
OVERSOLD_RSI = 30   # RSI at or below this level is treated as oversold (common trader usage + matches the "oversold on hourly" language)

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


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_intraday(ticker: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    """Load recent intraday bars. 1h capped by yfinance (~60-730d max); graceful empty on fail."""
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


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


def _get_band_position(close: float, bb: pd.DataFrame) -> str:
    """Return 'above_upper', 'below_lower', 'upper_half', or 'lower_half'."""
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


def _get_band_position(close: float, bb: pd.DataFrame) -> str:
    """Determine if price is above upper, below lower, or in the middle band area."""
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


def hourly_oversold(close_h: pd.Series, rsi_period: int = 14) -> dict:
    """Return hourly oversold status for surfacing 'oversold on hourly' like the Slack trade example.
    Returns dict with available, last_rsi, below_lower_bb, rsi_oversold, is_oversold, last_ts.
    Safe on short series (needs ~30 bars for stable RSI)."""
    if close_h is None or len(close_h) < 30:
        return {"available": False, "is_oversold": False}
    r = rsi(close_h, rsi_period)
    bb = bollinger(close_h)
    last_rsi = float(r.iloc[-1])
    last_close = float(close_h.iloc[-1])
    lower = bb["lower"].iloc[-1]
    below_lower = last_close < lower if not pd.isna(lower) else False
    rsi_os = last_rsi <= OVERSOLD_RSI
    return {
        "available": True,
        "last_rsi": last_rsi,
        "below_lower_bb": bool(below_lower),
        "rsi_oversold": bool(rsi_os),
        "is_oversold": bool(rsi_os or below_lower),
        "last_ts": close_h.index[-1],
    }


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


# ---------------------- Narrative (defensive) ----------------------

def generate_narrative(close, rsi, bb, current_streak, vol_percentile, regime, drawdown, last_z, intraday=None):
    last_rsi = rsi.iloc[-1]
    streak_len = current_streak.get("length", 0)
    band_pos = _get_band_position(close.iloc[-1], bb)

    reversal_signals = []
    momentum_signals = []

    # Detect active signals
    if last_rsi < 30:
        reversal_signals.append("RSI deeply oversold")
    if band_pos == "below_lower":
        reversal_signals.append("Price at lower Bollinger Band")
    if last_rsi > 70:
        momentum_signals.append("RSI deeply overbought")
    if band_pos == "above_upper":
        momentum_signals.append("Price at upper Bollinger Band")
    if streak_len >= 6:
        if regime == "uptrend":
            momentum_signals.append(f"Very long {streak_len}-day up streak")
        else:
            reversal_signals.append(f"Very long {streak_len}-day down streak")

    # Core synthesis focused on reversal likelihood (directly addressing the feedback)
    if len(reversal_signals) >= 2:
        if regime == "uptrend":
            summary = "Strong uptrend, but multiple reversal signals are now active. The stock is statistically stretched and mean-reversion risk is elevated."
            observations = [
                "Reversal pressure: " + ", ".join(reversal_signals),
                "These conditions have historically led to digestion or pullbacks more often than continuation in similar uptrends."
            ]
        else:
            summary = "Downtrend with multiple oversold signals firing. A short-term bounce or reversal is the more common historical outcome from this setup."
            observations = [
                "Reversal setups active: " + ", ".join(reversal_signals),
                "Watch for stabilization — these setups have often produced bounces in the current regime."
            ]
    elif len(momentum_signals) >= 2 and regime == "uptrend":
        summary = "Powerful uptrend with strong momentum. Conditions are stretched, but continuation has historically been the dominant outcome in similar setups."
        observations = [
            "Momentum still dominant: " + ", ".join(momentum_signals),
            "These can persist for a while in strong trends (especially if fundamentals are supportive)."
        ]
    else:
        summary = f"Clear {regime} with no strong reversal signals currently active."
        observations = ["The dominant trend remains in control based on these indicators."]

    # Intraday hourly context (addresses the exact gap: "oversold on hourly" trades that daily misses)
    if intraday and intraday.get("available") and intraday.get("is_oversold"):
        h_rsi = intraday.get("last_rsi", 0)
        note = f"Hourly oversold (RSI {h_rsi:.0f}"
        if intraday.get("below_lower_bb"):
            note += ", below lower band"
        note += ")."
        if last_rsi >= 35:
            note += " Daily not yet oversold — this can be a short-term timing signal for a potential bounce entry (as in the 'oversold on hourly' options trade example)."
        else:
            note += " Daily also oversold — stronger multi-timeframe confluence."
        observations.append(note)
        observations.append("⚠️ Note: Intraday signals are noisier and have weaker, shorter-lived edges than daily. This is useful for timing within a broader thesis, not as standalone conviction. Structural regime shifts can invalidate historical intraday mean-reversion too.")

    return {"summary": summary, "observations": observations, "bucket": "assessment"}


# ---------------------- Verdict (with defensive note) ----------------------

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


# ------------------------- Rich Findings -------------------------

def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def build_findings(close, rsi_period=14, intraday=None):
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

    # Key new item: surface hourly oversold even when daily is quiet (directly for the Slack example)
    if intraday and intraday.get("available") and intraday.get("is_oversold"):
        h_rsi = intraday.get("last_rsi", 0)
        if last_rsi > OVERSOLD_RSI:
            findings.append(("⚠️", f"Oversold on hourly (RSI {h_rsi:.0f}) — daily RSI not oversold. This is the exact class of short-term signal used in the 'Bought ... Oversold on hourly' trade."))
        else:
            findings.append(("•", f"Hourly also oversold (RSI {h_rsi:.0f})."))

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

        /* === GREEN PRIMARY BUTTON + SEARCH FIELD (trivial color-only, layout is handled by columns) ===
           Streamlit primary buttons and input focus rings inherit the theme primaryColor (which can be red or blue).
           We override ONLY colors here with !important. This is the easy part.
           (The hard mobile layout fights for button positioning were deliberately avoided by using columns + vertical_alignment.)
        */
        /* Search button green (matches Bullish verdict #16a34a) */
        button[kind="primary"],
        button[data-testid="baseButton-primary"] {
          background-color: #16a34a !important;
          border-color: #16a34a !important;
          color: white !important;
        }
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
          background-color: #15803d !important;
          border-color: #15803d !important;
        }
        button[kind="primary"]:active,
        button[data-testid="baseButton-primary"]:active {
          background-color: #166534 !important;
          border-color: #166534 !important;
        }

        /* Green focus ring / outline on the search text input (instead of red or default primary) */
        [data-testid="stTextInput"] input:focus,
        div[data-baseweb="input"] input:focus {
          border-color: #16a34a !important;
          box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.25) !important;
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

    # Intraday (1h) load — short TTL cache; gracefully skips if yf can't deliver (common for some symbols)
    intraday = None
    try:
        idf = load_intraday(ticker, "1h", "60d")
        if not idf.empty:
            h_close = idf["Close"].dropna()
            intraday = hourly_oversold(h_close, st.session_state.rsi_period)
    except Exception:
        intraday = None

    # === PROMINENT OVERSOLD ALARM BELLS ===
    # Placed high and loud right after the verdict so oversold conditions (daily or hourly) are impossible to miss.
    daily_oversold = r.iloc[-1] <= OVERSOLD_RSI
    hourly_oversold_now = intraday and intraday.get("available") and intraday.get("is_oversold")
    if daily_oversold or hourly_oversold_now:
        alarm_lines = []
        if daily_oversold:
            alarm_lines.append(f"🚨 DAILY OVERSOLD — RSI {r.iloc[-1]:.0f} (classic reversal setup)")
        if hourly_oversold_now:
            h_r = intraday.get("last_rsi", 0)
            if daily_oversold:
                alarm_lines.append(f"🚨 HOURLY OVERSOLD — RSI {h_r:.0f} (multi-timeframe confluence)")
            else:
                alarm_lines.append(f"🚨 HOURLY OVERSOLD — RSI {h_r:.0f} (short-term bounce/timing signal while daily is NOT oversold)")

        alarm_html = (
            "<div style='background:#fef2f2;border:3px solid #b91c1c;border-radius:8px;"
            "padding:14px 16px;margin:12px 0 8px'>"
            "<div style='color:#991b1b;font-size:clamp(15px,3.8vw,18px);font-weight:900;"
            "letter-spacing:0.5px;margin-bottom:6px'>🚨 OVERSOLD ALERTS — TREAT THESE AS POTENTIAL REVERSAL / BOUNCE SIGNALS</div>"
        )
        for line in alarm_lines:
            alarm_html += f"<div style='color:#7f1d1d;font-size:clamp(14px,3.2vw,16px);font-weight:700;margin:3px 0 3px 4px'>{line}</div>"
        alarm_html += (
            "<div style='color:#9f1239;font-size:0.72rem;margin-top:8px;line-height:1.3'>"
            "These are statistical observations only. Mean-reversion can fail (especially in strong trends or during structural demand shifts). "
            "Hourly signals are noisier and best used for entry timing within a larger thesis, not standalone conviction."
            "</div></div>"
        )
        st.markdown(alarm_html, unsafe_allow_html=True)

    # Current Picture
    streak = current_streak(close)
    cur_vol, vp = vol_percentile(close)
    cur_dd = drawdown_series(close).iloc[-1]
    lz = last_day_z(close)

    narrative = generate_narrative(close, r, bb, streak, vp, reg_now, cur_dd, lz, intraday=intraday)

    st.markdown("### Current Picture")
    st.markdown(narrative["summary"])
    for obs in narrative["observations"]:
        st.markdown(f"• {obs}")

    # Minimal intraday badge line when data present (so user immediately sees hourly was considered)
    if intraday and intraday.get("available"):
        h_r = intraday.get("last_rsi", 0)
        h_flag = "⚠️ OVERSOLD" if intraday.get("is_oversold") else "neutral"
        st.caption(f"🕐 1h snapshot (last bar ~{intraday.get('last_ts'):%Y-%m-%d %H:%M}): RSI {h_r:.0f} — {h_flag}")

    # What's Unusual (rich version)
    st.markdown("### What's Unusual")
    for marker, text in build_findings(close, st.session_state.rsi_period, intraday=intraday):
        st.markdown(f"{marker} {text}")

    # Quick Numbers (collapsed)
    with st.expander("Quick Numbers", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        word = "up" if streak["sign"] > 0 else "down" if streak["sign"] < 0 else "flat"
        c1.metric("Streak", f"{streak['length']} days {word}")
        c2.metric("Drawdown", f"{cur_dd:.1%}" if not pd.isna(cur_dd) else "N/A")
        c3.metric("Last day", f"{close.pct_change().iloc[-1]:+.2%}")
        c4.metric("1-year", f"{trailing_return(close, 252):+.1%}")

    # Setup Performance (full tables)
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

    # See recent real cases (now functional)
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

    # Charts (with Bollinger restored prominently)
    st.markdown("### Price & Bollinger Bands (20, 2σ)")
    band_df = pd.DataFrame({
        "Close": close,
        "Upper": bb["upper"],
        "Mid": bb["mid"],
        "Lower": bb["lower"],
    }).iloc[-252:]
    st.line_chart(band_df, height=320)

    with st.expander("Additional Charts", expanded=False):
        st.markdown("### RSI (last 1 year)")
        st.line_chart(r.iloc[-252:], height=280)

        st.markdown("### Trailing Returns")
        rets = {
            "1 week (5d)": trailing_return(close, 5),
            "1 month (21d)": trailing_return(close, 21),
            "3 months (63d)": trailing_return(close, 63),
            "1 year (252d)": trailing_return(close, 252),
            "YTD": ytd_return(close),
        }
        st.dataframe(pd.DataFrame({"Return": rets}).style.format("{:+.2%}", na_rep="—"), use_container_width=True)

        st.markdown("### Drawdown")
        dd_df = drawdown_series(close).rename("Drawdown").reset_index().iloc[-252:]
        dd_df.columns = ["Date", "Drawdown"]
        underwater = (
            alt.Chart(dd_df)
            .mark_area(color="#3b82f6", opacity=0.85, line={"color": "#1d4ed8"})
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Drawdown:Q", title=None, axis=alt.Axis(format="%")),
            )
            .properties(height=240)
        )
        st.altair_chart(underwater, use_container_width=True)


if __name__ == "__main__":
    main()
