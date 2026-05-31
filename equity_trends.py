"""
Signal Lab
=============================================

Major cleanup of reversal-related information.

Changes in this version:
- Removed the awkward "Reversal Signals" section that was causing confusion.
- Strengthened the "Current Picture" narrative to better call out reversal pressure / exhaustion when relevant.
- Made "What's Unusual" the primary place for surfacing potential reversal risks in context.
- Improved logic so overbought conditions in strong uptrends are highlighted as possible exhaustion (instead of hiding them).
- Proper N/A handling where data isn't applicable.
- Kept the overall structure cleaner.
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


# --------------------------- Compute layer ----------------------------

def run_table(close: pd.Series) -> pd.DataFrame:
    sign = np.sign(close.diff()).fillna(0)
    run_id = (sign != sign.shift()).cumsum()
    run_len = sign.groupby(run_id).cumcount() + 1
    return pd.DataFrame(
        {"sign": sign, "run_id": run_id, "run_len": run_len}, index=close.index
    )


def completed_runs(close: pd.Series) -> pd.DataFrame:
    rt = run_table(close)
    return rt.groupby("run_id").agg(sign=("sign", "first"), length=("run_len", "max"))


def current_streak(close: pd.Series) -> dict:
    rt = run_table(close)
    sign = int(rt["sign"].iloc[-1])
    length = int(rt["run_len"].iloc[-1])
    runs = completed_runs(close)
    same_dir = runs[runs["sign"] == sign]
    n_runs = len(same_dir)
    n_at_least = int((same_dir["length"] >= length).sum()) if n_runs else 0
    return {
        "sign": sign,
        "length": length,
        "n_at_least": n_at_least,
        "n_runs": n_runs,
    }


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
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower,
         "pct_b": pct_b, "bandwidth": bandwidth},
        index=close.index,
    )


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


# ---------------------- Narrative Layer (Improved) ----------------------

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


def generate_narrative(
    close: pd.Series,
    rsi: pd.Series,
    bb: pd.DataFrame,
    current_streak: dict,
    vol_percentile: float,
    regime: str,
    drawdown: float,
    last_z: float,
) -> dict:
    last_close = close.iloc[-1]
    last_rsi = rsi.iloc[-1]
    band_pos = _get_band_position(last_close, bb)
    bandwidth_pct = bb["bandwidth"].iloc[-1] / bb["bandwidth"].rolling(252).mean().iloc[-1] if len(bb) > 252 else 1.0

    streak_len = current_streak.get("length", 0)
    streak_sign = current_streak.get("sign", 0)
    streak_word = "up" if streak_sign > 0 else "down" if streak_sign < 0 else "flat"

    observations = []
    bucket = "mixed"
    summary = "The current picture is mixed."

    # Strong exhaustion / potential reversal in uptrend
    if regime == "uptrend" and last_rsi > 75 and streak_len >= 5 and (band_pos == "above_upper" or last_z > 2.5):
        bucket = "uptrend_exhaustion"
        summary = ("Strong upward momentum with signs of exhaustion. The move has been extended, "
                   "with RSI deeply overbought and price pushing well outside the Bollinger Bands.")
        observations.append(f"Extreme overbought reading: RSI at {last_rsi:.0f}.")
        if streak_len >= 5:
            observations.append(f"Extended streak: {streak_len} days up.")
        observations.append("Price is stretched relative to recent volatility.")

    # Strong capitulation / potential reversal in downtrend
    elif regime == "downtrend" and last_rsi < 25 and streak_len >= 5 and (band_pos == "below_lower" or last_z < -2.5):
        bucket = "downtrend_exhaustion"
        summary = ("Sharp decline with signs of capitulation. RSI is deeply oversold and price has moved well outside the lower Bollinger Band.")
        observations.append(f"Extreme oversold reading: RSI at {last_rsi:.0f}.")
        if streak_len >= 5:
            observations.append(f"Extended down streak: {streak_len} days.")
        observations.append("Significant downside move relative to recent volatility.")

    # Classic mean-reversion setup (more relevant when against trend)
    elif (band_pos == "below_lower" and last_rsi < 30 and streak_sign <= 0) or \
         (band_pos == "above_upper" and last_rsi > 70 and streak_sign >= 0):
        bucket = "classic_reversal"
        direction = "downside" if band_pos == "above_upper" else "upside"
        summary = ("Price has reached a statistically stretched level after a prolonged move. "
                   f"Multiple signals are lining up that often precede a {direction} reaction.")
        observations.append(f"Stretched condition: Price at {band_pos.replace('_', ' ')} with RSI at extreme levels.")
        observations.append(f"The current move ranks as an outlier relative to recent volatility (z-score {last_z:+.1f}).")

    # Clean strong trend
    elif regime in ("uptrend", "downtrend") and streak_len >= 4 and 0.35 < vol_percentile < 0.80:
        if (regime == "uptrend" and band_pos in ("above_upper", "upper_half")) or \
           (regime == "downtrend" and band_pos in ("below_lower", "lower_half")):
            bucket = "clean_trend"
            summary = (f"Strong {regime} with sustained momentum. Price has been moving along the outer Bollinger Band.")
            observations.append(f"Clear trend: Price well {'above' if regime == 'uptrend' else 'below'} the 200-day average.")
            observations.append(f"Extended streak: {streak_len} days {streak_word}.")

    # Squeeze
    elif bandwidth_pct < 0.55:
        bucket = "squeeze_low_conviction"
        summary = "Volatility has compressed to unusually low levels. A significant move is likely, but direction is not yet clear."
        observations.append(f"Band width is in the bottom {bandwidth_pct:.0%} of its own history.")

    # High volatility outlier
    elif vol_percentile > 0.80 and abs(last_z) > 2.5:
        bucket = "high_vol_outlier"
        summary = "A very large move occurred recently in a high-volatility environment. The situation is noisy."
        observations.append(f"Extreme move: {last_z:+.1f} standard deviations.")
        observations.append(f"Volatility is in the top {vol_percentile:.0%} of its history.")

    else:
        bucket = "quiet_or_mixed"
        summary = "No dominant directional or reversal pressure stands out at the moment."
        observations.append("The market is in a relatively neutral or mixed state based on these indicators.")

    return {
        "summary": summary,
        "observations": observations,
        "bucket": bucket,
    }


# ---------------------- Original compute functions (kept) --------------------

MIN_OCCURRENCES = 20
SIGNAL_HORIZON = 10

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


def current_regime(close: pd.Series, window: int = 200) -> str:
    up, down = trend_regime(close, window)
    if up.iloc[-1]:
        return "uptrend"
    if down.iloc[-1]:
        return "downtrend"
    return "undefined"


def backtest_by_regime(close: pd.Series, signal: pd.Series, horizon: int, window: int = 200) -> dict[str, dict | None]:
    up, down = trend_regime(close, window)
    return {
        "All": backtest_setup(close, signal, horizon),
        "Uptrend (above 200-day)": backtest_setup(close, signal & up, horizon, up),
        "Downtrend (below 200-day)": backtest_setup(close, signal & down, horizon, down),
    }


SETUP_DIRECTION = {
    "RSI crossed below 30 (oversold)": "bullish",
    "Close dropped below lower Bollinger band": "bullish",
    "RSI crossed above 70 (overbought)": "bearish",
    "Close pushed above upper Bollinger band": "bearish",
}
PROPOSAL_LOOKBACK = 5


def last_trigger_date(signal: pd.Series):
    hits = signal[signal]
    return hits.index[-1] if len(hits) else None


def recent_trigger_returns(close: pd.Series, signal: pd.Series, horizon: int, k: int = 6) -> pd.DataFrame:
    vals = close.to_numpy()
    n = len(vals)
    idx = np.flatnonzero(signal.to_numpy())
    rows = []
    for i in idx[-k:]:
        fwd = vals[i + horizon] / vals[i] - 1 if i + horizon < n else np.nan
        rows.append({"Trigger date": close.index[i].strftime("%Y-%m-%d"),
                     f"Move over next {horizon}d": fwd})
    return pd.DataFrame(rows)


def build_trade_idea(close: pd.Series, rsi_series: pd.Series, bb: pd.DataFrame) -> tuple[str, int, list[str]]:
    regime = current_regime(close)
    reg_key = {"uptrend": "Uptrend (above 200-day)",
               "downtrend": "Downtrend (below 200-day)"}.get(regime, "All")
    reasons: list[str] = []
    score = 0

    if regime == "uptrend":
        score += 1
        reasons.append("🟢 **Trend:** above the 200-day average (supports bullish bias).")
    elif regime == "downtrend":
        score -= 1
        reasons.append("🔴 **Trend:** below the 200-day average (supports bearish bias).")

    m3 = trailing_return(close, 63)
    if not np.isnan(m3):
        if m3 > 0:
            score += 1
            reasons.append(f"🟢 **Momentum:** +{m3:.0%} over the last ~3 months.")
        elif m3 < 0:
            score -= 1
            reasons.append(f"🔴 **Momentum:** {m3:+.0%} over the last ~3 months.")

    reversal_setups = [
        "RSI crossed below 30 (oversold)",
        "Close dropped below lower Bollinger band"
    ]
    momentum_setups = [
        "RSI crossed above 70 (overbought)",
        "Close pushed above upper Bollinger band"
    ]

    reversal_pressure = 0
    momentum_pressure = 0

    for name, sig in setup_triggers(close, rsi_series, bb).items():
        if not bool(sig.iloc[-PROPOSAL_LOOKBACK:].any()):
            continue

        stats = backtest_by_regime(close, sig, SIGNAL_HORIZON).get(reg_key) or backtest_by_regime(close, sig, SIGNAL_HORIZON)["All"]

        if stats is None or stats["n"] < MIN_OCCURRENCES:
            continue

        if stats["mean"] > 0:
            if name in reversal_setups:
                reversal_pressure += 1
                reasons.append(f"🟢 **Reversal setup fired:** {name} (historically positive in this regime).")
            else:
                momentum_pressure += 1
                reasons.append(f"🟢 **Momentum setup fired:** {name} (historically positive in this regime).")
        else:
            if name in reversal_setups:
                reversal_pressure -= 1
                reasons.append(f"🔴 **Reversal setup fired:** {name} (historically negative in this regime).")
            else:
                momentum_pressure -= 1
                reasons.append(f"🔴 **Momentum setup fired:** {name} (historically negative in this regime).")

    if reversal_pressure > 0:
        score += 1
    if reversal_pressure < 0:
        score -= 1

    if momentum_pressure > 0:
        score += 1
    if momentum_pressure < 0:
        score -= 1

    if score >= 2:
        verdict = "Bullish"
    elif score <= -2:
        verdict = "Bearish"
    else:
        verdict = "Neutral"

    if not reasons:
        reasons.append("Not enough history yet to read the signals.")

    return verdict, score, reasons, reversal_pressure, momentum_pressure


# ------------------------- Findings heuristics -------------------------

STREAK_MIN = 4
DRAWDOWN_MIN = -0.05
TAIL = 0.10


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def build_findings(close: pd.Series, rsi_period: int = 14) -> list[tuple[str, str]]:
    findings = []
    dd = drawdown_series(close)

    streak = current_streak(close)
    if streak["sign"] != 0 and streak["length"] >= STREAK_MIN and streak["n_runs"] > 0:
        word = "up" if streak["sign"] > 0 else "down"
        share = streak["n_at_least"] / streak["n_runs"]
        msg = (f"On a {streak['length']}-day {word} streak — "
               f"{streak['n_at_least']} of {streak['n_runs']} {word}-runs in this "
               f"window have reached that length.")
        findings.append(("⚠️" if share <= TAIL else "•", msg))

    cur_dd = dd.iloc[-1]
    worse_share = float((dd.dropna() < cur_dd).mean())
    if cur_dd <= DRAWDOWN_MIN and worse_share <= TAIL:
        uw = days_underwater(close)
        findings.append(("⚠️",
            f"Down {abs(cur_dd):.1%} from its peak ({_plural(uw, 'trading day')} "
            f"underwater) — among its deepest stretches: only {worse_share:.0%} of "
            f"days in this window have been worse."))

    cur_vol, vp = vol_percentile(close)
    if vp >= 1 - TAIL:
        findings.append(("⚠️", f"Volatility ({cur_vol:.0%} annualized) is in the "
                              f"top {1 - vp:.0%} of its own history — a stormy regime."))
    elif vp <= TAIL:
        findings.append(("•", f"Volatility ({cur_vol:.0%} annualized) is unusually "
                             f"calm — bottom {vp:.0%} of its history."))

    z = last_day_z(close)
    if abs(z) >= 2.5:
        findings.append(("⚠️", f"The most recent day was a {z:+.1f}-sigma move — "
                              f"a statistical outlier."))

    bb = bollinger(close)
    bw_pct = percentile_of(bb["bandwidth"], bb["bandwidth"].iloc[-1])
    if bw_pct <= TAIL:
        findings.append(("⚠️", f"Bollinger squeeze — bands are tighter than "
                              f"{1 - bw_pct:.0%} of this stock's history. Tight bands "
                              f"often precede a sharp move (direction unknown)."))
    walk = upper_band_walk(close)
    if walk >= 3:
        findings.append(("•", f"'Walking the upper band' — {_plural(walk, 'day')} of "
                             f"closes above the upper Bollinger band, the strong-uptrend "
                             f"behaviour. (Not a forecast — strong trends also end.)"))

    r = rsi(close, rsi_period)
    cur_rsi = r.iloc[-1]
    if pd.notna(cur_rsi):
        rsi_pct = percentile_of(r, cur_rsi)
        if cur_rsi >= 70:
            findings.append(("⚠️", f"RSI is {cur_rsi:.0f} (overbought territory) — "
                                  f"higher than on {rsi_pct:.0%} of days in this window."))
        elif cur_rsi <= 30:
            findings.append(("⚠️", f"RSI is {cur_rsi:.0f} (oversold territory) — "
                                  f"lower than on {1 - rsi_pct:.0%} of days in this window."))

    regime = current_regime(close)
    bt_signals = setup_triggers(close, r, bb)
    for name, sig in bt_signals.items():
        if not bool(sig.iloc[-1]):
            continue
        by_reg = backtest_by_regime(close, sig, SIGNAL_HORIZON)
        reg_key = {"uptrend": "Uptrend (above 200-day)",
                   "downtrend": "Downtrend (below 200-day)"}.get(regime, "All")
        stats = by_reg.get(reg_key) or by_reg["All"]
        reg_label = "" if regime == "undefined" else f" while in an {regime}"
        if stats is None or stats["n"] < MIN_OCCURRENCES:
            n_seen = 0 if stats is None else stats["n"]
            findings.append(("•", f"Setup triggered today: {name}{reg_label}. Too few "
                                 f"comparable past occurrences ({n_seen}) to judge it."))
            continue
        direction = "beat" if stats["edge"] > 0 else "lagged"
        marker = "⚠️" if stats["edge"] > 0 else "•"
        findings.append((marker,
            f"Setup triggered today: {name}{reg_label}. Over the next {SIGNAL_HORIZON} "
            f"days historically ({stats['n']} comparable times), this stock averaged "
            f"{stats['mean']:+.1%} — it {direction} its {stats['base_mean']:+.1%} "
            f"same-regime baseline by {stats['edge']:+.1%}; up {stats['hit_rate']:.0%} "
            f"of the time. Historical, not a promise."))

    win = close.iloc[-252:]
    from_hi = close.iloc[-1] / win.max() - 1
    from_lo = close.iloc[-1] / win.min() - 1
    if from_hi >= -0.005:
        findings.append(("•", "At or near a fresh 52-week high."))
    elif from_lo <= 0.01:
        findings.append(("•", "At or near a 52-week low."))

    if not findings:
        findings.append(("•", "Nothing notably unusual right now by these measures."))
    return findings


# ------------------------------- UI ------------------------------------

def _run_search() -> None:
    st.session_state.active_ticker = st.session_state.ticker.strip().upper()


def main():
    st.set_page_config(page_title="Signal Lab", layout="wide")

    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
          .block-container,
          [data-testid="stMainBlockContainer"],
          [data-testid="stAppViewBlockContainer"] {
            padding: 2.5rem 0.9rem 3rem !important;
          }
          div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
          div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
        }
        div[data-testid="stDataFrame"] > div {
          overflow-x: auto;
        }
        div[data-testid="stButton"] button {
          white-space: nowrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.8rem;color:#888;font-weight:600;letter-spacing:.5px;"
        "margin-bottom:4px'>SIGNAL LAB "
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
                      placeholder="Search for tickers (e.g. AAPL, NVDA)",
                      on_change=_run_search)
    with btn_col:
        st.button("Search", type="primary", on_click=_run_search, use_container_width=True)

    rsi_period = st.session_state.rsi_period
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
    st.markdown(
        f"<div style='font-size:clamp(36px,9vw,54px);font-weight:800;line-height:1.02;"
        f"margin:2px 0 0'>{ticker}</div>"
        f"<div style='color:#888;font-size:0.85rem;margin:2px 0 14px'>"
        f"through {last_date} · {len(close):,} trading days</div>",
        unsafe_allow_html=True,
    )

    bb = bollinger(close)
    r = rsi(close, rsi_period)
    reg_now = current_regime(close)

    verdict, score, reasons, reversal_pressure, momentum_pressure = build_trade_idea(close, r, bb)

    badge = {"Bullish": "#16a34a", "Neutral": "#6b7280", "Bearish": "#dc2626"}[verdict]
    st.markdown(
        f"<div style='border-left:6px solid {badge};background:{badge}1f;"
        f"padding:clamp(10px,2.5vw,14px) clamp(14px,3.5vw,20px);border-radius:4px;"
        f"color:{badge};font-weight:800;font-size:clamp(22px,6vw,34px);"
        f"letter-spacing:.5px'>{verdict.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if reversal_pressure > 0:
        st.markdown("⚠️ **Reversal pressure detected** — multiple reversal setups have fired recently.")
    elif reversal_pressure < 0:
        st.markdown("🔴 **Reversal setups firing against the trend** — this is creating conflicting signals.")

    for line in reasons:
        st.markdown(line)
    st.caption("Not financial advice.")

    # ===================== Current Picture (Narrative) =====================
    streak = current_streak(close)
    cur_vol, vp = vol_percentile(close)
    cur_dd = drawdown_series(close).iloc[-1]
    lz = last_day_z(close)

    narrative = generate_narrative(
        close=close,
        rsi=r,
        bb=bb,
        current_streak=streak,
        vol_percentile=vp,
        regime=reg_now,
        drawdown=cur_dd,
        last_z=lz,
    )

    st.markdown("### Current Picture")
    st.markdown(narrative["summary"])
    for obs in narrative["observations"]:
        st.markdown(f"• {obs}")

    # ===================== Snapshot =====================
    c1, c2, c3, c4 = st.columns(4)
    word = "up" if streak["sign"] > 0 else "down" if streak["sign"] < 0 else "flat"
    c1.metric("Streak", f"{_plural(streak['length'], 'day')} {word}")
    c2.metric("Drawdown", f"{cur_dd:.1%}" if not pd.isna(cur_dd) else "N/A")
    c3.metric("Last day", f"{close.pct_change().iloc[-1]:+.2%}" if not pd.isna(close.pct_change().iloc[-1]) else "N/A")
    c4.metric("1-year", f"{trailing_return(close, 252):+.1%}" if not pd.isna(trailing_return(close, 252)) else "N/A")

    cur_rsi = r.iloc[-1]
    i1, i2, i3 = st.columns(3)
    pb = bb["pct_b"].iloc[-1]
    i1.metric("%B", f"{pb:.2f}" if not pd.isna(pb) else "N/A")
    i2.metric(f"RSI ({rsi_period})", "—" if pd.isna(cur_rsi) else f"{cur_rsi:.0f}")
    bw_pct = percentile_of(bb["bandwidth"], bb["bandwidth"].iloc[-1])
    i3.metric("Band width", f"{bw_pct:.0%}" if not pd.isna(bw_pct) else "N/A")

    # ===================== What's Unusual (primary place for unusual/reversal info) =====================
    st.markdown("### What's Unusual")
    for marker, text in build_findings(close, rsi_period):
        st.markdown(f"{marker} {text}")

    # ===================== All Setup Backtests =====================
    st.markdown("### All Setup Backtests")
    st.caption("Historical performance after every tracked setup (for reference).")

    if reg_now != "undefined":
        st.markdown(f"**Current regime:** {reg_now.capitalize()}")

    triggers = setup_triggers(close, r, bb)
    up_mask, down_mask = trend_regime(close)

    def _row(setup_name, regime_label, stats, last_dt):
        last_str = "—" if last_dt is None else last_dt.strftime("%Y-%m-%d")
        if stats is None:
            return {"Setup": setup_name, "Regime": regime_label, "Times triggered": 0,
                    "Avg next move": np.nan, "Baseline": np.nan, "Edge vs hold": np.nan,
                    "Win rate": np.nan, "Last triggered": last_str}
        return {"Setup": setup_name, "Regime": regime_label,
                "Times triggered": stats["n"], "Avg next move": stats["mean"],
                "Baseline": stats["base_mean"], "Edge vs hold": stats["edge"],
                "Win rate": stats["hit_rate"], "Last triggered": last_str}

    cols = ["Setup", "Regime", "Times triggered", "Avg next move", "Baseline",
            "Edge vs hold", "Win rate", "Last triggered"]
    fmt = {"Times triggered": "{:.0f}", "Avg next move": "{:+.1%}", "Baseline": "{:+.1%}",
           "Edge vs hold": "{:+.1%}", "Win rate": "{:.0%}"}

    def render_group(title, names):
        st.markdown(f"**{title}**")
        rows = []
        for name in names:
            sig = triggers[name]
            by_reg = backtest_by_regime(close, sig, 10)
            rows.append(_row(name, "Uptrend (above 200-day)",
                             by_reg["Uptrend (above 200-day)"],
                             last_trigger_date(sig & up_mask)))
            rows.append(_row(name, "Downtrend (below 200-day)",
                             by_reg["Downtrend (below 200-day)"],
                             last_trigger_date(sig & down_mask)))
        df_g = pd.DataFrame(rows)[cols].set_index(["Setup", "Regime"])
        st.dataframe(df_g.style.format(fmt, na_rep="—"), use_container_width=True)

    bullish = [n for n in triggers if SETUP_DIRECTION[n] == "bullish"]
    bearish = [n for n in triggers if SETUP_DIRECTION[n] == "bearish"]
    render_group("Buy-side setups", bullish)
    render_group("Sell-side setups", bearish)

    with st.expander("Historical Examples After These Setups"):
        st.caption(
            "This shows the actual dates when each setup last triggered, "
            "plus what happened to the stock over the following 10 trading days. "
            "Useful for seeing real-world outcomes instead of just averages."
        )
        for name in triggers:
            rec = recent_trigger_returns(close, triggers[name], 10, k=5)
            st.markdown(f"**{name}**")
            if rec.empty:
                st.caption("Never triggered in this history.")
            else:
                st.dataframe(
                    rec.style.format({f"Move over next 10d": "{:+.1%}"}, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )

    # Charts
    tail = close.iloc[-252:].index

    st.markdown("### RSI · 1Y")
    st.slider("RSI period", min_value=2, max_value=50, key="rsi_period")
    rsi_df = pd.DataFrame({"RSI": r, "Overbought (70)": 70, "Oversold (30)": 30}).loc[tail]
    st.line_chart(rsi_df, height=320)

    st.markdown("### Price · 1Y")
    st.line_chart(close.loc[tail], height=320)

    st.markdown("### Bollinger Bands (20, 2σ) · 1Y")
    band_df = pd.DataFrame({
        "Close": close, "Upper": bb["upper"], "Mid": bb["mid"], "Lower": bb["lower"],
    }).loc[tail]
    st.line_chart(band_df, height=320)

    st.markdown("### Trailing returns")
    rets = {
        "1 week (5d)": trailing_return(close, 5),
        "1 month (21d)": trailing_return(close, 21),
        "3 months (63d)": trailing_return(close, 63),
        "1 year (252d)": trailing_return(close, 252),
        "YTD": ytd_return(close),
    }
    st.dataframe(
        pd.DataFrame({"Return": rets}).style.format("{:+.2%}", na_rep="—"),
        use_container_width=True,
    )

    st.markdown("### Price · full history")
    st.line_chart(close, height=280)

    st.markdown("### Drawdown")
    dd_df = drawdown_series(close).rename("Drawdown").reset_index()
    dd_df.columns = ["Date", "Drawdown"]
    underwater = (
        alt.Chart(dd_df)
        .mark_area(color="#3b82f6", opacity=0.9, line={"color": "#1d4ed8"})
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Drawdown:Q", title=None, axis=alt.Axis(format="%")),
        )
        .properties(height=260)
    )
    st.altair_chart(underwater, use_container_width=True)

    st.markdown("### Run lengths")
    runs = completed_runs(close)
    up = runs[runs.sign > 0].length.value_counts().sort_index()
    down = runs[runs.sign < 0].length.value_counts().sort_index()
    hist = pd.DataFrame({"Up runs": up, "Down runs": down}).fillna(0).astype(int)
    hist.index.name = "Run length (days)"
    st.bar_chart(hist, height=320)
    st.caption("Up- and down-streak counts by length.")


if __name__ == "__main__":
    main()