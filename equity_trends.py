"""
Signal Lab
=============================================

A "what's unusual right now" tool for a single equity, rather than yet another
charting app. It pulls daily history and computes streaks + their historical
rarity, drawdowns, trailing returns, and a couple of cheap-but-useful extras
(rolling volatility percentile, last-day z-score), then surfaces the notable
ones in a "Findings" panel.

IMPORTANT framing: everything here is DESCRIPTIVE, not predictive. A long
losing streak does not make a bounce more likely (that's the gambler's
fallacy). Treat this as an exploration tool, not a signal generator.

HOW TO RUN (copy-paste these into Terminal, in the folder holding this file):

    # first time only — installs the libraries:
    python3 -m pip install streamlit yfinance pandas numpy

    # every time — starts the app (opens in your browser):
    python3 -m streamlit run equity_trends.py

(We use `python3 -m streamlit ...` rather than bare `streamlit ...` because the
streamlit launcher isn't on the PATH on a default macOS Python setup.)
To stop it: click the Terminal window and press Ctrl-C.

Data note: yfinance is an unofficial Yahoo scraper and breaks periodically.
If it stops returning data, swap load_history() for an Alpha Vantage / Tiingo /
Stooq fetch — the rest of the file doesn't care where the DataFrame comes from.
"""

from __future__ import annotations  # lets type hints like `dict | None` work on Python 3.9

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
    """Fetch adjusted daily OHLCV. Cached for an hour so re-runs are instant."""
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)  # drop tz, keep wall-clock dates
    return df[["Open", "High", "Low", "Close", "Volume"]]


# --------------------------- Compute layer ----------------------------
# Pure functions: no Streamlit, no I/O — so they're trivially testable.

def run_table(close: pd.Series) -> pd.DataFrame:
    """Per-day run info. A 'run' is a maximal stretch of same-direction days.

    sign:        +1 up day, -1 down day, 0 flat (vs previous close)
    run_id:      increments each time direction flips
    run_len:     length of the run *up to and including* this day
    """
    sign = np.sign(close.diff()).fillna(0)
    run_id = (sign != sign.shift()).cumsum()
    run_len = sign.groupby(run_id).cumcount() + 1
    return pd.DataFrame(
        {"sign": sign, "run_id": run_id, "run_len": run_len}, index=close.index
    )


def completed_runs(close: pd.Series) -> pd.DataFrame:
    """One row per run: its direction (sign) and its full length."""
    rt = run_table(close)
    return rt.groupby("run_id").agg(sign=("sign", "first"), length=("run_len", "max"))


def current_streak(close: pd.Series) -> dict:
    """Direction + length of the streak ending on the most recent day, plus
    how many historical same-direction runs reached at least this length."""
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
        "n_at_least": n_at_least,   # incl. the in-progress run
        "n_runs": n_runs,
    }


def drawdown_series(close: pd.Series) -> pd.Series:
    """Percent below the running all-time-high, as a (<=0) series."""
    return close / close.cummax() - 1.0


def days_underwater(close: pd.Series) -> int:
    """Trading days since the price last sat at a fresh peak (0 if at peak)."""
    dd = drawdown_series(close)
    at_peak = dd >= -1e-12
    if at_peak.iloc[-1]:
        return 0
    last_peak = at_peak[at_peak].index.max()
    return int((close.index > last_peak).sum())


def trailing_return(close: pd.Series, days: int) -> float:
    """Simple return over the last `days` *trading* days."""
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
    """Return (current 20d annualized vol, its percentile vs own history)."""
    ret = close.pct_change()
    rv = ret.rolling(window).std() * np.sqrt(252)
    cur = rv.iloc[-1]
    pct = float((rv.dropna() < cur).mean())
    return cur, pct


def last_day_z(close: pd.Series) -> float:
    """Z-score of the most recent daily return vs the full-history distribution."""
    ret = close.pct_change().dropna()
    if ret.std() == 0:
        return 0.0
    return float((ret.iloc[-1] - ret.mean()) / ret.std())


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Standard Bollinger Bands plus the two readings a chart doesn't give you:

    pct_b:     where price sits in the band (1 = upper, 0 = lower, >1 = above it)
    bandwidth: (upper - lower) / middle — how wide the band is; a low value vs
               its own history is a 'squeeze' that often precedes a big move.
    """
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
    """How many consecutive recent closes have sat above the upper band
    ('walking the band' — the strong-trend behaviour traders watch for)."""
    above = close > bollinger(close, window, n_std)["upper"]
    count = 0
    for val in reversed(above.tolist()):
        if val:
            count += 1
        else:
            break
    return count


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (the standard)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - 100 / (1 + rs)).rename("rsi")


def percentile_of(series: pd.Series, value: float) -> float:
    """Fraction of historical values at or below `value` (0..1)."""
    s = series.dropna()
    return float((s <= value).mean()) if len(s) else float("nan")


# ---------------------- Conditional backtest engine --------------------
# The honest core of "edge finding": for a setup, what happened NEXT, and did
# it beat simply holding the stock over the same horizon? We count a setup only
# on the day it TRIGGERS (crosses into true) to avoid double-counting the same
# event, and always compare to a baseline so a stock's general drift doesn't
# masquerade as a signal.

MIN_OCCURRENCES = 20   # below this, we say "too few to judge" rather than pretend
SIGNAL_HORIZON = 10    # trading days ahead used for the live "setup active" flags


def becomes_true(cond: pd.Series) -> pd.Series:
    """True only on the day a condition flips from false to true (the trigger)."""
    return cond & ~cond.shift(1, fill_value=False)


def setup_triggers(close: pd.Series, rsi_series: pd.Series,
                   bb: pd.DataFrame) -> dict[str, pd.Series]:
    """The canonical, widely-watched setups — deliberately few, to avoid the
    data-mining trap of testing dozens and cherry-picking the luckiest."""
    return {
        "RSI crossed below 30 (oversold)": becomes_true(rsi_series < 30),
        "RSI crossed above 70 (overbought)": becomes_true(rsi_series > 70),
        "Close dropped below lower Bollinger band": becomes_true(close < bb["lower"]),
        "Close pushed above upper Bollinger band": becomes_true(close > bb["upper"]),
    }


def forward_returns(close: pd.Series, signal: pd.Series, horizon: int) -> np.ndarray:
    """Return over `horizon` days following each trigger day."""
    vals = close.to_numpy()
    n = len(vals)
    idx = np.flatnonzero(signal.to_numpy())
    return np.array([vals[i + horizon] / vals[i] - 1
                     for i in idx if i + horizon < n])


def baseline_forward(close: pd.Series, horizon: int,
                     mask: pd.Series | None = None) -> np.ndarray:
    """Overlapping `horizon`-day returns — the 'just hold it' comparison. With a
    mask, only days where mask is True start a baseline window (used to build a
    regime-matched baseline, e.g. 'all uptrend days')."""
    vals = close.to_numpy()
    n = len(vals)
    if mask is None:
        return vals[horizon:] / vals[:-horizon] - 1
    idx = np.flatnonzero(mask.to_numpy())
    return np.array([vals[i + horizon] / vals[i] - 1 for i in idx if i + horizon < n])


def backtest_setup(close: pd.Series, signal: pd.Series, horizon: int,
                   baseline_mask: pd.Series | None = None) -> dict | None:
    """Conditional forward-return stats vs baseline. None if it never triggered.
    `baseline_mask` lets the baseline be regime-matched instead of all-days."""
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
        "edge": float(fwd.mean() - base_mean),  # excess over holding (same regime)
    }


def trend_regime(close: pd.Series, window: int = 200) -> tuple[pd.Series, pd.Series]:
    """Boolean masks (uptrend, downtrend) by price vs its `window`-day average.
    Both are False before the average is defined, so those early days are simply
    left unclassified rather than mislabelled."""
    sma = close.rolling(window).mean()
    defined = sma.notna()
    up = (close > sma) & defined
    down = (close <= sma) & defined
    return up, down


def current_regime(close: pd.Series, window: int = 200) -> str:
    """'uptrend' / 'downtrend' / 'undefined' for the most recent bar."""
    up, down = trend_regime(close, window)
    if up.iloc[-1]:
        return "uptrend"
    if down.iloc[-1]:
        return "downtrend"
    return "undefined"


def backtest_by_regime(close: pd.Series, signal: pd.Series, horizon: int,
                       window: int = 200) -> dict[str, dict | None]:
    """Same setup, split by trend regime, each vs its OWN regime's baseline — so
    'worked in uptrends' can't just be 'uptrends drift up'."""
    up, down = trend_regime(close, window)
    return {
        "All": backtest_setup(close, signal, horizon),
        "Uptrend (above 200-day)": backtest_setup(close, signal & up, horizon, up),
        "Downtrend (below 200-day)": backtest_setup(close, signal & down, horizon, down),
    }


# Which way each setup is conventionally "looking" — used only to GROUP the
# report into buy-side / sell-side. The actual lean is decided by measured edge.
SETUP_DIRECTION = {
    "RSI crossed below 30 (oversold)": "bullish",
    "Close dropped below lower Bollinger band": "bullish",
    "RSI crossed above 70 (overbought)": "bearish",
    "Close pushed above upper Bollinger band": "bearish",
}
PROPOSAL_LOOKBACK = 5  # a setup counts as "live" if it fired within this many sessions


def last_trigger_date(signal: pd.Series):
    """Most recent date the signal fired, or None."""
    hits = signal[signal]
    return hits.index[-1] if len(hits) else None


def recent_trigger_returns(close: pd.Series, signal: pd.Series, horizon: int,
                           k: int = 6) -> pd.DataFrame:
    """The last `k` trigger dates and the move over the following `horizon` days,
    so you can pull up a chart and eyeball the actual instances."""
    vals = close.to_numpy()
    n = len(vals)
    idx = np.flatnonzero(signal.to_numpy())
    rows = []
    for i in idx[-k:]:
        fwd = vals[i + horizon] / vals[i] - 1 if i + horizon < n else np.nan
        rows.append({"Trigger date": close.index[i].strftime("%Y-%m-%d"),
                     f"Move over next {horizon}d": fwd})
    return pd.DataFrame(rows)


def build_trade_idea(close: pd.Series, rsi_series: pd.Series,
                     bb: pd.DataFrame) -> tuple[str, int, list[str]]:
    """Aggregate the directional signals into a transparent Bullish / Neutral /
    Bearish lean. Every component is returned as a reason, so it stays a readout
    you can sanity-check — not a black-box oracle. NOT advice."""
    regime = current_regime(close)
    reg_key = {"uptrend": "Uptrend (above 200-day)",
               "downtrend": "Downtrend (below 200-day)"}.get(regime, "All")
    reasons: list[str] = []
    score = 0

    # 1) Trend backdrop — price vs its 200-day average.
    if regime == "uptrend":
        score += 1
        reasons.append("🟢 **Trend:** above the 200-day average (uptrend backdrop).")
    elif regime == "downtrend":
        score -= 1
        reasons.append("🔴 **Trend:** below the 200-day average (downtrend backdrop).")

    # 2) Medium-term momentum — trailing ~3 months.
    m3 = trailing_return(close, 63)
    if not np.isnan(m3):
        if m3 > 0:
            score += 1
            reasons.append(f"🟢 **Momentum:** {m3:+.0%} over the last ~3 months.")
        elif m3 < 0:
            score -= 1
            reasons.append(f"🔴 **Momentum:** {m3:+.0%} over the last ~3 months.")

    # 3) Setups that fired recently, weighted by their measured same-regime history.
    for name, sig in setup_triggers(close, rsi_series, bb).items():
        if not bool(sig.iloc[-PROPOSAL_LOOKBACK:].any()):
            continue
        stats = backtest_by_regime(close, sig, SIGNAL_HORIZON).get(reg_key)
        if stats is None or stats["n"] < MIN_OCCURRENCES:
            reasons.append(f"⚪ **{name}** fired recently — too few comparable cases "
                           f"({0 if stats is None else stats['n']}) to weigh.")
            continue
        if stats["mean"] > 0:
            score += 1
            reasons.append(f"🟢 **{name}** fired: historically {stats['mean']:+.1%} over "
                           f"{SIGNAL_HORIZON}d in this {regime} ({stats['n']} cases).")
        elif stats["mean"] < 0:
            score -= 1
            reasons.append(f"🔴 **{name}** fired: historically {stats['mean']:+.1%} over "
                           f"{SIGNAL_HORIZON}d in this {regime} ({stats['n']} cases).")

    if score >= 2:
        verdict = "Bullish"
    elif score <= -2:
        verdict = "Bearish"
    else:
        verdict = "Neutral"

    if not reasons:
        reasons.append("Not enough history yet to read the signals.")
    return verdict, score, reasons


# ------------------------- Findings heuristics -------------------------
# Transparent, descriptive flags. The panel stays quiet unless something
# clears these bars — tune them here, no magic buried in the logic.

STREAK_MIN = 4        # only flag streaks at least this long
DRAWDOWN_MIN = -0.05  # only mention drawdowns at least this deep (-5%)
TAIL = 0.10           # "unusual" = sitting in this fraction of the tail


def _plural(n: int, unit: str) -> str:
    """'1 day' vs '4 days' — no more '1 days'."""
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def build_findings(close: pd.Series, rsi_period: int = 14) -> list[tuple[str, str]]:
    findings = []
    dd = drawdown_series(close)

    # Streak: only interesting once it's run a while. A 1-day streak is noise.
    streak = current_streak(close)
    if streak["sign"] != 0 and streak["length"] >= STREAK_MIN and streak["n_runs"] > 0:
        word = "up" if streak["sign"] > 0 else "down"
        share = streak["n_at_least"] / streak["n_runs"]
        msg = (f"On a {streak['length']}-day {word} streak — "
               f"{streak['n_at_least']} of {streak['n_runs']} {word}-runs in this "
               f"window have reached that length.")
        findings.append(("⚠️" if share <= TAIL else "•", msg))

    # Drawdown: only when it's both genuinely deep AND deep relative to its history.
    cur_dd = dd.iloc[-1]
    worse_share = float((dd.dropna() < cur_dd).mean())  # fraction of days deeper
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

    # Bollinger: squeeze (tight bands) and walking the upper band.
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

    # RSI: only when stretched, and reported with how rare that reading is.
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

    # Live "setup just triggered" flags, each carrying its historical edge —
    # measured within the CURRENT trend regime, since that's what's in force now.
    regime = current_regime(close)
    bt_signals = setup_triggers(close, r, bb)
    for name, sig in bt_signals.items():
        if not bool(sig.iloc[-1]):
            continue  # didn't trigger today
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

# Mag7 + a few extras, as one-tap sidebar buttons.
QUICK_PICKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
               "ORCL", "MU", "PLTR"]


def _pick_ticker(symbol: str) -> None:
    """Button callback — runs before rerun, so the text box picks it up cleanly."""
    st.session_state.ticker = symbol
    st.session_state.qp_open = False  # collapse the Quick picks pane on selection


def main():
    st.set_page_config(page_title="Signal Lab", layout="wide")
    # Streamlit keeps st.columns side-by-side on phones, so metric strips and
    # control rows cram together. This makes column rows WRAP (2-up) below 640px,
    # tightens page margins, scales metric text down, and lets wide tables scroll.
    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
          .block-container { padding: 2.5rem 0.9rem 3rem !important; }
          div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
          }
          div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: calc(50% - 0.5rem) !important;
          }
          div[data-testid="stMetricValue"] { font-size: 1.15rem !important; }
          div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
        }
        /* Wide backtest tables: scroll sideways instead of squishing to nothing. */
        div[data-testid="stDataFrame"] > div { overflow-x: auto; }
        /* Search (form submit) button: green = "go", not the default red = "danger". */
        div[data-testid="stFormSubmitButton"] button {
          background-color: #16a34a !important;
          border-color: #16a34a !important;
          color: #fff !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
          background-color: #15803d !important;
          border-color: #15803d !important;
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
    if "qp_open" not in st.session_state:
        st.session_state.qp_open = False

    # Primary controls live in the page body (not a sidebar). A form means typing
    # or tapping a quick pick only PRE-FILLS the field — analysis runs when Search
    # is pressed (or Enter), so a quick pick never fires off a search on its own.
    with st.form("search"):
        sc1, sc2 = st.columns([3, 1])
        sc1.text_input("Ticker", key="ticker", label_visibility="collapsed",
                       placeholder="Search for tickers")
        submitted = sc2.form_submit_button("Search", type="primary",
                                           use_container_width=True)
    if submitted:
        st.session_state.active_ticker = st.session_state.ticker.strip().upper()

    with st.expander("Quick picks", expanded=st.session_state.qp_open):
        qcols = st.columns(5)
        for i, sym in enumerate(QUICK_PICKS):
            qcols[i % 5].button(sym, key=f"qp_{sym}", use_container_width=True,
                                on_click=_pick_ticker, args=(sym,))

    rsi_period = st.session_state.rsi_period  # set by the slider down at the RSI chart
    ticker = st.session_state.active_ticker
    if not ticker:
        return

    try:
        df = load_history(ticker, "max")
    except Exception as e:  # noqa: BLE001 — surface any fetch/parse error plainly
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

    # Core indicators computed once, up front — everything below reuses them.
    bb = bollinger(close)
    r = rsi(close, rsi_period)
    reg_now = current_regime(close)

    # --- TRADE IDEA: the verdict, big and first. ---
    verdict, score, reasons = build_trade_idea(close, r, bb)
    badge = {"Bullish": "#16a34a", "Neutral": "#6b7280", "Bearish": "#dc2626"}[verdict]
    st.markdown(
        f"<div style='border-left:6px solid {badge};background:{badge}1f;"
        f"padding:clamp(10px,2.5vw,14px) clamp(14px,3.5vw,20px);border-radius:4px;"
        f"color:{badge};font-weight:800;font-size:clamp(22px,6vw,34px);"
        f"letter-spacing:.5px'>{verdict.upper()}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    for line in reasons:
        st.markdown(line)
    st.caption("Not financial advice.")

    # --- Snapshot strip: quick orientation ---
    c1, c2, c3, c4 = st.columns(4)
    streak = current_streak(close)
    word = "up" if streak["sign"] > 0 else "down" if streak["sign"] < 0 else "flat"
    c1.metric("Streak", f"{_plural(streak['length'], 'day')} {word}")
    c2.metric("Drawdown", f"{drawdown_series(close).iloc[-1]:.1%}")
    c3.metric("Last day", f"{close.pct_change().iloc[-1]:+.2%}")
    c4.metric("1-year", f"{trailing_return(close, 252):+.1%}")

    cur_rsi = r.iloc[-1]
    i1, i2, i3 = st.columns(3)
    pb = bb["pct_b"].iloc[-1]
    i1.metric("%B", f"{pb:.2f}",
              help="Bollinger %B. 1.0 = at the upper band, 0.0 = at the lower band, >1 = above it.")
    i2.metric(f"RSI ({rsi_period})", "—" if pd.isna(cur_rsi) else f"{cur_rsi:.0f}",
              help="Above 70 = overbought, below 30 = oversold (by convention).")
    bw_pct = percentile_of(bb["bandwidth"], bb["bandwidth"].iloc[-1])
    i3.metric("Band width", f"{bw_pct:.0%}",
              help="Where band width sits in its own history. Low = a squeeze.")

    # --- Backtested setups: the evidence behind the verdict ---
    st.markdown("### Setup backtests")
    st.caption("Forward return after each historical trigger, vs. just holding.")
    if reg_now != "undefined":
        st.markdown(f"**{reg_now.capitalize()}** · price "
                    f"{'above' if reg_now == 'uptrend' else 'below'} the 200-day average.")

    lc, rc = st.columns([1, 2])
    horizon = lc.selectbox("Days ahead", [5, 10, 21, 42], index=1,
                           help="How many trading days after each trigger to measure.")
    split = rc.checkbox("Split by regime", value=True,
                        help="Compare each setup in uptrends vs downtrends, each "
                             "against its OWN regime's baseline.")

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
            if split:
                by_reg = backtest_by_regime(close, sig, horizon)
                rows.append(_row(name, "Uptrend (above 200-day)",
                                 by_reg["Uptrend (above 200-day)"],
                                 last_trigger_date(sig & up_mask)))
                rows.append(_row(name, "Downtrend (below 200-day)",
                                 by_reg["Downtrend (below 200-day)"],
                                 last_trigger_date(sig & down_mask)))
            else:
                rows.append(_row(name, "All", backtest_setup(close, sig, horizon),
                                 last_trigger_date(sig)))
        df_g = pd.DataFrame(rows)[cols].set_index(["Setup", "Regime"])
        st.dataframe(df_g.style.format(fmt, na_rep="—"), use_container_width=True)

    bullish = [n for n in triggers if SETUP_DIRECTION[n] == "bullish"]
    bearish = [n for n in triggers if SETUP_DIRECTION[n] == "bearish"]
    render_group("📈 Buy-side", bullish)
    render_group("📉 Sell-side", bearish)

    st.caption(f"Under {MIN_OCCURRENCES} triggers is too thin to trust.")

    with st.expander("Recent triggers"):
        st.caption("The last few times each setup fired, and the move that followed.")
        for name in triggers:
            rec = recent_trigger_returns(close, triggers[name], horizon, k=6)
            st.markdown(f"**{name}**")
            if rec.empty:
                st.caption("Never triggered in this history.")
            else:
                st.dataframe(
                    rec.style.format({f"Move over next {horizon}d": "{:+.1%}"}, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )

    # --- What's unusual right now (situational context) ---
    st.markdown("### What's unusual")
    for marker, text in build_findings(close, rsi_period):
        st.markdown(f"{marker} {text}")

    # --- Indicator charts (supporting visual), full width, RSI first ---
    tail = close.iloc[-252:].index

    st.markdown("### RSI · 1Y")
    st.slider("RSI period", min_value=2, max_value=50, key="rsi_period",
              help="Lookback window for RSI. Lower = jumpier and hits 30/70 often; "
                   "higher = smoother and rarely reaches the extremes.")
    rsi_df = pd.DataFrame({"RSI": r, "Overbought (70)": 70, "Oversold (30)": 30}).loc[tail]
    st.line_chart(rsi_df, height=320)

    # Price in the SAME window, directly beneath RSI, so the x-axes align and you
    # can eyeball divergences (price high vs RSI not confirming, and vice versa).
    st.markdown("### Price · 1Y")
    st.line_chart(close.loc[tail], height=320)

    st.markdown("### Bollinger Bands (20, 2σ) · 1Y")
    band_df = pd.DataFrame({
        "Close": close, "Upper": bb["upper"], "Mid": bb["mid"], "Lower": bb["lower"],
    }).loc[tail]
    st.line_chart(band_df, height=320)

    # --- Trailing returns ---
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

    # --- Supporting charts: full width, stacked ---
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
            # mark_area fills between the line and the 0 baseline, so the colored
            # region is the drawdown itself, hanging DOWN from zero (the fix).
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
