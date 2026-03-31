import os
import random
from datetime import datetime, timedelta, date
from pathlib import Path
import sys
import textwrap

import requests  # NEW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from components.news_panel import render_news_panel
from collections import defaultdict
from dotenv import load_dotenv 
from src.live.news_fetcher import fetch_latest_news, QUERIES
# ... rest of your app

# Ensure project root is on sys.path so `src` package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live.infer_xgb import get_predictions_for_chart

load_dotenv()  # loads from .env in project root

from groq import Groq

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if "ai_response" not in st.session_state:
    st.session_state.ai_response = None

# -------------------------------------------------
# Raw OHLC sources used for mini chart context
# -------------------------------------------------
RAW_FEATURE_CSV_BY_TF = {
    "15m": "data/raw/btc_15m_raw.csv",
    "1h":  "data/raw/btc_1h_raw.csv",
    "4h":  "data/raw/btc_4h_raw.csv",
}

# Map chart timeframe → LIVE features CSV
FEATURE_CSV_BY_TF = {
    "5m":  "data/processed/btc_5m_live_features.csv",
    "15m": "data/processed/btc_15m_live_features.csv",
    "1h":  "data/processed/btc_1h_live_features.csv",
    "4h":  "data/processed/btc_4h_live_features.csv",
    "1d":  "data/processed/btc_1d_live_features.csv",
}

def _pick_col(row, candidates, default=None):
    """Find first matching column name in row keys."""
    for c in candidates:
        if c in row.index:
            return c
    return default

def load_live_features(tf):
    csv_path = Path(f"data/processed/btc_{tf}_live_features.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.iloc[-1]  # Returns Series

def to_num(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def badge_from_score(score):
    if score >= 0.67:
        return "Bullish"
    if score <= 0.33:
        return "Bearish"
    return "Neutral"

def score_from_signal(signal):
    signal = str(signal).lower()
    if "bullish" in signal or "buy" in signal or "above" in signal or "positive" in signal:
        return 1.0
    if "bearish" in signal or "sell" in signal or "below" in signal or "negative" in signal:
        return 0.0
    return 0.5

def safe_status_text(label, value, note=""):
    if pd.isna(value):
        return "N/A"
    if note:
        return f"{label}: {value:.4f} ({note})"
    return f"{label}: {value:.4f}"

def compute_indicator_snapshot(row):
    """
    row is pandas Series (single row from df.iloc[-1])
    Returns:
      indicators: list of dicts for table + per-indicator bars
      overall_score: 0..1 aggregated score across indicators
    """
    cols = row.index.tolist()

    close = to_num(row.get(_pick_col(row, ["close", "Close"]), np.nan))
    volume = to_num(row.get(_pick_col(row, ["volume", "Volume"]), np.nan))

    # RSI
    rsi_col = _pick_col(row, ["rsi", "RSI", "RSI_14", "rsi14"])
    rsi = to_num(row.get(rsi_col, np.nan))
    if pd.isna(rsi):
        rsi_status = "N/A"
        rsi_score = 0.5
    elif rsi > 55:
        rsi_status = f"{rsi:.2f} (Bullish)"
        rsi_score = 1.0
    elif rsi < 45:
        rsi_status = f"{rsi:.2f} (Bearish)"
        rsi_score = 0.0
    else:
        rsi_status = f"{rsi:.2f} (Neutral)"
        rsi_score = 0.5

    # MACD
    macd_col = _pick_col(row, ["macd", "MACD", "MACD_12_26_9"])
    macds_col = _pick_col(row, ["macdsignal", "MACDs", "MACDs_12_26_9"])
    macdh_col = _pick_col(row, ["macdhistogram", "MACDh", "MACDh_12_26_9"])
    macd = to_num(row.get(macd_col, np.nan))
    macds = to_num(row.get(macds_col, np.nan))
    macdh = to_num(row.get(macdh_col, np.nan))

    if pd.isna(macd) and pd.isna(macds) and pd.isna(macdh):
        macd_status = "N/A"
        macd_score = 0.5
    else:
        if not pd.isna(macdh):
            if macdh > 0:
                macd_status = f"{macd:.4f}/{macds:.4f}/{macdh:.4f} (Bullish)"
                macd_score = 1.0
            elif macdh < 0:
                macd_status = f"{macd:.4f}/{macds:.4f}/{macdh:.4f} (Bearish)"
                macd_score = 0.0
            else:
                macd_status = f"{macd:.4f}/{macds:.4f}/{macdh:.4f} (Neutral)"
                macd_score = 0.5
        elif not pd.isna(macd) and not pd.isna(macds):
            if macd > macds:
                macd_status = f"{macd:.4f}/{macds:.4f} (Bullish)"
                macd_score = 1.0
            elif macd < macds:
                macd_status = f"{macd:.4f}/{macds:.4f} (Bearish)"
                macd_score = 0.0
            else:
                macd_status = f"{macd:.4f}/{macds:.4f} (Neutral)"
                macd_score = 0.5
        else:
            macd_status = f"{macd:.4f} (Neutral)"
            macd_score = 0.5

    # Bollinger Bands
    bbu_col = _pick_col(row, ["bbu", "BBU", "BBU_20_2.0", "BBU_5_2.0"])
    bbm_col = _pick_col(row, ["bbm", "BBM", "BBM_20_2.0", "BBM_5_2.0"])
    bbl_col = _pick_col(row, ["bbl", "BBL", "BBL_20_2.0", "BBL_5_2.0"])
    bbu = to_num(row.get(bbu_col, np.nan))
    bbm = to_num(row.get(bbm_col, np.nan))
    bbl = to_num(row.get(bbl_col, np.nan))

    if pd.isna(close) or (pd.isna(bbu) and pd.isna(bbm) and pd.isna(bbl)):
        bb_status = "N/A"
        bb_score = 0.5
    else:
        if not pd.isna(bbu) and not pd.isna(bbl) and not pd.isna(bbm):
            if close > bbm:
                bb_status = f"{close:.2f} vs {bbm:.2f}/{bbu:.2f}/{bbl:.2f} (Bullish)"
                bb_score = 1.0
            elif close < bbl:
                bb_status = f"{close:.2f} vs {bbm:.2f}/{bbu:.2f}/{bbl:.2f} (Bearish)"
                bb_score = 0.0
            else:
                bb_status = f"{close:.2f} vs {bbm:.2f}/{bbu:.2f}/{bbl:.2f} (Neutral)"
                bb_score = 0.5
        else:
            bb_status = f"{close:.2f} (Neutral)"
            bb_score = 0.5

    # Volume
    vol_ratio_col = _pick_col(row, ["volumeratio", "volume_ratio", "VolumeRatio"])
    vol_sma_col = _pick_col(row, ["volume_sma_20", "volumesma20", "vol_sma_20"])
    vol_ratio = to_num(row.get(vol_ratio_col, np.nan))
    vol_sma = to_num(row.get(vol_sma_col, np.nan))

    if pd.isna(volume):
        vol_status = "N/A"
        vol_score = 0.5
    elif not pd.isna(vol_ratio):
        if vol_ratio > 1.1:
            vol_status = f"{volume:.1f} (High, {vol_ratio:.2f}x)"
            vol_score = 1.0
        elif vol_ratio < 0.9:
            vol_status = f"{volume:.1f} (Low, {vol_ratio:.2f}x)"
            vol_score = 0.0
        else:
            vol_status = f"{volume:.1f} (Normal, {vol_ratio:.2f}x)"
            vol_score = 0.5
    elif not pd.isna(vol_sma) and vol_sma > 0:
        ratio = volume / vol_sma
        if ratio > 1.1:
            vol_status = f"{volume:.1f} (High, {ratio:.2f}x)"
            vol_score = 1.0
        elif ratio < 0.9:
            vol_status = f"{volume:.1f} (Low, {ratio:.2f}x)"
            vol_score = 0.0
        else:
            vol_status = f"{volume:.1f} (Normal, {ratio:.2f}x)"
            vol_score = 0.5
    else:
        vol_status = f"{volume:.1f} (Neutral)"
        vol_score = 0.5

    # SMA / trend
    sma20_col = _pick_col(row, ["sma20", "SMA20", "SMA_20"])
    sma50_col = _pick_col(row, ["sma50", "SMA50", "SMA_50"])
    sma20 = to_num(row.get(sma20_col, np.nan))
    sma50 = to_num(row.get(sma50_col, np.nan))
    sma_ratio = to_num(row.get(_pick_col(row, ["smaratio", "sma_ratio"]), np.nan))

    if pd.isna(sma20) and pd.isna(sma50) and pd.isna(sma_ratio):
        sma_status = "N/A"
        sma_score = 0.5
    else:
        if not pd.isna(sma_ratio):
            if sma_ratio > 1:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Bullish)"
                sma_score = 1.0
            elif sma_ratio < 1:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Bearish)"
                sma_score = 0.0
            else:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Neutral)"
                sma_score = 0.5
        elif not pd.isna(sma20) and not pd.isna(sma50):
            if sma20 > sma50:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Bullish)"
                sma_score = 1.0
            elif sma20 < sma50:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Bearish)"
                sma_score = 0.0
            else:
                sma_status = f"{sma20:.2f}/{sma50:.2f} (Neutral)"
                sma_score = 0.5
        else:
            sma_status = "N/A"
            sma_score = 0.5

    indicators = [
        {"Indicator": "RSI", "Status": rsi_status, "Score": rsi_score},
        {"Indicator": "MACD", "Status": macd_status, "Score": macd_score},
        {"Indicator": "Bollinger Bands", "Status": bb_status, "Score": bb_score},
        {"Indicator": "Volume", "Status": vol_status, "Score": vol_score},
        {"Indicator": "SMA", "Status": sma_status, "Score": sma_score},
    ]

    overall_score = float(np.mean([x["Score"] for x in indicators]))
    return indicators, overall_score

def render_indicator_bar(title, score):
    label = "Bullish" if score >= 0.67 else "Bearish" if score <= 0.33 else "Neutral"
    render_ta_gauge(title, label, score)

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are CryptoPulse AI, a professional crypto trader and quantitative analyst.
    You advise on Bitcoin trading scenarios using:
    - Model prediction (direction + confidence)
    - Market structure (HH/HL, LH/LL, combined_direction from trend_direction.py), candlestick patterns
    - RSI, MACD, volume, moving averages, Bollinger Bands
    - Multi-timeframe context (5m, 15m, 1h, 4h, 1d)

    Respond with:
    - Clear, actionable guidance for the exact scenario and level the user mentions.
    - Explicitly comment on whether the model prediction and structure SUPPORT or CONTRADICT the scenario.
    - Mention how RSI, MACD, volume and structure should ideally look.
    - Be concise (3–7 short bullets), no fluff, no disclaimers.
    - Only refer to the level the user gives.
    - Suggest next steps to trade with clear entry points, SL and targets. 
    - Advise clearly if model prediction is not clear, validate other signals, candlestick patterns and market structure.
    - At the end, add clear note saying "This is AI generated analysis. There is no guarantee in financial markets.
    - Trade responsibly, Capital at risk. 
    """
)

def call_ai_trade_assistant(
    level: float,
    direction: str,
    timeframe: str,
    message: str,
    latest_indicators: dict | None = None,
    model_direction: str | None = None,
    model_confidence: float | None = None,
    trend_structure: dict | None = None,
) -> str:
    """
    Call Groq LLM to get a trade explanation for this scenario.
    Includes model prediction + structure context.
    """
    client = get_groq_client()
    if client is None:
        return "AI backend is not configured. Please set GROQ_API_KEY to enable the assistant."

    indicators_text = ""
    if latest_indicators:
        indicators_text = (
            f"\nLatest indicators for BTC {timeframe}:\n"
            f"- RSI: {latest_indicators.get('rsi', 'N/A'):.1f}\n"
            f"- MACD: {latest_indicators.get('macd', 'N/A'):.4f} "
            f"(signal {latest_indicators.get('macd_signal', 'N/A'):.4f})\n"
            f"- Volume ratio: {latest_indicators.get('volume_ratio', 'N/A'):.2f}x\n"
            f"- SMA ratio (price / SMA): {latest_indicators.get('sma_ratio', 'N/A'):.3f}\n"
            f"- Bollinger position (0=lower,1=upper): {latest_indicators.get('bb_position', 'N/A'):.2f}\n"
        )

    model_text = ""
    if model_direction is not None:
        conf_pct = f"{model_confidence * 100:.1f}%" if model_confidence is not None else "N/A"
        model_text += (
            f"\nModel prediction:\n"
            f"- Direction: {model_direction}\n"
            f"- Confidence: {conf_pct}\n"
        )

    if trend_structure:
        model_text += (
            "\nTrend structure (from HH/HL vs LH/LL):\n"
            f"- Label: {trend_structure.get('label', 'N/A')}\n"
            f"- Strength: {trend_structure.get('strength', 0.0):.2f}\n"
            f"- Combined direction: {trend_structure.get('combined_direction', 'N/A')}\n"
            f"- HH/HL vs LH/LL counts: "
            f"HH {trend_structure.get('hh_count', 0)}, "
            f"HL {trend_structure.get('hl_count', 0)}, "
            f"LH {trend_structure.get('lh_count', 0)}, "
            f"LL {trend_structure.get('ll_count', 0)}\n"
        )

    user_prompt = textwrap.dedent(
        f"""
        User scenario:
        - Timeframe: {timeframe}
        - Key level: {level}
        - Scenario: {direction} (price {'breaking below' if direction=='Breakdown' else 'breaking above'} this level)

        User question:
        {message}

        {indicators_text}
        {model_text}
        """
    )

    try:
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        return f"Error contacting AI backend: {e}"

@st.cache_data(ttl=120)  # 2min cache, keeps rationale in sync with 2min UI refresh
def get_live_indicators(timeframe: str):
    """Load latest indicators from LIVE features CSV for the given timeframe."""
    csv_path = FEATURE_CSV_BY_TF.get(timeframe)
    if not csv_path or not Path(csv_path).exists():
        return {}

    df = pd.read_csv(csv_path).tail(1).iloc[0]
    return {
        "rsi": df.get("rsi", 50),
        "macd": df.get("macd", 0),
        "macd_signal": df.get("macd_signal", 0),
        "volume_ratio": df.get("volume_ratio", 1.0),
        "sma_ratio": df.get("sma_ratio", 1.0),
        "bb_position": df.get("bb_position", 0.5),
    }

# ============ Simple live price helper ============

def get_live_btc_price():
    """
    Fetch BTC/USDT price and 24h change from Binance public API.
    Falls back to static values if request fails.
    """
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbol": "BTCUSDT"},
            timeout=3,
        )
        if r.status_code == 200:
            data = r.json()
            last_price = float(data["lastPrice"])
            price_change = float(data["priceChange"])
            price_change_pct = float(data["priceChangePercent"])
            return last_price, price_change, price_change_pct
    except Exception:
        pass
    # Fallback (same numbers you had before)
    return 69936.8, 477.2, 0.69  # static mock

# ============ Page config & CSS ============

st.set_page_config(page_title="CryptoPulse AI – BTC/USD", layout="wide")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div.block-container {padding-top: 0.5rem;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

blink_css = """
<style>
@keyframes blink-green { 50% { color: #bbf7d0; } }
@keyframes blink-red { 50% { color: #fecaca; } }
@keyframes blink-amber { 50% { color: #fef3c7; } }
.blink-green { color:#22c55e; animation: blink-green 1.2s infinite; }
.blink-red { color:#ef4444; animation: blink-red 1.2s infinite; }
.blink-amber { color:#f59e0b; animation: blink-amber 1.2s infinite; }
</style>
"""
st.markdown(blink_css, unsafe_allow_html=True)

# ============ Helper components ============

def render_range_bar(label: str, low: float, high: float, current: float):
    """Compact red->green range bar with centered label and prices close to bar."""
    pct = 0.0 if high == low else max(0.0, min(1.0, (current - low) / (high - low)))
    bar_html = f"""
    <div style="margin-top:0.8rem; margin-bottom:0.2rem;">
      <div style="font-size:0.90rem; font-weight:600; color:#e5e7eb; text-align:center; margin-bottom:0.35rem;">
        {label}
      </div>
      <div style="display:flex; align-items:center; gap:0.5rem;">
        <div style="flex:1; background-color:#e5e7eb;
                    border-radius:999px; height:4px;
                    position:relative; overflow:hidden;">
          <div style="
              position:absolute; left:0; top:0; bottom:0;
              width:{pct*100:.1f}%;
              background:linear-gradient(90deg,#ef4444,#22c55e);
              border-radius:999px;">
          </div>
        </div>
      </div>
      <div style="display:flex; justify-content:space-between;
                  font-size:0.78rem; color:#9ca3af; margin-top:0.25rem;">
        <span>{low:,.0f}</span>
        <span>{high:,.0f}</span>
      </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def render_ta_gauge(title: str, label: str, score: float):
    """Simple horizontal gauge (0–1) with a pointer and a label like Buy / Sell."""
    pct = max(0.0, min(1.0, score)) * 100
    if "Strong Sell" in label or label == "Sell":
        pill_bg = "rgba(248,113,113,0.12)"
        pill_fg = "#b91c1c"
    elif "Strong Buy" in label or label == "Buy":
        pill_bg = "rgba(34,197,94,0.12)"
        pill_fg = "#15803d"
    else:
        pill_bg = "rgba(148,163,184,0.12)"
        pill_fg = "#4b5563"

    html = f"""
    <div style="text-align:center; margin:0.5rem 0 1.0rem 0%;">
      <div style="font-size:0.95rem; font-weight:600; margin-bottom:0.4rem;">
        {title}
      </div>
      <div style="position:relative; height:12px; border-radius:999px;
                  background:linear-gradient(90deg,#ef4444,#f97316,#22c55e);">
        <div style="position:absolute; top:-4px; left:{pct:.0f}%;
                    transform:translateX(-50%);
                    width:8px; height:20px; border-radius:999px; background:#111827;">
        </div>
      </div>
      <div style="margin-top:0.35rem;">
        <span style="font-size:0.85rem; padding:0.25rem 0.8rem; border-radius:999px;
                     background-color:{pill_bg}; color:{pill_fg}; font-weight:600;">
          {label}
        </span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---- Candlestick pattern helpers ----

def classify_candle_pattern(open_p: float, high_p: float, low_p: float, close_p: float) -> str:
    """Very simple candlestick pattern classifier (mock but consistent)."""
    body = abs(close_p - open_p)
    rng = high_p - low_p if high_p != low_p else 1e-6
    upper = high_p - max(open_p, close_p)
    lower = min(open_p, close_p) - low_p

    body_pct = body / rng
    upper_pct = upper / rng
    lower_pct = lower / rng

    if body_pct < 0.15 and upper_pct > 0.35 and lower_pct > 0.35:
        return "Doji"
    if body_pct > 0.7 and upper_pct < 0.1 and lower_pct < 0.1:
        return "Marubozu"
    if body_pct < 0.3 and upper_pct > 0.4 and lower_pct < 0.2:
        return "Hanging Man"
    return "Standard"

def compatibility_score(prev_pattern: str, next_pattern: str) -> str:
    """Mock compatibility between previous candle pattern and next predicted candle."""
    bullish_continuation = {"Marubozu", "Standard"}
    bearish_reversal = {"Hanging Man", "Doji"}

    if prev_pattern in bullish_continuation and next_pattern in bullish_continuation:
        return "High"
    if prev_pattern in bearish_reversal and next_pattern in bullish_continuation:
        return "Low"
    return "Medium"

# ============ Real prediction helpers ============

def get_next_candle_prediction(chart_tf: str, horizon_mode: str = "current") -> dict:
    """
    Wrap infer_xgb.get_predictions_for_chart:
      chart_tf='15m' -> [15m->15m, 15m->1h]
      chart_tf='1h'  -> [1h->1h, 1h->4h]
      chart_tf='4h'  -> [4h->4h, 4h->1d]
    """
    if chart_tf not in {"15m", "1h", "4h"}:
        raise ValueError(f"Prediction not supported for timeframe: {chart_tf}")
    preds = get_predictions_for_chart(chart_tf)
    return preds[0] if horizon_mode == "current" else preds[1]

def load_last_n_candles(tf: str, n: int = 6) -> list[dict]:
    """
    Load last N OHLC candles for selected timeframe from raw CSV.
    """
    path_str = RAW_FEATURE_CSV_BY_TF.get(tf)
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time")
    else:
        df = df.sort_index()

    tail = df.tail(n)
    candles = []
    for _, row in tail.iterrows():
        candles.append(
            {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return candles

def build_mini_prediction_chart(last_candles: list[dict], pred: dict) -> go.Figure:
    """Mini chart: last 5 real candles + 1 predicted (bold color, closer spacing)."""
    if len(last_candles) < 1:
        return go.Figure()

    # Context candles (5 real)
    ctx = last_candles[-5:]
    opens, highs, lows, closes = [c["open"] for c in ctx], [c["high"] for c in ctx], \
                                [c["low"] for c in ctx], [c["close"] for c in ctx]

    prev_close = closes[-1]
    prev_range = max(highs[-1], lows[-1]) - min(highs[-1], lows[-1])
    avg_body = abs(closes[-1] - opens[-1]) or prev_range * 0.6
    base_range = prev_range or avg_body * 1.8

    # Predicted candle (clear color + nice body)
    pred_ret = pred.get("predicted_return") or 0.003 * (1 if pred.get("direction") in ["Bullish"] else -1 if pred.get("direction") in ["Bearish"] else 0)
    direction = pred.get("direction")
    target_close = prev_close * (1 + float(pred_ret))
    scale = 3.0 if pred.get("horizon") in ["1h", "4h", "1d"] and pred.get("timeframe") == "15m" else 1.5

    body_size = avg_body * 1.2 * scale  # Nice prominent body
    if direction in ["Bullish", "SideBull"]:
        pred_open, pred_close = target_close - body_size * 0.7, target_close
        pred_color = "#10b981"  # Bold green
    elif direction in ["Bearish", "SideBear"]:
        pred_open, pred_close = target_close, target_close + body_size * 0.7
        pred_color = "#ef4444"  # Bold red
    else:
        pred_open = target_close - body_size * 0.3
        pred_close = target_close + body_size * 0.3
        #pred_color = "#6b7280"  # Grey neutral
        pred_color = "#d97706" # Amber

    wick_range = base_range * 0.5 * scale
    pred_high = max(pred_open, pred_close) + wick_range * 0.4
    pred_low = min(pred_open, pred_close) - wick_range * 0.6

    # Append predicted
    opens.append(pred_open); highs.append(pred_high)
    lows.append(pred_low); closes.append(pred_close)

    # Tighter x-spacing
    #xs = list(range(len(opens)))  # Integer indices
    #xs = [x * 0.85 for x in xs]   # Scale closer (0.85 spacing)
    xs = [i * 0.82 for i in range(len(opens))]  # Uniform 0.82 spacing

    fig = go.Figure()

    # Context (light grey, thin)
    fig.add_trace(go.Candlestick(
        x=xs[:-1], open=opens[:-1], high=highs[:-1], low=lows[:-1], close=closes[:-1],
        increasing_line_color="#9ca3af", decreasing_line_color="#9ca3af",
        increasing_fillcolor="rgba(156,163,175,0.3)", decreasing_fillcolor="rgba(156,163,175,0.3)",
        line=dict(width=0.8), name="Context", showlegend=False
    ))

    # Predicted (bold color, thick lines)
    fig.add_trace(go.Candlestick(
        x=[xs[-1]], open=[pred_open], high=[pred_high], low=[pred_low], close=[pred_close],
        increasing_line_color=pred_color, decreasing_line_color=pred_color,
        increasing_fillcolor=pred_color, decreasing_fillcolor=pred_color,
        line=dict(width=2.5), name="Predicted", showlegend=False
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_visible=False, xaxis_showgrid=False, xaxis_zeroline=False,
        yaxis_title="Price", yaxis_showgrid=True, yaxis_gridcolor="#1f2937", yaxis_zeroline=False,
        paper_bgcolor="#020617", plot_bgcolor="#020617", height=220,
        # Tighter layout
        font_size=10
    )
    return fig


# ============ Mock backend functions ============

def get_news_heatmap_data(symbol: str, timeframe: str) -> pd.DataFrame:
    now = datetime.utcnow()
    rows = [
        {
            "bucket": "Last 30m",
            "timestamp": now - timedelta(minutes=10),
            "headline": "BTC ETF inflows hit new weekly high",
            "sentiment": "positive",
            "impact": 0.95,
        },
        {
            "bucket": "Last 2h",
            "timestamp": now - timedelta(hours=1, minutes=15),
            "headline": "Major exchange experiences brief outage",
            "sentiment": "negative",
            "impact": 0.8,
        },
        {
            "bucket": "Last 6h",
            "timestamp": now - timedelta(hours=4),
            "headline": "Whale moves large BTC tranche to exchange",
            "sentiment": "negative",
            "impact": 0.65,
        },
        {
            "bucket": "Last 24h",
            "timestamp": now - timedelta(hours=14),
            "headline": "On-chain activity rises amid renewed interest",
            "sentiment": "neutral",
            "impact": 0.4,
        },
        {
            "bucket": "Older",
            "timestamp": now - timedelta(days=1, hours=5),
            "headline": "Macro data comes in line with expectations",
            "sentiment": "neutral",
            "impact": 0.2,
        },
    ]
    return pd.DataFrame(rows)

def get_indicator_states(symbol: str, timeframe: str):
    return {
        "RSI": "Bullish (above 55)", "MACD": "Sideways (flat histogram)",
        "MA50 vs MA200": "Bullish (golden cross)",
        "Bollinger Bands": "High volatility (near upper band)",
        "Volume": "Above recent average",
    }

def get_current_trend(symbol: str, timeframe: str) -> str:
    return "Uptrend"

def get_mock_prediction_history() -> pd.DataFrame:
    records = []
    today = date.today()
    for i in range(180):
        d = today - timedelta(days=179 - i)
        prediction = random.choice(["Bullish", "Bearish"])
        actual = random.choice(["Up", "Down"])
        correct = (prediction == "Bullish" and actual == "Up") or (prediction == "Bearish" and actual == "Down")
        pnl = 0.01 if correct else -0.005
        records.append({"date": d, "timeframe": "1h", "prediction": prediction, "actual_move": actual, "correct": correct, "pnl_pct": pnl})
    return pd.DataFrame(records)

def ask_ai_trade_assistant(level: float, direction: str, timeframe: str, message: str):
    if level is None:
        return "Please provide a key price level so I can reason about a breakdown / breakout."
    direction_txt = "breakdown below" if direction == "Breakdown" else "breakout above"
    return (
        f"Assuming a {direction_txt} {level:.0f} on the {timeframe} chart:\n\n"
        f"- RSI should ideally confirm the move (dropping below 50 on breakdown, above 60 on breakout).\n"
        f"- MACD should either be already crossed in the {direction.lower()} direction or crossing at the move.\n"
        f"- Volume should expand compared to the last few candles to avoid a fake move.\n\n"
        "You can tighten or relax these conditions depending on your risk tolerance."
    )

# ============ YOUR NEW NEWS FUNCTIONS (MOCK IMPLEMENTATION) ============

def fetch_categorized_news(symbol: str) -> dict:
    """MOCK function to simulate fetching categorized news articles."""
    now = datetime.utcnow()

    raw_items = [
        {"category": "crypto", "bucket": "Last 30m", "sentiment": "positive", "headline": "BTC ETF inflows hit new weekly high", "impact": 0.95},
        {"category": "financial", "bucket": "Last 2h", "sentiment": "negative", "headline": "Major exchange experiences brief outage", "impact": 0.8},
        {"category": "geopolitical", "bucket": "Last 6h", "sentiment": "negative", "headline": "Regulatory uncertainty clouds market sentiment", "impact": 0.65},
        {"category": "crypto", "bucket": "Last 24h", "sentiment": "neutral", "headline": "On-chain activity rises amid renewed interest", "impact": 0.4},
        {"category": "financial", "bucket": "Older", "sentiment": "neutral", "headline": "Macro data comes in line with expectations", "impact": 0.2},
    ]

    grouped = defaultdict(list)
    for item in raw_items:
        grouped[item["category"]].append(item)

    return dict(grouped)

def render_news_list(articles: list[dict]):
    """MOCK function to render a list of news articles."""
    if not articles:
        st.write("No news in this category at the moment.")
        return

    for row in articles:
        impact = row["impact"]
        if impact > 0.8: bg, border = "#fee2e2", "#ef4444"
        elif impact > 0.5: bg, border = "#fef3c7", "#f59e0b"
        else: bg, border = "#dcfce7", "#22c55e"
        
        st.markdown(f"""
            <div style="border-left:4px solid {border}; background-color:{bg}; padding:0.5rem 0.75rem; margin-bottom:0.4rem; border-radius:4px;">
              <div style="font-size:0.8rem; color:#4b5563;">{row['bucket']} · {row['sentiment'].capitalize()}</div>
              <div style="font-size:0.95rem; font-weight:500; margin-top:0.05rem; color:#111827;">{row['headline']}</div>
            </div>
            """, unsafe_allow_html=True)

# ============ TradingView chart embed ============

def render_tradingview_chart(symbol: str = "BTCUSD", interval: str = "60", height: int = 720):
    symbol_tv = symbol.replace("USDT", "USD").replace("USDC", "USD")
    html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div id="tradingview_chart" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%", "height": {height}, "symbol": "{symbol_tv}",
          "interval": "{interval}", "timezone": "Etc/UTC", "theme": "light",
          "style": "1", "locale": "en", "toolbar_bg": "#f1f3f6",
          "hide_top_toolbar": false, "hide_side_toolbar": false,
          "allow_symbol_change": false, "withdateranges": true, "details": false,
          "hotlist": false, "calendar": false, "studies": [],
          "container_id": "tradingview_chart"
        }});
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(html, height=height + 40, scrolling=True)

# ============ UI ============

def main():
    # Fetch live BTC price once per run (used for top bar and day's range)
    live_price, live_change, live_change_pct = get_live_btc_price()
    change_color = "#16a34a" if live_change >= 0 else "#dc2626"
    change_sign = "+" if live_change >= 0 else ""

    # ----- Top bar -----
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; padding:0.75rem 0.5rem 0.5rem 0.5rem; border-bottom:1px solid #e5e7eb;">
          <div style="display:flex; flex-direction:column;">
            <div style="font-size:1.3rem; font-weight:600;">Bitcoin <span style="color:#6b7280; font-weight:400;">(BTC/USD)</span></div>
            <div style="margin-top:0.4rem; display:flex; align-items:baseline; gap:0.6rem;">
              <span style="font-size:2rem; font-weight:600;">{live_price:,.1f}</span>
              <span style="color:{change_color}; font-weight:600;">{change_sign}{live_change:,.1f} ({change_sign}{live_change_pct:.2f}%)</span>
            </div>
            <div style="margin-top:0.1rem; color:#6b7280; font-size:0.85rem;">Real-time data · Binance feed</div>
          </div>
          <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.4rem;">
            <div style="display:flex; gap:0.5rem;"><button style="background-color:#2563eb;color:white;border:none; padding:0.35rem 0.8rem;border-radius:4px;font-size:0.85rem;">★ Add to Watchlist</button></div>
            <div style="display:flex; gap:0.5rem; margin-top:0.25rem;">
              <button style="background-color:#16a34a;color:white;border:none; padding:0.4rem 1.1rem;border-radius:4px;font-weight:600;">Buy</button>
              <button style="background-color:#dc2626;color:white;border:none; padding:0.4rem 1.1rem;border-radius:4px;font-weight:600;">Sell</button>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Controls row (Symbol + Timeframe) -----
    col_l, col_r = st.columns([3, 1]) 
    with col_l:
        c1, _, c2 = st.columns([1.2, 0.4, 1.2])
        with c1:
            st.markdown("<div style='font-size:0.9rem; color:#e5e7eb; margin-bottom:0.9rem;'>Symbol</div>", unsafe_allow_html=True)
            symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0, label_visibility="collapsed", key="symbol_dd")
        with c2:
            st.markdown("<div style='font-size:0.9rem; color:#e5e7eb; margin-bottom:0.9rem;'>Timeframe</div>", unsafe_allow_html=True)
            timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h"], index=0, label_visibility="collapsed", key="tf_dd")



    with col_r:
        # Day's range now dynamic from intraday approx (using 24h stats as proxy)
        # Hard upper/lower bounds from Binance 24h stats
        # low/high are used as today's range; live_price as current
        day_low_approx = live_price * 0.98
        day_high_approx = live_price * 1.02

    # Auto-refresh every 2 minutes
    _ = st_autorefresh(interval=2 * 60 * 1000, key="dashboard_refresh")

    # 52 week range still mock
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        render_range_bar("Day's Range", day_low_approx, day_high_approx, live_price)
    with col_r2:
        render_range_bar("52 wk Range", 60187.0, 126186.0, live_price)

    # ----- Tabs -----
    tab_general, tab_chart, tab_news, tab_tech, tab_history = st.tabs(["Home", "Chart", "News", "Technical Indicators", "Historical Data"])

    with tab_general:
        st.subheader("About Bitcoin", anchor=False)
        st.write("Bitcoin (BTC) is the first and most well-known cryptocurrency—a type of digital money that runs on a decentralized network instead of a bank or government.")
        st.caption("Educational overview only – not investment advice.")
        st.markdown("---")
        st.markdown("### How do you feel today about Bitcoin?", unsafe_allow_html=True)
        sentiment = st.radio(" ", ["Bullish?", "Bearish?"], index=None, horizontal=True, key="general_sentiment")
        if sentiment:
            st.markdown(
                "Cool, let's validate your view with our prediction model on the **Chart** tab.",
                unsafe_allow_html=True,
            )

    with tab_chart:
        # real last 6 candles for pattern analysis + mini chart
        last_6 = load_last_n_candles(timeframe, n=6)

        prev_patterns = []
        for row in last_6[:-1]:
            patt = classify_candle_pattern(row["open"], row["high"], row["low"], row["close"])
            prev_patterns.append(patt)
        prev_pattern = prev_patterns[-1] if prev_patterns else "Standard"

        chart_container, rationale_container, ai_container = st.container(), st.container(), st.container()

        FEATURE_CSV_BY_TF = {
            "15m": "data/processed/btc_15m_features.csv",
            "1h": "data/processed/btc_1h_features.csv", 
            "4h": "data/processed/btc_4h_features.csv",
            "1d": "data/processed/btc_1d_features.csv",
            "5m": "data/processed/btc_15m_features.csv" 
        }

        with chart_container:
            left, right = st.columns([3, 1])
            with left:
                st.markdown(
                    """
                    <div style='font-size:1.1rem; font-weight:600;'>
                      ● <span style='color:#22c55e;'>Live</span>
                      <span style='margin-left:0.35rem;'>BTC</span>
                      <span style='font-size:0.9rem; color:#6b7280;'> (BTC/USDT)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                interval_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
                interval = interval_map.get(timeframe, "60")
                render_tradingview_chart(symbol.replace("USDT", "USD"), interval=interval)

            with right:
                # Header + red Refresh button only (no Horizon)
                # Title row with red Refresh button inline (perfect alignment)
                col_title, col_refresh = st.columns([3, 1])
                
                with col_refresh:

                    if st.button("🔄", key=f"refresh_icon_{timeframe}", 
                                help="Refresh predictions"):
                        st.markdown(
                            """
                            <style>
                            div[aria-label="🔄"] button { background-color: #ef4444 !important; color: white !important; padding: 0.2rem 0.4rem !important; }
                            </style>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.cache_data.clear()
                        st.rerun()

                horizon_key = "current" 

                if timeframe == "5m":
                    st.caption("🛈 Live model available for 15m, 1h, 4h timeframes only")
                else:
                    try:
                        pred = get_next_candle_prediction(timeframe, horizon_key)
                    except ValueError:
                        st.caption("🛈 Live model available for 15m, 1h, 4h timeframes only")
                        pred = {}


                # Horizon toggle
                horizon_mode = st.radio(
                    "Horizon", ["Current TF", "Next TF"], horizontal=True, key=f"pred_mode_{timeframe}_v5"
                )

                # Dynamic horizon name mapping
                horizon_map = {
                    "15m": {"current": "15m", "next": "1h"},
                    "1h": {"current": "1h", "next": "4h"},
                    "4h": {"current": "4h", "next": "1d"},
                    "5m": {"current": "5m", "next": "15m"}
                }

                horizon_key = "current" if horizon_mode == "Current TF" else "next"
                display_horizon = horizon_map.get(timeframe, {}).get(horizon_key, timeframe)

                # Title with st.rerun() trigger
                st.markdown(f"**Next {display_horizon} prediction**")

                # Real model prediction
                if timeframe == "5m":
                    st.info("Live model is available for 15m, 1h and 4h timeframes.")
                    return
                try:
                    pred = get_next_candle_prediction(timeframe, horizon_key)
                except ValueError:
                    st.info("Live model is available for 15m, 1h and 4h timeframes.")
                    return

                direction = pred.get("direction")
                conf = float(pred.get("confidence", 0.0))
                struct = pred.get("structure", {})

                if direction in ["Bullish", "SideBull"]:
                    dir_class = "blink-green"
                elif direction in ["Bearish", "SideBear"]:
                    dir_class = "blink-red"
                else:
                    dir_class = "blink-amber"

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;">
                    <div>
                        <div style="font-size:0.8rem;color:#9ca3af;">Direction</div>
                        <div class="{dir_class}" style="font-size:1.3rem;font-weight:700;">
                        {direction}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem;color:#9ca3af;">Confidence</div>
                        <div style="font-size:1.3rem;font-weight:600;color:#e5e7eb;">
                        {conf*100:.1f}%
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # **NEW: Structure row**
                struct_label = struct.get('label', 'Ranging')
                if "Uptrend" in struct_label:
                    struct_class = "blink-green"
                    struct_color = "#059669"
                elif "Downtrend" in struct_label:
                    struct_class = "blink-red"
                    struct_color = "#dc2626"
                else:  # Ranging or other
                    struct_class = "blink-amber"
                    struct_color = "#d97706"

                st.markdown(
                    f"""
                    <div style="background:rgba(59,130,246,0.08); padding:0.6rem; border-radius:8px; margin:0.8rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                        <div style="font-size:0.75rem; color:#f9fafb; font-weight:500;">TREND STRUCTURE</div>
                        <div class="{struct_class}" style="font-size:1.05rem; font-weight:700; color:{struct_color};">
                            {struct_label} <span style="font-size:0.85rem; opacity:0.8;">({struct.get('strength', 0)*100:.0f}%)</span>
                        </div>
                        </div>
                        <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:#f9fafb;">COMBINED</div>
                        <div style="font-size:1.0rem; font-weight:600; color:#f9fafb;">
                            {struct.get('combined_direction', 'N/A')}
                        </div>
                        </div>
                    </div>
                    <div style="font-size:0.75rem; color:#f9fafb; margin-top:0.3rem;">
                        HH/HL: {struct.get('hh_count',0)}/{struct.get('hl_count',0)} | LH/LL: {struct.get('lh_count',0)}/{struct.get('ll_count',0)}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"Model input: {pred.get('as_of')} · {pred.get('timeframe')} → {pred.get('horizon')}",
                    unsafe_allow_html=True,
                )

                # Mini prediction chart: last 5 real candles + 1 forecast candle
                recent = load_last_n_candles(timeframe, n=6)
                mini_fig = build_mini_prediction_chart(recent, pred)
                st.plotly_chart(mini_fig, use_container_width=True)

                st.caption(
                    "Mini prediction chart: last 5 closed candles (grey) plus the "
                    "model's forecast candle for the next bar of the selected timeframe."
                )

            with rationale_container:
                st.markdown("---")

                # Centered, white, bigger heading
                st.markdown(
                    """
                    <div style="
                        text-align:center;
                        font-weight:700;
                        font-size:1.4rem;
                        margin-bottom:0.75rem;
                        color:#ffffff;
                    ">
                        Our Rationale
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # --- load indicators for the CURRENT timeframe ---
                # If you already have @st.cache_data get_live_indicators(timeframe), use it:
                inds = get_live_indicators(timeframe) if "get_live_indicators" in globals() else {}

                if inds:
                    rsi = inds["rsi"]
                    macd = inds["macd"]
                    macd_signal = inds["macd_signal"]
                    volume_ratio = inds["volume_ratio"]
                    sma_ratio = inds["sma_ratio"]
                    bb_position = inds["bb_position"]
                    struct_label = struct.get("label", "Ranging") if "struct" in locals() else "Ranging"
                else:
                    # Fallback: direct CSV read per TF (still tied to selected timeframe)
                    feature_csv = FEATURE_CSV_BY_TF.get(timeframe)
                    if feature_csv and Path(feature_csv).exists():
                        df = pd.read_csv(feature_csv).tail(1)
                        latest = df.iloc[0]
                        rsi = latest.get("rsi", 50)
                        macd = latest.get("macd", 0)
                        macd_signal = latest.get("macd_signal", 0)
                        volume_ratio = latest.get("volume_ratio", 1.0)
                        sma_ratio = latest.get("sma_ratio", 1.0)
                        bb_position = latest.get("bb_position", 0.5)
                        struct_label = struct.get("label", "Ranging") if "struct" in locals() else "Ranging"
                    else:
                        rsi, macd, macd_signal, volume_ratio, sma_ratio, bb_position = 50, 0, 0, 1.0, 1.0, 0.5
                        struct_label = "Data unavailable"

                # --- Dynamic rationale bullets with one-line explanations ---
                rationale_points: list[str] = []
                rationale_points.append(
                    f"- **Structure**: {struct_label} — recent swing highs and lows suggest this overall trend."
                )

                # SMA relationship
                if sma_ratio > 1.0:
                    rationale_points.append(
                        "- Price > SMA20/50: Bullish bias — price trades above key moving averages, showing buyers in control."
                    )
                elif sma_ratio < 0.98:
                    rationale_points.append(
                        "- Price < SMAs: Bearish pressure — price sits below key moving averages, indicating sellers dominating."
                    )
                else:
                    rationale_points.append(
                        "- Price ~ SMAs: Neutral — price is hovering around moving averages, signalling indecision."
                    )

                # RSI state
                if rsi > 70:
                    rationale_points.append(
                        f"- RSI {rsi:.0f}: Overbought — upside momentum is strong but risk of a pullback increases."
                    )
                elif rsi < 30:
                    rationale_points.append(
                        f"- RSI {rsi:.0f}: Oversold — downside momentum is stretched and a bounce becomes more likely."
                    )
                elif rsi > 50:
                    rationale_points.append(
                        f"- RSI {rsi:.0f}: Bullish — momentum tilts to the upside, favouring long setups."
                    )
                else:
                    rationale_points.append(
                        f"- RSI {rsi:.0f}: Neutral — momentum is balanced with no strong edge to either side."
                    )

                # MACD
                if macd > macd_signal:
                    rationale_points.append(
                        f"- MACD bullish ({macd:.3f}): The faster line is above the signal line, confirming bullish momentum."
                    )
                else:
                    rationale_points.append(
                        f"- MACD bearish ({macd:.3f}): The faster line is below the signal line, confirming bearish momentum."
                    )

                # Volume
                if volume_ratio > 1.2:
                    rationale_points.append(
                        f"- Volume ↑ {volume_ratio:.1f}x: Moves are happening on elevated activity, making the signal more reliable."
                    )
                elif volume_ratio < 0.8:
                    rationale_points.append(
                        f"- Volume ↓ {volume_ratio:.1f}x: Price moves are on thin volume, so signals may be weaker."
                    )
                else:
                    rationale_points.append(
                        f"- Volume normal {volume_ratio:.1f}x: Activity is typical, giving average reliability to the setup."
                    )

                # Bollinger Bands position
                if bb_position > 0.8:
                    rationale_points.append(
                        f"- Upper BB {bb_position:.0%}: Price is pushing near the upper band, indicating strong buying pressure."
                    )
                elif bb_position < 0.2:
                    rationale_points.append(
                        f"- Lower BB {bb_position:.0%}: Price is near the lower band, reflecting strong selling pressure."
                    )
                else:
                    rationale_points.append(
                        f"- BB mid {bb_position:.0%}: Price is around the middle band, consistent with a more balanced market."
                    )

                # --- layout: two columns of bullets ---
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    for point in rationale_points[:4]:
                        st.markdown(f"• {point}")
                with col_r2:
                    for point in rationale_points[4:]:
                        st.markdown(f"• {point}")


            # For the timeframes you support in the UI
            SUPPORTED_CHART_TFS = ["15m", "1h", "4h"]

            predictions_by_chart_tf: dict[str, list[dict]] = {}

            for tf in SUPPORTED_CHART_TFS:
                try:
                    predictions_by_chart_tf[tf] = get_predictions_for_chart(tf)
                except Exception:
                    predictions_by_chart_tf[tf] = []

            def get_primary_prediction_for_chart_tf(
                chart_tf: str,
                predictions_by_chart_tf: dict[str, list[dict]],
            ) -> tuple[str | None, float | None, dict | None]:
                preds = predictions_by_chart_tf.get(chart_tf, [])
                if not preds:
                    return None, None, None

                primary = preds[0]  # e.g. 1h/1h for 1h chart
                return (
                    primary.get("direction"),          # class_name, e.g. "Bullish"
                    primary.get("confidence"),         # float 0..1
                    primary.get("structure"),          # dict from structure_result.to_dict()
                )


            with ai_container:
                st.markdown("---")
                st.markdown("### AI trade assistant", unsafe_allow_html=True)

                col_left, col_right = st.columns([2, 1])

                with col_left:
                    st.write(
                        "Describe your scenario, e.g. "
                        "`If we get a breakdown of 68,000 on 1h, what should RSI/MACD/volume look like?`"
                    )
                    key_level = st.number_input(
                        "Key level (trendline / support / resistance)", value=68000.0, step=100.0
                    )
                    move_type = st.selectbox("Scenario", ["Breakdown", "Breakout"], index=0)
                    user_msg = st.text_area(
                        "Your question to the AI",
                        value=(
                            "If we get a breakdown of this level, what should indicators "
                            "look like to justify a short?"
                        ),
                        height=100,
                    )

                    col_btn_ask, col_btn_clear = st.columns([1, 1])
                    with col_btn_ask:
                        ask_btn = st.button("Ask AI about this scenario", use_container_width=True)
                    with col_btn_clear:
                        clear_btn = st.button("Clear AI response", use_container_width=True)

                    # Clear stored response
                    if clear_btn:
                        st.session_state.ai_response = None

                    # Call AI and persist response
                    if ask_btn:
                        try:
                            latest_inds = get_live_indicators(timeframe)
                        except Exception:
                            latest_inds = None

                        # Get model prediction + structure for this chart TF
                        model_direction, model_confidence, trend_structure = get_primary_prediction_for_chart_tf(
                            timeframe,
                            predictions_by_chart_tf,
                        )

                        st.session_state.ai_response = call_ai_trade_assistant(
                            level=key_level,
                            direction=move_type,
                            timeframe=timeframe,
                            message=user_msg,
                            latest_indicators=latest_inds,
                            model_direction=model_direction,
                            model_confidence=model_confidence,
                            trend_structure=trend_structure,
                        )
                    # Show persisted response (survives reruns/auto-refresh)
                    if st.session_state.ai_response:
                        st.markdown("**AI response:**")
                        st.write(st.session_state.ai_response)

                with col_right:
                    uploaded_img = st.file_uploader(
                        "Attach current chart screenshot",
                        type=["png", "jpg", "jpeg"],
                    )
                    if uploaded_img:
                        st.image(uploaded_img, caption="Attached chart for AI context")
                        st.caption("Image support not wired yet, but will be sent to the AI backend later.")


    with tab_news:
        st.header("Latest Hot News")

        # Tab container matching screenshot
        tab_crypto, tab_financial, tab_geo = st.tabs(["Crypto", "Financial", "Geopolitical"])

        with tab_crypto:
            
            crypto_news = fetch_latest_news(QUERIES['crypto'])
            for i, art in enumerate(crypto_news):
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("📈")
                    with col2:
                        st.markdown(f"**{art['title']}**")
                        if art['description']:
                            st.caption(art['description'])
                        st.caption(f"*{art['source']}* • {art['publishedAt'][:10]}")
                        st.markdown(f"[Read more]({art['url']})")
                    st.divider()

        with tab_financial:
           
            financial_news = fetch_latest_news(QUERIES['financial'])
            for art in financial_news:
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("💹")
                    with col2:
                        st.markdown(f"**{art['title']}**")
                        if art['description']:
                            st.caption(art['description'])
                        st.caption(f"*{art['source']}* • {art['publishedAt'][:10]}")
                        st.markdown(f"[Read more]({art['url']})")
                    st.divider()

        with tab_geo:
            
            geo_news = fetch_latest_news(QUERIES['geopolitical'])
            for art in geo_news:
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("🌍")
                    with col2:
                        st.markdown(f"**{art['title']}**")
                        if art['description']:
                            st.caption(art['description'])
                        st.caption(f"*{art['source']}* • {art['publishedAt'][:10]}")
                        st.markdown(f"[Read more]({art['url']})")
                    st.divider()

        


        #categorized_news = fetch_categorized_news(symbol)
        #render_news_panel(categorized_news)

    with tab_tech:
                st.subheader("Technical Analysis", anchor=False)
                ta_tf = st.radio(
                    "Timeframe",
                    ["15m", "1H", "4H"],
                    horizontal=True,
                    index=1,
                    key="tech_ta_tf",
                )

                row = load_live_features(ta_tf)

                if row is None:
                    st.warning(f"No live features found for btc_{ta_tf}_live_features.csv")
                    indicators = [
                        {"Indicator": "RSI",             "Status": "N/A", "Score": 0.5},
                        {"Indicator": "MACD",            "Status": "N/A", "Score": 0.5},
                        {"Indicator": "Bollinger Bands", "Status": "N/A", "Score": 0.5},
                        {"Indicator": "Volume",          "Status": "N/A", "Score": 0.5},
                        {"Indicator": "SMA",             "Status": "N/A", "Score": 0.5},
                    ]
                    overall_score = 0.5
                else:
                    indicators, overall_score = compute_indicator_snapshot(row)

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    render_indicator_bar("RSI", indicators[0]["Score"])
                with c2:
                    render_indicator_bar("MACD", indicators[1]["Score"])
                with c3:
                    render_indicator_bar("Bollinger Bands", indicators[2]["Score"])
                with c4:
                    render_indicator_bar("Volume", indicators[3]["Score"])
                with c5:
                    render_indicator_bar("SMA", indicators[4]["Score"])
                with c6:
                    render_indicator_bar("Overall Technical", overall_score)

                st.markdown("---")
                st.subheader("Key technical indicators", anchor=False)

                trend_support = (
                    "Supports uptrend" if overall_score >= 0.67
                    else "Supports downtrend" if overall_score <= 0.33
                    else "Neutral / mixed"
                )

                rows = [
                    {
                        "Indicator": item["Indicator"],
                        "Status": item["Status"],
                        "Trend support": trend_support,
                    }
                    for item in indicators
                ]
                tech_df = pd.DataFrame(rows)
                st.dataframe(tech_df, use_container_width=True)

                if row is not None:
                    st.caption(f"Live data from btc_{ta_tf}_live_features.csv (latest row).")
                else:
                    st.caption("Using fallback neutral values because the live feature file was not found.")


    with tab_history:
        st.subheader("Historical analysis (mock)", anchor=False)
        st.markdown("### Previous model predictions – hit/miss (mock)", unsafe_allow_html=True)
        hist_df = get_mock_prediction_history()
        lookback_label = st.selectbox("Lookback window", ["1 month", "3 months", "6 months"], index=2)
        days_hist = {"1 month": 30, "3 months": 90, "6 months": 180}[lookback_label]
        filt = hist_df[hist_df["date"] >= (date.today() - timedelta(days=days_hist))]
        if not filt.empty:
            st.write(f"- Total trades: **{len(filt)}**, correct: **{int(filt['correct'].sum())}**, accuracy: **{filt['correct'].mean():.1%}**.")
            st.dataframe(filt.sort_values("date", ascending=False), use_container_width=True, height=220)
        
        st.markdown("---")
        st.markdown("### Hypothetical returns calculator (mock)", unsafe_allow_html=True)
        col_cap, col_lb = st.columns(2)
        initial_capital = col_cap.number_input("Starting capital (€)", min_value=1000.0, value=10000.0, step=500.0)
        rb_label = col_lb.selectbox("Backtest window", ["1 month", "3 months", "6 months"], index=1)
        rb_days = {"1 month": 30, "3 months": 90, "6 months": 180}[rb_label]
        rb_hist = hist_df[hist_df["date"] >= (date.today() - timedelta(days=rb_days))]
        if not rb_hist.empty:
            final_capital = initial_capital * (1 + rb_hist["pnl_pct"]).prod()
            st.write(f"A starting capital of **€{initial_capital:,.0f}** would now be **€{final_capital:,.0f}** (**{(final_capital/initial_capital - 1):.1%}** return).")

if __name__ == "__main__":
    main()