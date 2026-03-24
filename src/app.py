import random
from datetime import datetime, timedelta, date
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# Ensure project root is on sys.path so `src` package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live.infer_xgb import get_predictions_for_chart

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

# ============ Real prediction helper ============

MODEL_TF_MAP = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "4h",  # 1d chart uses 4h models (4h→4h, 4h→1d)
}

def get_next_candle_prediction(chart_tf: str, horizon_mode: str = "current") -> dict:
    """Get real prediction from XGBoost models for the selected chart TF."""
    preds = get_predictions_for_chart(chart_tf)  # [current_tf, next_tf]
    return preds[0] if horizon_mode == "current" else preds[1]

# ============ Real-time range helpers ============

def get_day_and_52w_range() -> tuple[float, float, float, float, float]:
    """
    Compute day's low/high and 52w low/high from 1d features CSV.
    Returns: (day_low, day_high, wk_low, wk_high, current_close)
    """
    path = Path("data/processed/btc_1d_features.csv")
    if not path.exists():
        # Fallback demo numbers if file missing
        return 69493.2, 71347.1, 60187.0, 126186.0, 70456.0

    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time")

    last = df.iloc[-1]
    day_low = float(last["low"])
    day_high = float(last["high"])
    current = float(last["close"])

    cutoff = df["open_time"].max() - pd.Timedelta(days=365)
    df_52 = df[df["open_time"] >= cutoff]
    wk_low = float(df_52["low"].min())
    wk_high = float(df_52["high"].max())

    return day_low, day_high, wk_low, wk_high, current

def load_last_6_candles(tf: str) -> list[dict]:
    """
    Load last 6 OHLC candles for selected timeframe from processed feature CSV.
    Used for rationale pattern analysis.
    """
    csv_map = {
        "15m": "data/processed/btc_15m_features.csv",
        "1h": "data/processed/btc_1h_features.csv",
        "4h": "data/processed/btc_4h_features.csv",
        "1d": "data/processed/btc_1d_features.csv",
    }
    path_str = csv_map.get(tf)
    if not path_str or not Path(path_str).exists():
        return []

    df = pd.read_csv(path_str)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time")
    tail = df.tail(6)

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

def build_rationale(pred: dict, indicator_states: dict, prev_pattern: str) -> tuple[str, str]:
    """Build dynamic rationale text from model output + indicators + pattern."""
    direction = pred.get("direction", "Neutral")
    conf = float(pred.get("confidence", 0.0))
    if conf < 0.55:
        conf_label = "low"
    elif conf < 0.7:
        conf_label = "moderate"
    else:
        conf_label = "high"

    rsi_state = indicator_states.get("RSI", "")
    macd_state = indicator_states.get("MACD", "")
    bb_state = indicator_states.get("Bollinger Bands", "")
    vol_state = indicator_states.get("Volume", "")

    left_lines = []
    right_lines = []

    left_lines.append(f"- Model bias is **{direction}** with {conf_label} confidence.")
    if "Bullish" in rsi_state:
        left_lines.append("- RSI is in bullish territory, supporting upside.")
    elif "Bearish" in rsi_state:
        left_lines.append("- RSI is in bearish territory, limiting upside.")
    else:
        left_lines.append("- RSI is in neutral territory, not strongly directional.")

    left_lines.append(f"- Recent candlesticks form a **{prev_pattern}**-type setup.")
    right_lines.append(f"- MACD: {macd_state}.")
    right_lines.append(f"- Volumes: {vol_state}.")
    right_lines.append(f"- Volatility / bands: {bb_state}.")

    return "\n".join(left_lines), "\n".join(right_lines)

# ============ Mini prediction chart (Story 6.3) ============
def build_mini_prediction_chart(ui_timeframe: str, pred: dict) -> go.Figure:
    """
    Single synthetic next candle, shaped as a candlestick pattern archetype.
    This is a visual explainer, not a literal OHLC forecast.
    """
    chart_tf = MODEL_TF_MAP[ui_timeframe]
    
    FEATURE_CSV_BY_TF = {
        "15m": "data/processed/btc_15m_features.csv",
        "1h": "data/processed/btc_1h_features.csv",
        "4h": "data/processed/btc_4h_features.csv",
        "1d": "data/processed/btc_1d_features.csv",
    }
    csv_path = FEATURE_CSV_BY_TF.get(chart_tf)

    last_close, last_high, last_low, vol_proxy = 70000.0, 70100.0, 69900.0, 0.002

    if csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path).sort_values("open_time")
            if not df.empty:
                last_row = df.iloc[-1]
                last_close = float(last_row["close"])
                last_high = float(last_row["high"])
                last_low = float(last_row["low"])
                if "volatility" in df.columns:
                    vol_proxy = float(last_row["volatility"])
        except Exception:
            pass

    recent_range = max(last_high - last_low, last_close * 0.001)
    base_range = max(recent_range, last_close * max(vol_proxy, 0.001))

    direction = pred.get("direction", "Neutral")
    confidence = float(pred.get("confidence", 0.0))

    if direction in ["Bullish", "SideBull"]:
        if confidence >= 0.60: pattern_type = "bull_marubozu"
        elif confidence >= 0.45: pattern_type = "bull_trend"
        else: pattern_type = "dragonfly"
    elif direction in ["Bearish", "SideBear"]:
        if confidence >= 0.60: pattern_type = "bear_marubozu"
        elif confidence >= 0.45: pattern_type = "bear_trend"
        else: pattern_type = "gravestone"
    else:
        pattern_type = "doji" if confidence <= 0.30 else "hammer"

    def pattern_bull_marubozu():
        low = last_close - 0.2 * base_range
        high = last_close + 0.8 * base_range
        return low + 0.05 * (high-low), high, low, high - 0.02 * (high-low)

    def pattern_bear_marubozu():
        low = last_close - 0.8 * base_range
        high = last_close + 0.2 * base_range
        return high - 0.05 * (high-low), high, low, low + 0.02 * (high-low)

    def pattern_bull_trend():
        low = last_close - 0.3 * base_range
        high = last_close + 0.7 * base_range
        return low + 0.20 * (high-low), high, low, low + 0.75 * (high-low)

    def pattern_bear_trend():
        low = last_close - 0.7 * base_range
        high = last_close + 0.3 * base_range
        return low + 0.80 * (high-low), high, low, low + 0.25 * (high-low)

    def pattern_doji():
        low = last_close - 0.6 * base_range
        high = last_close + 0.6 * base_range
        mid = (low + high) / 2.0
        return mid - 0.015 * base_range, high, low, mid + 0.015 * base_range

    def pattern_dragonfly():
        high = last_close + 0.15 * base_range
        low = last_close - 0.85 * base_range
        o = c = high - 0.05 * (high - low)
        return o, high, low, c

    def pattern_gravestone():
        high = last_close + 0.85 * base_range
        low = last_close - 0.15 * base_range
        o = c = low + 0.05 * (high - low)
        return o, high, low, c

    def pattern_hammer():
        high = last_close + 0.2 * base_range
        low = last_close - 0.8 * base_range
        body_top = low + 0.65 * (high-low)
        body_bottom = low + 0.55 * (high-low)
        return (body_bottom, body_top) if direction not in ["Bearish", "SideBear"] else (body_top, body_bottom), high, low, (body_top if direction not in ["Bearish", "SideBear"] else body_bottom)

    pattern_funcs = {
        "bull_marubozu": pattern_bull_marubozu, "bear_marubozu": pattern_bear_marubozu,
        "bull_trend": pattern_bull_trend, "bear_trend": pattern_bear_trend,
        "doji": pattern_doji, "dragonfly": pattern_dragonfly,
        "gravestone": pattern_gravestone, "hammer": pattern_hammer
    }
    
    o, high, low, c = pattern_funcs.get(pattern_type, pattern_doji)()

    if direction in ["Bullish", "SideBull"]: color = "#22c55e"
    elif direction in ["Bearish", "SideBear"]: color = "#ef4444"
    else: color = "#9ca3af"

    fig = go.Figure(data=[go.Candlestick(
        x=[0], open=[o], high=[high], low=[low], close=[c],
        increasing_line_color=color, decreasing_line_color=color,
        increasing_fillcolor=color, decreasing_fillcolor=color,
        showlegend=False,
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(gridcolor="#1f2937"), paper_bgcolor="#020617", plot_bgcolor="#020617", height=220)
    return fig

# ============ Mock backend functions ============

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

def fetch_categorized_news() -> list[dict]:
    """MOCK function to simulate fetching categorized news articles."""
    now = datetime.utcnow()
    return [
        {"category": "crypto", "bucket": "Last 30m", "sentiment": "positive", "headline": "BTC ETF inflows hit new weekly high", "impact": 0.95},
        {"category": "financial", "bucket": "Last 2h", "sentiment": "negative", "headline": "Major exchange experiences brief outage", "impact": 0.8},
        {"category": "geopolitical", "bucket": "Last 6h", "sentiment": "negative", "headline": "Regulatory uncertainty clouds market sentiment", "impact": 0.65},
        {"category": "crypto", "bucket": "Last 24h", "sentiment": "neutral", "headline": "On-chain activity rises amid renewed interest", "impact": 0.4},
        {"category": "financial", "bucket": "Older", "sentiment": "neutral", "headline": "Macro data comes in line with expectations", "impact": 0.2},
    ]

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

def render_tradingview_chart(symbol: str = "BTCUSD", interval: str = "60", height: int = 520):
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
    components.html(html, height=height + 40, scrolling=False)

# ============ UI ============

def main():
    # ----- Top bar -----
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; padding:0.75rem 0.5rem 0.5rem 0.5rem; border-bottom:1px solid #e5e7eb;">
          <div style="display:flex; flex-direction:column;">
            <div style="font-size:1.3rem; font-weight:600;">Bitcoin <span style="color:#6b7280; font-weight:400;">(BTC/USD)</span></div>
            <div style="margin-top:0.4rem; display:flex; align-items:baseline; gap:0.6rem;">
              <span style="font-size:2rem; font-weight:600;">69,936.8</span>
              <span style="color:#16a34a; font-weight:600;">+477.2 (+0.69%)</span>
            </div>
            <div style="margin-top:0.1rem; color:#6b7280; font-size:0.85rem;">Real-time data · Mock feed</div>
          </div>
          <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.4rem;">
            <div style="display:flex; gap:0.5rem;"><button style="background-color:#2563eb;color:white;border:none; padding:0.35rem 0.8rem;border-radius:4px;font-size:0.85rem;">★ Add to Watchlist</button></div>
            <div style="display:flex; gap:0.5rem; margin-top:0.25rem;">
              <button style="background-color:#16a34a;color:white;border:none; padding:0.4rem 1.1rem;border-radius:4px;font-weight:600;">Buy</button>
              <button style="background-color:#dc2626;color:white;border:none; padding:0.4rem 1.1rem;border-radius:4px;font-weight:600;">Sell</button>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ----- Controls row + ranges -----
    col_l, col_r = st.columns([3, 1])
    with col_l:
        c1, _, c2 = st.columns([1.2, 0.4, 1.2])
        with c1:
            st.markdown("<div style='font-size:0.9rem; color:#e5e7eb; margin-bottom:0.2rem;'>Symbol</div>", unsafe_allow_html=True)
            symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=0, label_visibility="collapsed", key="symbol_dd")
        with c2:
            st.markdown("<div style='font-size:0.9rem; color:#e5e7eb; margin-bottom:0.2rem;'>Timeframe</div>", unsafe_allow_html=True)
            ui_timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1, label_visibility="collapsed", key="tf_dd")
    with col_r:
        st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)
        bcol, icol = st.columns([1.0, 0.25])
        with bcol:
            st.button("Refresh", type="primary")
        with icol:
            st.markdown("""<div style="display:flex; align-items:center; height:100%;"><span title="Auto-refreshes every 5 minutes" style="font-size:1.0rem; color:#e5e7eb; cursor:help; margin-left:0.15rem;">ⓘ</span></div>""", unsafe_allow_html=True)

    _ = st_autorefresh(interval=5 * 60 * 1000, key="dashboard_refresh")
    day_low, day_high, wk_low, wk_high, cur_price = get_day_and_52w_range()
    col_r1, col_r2 = st.columns(2)
    with col_r1: render_range_bar("Day's Range", day_low, day_high, cur_price)
    with col_r2: render_range_bar("52 wk Range", wk_low, wk_high, cur_price)

    # ----- Tabs -----
    tab_general, tab_chart, tab_news, tab_tech, tab_history = st.tabs(["General", "Chart", "Latest Hot News", "Technical Analysis", "Historical Analysis"])

    with tab_general:
        st.subheader("About Bitcoin", anchor=False)
        st.write("Bitcoin is the first decentralized cryptocurrency... not financial advice.")
        st.caption("Educational overview only – not investment advice.")
        st.markdown("---")
        st.markdown("### How do you feel today about Bitcoin?", unsafe_allow_html=True)
        sentiment = st.radio(" ", ["Bullish (green)", "Bearish (red)"], index=None, horizontal=True, key="general_sentiment")
        if sentiment:
            st.markdown("<span style='font-size:0.9rem; color:#9ca3af;'>Cool, let's validate your view with our prediction model on the <b>Chart</b> tab.</span>", unsafe_allow_html=True)

    with tab_chart:
        last_6 = load_last_6_candles(ui_timeframe)
        prev_pattern = "Standard"
        if len(last_6) > 1:
            prev_pattern = classify_candle_pattern(last_6[-2]["open"], last_6[-2]["high"], last_6[-2]["low"], last_6[-2]["close"])

        chart_container, rationale_container, ai_container = st.container(), st.container(), st.container()

        with chart_container:
            left, right = st.columns([3, 1])
            with left:
                st.markdown("<div style='font-size:1.1rem; font-weight:600;'>● <span style='color:#22c55e;'>Live</span><span style='margin-left:0.35rem;'>BTC</span><span style='font-size:0.9rem; color:#6b7280;'> (BTC/USDT)</span></div>", unsafe_allow_html=True)
                interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
                render_tradingview_chart(symbol, interval=interval_map.get(ui_timeframe, "60"))
            with right:
                st.markdown(f"<div style='font-size:1.1rem; font-weight:600;'>Next {ui_timeframe} prediction</div>", unsafe_allow_html=True)
                pred_mode_label = st.radio("Horizon", ["Current TF", "Next TF"], horizontal=True, key=f"pred_mode_{ui_timeframe}")
                horizon_mode = "current" if pred_mode_label == "Current TF" else "next"
                pred = get_next_candle_prediction(ui_timeframe, horizon_mode=horizon_mode) if ui_timeframe in {"15m", "1h", "4h"} else {}
                
                if pred:
                    direction = pred.get("direction", "Neutral")
                    dir_class = "blink-green" if direction in ["Bullish", "SideBull"] else "blink-red" if direction in ["Bearish", "SideBear"] else "blink-amber"
                    st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:0.4rem;">
                          <div><div style="font-size:0.85rem; color:#9ca3af;">Direction</div><div class="{dir_class}" style="font-size:1.8rem; font-weight:700;">{direction}</div></div>
                          <div style="text-align:right;"><div style="font-size:0.85rem; color:#9ca3af;">Confidence</div><div style="font-size:1.4rem; font-weight:600;">{pred.get("confidence", 0.0):.1%}</div></div>
                        </div>
                        <div style="font-size:0.80rem; color:#9ca3af; margin-top:0.35rem;">Model input: <b>{pred.get("as_of", "n/a")}</b> · {pred.get("timeframe", ui_timeframe)} → {pred.get("horizon", ui_timeframe)}</div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("Model not trained for this timeframe yet.")
                
                st.plotly_chart(build_mini_prediction_chart(ui_timeframe, pred), use_container_width=True)

        with rationale_container:
            st.markdown("---")
            st.markdown("<div style='font-size:1.1rem; font-weight:600;'>Our rationale</div>", unsafe_allow_html=True)
            left, right = st.columns(2)
            left_text, right_text = build_rationale(pred, get_indicator_states(symbol, ui_timeframe), prev_pattern)
            left.markdown(left_text)
            right.markdown(right_text)

        with ai_container:
            st.markdown("---")
            st.markdown("### AI trade assistant (mock backend)", unsafe_allow_html=True)
            left, right = st.columns([2, 1])
            with left:
                st.write("Describe your scenario...")
                key_level = st.number_input("Key level", value=68000.0, step=100.0)
                move_type = st.selectbox("Scenario", ["Breakdown", "Breakout"])
                user_msg = st.text_area("Your question to the AI", "If we get a breakdown...")
                if st.button("Ask AI about this scenario"):
                    ai_response = ask_ai_trade_assistant(key_level, move_type, ui_timeframe, user_msg)
                    st.markdown("**AI response (mock):**")
                    st.write(ai_response)
            with right:
                if uploaded_img := st.file_uploader("Attach current chart screenshot", type=["png", "jpg"]):
                    st.image(uploaded_img, caption="Attached chart for AI context")
                    st.caption("This image would be sent to the AI backend.")

    with tab_news:
        st.subheader("Latest Hot News (mock)", anchor=False)
        tab_crypto, tab_finance, tab_geo = st.tabs(["Crypto", "Financials", "Geopolitics"])
        articles = fetch_categorized_news()
        with tab_crypto:
            render_news_list([a for a in articles if a["category"] == "crypto"])
        with tab_finance:
            render_news_list([a for a in articles if a["category"] == "financial"])
        with tab_geo:
            render_news_list([a for a in articles if a["category"] == "geopolitical"])
        st.caption("Red = hot/recent, amber = medium, green = older/lower impact.")

    with tab_tech:
        st.subheader("Technical Analysis", anchor=False)
        ta_tf = st.radio("Timeframe", ["15m", "1H", "4H", "1D"], horizontal=True, index=1, key="tech_ta_tf")
        
        if ta_tf == "15m": ti_label, ti_score, ma_label, ma_score, sum_label, sum_score = "Buy", 0.65, "Strong Buy", 0.80, "Buy", 0.70
        elif ta_tf == "1H": ti_label, ti_score, ma_label, ma_score, sum_label, sum_score = "Buy", 0.60, "Buy", 0.65, "Buy", 0.62
        elif ta_tf == "4H": ti_label, ti_score, ma_label, ma_score, sum_label, sum_score = "Neutral", 0.50, "Neutral", 0.50, "Neutral", 0.50
        else: ti_label, ti_score, ma_label, ma_score, sum_label, sum_score = "Sell", 0.35, "Sell", 0.30, "Sell", 0.32

        c_ti, c_sum, c_ma = st.columns(3)
        with c_ti: render_ta_gauge("Technical Indicators", ti_label, ti_score)
        with c_sum: render_ta_gauge("Summary", sum_label, sum_score)
        with c_ma: render_ta_gauge("Moving Averages", ma_label, ma_score)

        st.markdown("---")
        st.subheader("Key technical indicators", anchor=False)
        indicator_states = get_indicator_states(symbol, ui_timeframe)
        rows = [{"Indicator": name, "Status": state, "Trend support": "Supports uptrend" if "Bullish" in state else "Supports downtrend" if "Bearish" in state else "Neutral / mixed"} for name, state in indicator_states.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption("Mock values for now...")

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