import random
from datetime import datetime, timedelta, date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from components.news_panel import render_news_panel
from collections import defaultdict

# Ensure project root is on sys.path so `src` package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live.infer_xgb import get_predictions_for_chart

from src.live.infer_xgb import get_predictions_for_chart

# -------------------------------------------------
# Raw OHLC sources used for mini chart context
# -------------------------------------------------
RAW_FEATURE_CSV_BY_TF = {
    "15m": "data/raw/btc_15m_raw.csv",
    "1h":  "data/raw/btc_1h_raw.csv",
    "4h":  "data/raw/btc_4h_raw.csv",
}

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


def build_mini_prediction_chart(
    last_candles: list[dict],
    pred: dict,
) -> go.Figure:
    """
    Mini chart:
      - last 5 *real* candles (context)
      - 1 predicted candle from model output
    """
    if len(last_candles) < 1:
        return go.Figure()

    # Context candles
    ctx = last_candles[-5:]
    opens = [c["open"] for c in ctx]
    highs = [c["high"] for c in ctx]
    lows  = [c["low"]  for c in ctx]
    closes= [c["close"] for c in ctx]

    prev_open  = opens[-1]
    prev_close = closes[-1]
    prev_high  = highs[-1]
    prev_low   = lows[-1]

    prev_body  = abs(prev_close - prev_open)
    prev_range = max(prev_high, prev_low) - min(prev_high, prev_low)
    avg_body   = prev_body if prev_body > 0 else max(prev_range * 0.25, 1)
    base_range = prev_range if prev_range > 0 else avg_body * 1.5

    # Predicted return and direction
    pred_ret = pred.get("predicted_return", None)
    direction = pred.get("direction", "Neutral")

    if pred_ret is None:
        conf = float(pred.get("confidence", 0.0))
        mag = 0.002 + 0.004 * max(0.0, min(conf, 1.0))  # 0.2–0.6%
        if direction in ["Bullish", "SideBull"]:
            pred_ret = mag
        elif direction in ["Bearish", "SideBear"]:
            pred_ret = -mag
        else:
            pred_ret = 0.0

    target_close = prev_close * (1.0 + float(pred_ret))

    # Direction-based body shape
    if direction in ["Bullish", "SideBull"]:
        body_size = avg_body * (1.0 if direction == "SideBull" else 1.4)
        pred_open = target_close - body_size
        pred_close = target_close
    elif direction in ["Bearish", "SideBear"]:
        body_size = avg_body * (1.0 if direction == "SideBear" else 1.4)
        pred_open = target_close + body_size
        pred_close = target_close
    else:
        body_size = avg_body * 0.3
        pred_open = target_close - body_size / 2
        pred_close = target_close + body_size / 2

    # Scale body/wicks for next‑TF horizons (e.g. 15m->1h)
    scale = 4.0 if pred.get("horizon") == "1h" and pred.get("timeframe") == "15m" else 1.0
    body_size *= scale

    wick_scale = 0.4
    wick_range = base_range * wick_scale * scale

    pred_high = max(pred_open, pred_close) + wick_range * 0.6
    pred_low  = min(pred_open, pred_close) - wick_range * 0.4

    opens.append(pred_open)
    highs.append(pred_high)
    lows.append(pred_low)
    closes.append(pred_close)

    xs = list(range(len(opens)))
    fig = go.Figure()

    # Context candles (grey)
    fig.add_trace(
        go.Candlestick(
            x=xs[:-1],
            open=opens[:-1],
            high=highs[:-1],
            low=lows[:-1],
            close=closes[:-1],
            increasing_line_color="#9ca3af",
            decreasing_line_color="#9ca3af",
            increasing_fillcolor="rgba(156,163,175,0.4)",
            decreasing_fillcolor="rgba(156,163,175,0.4)",
            showlegend=False,
            name="Context",
        )
    )

    # Predicted candle color
    if direction in ["Bullish", "SideBull"]:
        inc_col = "#16a34a"
        dec_col = "#16a34a"
    elif direction in ["Bearish", "SideBear"]:
        inc_col = "#dc2626"
        dec_col = "#dc2626"
    else:
        inc_col = "#6b7280"
        dec_col = "#6b7280"

    fig.add_trace(
        go.Candlestick(
            x=xs[-1:],
            open=opens[-1:],
            high=highs[-1:],
            low=lows[-1:],
            close=closes[-1:],
            increasing_line_color=inc_col,
            decreasing_line_color=dec_col,
            increasing_fillcolor="rgba(22,163,74,0.7)",
            decreasing_fillcolor="rgba(220,38,38,0.7)",
            showlegend=False,
            name="Predicted",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="#1f2937",
            zeroline=False,
        ),
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        height=220,
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
            st.markdown(
                "<div style='font-size:0.9rem; color:#e5e7eb; margin-bottom:0.2rem;'>Timeframe</div>",
                unsafe_allow_html=True,
            )
            timeframe = st.selectbox(
                "Timeframe",
                ["5m", "15m", "1h", "4h", "1d"],
                index=0,
                label_visibility="collapsed",
                key="tf_dd",
            )
    with col_r:
        st.markdown("<div style='height:1.4rem;'></div>", unsafe_allow_html=True)
        bcol, icol = st.columns([1.0, 0.25])
        with bcol:
            st.button("Refresh", type="primary")
        with icol:
            st.markdown("""<div style="display:flex; align-items:center; height:100%;"><span title="Auto-refreshes every 5 minutes" style="font-size:1.0rem; color:#e5e7eb; cursor:help; margin-left:0.15rem;">ⓘ</span></div>""", unsafe_allow_html=True)

    # Auto-refresh every 2 minutes
    _ = st_autorefresh(interval=2 * 60 * 1000, key="dashboard_refresh")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        render_range_bar("Day's Range", 69493.2, 71347.1, 70456.0)
    with col_r2:
        render_range_bar("52 wk Range", 60187.0, 126186.0, 70456.0)

    # ----- Tabs -----
    tab_general, tab_chart, tab_news, tab_tech, tab_history = st.tabs(["General", "Chart", "Latest Hot News", "Technical Analysis", "Historical Analysis"])

    with tab_general:
        st.subheader("About Bitcoin", anchor=False)
        st.write("Bitcoin (BTC) is the first and most well-known cryptocurrency—a type of digital money that runs on a decentralized network instead of a bank or government.")
        st.caption("Educational overview only – not investment advice.")
        st.markdown("---")
        st.markdown("### How do you feel today about Bitcoin?", unsafe_allow_html=True)
        sentiment = st.radio(" ", ["Bullish (green)", "Bearish (red)"], index=None, horizontal=True, key="general_sentiment")
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
                st.markdown(
                    f"<div class='section-title'>Next {timeframe} prediction</div>",
                    unsafe_allow_html=True,
                )

                # Horizon toggle
                horizon_mode = st.radio(
                    "Horizon",
                    ["Current TF", "Next TF"],
                    horizontal=True,
                    key=f"pred_mode_{timeframe}",
                )
                horizon_key = "current" if horizon_mode == "Current TF" else "next"

                # Real model prediction
                if timeframe == "5m":
                    st.info("Live model is available for 15m, 1h and 4h timeframes.")
                    return
                try:
                    pred = get_next_candle_prediction(timeframe, horizon_key)
                except ValueError:
                    st.info("Live model is available for 15m, 1h and 4h timeframes.")
                    return

                direction = pred.get("direction", "Neutral")
                conf = float(pred.get("confidence", 0.0))

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
                        <div class="{dir_class}" style="font-size:1.4rem;font-weight:700;">
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
            st.markdown(
                "<div style='font-size:1.1rem; font-weight:600;'>Our rationale</div>",
                unsafe_allow_html=True,
            )
            col_r1, col_r2 = st.columns(2)

            compat = compatibility_score(prev_pattern, direction if pred else prev_pattern)

            with col_r1:
                st.markdown(
                    f"""
- Price is trading above key moving averages, supporting a bullish bias.
- RSI is in bullish territory without major divergence.
- Recent candlesticks form a **{prev_pattern}** setup, followed by a predicted **{direction if pred else prev_pattern}**.
- Pattern compatibility between previous and next candle is **{compat}**.
                    """
                )
            with col_r2:
                st.markdown(
                    """
- Recent candles show higher lows, consistent with an uptrend.
- No major negative headlines in the latest news heatmap buckets.
- Volumes are near or above recent averages.
                    """
                )

        with ai_container:
            st.markdown("---")
            st.markdown("### AI trade assistant (mock backend)", unsafe_allow_html=True)

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
                    value="If we get a breakdown of this level, what should indicators look like to justify a short?",
                    height=100,
                )
                ask_btn = st.button("Ask AI about this scenario")

                ai_response = None
                if ask_btn:
                    ai_response = ask_ai_trade_assistant(
                        level=key_level,
                        direction=move_type,
                        timeframe=timeframe,
                        message=user_msg,
                    )

                if ai_response:
                    st.markdown("**AI response (mock):**")
                    st.write(ai_response)
            with right:
                if uploaded_img := st.file_uploader("Attach current chart screenshot", type=["png", "jpg"]):
                    st.image(uploaded_img, caption="Attached chart for AI context")
                    st.caption("This image would be sent to the AI backend.")

    with tab_news:
        categorized_news = fetch_categorized_news(symbol)
        render_news_panel(categorized_news)

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

        indicator_states = get_indicator_states(symbol, timeframe)
        rows = []
        trend = get_current_trend(symbol, timeframe)
        for name, state in indicator_states.items():
            if "Bullish" in state and trend == "Uptrend":
                support = "Supports uptrend"
            elif "Bearish" in state and trend == "Downtrend":
                support = "Supports downtrend"
            else:
                support = "Neutral / mixed"
            rows.append(
                {
                    "Indicator": name,
                    "Status": state,
                    "Trend support": support,
                }
            )
        tech_df = pd.DataFrame(rows)
        st.dataframe(tech_df, use_container_width=True)
        st.caption(
            "Mock values for now – this view will later be driven by real indicator "
            "calculations for the selected symbol and timeframe."
        )

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