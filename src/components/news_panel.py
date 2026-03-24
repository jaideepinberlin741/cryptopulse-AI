import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone
from components.news_pipeline.fetch_news import fetch_categorized_news

def render_news_panel():
    st_autorefresh(interval=20000, key="news_refresh")

    tabs = st.tabs(["🪙 Crypto", "💰 Financial", "🌍 Geopolitics"])
    articles = fetch_categorized_news()

    if not articles:
        st.info("No news available right now.")
        return

    categories = {
        "crypto": tabs[0],
        "financial": tabs[1],
        "geopolitical": tabs[2],
    }

    for category, tab in categories.items():
        with tab:
            filtered = [a for a in articles if a.get("category") == category]
            if not filtered:
                st.caption("No articles in this category yet.")
                continue

            for article in filtered:
                render_heatmap_card(article)


def render_heatmap_card(article):
    title = article.get("title", "No title")
    source = article.get("source", {}).get("name", "Unknown source")
    url = article.get("url", "#")
    published = article.get("publishedAt")

    # --- Time bucket logic ---
    published_dt = None
    try:
        published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except:
        pass

    bucket, color = compute_bucket_and_color(published_dt)

    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {color};
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.5rem;
            background-color: rgba(0,0,0,0.03);
            border-radius: 4px;
        ">
            <div style="font-size:0.8rem; color:#6b7280;">
                {bucket} · {source}
            </div>
            <div style="font-size:0.95rem; font-weight:600; margin-top:0.15rem;">
                <a href="{url}" target="_blank" style="text-decoration:none; color:#111827;">
                    {title}
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_bucket_and_color(published_dt):
    if not published_dt:
        return "Unknown time", "#9ca3af"

    now = datetime.now(timezone.utc)
    diff = (now - published_dt).total_seconds() / 3600  # hours

    if diff <= 0.5:
        return "Last 30m", "#ef4444"  # red
    elif diff <= 2:
        return "Last 2h", "#f59e0b"  # amber
    elif diff <= 6:
        return "Last 6h", "#f59e0b"
    elif diff <= 24:
        return "Last 24h", "#22c55e"  # green
    else:
        return "Older", "#22c55e"
