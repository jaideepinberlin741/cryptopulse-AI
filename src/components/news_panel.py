import streamlit as st

def render_news_item(item):

    bucket = item.get("bucket", "")
    sentiment = item.get("sentiment", "").capitalize()
    headline = item.get("headline", "Untitled")

    st.markdown(
        f"""
        <div style="padding: 0.75rem 1rem; border-bottom: 1px solid #e5e7eb;">
            <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.25rem;">
                <span style="background: {color}; width: 10px; height: 10px; border-radius: 50%;"></span>
                <span style="font-size: 0.85rem; color: #6b7280;">{bucket} · {sentiment}</span>
            </div>
            <div style="font-size: 1rem; font-weight: 500;">
                {headline}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_news_panel(categorized_news):
    st.subheader("Live Market News", anchor=False)

    tab_crypto, tab_financial, tab_geo = st.tabs(["Crypto", "Financials", "Geopolitics"])

    with tab_crypto:
        for item in categorized_news.get("crypto", []):
            render_news_item(item)

    with tab_financial:
        for item in categorized_news.get("financial", []):
            render_news_item(item)

    with tab_geo:
        for item in categorized_news.get("geopolitical", []):
            render_news_item(item)
