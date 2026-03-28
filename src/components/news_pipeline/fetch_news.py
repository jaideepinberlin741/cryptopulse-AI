import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .mock_data import MOCK_NEWS

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2"

def fetch_categorized_news(symbol: str):
    """
    Returns categorized news for a given symbol.
    For now this uses mock data.
    """

    # 1. Fallback if API key crashes
    if not NEWS_API_KEY:
        print("NEWS_API_KEY missing → using legacy heatmap mock.")

        # Lazy import to avoid circular import
        from app import get_news_heatmap_data

        df = get_news_heatmap_data("BTC", "1h")

        converted = []
        for _, row in df.iterrows():
            converted.append({
                "title": row["headline"],
                "sentiment": row["sentiment"],
                "bucket": row["bucket"],
                "impact": row["impact"],
                "category": "crypto",
            })

        return converted

    # 2. real API-call
    headers = {"Authorization": f"Bearer {NEWS_API_KEY}"}
    all_articles = []

    # Financial news
    try:
        financial_params = {
            "language": "en",
            "sources": "bloomberg,the-wall-street-journal,reuters,financial-times"
        }
        response = requests.get(f"{BASE_URL}/top-headlines", headers=headers, params=financial_params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        for article in articles:
            article['category'] = 'financial'
        all_articles.extend(articles)
        print(f"Successfully fetched {len(articles)} financial articles.")
    except requests.exceptions.RequestException:
        print("Financial news unavailable (rate limit).")

    # Geopolitical news
    try:
        geopolitical_params = {
            "language": "en",
            "sources": "associated-press,bbc-news,politico,the-economist"
        }
        response = requests.get(f"{BASE_URL}/top-headlines", headers=headers, params=geopolitical_params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        for article in articles:
            article['category'] = 'geopolitical'
        all_articles.extend(articles)
        print(f"Successfully fetched {len(articles)} geopolitical articles.")
    except requests.exceptions.RequestException:
        print("Geopolitical news unavailable (rate limit).")

    # Crypto news
    try:
        crypto_params = {
            "q": "(crypto OR cryptocurrency OR bitcoin OR ethereum OR blockchain OR DeFi OR NFT) AND NOT (scam OR hack OR giveaway)",
            "language": "en",
            "sortBy": "publishedAt"
        }
        response = requests.get(f"{BASE_URL}/everything", headers=headers, params=crypto_params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        for article in articles:
            article['category'] = 'crypto'
        all_articles.extend(articles)
        print(f"Successfully fetched {len(articles)} crypto articles.")
    except requests.exceptions.RequestException:
        print("Crypto news unavailable (rate limit).")

    if not all_articles:
        print("API returned no data → using legacy heatmap mock.")

        from app import get_news_heatmap_data

        df = get_news_heatmap_data("BTC", "1h")

        converted = []
        for _, row in df.iterrows():
            converted.append({
                "title": row["headline"],
                "sentiment": row["sentiment"],
                "bucket": row["bucket"],
                "impact": row["impact"],
                "category": "crypto",
            })

        return converted

    print(f"Total articles fetched: {len(all_articles)}")
    return all_articles
