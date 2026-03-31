# news_fetcher.py - New module for Latest Hot News section
import os
import requests
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')  # Set in .env or Streamlit secrets
NEWSAPI_BASE = 'https://newsapi.org/v2'

# Cache storage (in-memory or Redis for production)
@st.cache_data(ttl=300)  # 5 min cache
def fetch_latest_news(query: str, num_articles: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch latest news from NewsAPI using keyword search.
    Returns top articles with title, description, url, publishedAt, source.
    """
    if not NEWSAPI_KEY:
        return [{"title": "API key missing. Set NEWSAPI_KEY env var.", "description": "Add your key from newsapi.org", "url": "https://newsapi.org"}]
    
    params = {
        'q': query,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles * 2,  # Fetch extra, slice top
        'apiKey': NEWSAPI_KEY,
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Recent week
    }
    
    try:
        resp = requests.get(f'{NEWSAPI_BASE}/everything', params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        articles = data.get('articles', [])
        # Clean and limit
        return [{
            'title': art['title'][:80] + '...' if len(art['title']) > 80 else art['title'],
            'description': art['description'][:120] + '...' if art.get('description') else '',
            'url': art['url'],
            'publishedAt': art['publishedAt'],
            'source': art['source']['name'],
            'image': art.get('urlToImage', '')
        } for art in articles[:num_articles]]
    except Exception as e:
        return [{"title": f"Fetch error: {str(e)[:50]}", "description": "Check API key/network", "url": "#"}]

# Define queries matching screenshot tabs
QUERIES = {
    'crypto': 'bitcoin OR ethereum OR crypto OR blockchain OR btc OR eth',
    'financial': 'stocks OR markets OR fed OR inflation OR economy OR rates',
    'geopolitical': 'geopolitics OR sanctions OR war OR diplomacy OR elections OR trump OR china OR russia'
}