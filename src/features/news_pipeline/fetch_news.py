'''
This script contains the function to fetch RELEVANT MACROECONOMIC & GEOPOLITICAL news 
from a curated list of top financial news sources, while EXCLUDING crypto-focused articles.
'''
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- OUR CURATED LIST OF TOP-TIER SOURCES ---
TOP_FINANCIAL_SOURCES = 'reuters,bloomberg,the-wall-street-journal,associated-press,financial-times,the-economist'

# --- NEW: COMPREHENSIVE QUERY FOR MACRO & GEOPOLITICAL NEWS ---
COMPREHENSIVE_QUERY = (
    "inflation OR rates OR 'central bank' OR fomc OR fed OR gdp OR "
    "election OR 'trade war' OR summit OR sanctions OR geopolitics"
)

# --- NEW: TERMS TO EXPLICITLY EXCLUDE ---
CRYPTO_EXCLUSION_TERMS = "NOT crypto NOT bitcoin NOT ethereum NOT blockchain"

# Securely get the API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_news_articles(sources: str = TOP_FINANCIAL_SOURCES):
    """
    Fetches relevant news articles, excluding crypto-specific topics.

    Args:
        sources: A comma-separated string of source identifiers.

    Returns:
        A list of articles or None if the request fails.
    """
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found. Please set it in your .env file.")
        return None

    # --- ADVANCED QUERY CONSTRUCTION ---
    # Final query combines the topics we want AND the topics we want to exclude.
    final_query = f"({COMPREHENSIVE_QUERY}) {CRYPTO_EXCLUSION_TERMS}"

    params = {
        'q': final_query,
        'sources': sources,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        print(f"Successfully fetched {len(articles)} relevant articles using advanced query.")
        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from NewsAPI: {e}")
        return None

if __name__ == '__main__':
    # The function will now use the advanced query logic by default
    test_articles = fetch_news_articles()
    
    if test_articles:
        print("\n--- LATEST RELEVANT MACRO/GEO-POLITICAL ARTICLES ---")
        for article in test_articles[:10]:
            source_name = article['source']['name']
            print(f"Title: {article['title']}")
            print(f"Source: {source_name}")
            print(f"URL: {article['url']}\n")