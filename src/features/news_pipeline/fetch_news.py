'''
This script contains the function to fetch RELEVANT MACROECONOMIC news 
from a curated list of top financial news sources via NewsAPI.
'''
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- NEW: A CURATED LIST OF TOP-TIER SOURCES ---
# We've created a constant to hold our list of trusted sources.
# This makes it easy to add or remove sources in the future.
TOP_FINANCIAL_SOURCES = 'reuters,bloomberg,the-wall-street-journal,associated-press,financial-times,the-economist'

# Securely get the API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_news_articles(query: str, sources: str = TOP_FINANCIAL_SOURCES):
    """
    Fetches news articles from NewsAPI based on a search query and specific sources.

    Args:
        query: The search term for the articles (e.g., "inflation OR interest rates").
        sources: A comma-separated string of source identifiers.

    Returns:
        A list of articles or None if the request fails.
    """
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found. Please set it in your .env file.")
        return None

    params = {
        'q': query,
        'sources': sources,  # We are now using our expanded list!
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        print(f"Successfully fetched {len(articles)} articles from top financial sources for query: '{query}'")
        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from NewsAPI: {e}")
        return None

if __name__ == '__main__':
    # A more sophisticated query to find relevant macroeconomic news
    macro_query = "inflation OR rates OR 'central bank' OR fomc OR fed"
    
    # The function will now use the TOP_FINANCIAL_SOURCES list by default
    test_articles = fetch_news_articles(query=macro_query)
    
    if test_articles:
        print("\n--- LATEST ARTICLES FROM TOP SOURCES ---")
        # Print the title, source, AND the clickable URL!
        for article in test_articles[:10]: # Let's show 10 now
            source_name = article['source']['name']
            print(f"Title: {article['title']}")
            print(f"Source: {source_name}")
            print(f"URL: {article['url']}\n")