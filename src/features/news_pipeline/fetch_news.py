import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Securely get the API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_news_articles(query: str):
    """
    Fetches news articles from NewsAPI based on a search query.

    Args:
        query: The search term for the articles (e.g., "Bitcoin").

    Returns:
        A list of articles or None if the request fails.
    """
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found. Please set it in your .env file.")
        return None

    params = {
        'q': query,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()  # Raise an error for bad responses

        data = response.json()
        articles = data.get("articles", [])

        print(f"Successfully fetched {len(articles)} articles for query: '{query}'")
        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from NewsAPI: {e}")
        return None

if __name__ == '__main__':
    # Test block: run directly with 'python src/features/news_pipeline/fetch_news.py'
    test_articles = fetch_news_articles(query="crypto")
    if test_articles:
        for article in test_articles[:5]:
            print(f"- {article['title']}")