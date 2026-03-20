import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
BASE_URL = "https://newsapi.org/v2"

def fetch_categorized_news():
    """
    Fetches news from newsapi.org and categorizes it into 'financial', 
    'geopolitical', and 'crypto'.
    """
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY environment variable not set.")
        return []

    headers = {"Authorization": f"Bearer {NEWS_API_KEY}"}
    all_articles = []

    # 1. Fetch Financial News
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching financial news: {e}")

    # 2. Fetch Geopolitical News
    try:
        # **UPDATE:** Added 'politico' and 'the-economist' to the sources.
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching geopolitical news: {e}")

    # 3. Fetch Crypto News
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
    except requests.exceptions.RequestException as e:
        print(f"Error fetching crypto news: {e}")

    print(f"Total articles fetched: {len(all_articles)}")
    return all_articles

if __name__ == '__main__':
    categorized_articles = fetch_categorized_news()
    if categorized_articles:
        print("\n--- Sample of Fetched Articles ---")
        for article in categorized_articles[:15]:
            print(f"[{article.get('category')}] {article.get('title')}")