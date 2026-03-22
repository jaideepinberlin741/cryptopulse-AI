from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from src.components.news_pipeline.fetch_news import fetch_news_articles

router = APIRouter()

@router.get(
    "/",
    summary="Fetch processed news articles for the heatmap",
    response_model=List[Dict[str, Any]] 
)
async def get_news_for_heatmap():
    """
    Fetch the latest macroeconomic, geopolitical, and crypto-related news
    and return it in a format suitable for the heatmap visualization.
    """
    print("API endpoint /v1/news/ hit. Fetching articles...")
    
    articles = fetch_news_articles()

    if articles is None:
        raise HTTPException(status_code=500, detail="Failed to fetch articles from the news provider.")

    print(f"Successfully fetched {len(articles)} articles. Returning to client.")
    return articles