'''
This file contains the API endpoint for the news feature.
'''
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

# --- IMPORTANT: We are importing the function you just built! ---
from src.components.news_pipeline.fetch_news import fetch_news_articles

# Create a new router for this feature
router = APIRouter()

@router.get(
    "/",
    summary="Fetch processed news articles for the heatmap",
    response_model=List[Dict[str, Any]] # For now, we return the list of articles
)
async def get_news_for_heatmap():
    """
    This endpoint fetches the latest relevant macroeconomic and geopolitical news,
    processes it, and returns it in a format ready for the heatmap visualization.
    
    (Note: Aggregation logic will be added next.)
    """
    print("API endpoint /v1/news/ hit. Fetching articles...")
    
    # --- HERE IS THE MAGIC ---
    # We call the function from your other file to get the articles.
    articles = fetch_news_articles()
    # -------------------------

    if articles is None:
        # If the fetch fails, we send a clear error back to the frontend.
        raise HTTPException(status_code=500, detail="Failed to fetch articles from the news provider.")

    # For now, we will just return the raw articles.
    # In the next step, we will add the logic to aggregate these into heatmap data.
    print(f"Successfully fetched {len(articles)} articles. Returning to client.")
    return articles