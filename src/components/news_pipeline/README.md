# Epic 5: News Data Pipeline

This document outlines the purpose and technical implementation of the news data pipeline for the CryptoPulse AI project.

## 1. Goal

The primary goal of this feature is to collect and categorize high-quality, relevant global news to provide a multi-faceted context for cryptocurrency market movements. The pipeline will gather news from three distinct categories: **Financial**, **Geopolitical**, and **Crypto**. This data is the foundation for the "News Heatmap" visualization (User Story #5.2).

## 2. Key Components

This feature consists of three main files:

-   `src/features/news_pipeline/fetch_news.py`: A standalone Python script responsible for connecting to the NewsAPI and fetching articles based on a sophisticated query strategy.
-   `src/api/v1/news.py`: A FastAPI router that defines the `/v1/news` API endpoint. It uses the `fetch_news` function to get the data.
-   `src/main.py`: The main FastAPI application entry point, which includes and serves the `/v1/news` endpoint.

## 3. Core Logic & Design Decisions

To provide a comprehensive view for the heatmap, the fetching strategy is designed to pull articles from three distinct categories by making separate, targeted calls to `newsapi.org`.

### 1. Financial News
This feed captures broad financial and economic news from top-tier sources.

*   **Endpoint:** `/v2/top-headlines`
*   **Parameters:**
    *   `category`: `business`
    *   `sources`: (Recommended) e.g., `bloomberg`, `the-wall-street-journal`, `financial-post`

### 2. Geopolitical News
This feed acts as a proxy for major world events that can impact markets, pulling from trusted international sources.

*   **Endpoint:** `/v2/top-headlines`
*   **Parameters:**
    *   `category`: `general`
    *   `sources`: (Recommended) e.g., `reuters`, `associated-press`

### 3. Crypto News
This feed provides highly specific, targeted news about the cryptocurrency ecosystem, which was previously excluded.

*   **Endpoint:** `/v2/everything`
*   **Parameters:**
    *   `q`: A detailed query string using boolean operators.
    *   **Example `q` parameter:** `(crypto OR cryptocurrency OR bitcoin OR ethereum OR blockchain OR DeFi OR NFT) AND NOT (scam OR hack OR giveaway)`

## 4. How to Run & Test

1.  **Get API Key:** Obtain a free API key from [NewsAPI.org](https://newsapi.org/).
2.  **Set Environment Variable:** Create a `.env` file in the project root and add the key: `NEWS_API_KEY="YOUR_KEY_HERE"`.
3.  **Install Dependencies:** Ensure you have installed the necessary libraries: `pip install fastapi "uvicorn[standard]" python-dotenv requests`.
4.  **Run the Server:** From the project root, run the command: `uvicorn src.main:app --reload`.
5.  **Test the Endpoint:** Open a browser or Postman and access `http://127.0.0.1:8000/v1/news/`.