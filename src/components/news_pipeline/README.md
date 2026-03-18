# Epic 5: News Data Pipeline

This document outlines the purpose and technical implementation of the news data pipeline for the CryptoPulse AI project.

## 1. Goal

The primary goal of this feature is to collect high-quality, relevant global news that can provide context for cryptocurrency market movements. The data gathered here is the foundation for the "News Heatmap" visualization (User Story #5.2).

## 2. Key Components

This feature consists of three main files:

-   `src/features/news_pipeline/fetch_news.py`: A standalone Python script responsible for connecting to the NewsAPI and fetching the articles based on a sophisticated query.
-   `src/api/v1/news.py`: A FastAPI router that defines the `/v1/news` API endpoint. It uses the `fetch_news` function to get the data.
-   `src/main.py`: The main FastAPI application entry point, which includes and serves the `/v1/news` endpoint.

## 3. Core Logic & Design Decisions

The effectiveness of this feature comes from a series of strategic data filtering decisions:

-   **Curated Sources:** We exclusively pull data from a curated list of top-tier, trusted financial news sources (`reuters`, `bloomberg`, `the-wall-street-journal`, etc.) to ensure signal quality.
-   **Macro & Geopolitical Focus:** The search query is specifically designed to find articles related to broad macroeconomic and geopolitical events (e.g., inflation, central bank policy, elections, trade wars).
-   **Crypto Noise Filtration:** Crucially, the query uses a `NOT` operator to **explicitly exclude** articles that are primarily about "crypto," "bitcoin," or "blockchain." This ensures the data reflects external world events that *influence* crypto, rather than the crypto world talking about itself.

## 4. How to Run & Test

1.  **Get API Key:** Obtain a free API key from [NewsAPI.org](https://newsapi.org/).
2.  **Set Environment Variable:** Create a `.env` file in the project root and add the key: `NEWS_API_KEY="YOUR_KEY_HERE"`.
3.  **Install Dependencies:** Ensure you have installed the necessary libraries: `pip install fastapi "uvicorn[standard]" python-dotenv requests`.
4.  **Run the Server:** From the project root, run the command: `uvicorn src.main:app --reload`.
5.  **Test the Endpoint:** Open a browser or Postman and access `http://127.0.0.1:8000/v1/news/`.