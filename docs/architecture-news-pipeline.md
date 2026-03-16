# Architecture Design: News Heatmap Pipeline

This document outlines the technical design for the data pipeline that will power the News Heatmap feature (Epic 5).

## Goal of the Pipeline
To periodically fetch curated crypto news from the Cryptopanic API, store it in our own database, and expose it through our backend API for the frontend to consume.

## Core Architectural Components
Our pipeline will consist of three main services:

1.  **The Collector:** A scheduled Python script responsible for fetching data from the Cryptopanic API.
2.  **The Database:** A dedicated table in our project's database to store the news data permanently.
3.  **The API Server:** Our existing backend application (e.g., FastAPI) will have a new endpoint to serve this stored data to the frontend.

## Architectural Flow
The following diagram illustrates the flow of data through the system:

flowchart TD

    A[Scheduler<br/>(cron job)]
    B[Collector Script<br/>fetch_news.py]
    C[Cryptopanic API]
    D[(PostgreSQL<br/>news_articles)]
    E[Backend API<br/>GET /v1/news]
    F[Frontend Heatmap]

    A --> B
    B -- GET request --> C
    B --> D
    D --> E
    F -- HTTP Request --> E


---

## Detailed Component Breakdown

### 1. The Collector Script (`fetch_news.py`)
This is the heart of our data ingestion. It is a standalone Python script that will be run on a recurring schedule.

*   **Trigger:** It will be run every 15 minutes by a server scheduler like `cron`.
*   **Logic:**
    1.  **Load API Key:** Securely load the Cryptopanic API token from an environment variable.
    2.  **API Call:** Make a `GET` request to the Cryptopanic `/posts/` endpoint.
    3.  **Deduplication:** For each article received, check if its unique ID already exists in our `news_articles` database table to prevent duplicate entries.
    4.  **Transform & Load:** If the article is new, transform its structure to match our database schema and insert it as a new row.

### 2. The Database Schema (`news_articles` table)
This schema defines the structure of our internal data model for storing news articles in PostgreSQL.

```sql
CREATE TABLE news_articles (
    id VARCHAR(255) PRIMARY KEY,       -- The unique ID from the Cryptopanic API
    title TEXT NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    article_url TEXT,
    sentiment VARCHAR(50),             -- e.g., 'bullish', 'bearish', 'neutral'
    impact VARCHAR(50),                -- e.g., 'high', 'medium', 'low'
    source_name VARCHAR(255),
    mentioned_assets TEXT[]            -- A PostgreSQL array of asset symbols like {'BTC', 'ETH'}
);
3. The API Endpoint (GET /v1/news)
This is the endpoint our frontend will use to retrieve data for the heatmap.

Framework: To be added to our existing FastAPI application.
Endpoint: GET /v1/news
Query Parameters:
start_date: ISO 8601 timestamp (e.g., 2024-03-10T00:00:00Z)
end_date: ISO 8601 timestamp (e.g., 2024-03-16T23:59:59Z)
asset: Asset symbol (e.g., BTC)
Response: A JSON array of news articles that match the filter criteria.
The Future-Proofing Strategy
This design is robust because its components are decoupled.

The API Server and the Frontend only ever communicate with our database. They have no knowledge of the Cryptopanic API. The Collector Script is the only component that interacts with the external API.

This decoupling means that for a future version, we can completely rewrite the fetch_news.py script (e.g., to use a different data source and our own sentiment model) without needing to change any other part of the application, as long as the data continues to conform to the news_articles table schema.

