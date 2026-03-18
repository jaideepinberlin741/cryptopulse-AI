# Architecture Design: News Pipeline (Capstone v1)

This document outlines the simplified technical design for the data pipeline that will power the news feature for the capstone project (Epic 5).

## Goal of the Pipeline

For the capstone, our goal is to periodically fetch relevant crypto news from a **general news API** and expose it through our backend API for the frontend to consume and display as a list of headlines.

## Core Architectural Components (Simplified for Capstone)

Our pipeline will consist of two main services for the initial version:

1.  **The Collector:** A scheduled Python script responsible for fetching data from the selected general news API (e.g., NewsAPI.org, GNews).
2.  **The API Server:** Our existing backend application (e.g., FastAPI) will have a new endpoint to trigger the collector and serve the fresh data to the frontend.

*Note: A database for persistent storage has been deferred to a future version.*

## Architectural Flow (Simplified for Capstone)

The following text-based diagram illustrates the simplified flow of data for the capstone:

```text
1. Frontend UI --(HTTP Request)--> Backend API (/v1/news)
2. Backend API --(Triggers)--> Collector Script (fetch_news.py)
3. Collector Script --(GET Request)--> General News API
4. General News API --(News Data)--> Collector Script
5. Collector Script --(Headlines)--> Backend API
6. Backend API --(JSON Response)--> Frontend UI

### 3. Detailed Component Breakdown

#### 1. The Collector Script (`fetch_news.py`)

This is the heart of our data ingestion. It is a standalone Python script that will be triggered by our backend.

- **Trigger:** It will be executed when a request is made to our backend's `/v1/news` endpoint.
- **Logic:**
    1.  **Load API Key:** Securely load the API token for the chosen news service from an environment variable.
    2.  **API Call:** Make a `GET` request to the news API's endpoint, using keywords like "crypto," "Bitcoin," and "Ethereum" to get relevant articles.
    3.  **Transform:** Process the JSON response to create a clean list of news headlines, URLs, and publication dates.
    4.  **Return Data:** Return the list of headlines to the backend API.

#### 2. The API Endpoint (GET /v1/news)

This is the endpoint our frontend will use to retrieve fresh news headlines.

- **Framework:** To be added to our existing FastAPI application.
- **Endpoint:** `GET /v1/news`
- **Logic:**
    1.  On receiving a request, the endpoint will call the `fetch_news.py` script.
    2.  It will receive the list of headlines from the script.
    3.  It will return this list as a JSON array to the frontend.
- **Query Parameters:**
    - `query`: A string for the search keywords (e.g., "BTC").
- **Response:** A JSON array of fresh news articles.

The Future-Proofing Strategy (Revised)
This simplified design allows us to deliver the core feature for the capstone while building a solid foundation.

The fetch_news.py script is a modular component. For a future version, we can easily enhance this pipeline by:

Adding a database to store the news permanently.
Modifying the script to load data into the database.
Implementing our own sentiment analysis model to enrich the data.
This approach allows us to iterate and add complexity in a controlled way after the capstone is complete.