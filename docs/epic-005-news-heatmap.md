# Epic 5: News Heatmap Feature

**Status:** In Planning
**Labels:** epic, feature, backend

## Goal
To provide users with contextual news information to understand market movements by integrating a news API and displaying the data as a heatmap.

---

## User Story 5.1: Collect News Headlines

**As a user, I want crypto news headlines collected from a variety of sources so that I can see the events that may be affecting market prices.**

### Tasks:
- Research and select a suitable news aggregator API.
- Create a data pipeline to periodically fetch and store news articles.
- Filter for crypto-relevant headlines and extract key metadata (sentiment, impact, etc.).
- Store the processed headlines in our database.

### Acceptance Criteria:
- News headlines are successfully stored in a dedicated database table.
- The stored data includes, at a minimum: a unique ID, title, publication timestamp, source, and sentiment.

---

## User Story 5.2: Display News Heatmap

**As a user, I want to see a news sentiment heatmap so that I can quickly visualize the intensity and nature of news flow over time.**

### Tasks:
- Create a new API endpoint (`/v1/news`) to serve the stored news data.
- The endpoint must support filtering by date range and asset.
- The frontend will consume this endpoint to generate a heatmap visualization.

### Acceptance Criteria:
- The `/v1/news` endpoint is live and returns filtered data correctly.
- The heatmap on the frontend accurately visualizes the sentiment and volume of news over the selecte