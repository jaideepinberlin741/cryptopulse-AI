# Initial Reconnaissance Report: Top Crypto News APIs

This document outlines the initial research into finding a suitable news API for providing contextual news information, as required by Epic 5.

## Goal

The primary goal is to find an API that can fulfill User Story 5.1 (collecting headlines with timestamps and asset mentions) while also keeping future needs, like sentiment analysis (User Story 5.2), in mind.

## Top Contenders

### 1. Cryptopanic API

*   **Description:** A service specifically designed to aggregate cryptocurrency news for traders and analysts.
*   **Pros:**
    *   Built for our exact use case.
    *   Appears to support sentiment analysis, aligning with future goals.
    *   Highly regarded in the crypto development community.
*   **Cons:**
    *   Requires signing up for a free authentication token.

### 2. CryptoControl Python Client

*   **Description:** A Python package that provides access to formatted news articles.
*   **Pros:**
    *   The promise of "formatted articles" suggests clean, easy-to-parse data.
    *   Directly focused on headlines.
*   **Cons:**
    *   Requires installing a specific third-party package.
    *   May be less feature-rich than dedicated API services.

### 3. Free Crypto News API

*   **Description:** A simple, open-source API that does not require an API key.
*   **Pros:**
    *   Zero barrier to entry; perfect for rapid prototyping.
*   **Cons:**
    *   Likely lacks advanced features like filtering, sentiment analysis, or deep archives.

## Recommendation (Revised)

Based on team alignment and a review of our capstone goals, the recommended choice is the **NewsAPI.org** API.

### Justification for Recommendation

NewsAPI is the strongest choice for this project because it directly supports the educational and technical goals of the epic:

*   **Build Custom Logic:** It requires us to build our own filtering logic, which is an excellent way to demonstrate foundational engineering skills.
*   **Enable Future Expansion:** It provides a raw data stream that is perfect for building our own sentiment analysis pipeline in a future version.
*   **Control the Data Model:** We define our own data model from the start, rather than relying on the structure of a pre-built crypto aggregator.
*   **Stability and Predictability:** It is a well-known, stable API, which will minimize time spent debugging API quirks and maximize time spent building features.

This approach aligns perfectly with our decision to prioritize building a "foundational data pipeline" and gives us full control over the data sources from the beginning. Using a general-purpose news API allows us to demonstrate this skill effectively and intentionally defers the complexity of crypto-specific filtering, which aligns with our simplified scope.
