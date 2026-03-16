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

## Recommendation

Based on this initial research, the **Cryptopanic API** is the recommended choice.

A review of the initial product prototype confirms that the required feature is not just a list of headlines, but a curated and analyzed stream of *crypto-relevant* news from a wide variety of sources (including major outlets like Reuters). A general-purpose news API would require us to build a complex filtering, entity-recognition, and sentiment-analysis engine from scratch.

An aggregator service like Cryptopanic has already performed this essential curation work. It represents the most direct and efficient path to implementing the news heatmap feature as designed, fulfilling our immediate needs for headlines while providing a clear path toward the sentiment analysis features planned for later stages of the project.