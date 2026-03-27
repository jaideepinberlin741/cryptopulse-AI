# Epic []: Historical Heatmap Timelines

**User Story : As a user, I want to see how the intensity of news topics has trended over time so that I can understand the narrative and momentum behind market-moving events.**

### V2 API Response with Time-Series Data

To support this, the `GET /v1/news/grouped` endpoint will be enhanced. The `heatmaps` object will be extended to include a `time_series` array for each topic, providing historical heat and sentiment scores.

```json
{
  "heatmaps": {
    "financial": [
      {
        "topic": "inflation",
        "current_heat_score": 0.9,
        "current_sentiment_score": -0.6,
        "time_series": [
          { "timestamp": "2026-03-18T09:00:00Z", "heat_score": 0.7, "sentiment_score": -0.5 },
          { "timestamp": "2026-03-18T12:00:00Z", "heat_score": 0.8, "sentiment_score": -0.5 },
          { "timestamp": "2026-03-18T15:00:00Z", "heat_score": 0.85, "sentiment_score": -0.6 },
          { "timestamp": "2026-03-18T18:00:00Z", "heat_score": 0.9, "sentiment_score": -0.6 }
        ]
      }
    ]
  },
  "financial": [ 
    // ... article list ...
  ]
}
```
