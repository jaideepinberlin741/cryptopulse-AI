import json
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET

import requests

CACHE_DIR = Path("data/news_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "latest_hot_news_rss.json"
CACHE_TTL_SECONDS = 1800  # 30 minutes

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

RSS_FEEDS = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    ],
    "financial": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI,%5EGSPC,%5EIXIC&region=US&lang=en-US",
        "https://www.ecb.europa.eu/rss/press.html",
    ],
    "geopolitical": [
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
}


def safe_get(article: Dict, key: str, default: str = "") -> str:
    value = article.get(key, default)
    if value is None:
        return default
    return str(value)



def load_cache() -> Dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}



def save_cache(payload: Dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_TTL_SECONDS



def format_pub_date(raw_date: str) -> str:
    if not raw_date:
        return ""
    try:
        dt = parsedate_to_datetime(raw_date)
        return dt.strftime("%b %d, %Y")
    except Exception:
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            return dt.strftime("%b %d, %Y")
        except Exception:
            return raw_date[:16]



def strip_html(text: str) -> str:
    if not text:
        return ""
    import re
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def parse_rss_feed(feed_url: str, limit: int = 10) -> List[Dict]:
    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    articles = []

    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title", default="No title")
        link = item.findtext("link", default="#")
        description = item.findtext("description", default="")
        pub_date = item.findtext("pubDate", default="")
        source = item.findtext("source", default="")

        if not source:
            channel_title = root.findtext(".//channel/title", default="Unknown")
            source = channel_title

        article = {
            "title": title.strip()[:140],
            "description": strip_html(description)[:220],
            "url": link.strip(),
            "publishedAt": format_pub_date(pub_date),
            "source": source.strip() if source else "Unknown",
            "image": "",
        }

        if article["title"] and article["url"]:
            articles.append(article)

    return articles



def dedupe_articles(articles: List[Dict], limit: int = 8) -> List[Dict]:
    seen = set()
    unique = []

    for article in articles:
        key = (article.get("title", "").strip().lower(), article.get("url", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
        if len(unique) >= limit:
            break

    return unique



def fetch_category(category: str, per_feed_limit: int = 6, final_limit: int = 8) -> List[Dict]:
    feeds = RSS_FEEDS.get(category, [])
    collected = []

    for feed_url in feeds:
        collected.extend(parse_rss_feed(feed_url, per_feed_limit))

    return dedupe_articles(collected, limit=final_limit)



def update_news_cache() -> Dict:
    payload = {
        "updated_at": datetime.utcnow().isoformat(),
        "crypto": fetch_category("crypto"),
        "financial": fetch_category("financial"),
        "geopolitical": fetch_category("geopolitical"),
    }
    save_cache(payload)
    return payload



def get_news(category: str) -> List[Dict]:
    cached = load_cache()

    if cache_is_fresh() and category in cached and cached.get(category):
        return cached.get(category, [])

    if category in cached and cached.get(category):
        return cached.get(category, [])

    return [{
        "title": "No cached news yet",
        "description": "Click 'Refresh News' to fetch the latest headlines.",
        "url": "#",
        "publishedAt": "",
        "source": "System",
        "image": "",
    }]