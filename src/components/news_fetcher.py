import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests


CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "news_cache")
)
CACHE_FILE = os.path.join(CACHE_DIR, "latest_hot_news.json")

RSS_FEEDS = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    ],
    "financial": [
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "https://finance.yahoo.com/news/rssindex",
        "https://www.ecb.europa.eu/rss/press.html",
    ],
    "geopolitical": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoPulseAI/1.0; +https://localhost)"
}


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_cache() -> Dict[str, Any]:
    ensure_cache_dir()
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(payload: Dict[str, Any]):
    ensure_cache_dir()
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_get(d: Dict[str, Any], key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def parse_date(date_str: str) -> str:
    if not date_str:
        return ""

    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue

    return ""


def hostname_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def extract_source(feed_url: str, item) -> str:
    source = ""

    source_tag = item.find("source")
    if source_tag is not None and source_tag.text:
        source = source_tag.text.strip()

    if not source:
        source = hostname_from_url(feed_url)

    return source or "Unknown"


def extract_link(item) -> str:
    link = item.findtext("link", default="").strip()
    if link:
        return link

    for elem in item.iter():
        href = elem.attrib.get("href")
        rel = elem.attrib.get("rel")
        if href and (rel == "alternate" or "atom" in elem.tag.lower()):
            return href.strip()

    return ""


def extract_description(item) -> str:
    candidates = [
        item.findtext("description", default=""),
        item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded", default=""),
        item.findtext("{http://www.w3.org/2005/Atom}summary", default=""),
        item.findtext("{http://www.w3.org/2005/Atom}content", default=""),
    ]

    for text in candidates:
        cleaned = clean_html(text)
        if cleaned:
            return cleaned

    return ""


def extract_title(item) -> str:
    title = item.findtext("title", default="").strip()
    return clean_html(title)


def extract_published(item) -> str:
    candidates = [
        item.findtext("pubDate", default=""),
        item.findtext("published", default=""),
        item.findtext("updated", default=""),
        item.findtext("{http://www.w3.org/2005/Atom}published", default=""),
        item.findtext("{http://www.w3.org/2005/Atom}updated", default=""),
    ]

    for raw in candidates:
        parsed = parse_date(raw.strip())
        if parsed:
            return parsed

    return ""


def fetch_rss_feed(feed_url: str, timeout: int = 15) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    items = []

    channel_items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for item in channel_items + atom_entries:
        title = extract_title(item)
        link = extract_link(item)
        description = extract_description(item)
        published_at = extract_published(item)
        source = extract_source(feed_url, item)

        if not title or not link:
            continue

        items.append(
            {
                "title": truncate(title, 120),
                "description": truncate(description, 180),
                "url": link,
                "publishedAt": published_at,
                "source": source,
                "image": "",
            }
        )

    return items


def dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []

    for art in articles:
        key = (
            safe_get(art, "url", "").strip().lower(),
            safe_get(art, "title", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(art)

    return deduped


def sort_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(x):
        published = x.get("publishedAt", "")
        return published if published else ""

    return sorted(articles, key=key_fn, reverse=True)


def fetch_category(category: str, limit: int = 8) -> List[Dict[str, Any]]:
    feeds = RSS_FEEDS.get(category, [])
    all_articles = []

    for feed_url in feeds:
        feed_articles = fetch_rss_feed(feed_url)
        all_articles.extend(feed_articles)
        time.sleep(0.2)

    all_articles = dedupe_articles(all_articles)
    all_articles = sort_articles(all_articles)

    return all_articles[:limit]


def update_news_cache() -> Dict[str, Any]:
    cached = load_cache()

    try:
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "crypto": fetch_category("crypto", limit=8),
            "financial": fetch_category("financial", limit=8),
            "geopolitical": fetch_category("geopolitical", limit=8),
        }
        save_cache(payload)
        return {"ok": True, "data": payload}
    except Exception as e:
        return {
            "ok": False,
            "error": type(e).__name__,
            "details": str(e),
            "data": cached,
        }


def placeholder_article(message: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "No cached news yet",
            "description": message,
            "url": "#",
            "publishedAt": "",
            "source": "System",
            "image": "",
        }
    ]


def get_news(category: str) -> List[Dict[str, Any]]:
    cached = load_cache()

    if cached and category in cached and cached.get(category):
        return cached.get(category, [])

    return placeholder_article("Click 'Refresh News' to fetch the latest headlines.")