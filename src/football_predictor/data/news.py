from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


@dataclass(slots=True)
class NewsItem:
    title: str
    link: str
    published: str
    source: str


class GoogleNewsRssClient:
    """Fallback news collector for team-specific context.

    This is not a substitute for a licensed news provider. It exists so the
    pipeline has a lightweight way to attach recent article text or titles to a
    future embedding model.
    """

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 10) -> list[NewsItem]:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append(
                NewsItem(
                    title=(item.findtext("title") or "").strip(),
                    link=(item.findtext("link") or "").strip(),
                    published=(item.findtext("pubDate") or "").strip(),
                    source=(item.findtext("source") or "").strip(),
                )
            )
        return items

    def multi_search(self, queries: Iterable[str], limit_per_query: int = 5) -> list[NewsItem]:
        combined: list[NewsItem] = []
        for query in queries:
            combined.extend(self.search(query, limit=limit_per_query))
        return combined
