from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, List

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class NewsCollector(BaseCollector):
    """Ingest headlines from a configurable news API."""

    def __init__(self, api_key: str, base_url: str, rate_limit_per_minute: int = 60, params: dict | None = None):
        default_params = {
            "language": "tr,en",
            "pageSize": 20,
            "from": (datetime.utcnow() - timedelta(hours=6)).isoformat(timespec="seconds") + "Z",
        }
        merged_params = {**default_params, **(params or {})}
        super().__init__("news", api_key, base_url, rate_limit_per_minute, params=merged_params)

    def _fetch(self) -> Iterable[Document]:
        response = self.session.get(self.base_url, params=self.params, headers=self._headers(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        articles: List[dict] = payload.get("articles", [])
        for article in articles:
            yield Document(
                text=article.get("title", "") + " " + article.get("description", ""),
                source="news",
                language=article.get("language"),
                metadata={
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                    "tickers": article.get("tickers", []),
                },
            )
