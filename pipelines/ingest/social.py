from __future__ import annotations

import logging
from typing import Iterable

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class SocialCollector(BaseCollector):
    """Fetch short-form social content (tweets/posts) via an API."""

    def __init__(self, api_key: str, base_url: str, rate_limit_per_minute: int = 300, params: dict | None = None):
        default_params = {"limit": 50, "order": "desc"}
        merged_params = {**default_params, **(params or {})}
        super().__init__("social", api_key, base_url, rate_limit_per_minute, params=merged_params)

    def _fetch(self) -> Iterable[Document]:
        response = self.session.get(self.base_url, params=self.params, headers=self._headers(), timeout=15)
        response.raise_for_status()
        payload = response.json()
        for post in payload.get("data", []):
            yield Document(
                text=post.get("text", ""),
                source="social",
                language=post.get("lang"),
                metadata={
                    "user": post.get("user", {}),
                    "engagement": post.get("metrics", {}),
                    "created_at": post.get("created_at"),
                },
            )
