from __future__ import annotations

import logging
from typing import Iterable

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class SocialCollector(BaseCollector):
    """Fetch short-form social content (tweets/posts) via an API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit_per_minute: int = 300,
        burst: int | None = None,
        params: dict | None = None,
    ):
        default_params = {"limit": 50, "order": "desc"}
        merged_params = {**default_params, **(params or {})}
        super().__init__(
            "social",
            api_key,
            base_url,
            rate_limit_per_minute=rate_limit_per_minute,
            burst=burst,
            params=merged_params,
        )

    def _fetch(self) -> Iterable[Document]:
        response = self.session.get(self.base_url, params=self.params, headers=self._headers(), timeout=15)
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").lower()
        body_preview = response.text[:200].strip()
        if "json" not in content_type and body_preview.lower().startswith("<!doctype html"):
            logger.warning(
                "Skipping social fetch because the endpoint returned HTML instead of JSON "
                "(content-type=%s). This usually means the URL requires authentication/consent. "
                "Response starts with: %s",
                content_type or "unknown",
                body_preview,
            )
            return []

        try:
            payload = response.json()
        except ValueError as exc:  # requests.exceptions.JSONDecodeError inherits ValueError
            logger.warning(
                "Skipping social fetch because response body is not valid JSON (%s). Body starts with: %s",
                exc,
                body_preview,
            )
            return []

        if isinstance(payload, dict):
            posts = payload.get("data", [])
            if isinstance(posts, dict):
                posts = [posts]
            if not isinstance(posts, list):
                logger.warning("Skipping social fetch because 'data' is not a list in response JSON")
                return []
        elif isinstance(payload, list):
            posts = payload
        else:
            logger.warning("Skipping social fetch because response JSON is neither an object nor a list: %s", type(payload).__name__)
            return []

        for post in posts:
            if not isinstance(post, dict):
                logger.debug("Skipping social post with unexpected type: %s", type(post).__name__)
                continue

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
