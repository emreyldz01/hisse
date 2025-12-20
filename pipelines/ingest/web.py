from __future__ import annotations

import logging
from collections import deque
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class WebCrawlerCollector(BaseCollector):
    """Crawl HTML pages without a dedicated API and extract text content."""

    def __init__(
        self,
        base_url: str,
        start_urls: list[str],
        rate_limit_per_minute: int = 30,
        burst: int | None = None,
        max_pages: int = 5,
        same_domain_only: bool = True,
    ):
        self.start_urls = start_urls
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        super().__init__(
            name="web-crawl",
            api_key="",
            base_url=base_url or (start_urls[0] if start_urls else ""),
            rate_limit_per_minute=rate_limit_per_minute,
            burst=burst,
            params={},
        )

    def _fetch(self) -> Iterable[Document]:
        if not self.start_urls:
            logger.warning("No start URLs provided for web crawler")
            return []

        visited: set[str] = set()
        queue: deque[str] = deque(self.start_urls[: self.max_pages])
        pages_fetched = 0
        allowed_netloc = urlparse(self.base_url).netloc if self.same_domain_only and self.base_url else None

        while queue and pages_fetched < self.max_pages:
            url = queue.popleft()
            if url in visited:
                continue

            if allowed_netloc and urlparse(url).netloc != allowed_netloc:
                logger.debug("Skipping %s because it is outside the allowed domain", url)
                continue

            visited.add(url)
            self.rate_limiter.wait()
            logger.info("Crawling %s", url)

            try:
                response = self.session.get(url, headers=self._headers(), timeout=15)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error("Failed to fetch %s: %s", url, exc)
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            paragraphs = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
            body_text = f"{title} {paragraphs}".strip()

            if body_text:
                pages_fetched += 1
                yield Document(
                    text=body_text,
                    source="web",
                    language=None,
                    metadata={"url": url, "title": title},
                )

            if pages_fetched < self.max_pages:
                for link in soup.find_all("a", href=True):
                    absolute = urljoin(url, link["href"])
                    if absolute not in visited:
                        queue.append(absolute)
