from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Iterable, List, Optional

import requests

from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, per_minute: int, burst: Optional[int] = None):
        self.per_minute = per_minute
        self.burst = burst or per_minute
        self.calls = deque()

    def wait(self) -> None:
        now = time.time()
        window_start = now - 60
        while self.calls and self.calls[0] < window_start:
            self.calls.popleft()

        if len(self.calls) >= self.burst:
            sleep_time = 60 - (now - self.calls[0])
            logger.debug("Throttling for %.2f seconds to respect rate limits", sleep_time)
            time.sleep(max(sleep_time, 0))

        self.calls.append(time.time())


class BaseCollector(ABC):
    """Base class shared by concrete collectors."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str,
        rate_limit_per_minute: int = 60,
        burst: Optional[int] = None,
        params: Optional[Dict] = None,
    ):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.params = params or {}
        self.rate_limiter = RateLimiter(per_minute=rate_limit_per_minute, burst=burst)
        self.session = requests.Session()

    def _is_placeholder_url(self) -> bool:
        return not self.base_url or "example.com" in self.base_url

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "hisse-ingest/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def collect(self) -> List[Document]:
        if self._is_placeholder_url():
            logger.warning("Skipping %s collector because base_url is not configured", self.name)
            return []

        self.rate_limiter.wait()
        logger.info("Collecting data from %s", self.name)
        try:
            return list(self._fetch())
        except requests.RequestException as exc:
            logger.error("Network error while collecting %s: %s", self.name, exc)
            return []
        except Exception:
            logger.exception("Unexpected error while collecting %s", self.name)
            return []

    @abstractmethod
    def _fetch(self) -> Iterable[Document]:
        ...
