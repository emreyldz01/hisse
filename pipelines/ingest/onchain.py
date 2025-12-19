from __future__ import annotations

import logging
from typing import Iterable

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class OnChainCollector(BaseCollector):
    """Collect on-chain metrics such as transfers or gas usage."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit_per_minute: int = 120,
        burst: int | None = None,
        params: dict | None = None,
    ):
        default_params = {"window": "1h"}
        merged_params = {**default_params, **(params or {})}
        super().__init__(
            "onchain",
            api_key,
            base_url,
            rate_limit_per_minute=rate_limit_per_minute,
            burst=burst,
            params=merged_params,
        )
=======
    def __init__(self, api_key: str, base_url: str, rate_limit_per_minute: int = 120, params: dict | None = None):
        default_params = {"window": "1h"}
        merged_params = {**default_params, **(params or {})}
        super().__init__("onchain", api_key, base_url, rate_limit_per_minute, params=merged_params)

    def _fetch(self) -> Iterable[Document]:
        response = self.session.get(self.base_url, params=self.params, headers=self._headers(), timeout=20)
        response.raise_for_status()
        payload = response.json()
        for metric in payload.get("metrics", []):
            text = f"{metric.get('name', 'metric')} value={metric.get('value')} chain={metric.get('chain')}"
            yield Document(
                text=text,
                source="onchain",
                language="en",
                metadata={
                    "chain": metric.get("chain"),
                    "height": metric.get("height"),
                    "window": payload.get("window"),
                },
            )
