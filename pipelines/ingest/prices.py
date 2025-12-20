from __future__ import annotations

import logging
from typing import Iterable

from .base import BaseCollector
from pipelines.processing.schemas import Document

logger = logging.getLogger(__name__)


class PriceCollector(BaseCollector):
    """Collect price candles or ticks from a market data API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit_per_minute: int = 60,
        burst: int | None = None,
        params: dict | None = None,
    ):
        default_params = {"symbol": "BTCUSD", "interval": "1m", "limit": 50}
        merged_params = {**default_params, **(params or {})}
        super().__init__(
            "prices",
            api_key,
            base_url,
            rate_limit_per_minute=rate_limit_per_minute,
            burst=burst,
            params=merged_params,
        )

    def _fetch(self) -> Iterable[Document]:
        response = self.session.get(self.base_url, params=self.params, headers=self._headers(), timeout=10)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict):
            candles = payload.get("candles", [])
        elif isinstance(payload, list):
            candles = payload
        else:
            logger.warning("Unexpected payload type for prices collector: %s", type(payload).__name__)
            return

        for candle in candles:
            if not isinstance(candle, dict):
                logger.debug("Skipping candle with unexpected type: %s", type(candle).__name__)
                continue

            text = f"{candle.get('symbol')} close={candle.get('close')} volume={candle.get('volume')}"
            yield Document(
                text=text,
                source="prices",
                language="en",
                metadata={
                    "symbol": candle.get("symbol"),
                    "timestamp": candle.get("timestamp"),
                    "open": candle.get("open"),
                    "high": candle.get("high"),
                    "low": candle.get("low"),
                    "close": candle.get("close"),
                    "volume": candle.get("volume"),
                },
            )
