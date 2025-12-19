from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RateLimitConfig:
    """Simple per-minute rate-limit configuration."""

    per_minute: int = 60
    burst: Optional[int] = None


@dataclass
class CollectorConfig:
    """Configuration shared by all collectors."""

    name: str
    api_key: str
    base_url: str
    params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


@dataclass
class QueueConfig:
    """Queue backend selection and connection information."""

    backend: str = "local"  # local | redis | kafka
    redis_url: str = "redis://localhost:6379/0"
    redis_stream: str = "ingest"
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topic: str = "ingest"


@dataclass
class ScheduleConfig:
    """Cron or orchestrator-based schedule for each collector."""

    cron: Optional[str] = None
    prefect_deployment_name: Optional[str] = None
    airflow_dag_id: Optional[str] = None


@dataclass
class IngestJobConfig:
    """Full definition of a collector and its orchestration settings."""

    collector: Any
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    target_language: str = "en"
    normalize: bool = True
    deduplicate: bool = True
    spam_filter: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the ingest pipeline."""

    jobs: List[IngestJobConfig]
    queue: QueueConfig = field(default_factory=QueueConfig)
