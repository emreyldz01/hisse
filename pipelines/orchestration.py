from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Iterable, List

from pipelines.config import IngestJobConfig, PipelineConfig, QueueConfig
from pipelines.processing.text_processing import TextProcessor
from pipelines.processing.schemas import ProcessedDocument

logger = logging.getLogger(__name__)


class QueueBackend:
    def enqueue(self, document: ProcessedDocument) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LocalQueueBackend(QueueBackend):
    """In-memory queue for development/testing."""

    def __init__(self):
        self.buffer: List[ProcessedDocument] = []

    def enqueue(self, document: ProcessedDocument) -> None:
        self.buffer.append(document)
        logger.info("Queued locally: %s", document.id)


class RedisQueueBackend(QueueBackend):
    """Redis stream backend."""

    def __init__(self, url: str, stream: str):
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("redis library is required for RedisQueueBackend") from exc

        self.client = redis.Redis.from_url(url)
        self.stream = stream

    def enqueue(self, document: ProcessedDocument) -> None:
        payload = json.loads(json.dumps(asdict(document)))
        self.client.xadd(self.stream, payload)
        logger.info("Queued to redis stream=%s id=%s", self.stream, document.id)


class KafkaQueueBackend(QueueBackend):
    """Kafka topic backend."""

    def __init__(self, servers: list[str], topic: str):
        try:
            from kafka import KafkaProducer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("kafka-python is required for KafkaQueueBackend") from exc

        self.producer = KafkaProducer(bootstrap_servers=servers, value_serializer=lambda v: json.dumps(v).encode("utf-8"))
        self.topic = topic

    def enqueue(self, document: ProcessedDocument) -> None:
        payload = json.loads(json.dumps(asdict(document)))
        self.producer.send(self.topic, payload)
        logger.info("Queued to kafka topic=%s id=%s", self.topic, document.id)


def create_queue_backend(config: QueueConfig) -> QueueBackend:
    backend = config.backend.lower()
    if backend == "redis":
        return RedisQueueBackend(config.redis_url, config.redis_stream)
    if backend == "kafka":
        return KafkaQueueBackend(config.kafka_bootstrap_servers, config.kafka_topic)
    return LocalQueueBackend()


class CollectorRunner:
    """Wrap collectors, feed them to processors, and push into queues."""

    def __init__(self, jobs: Iterable[IngestJobConfig], queue_backend: QueueBackend):
        self.jobs = list(jobs)
        self.queue = queue_backend

    def run_once(self) -> None:
        for job in self.jobs:
            collector = job.collector
            if not hasattr(collector, "collect"):
                raise TypeError(f"Collector for job {getattr(job.collector, 'name', 'unknown')} is not runnable")
            processor = TextProcessor(
                target_language=job.target_language,
                enable_deduplication=job.deduplicate,
                enable_spam_filter=job.spam_filter,
                enable_normalization=job.normalize,
            )

            raw_docs = collector.collect()
            processed_docs = processor.process_many(raw_docs)
            for doc in processed_docs:
                self.queue.enqueue(doc)


def create_prefect_flow(runner: CollectorRunner, flow_name: str = "hisse-ingest"):  # pragma: no cover - optional dependency
    try:
        from prefect import flow
    except ImportError as exc:
        raise RuntimeError("prefect must be installed to build flows") from exc

    @flow(name=flow_name)
    def ingest_flow():
        runner.run_once()

    return ingest_flow
