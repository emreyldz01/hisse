from __future__ import annotations

import os

from pipelines.config import CollectorConfig, IngestJobConfig, PipelineConfig, QueueConfig, RateLimitConfig, ScheduleConfig
from pipelines.ingest.news import NewsCollector
from pipelines.ingest.onchain import OnChainCollector
from pipelines.ingest.prices import PriceCollector
from pipelines.ingest.social import SocialCollector
from pipelines.orchestration import CollectorRunner, create_prefect_flow, create_queue_backend


NEWS_DEFAULT_PRESETS = {
    "tr": {
        "params": {"country": "tr", "category": "business", "language": "tr", "pageSize": 50},
        "cron": "*/5 * * * *",
        "description": "Türkiye iş/dünya gündemi (NewsAPI top-headlines)",
    },
    "us": {
        "params": {"country": "us", "category": "business", "language": "en", "pageSize": 50},
        "cron": "*/5 * * * *",
        "description": "ABD iş/dünya gündemi (NewsAPI top-headlines)",
    },
    "global-tech": {
        "params": {"category": "technology", "language": "en", "pageSize": 50},
        "cron": "*/10 * * * *",
        "description": "Global teknoloji başlıkları (NewsAPI top-headlines)",
    },
}


def build_news_jobs(news_api_key: str, base_url: str, rate_limit: RateLimitConfig) -> list[IngestJobConfig]:
    regions_env = os.getenv("NEWS_REGIONS", "tr,us,global-tech")
    selected_regions = [region.strip() for region in regions_env.split(",") if region.strip()]

    jobs: list[IngestJobConfig] = []
    for region in selected_regions:
        preset = NEWS_DEFAULT_PRESETS.get(region)
        if not preset:
            continue

        params = {**preset["params"], "apiKey": news_api_key}
        collector = CollectorConfig(
            name=f"news-{region}",
            api_key="",  # API anahtarı query param olarak gönderiliyor
            base_url=base_url,
            params=params,
            rate_limit=rate_limit,
        )
        jobs.append(
            IngestJobConfig(
                collector=collector,
                schedule=ScheduleConfig(cron=preset["cron"], prefect_deployment_name=f"news-{region}-ingest"),
            )
        )

    return jobs


def load_config_from_env() -> PipelineConfig:
    queue_config = QueueConfig(
        backend=os.getenv("QUEUE_BACKEND", "local"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_stream=os.getenv("REDIS_STREAM", "ingest"),
        kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092").split(","),
        kafka_topic=os.getenv("KAFKA_TOPIC", "ingest"),
    )

    news_api_key = os.getenv("NEWS_API_KEY", "")
    news_base_url = os.getenv("NEWS_API_URL", "https://newsapi.org/v2/top-headlines")
    news_rate_limit = RateLimitConfig(per_minute=int(os.getenv("NEWS_RATE_LIMIT", "60")))

    jobs: list[IngestJobConfig] = []

    if news_api_key:
        jobs.extend(build_news_jobs(news_api_key, news_base_url, news_rate_limit))
    else:
        jobs.append(
            IngestJobConfig(
                collector=CollectorConfig(
                    name="news",
                    api_key="",
                    base_url="https://news-api.example.com/v1/headlines",
                    params={"category": "business"},
                    rate_limit=news_rate_limit,
                ),
                schedule=ScheduleConfig(cron="*/5 * * * *", prefect_deployment_name="news-ingest"),
            )
        )

    jobs.extend(
        [
            IngestJobConfig(
                collector=CollectorConfig(
                    name="social",
                    api_key=os.getenv("SOCIAL_API_KEY", ""),
                    base_url=os.getenv("SOCIAL_API_URL", "https://social-api.example.com/v1/posts"),
                    rate_limit=RateLimitConfig(per_minute=int(os.getenv("SOCIAL_RATE_LIMIT", "300"))),
                ),
                schedule=ScheduleConfig(cron="*/2 * * * *", prefect_deployment_name="social-ingest"),
            ),
            IngestJobConfig(
                collector=CollectorConfig(
                    name="onchain",
                    api_key=os.getenv("ONCHAIN_API_KEY", ""),
                    base_url=os.getenv("ONCHAIN_API_URL", "https://onchain.example.com/v1/metrics"),
                    params={"chain": os.getenv("ONCHAIN_CHAIN", "eth")},
                    rate_limit=RateLimitConfig(per_minute=int(os.getenv("ONCHAIN_RATE_LIMIT", "120"))),
                ),
                schedule=ScheduleConfig(cron="*/10 * * * *", prefect_deployment_name="onchain-ingest"),
            ),
            IngestJobConfig(
                collector=CollectorConfig(
                    name="prices",
                    api_key=os.getenv("PRICES_API_KEY", ""),
                    base_url=os.getenv("PRICES_API_URL", "https://prices.example.com/v1/candles"),
                    params={"symbol": os.getenv("PRICES_SYMBOL", "BTCUSD")},
                    rate_limit=RateLimitConfig(per_minute=int(os.getenv("PRICES_RATE_LIMIT", "90"))),
                ),
                schedule=ScheduleConfig(cron="*/1 * * * *", prefect_deployment_name="prices-ingest"),
            ),
        ]
    )

    return PipelineConfig(jobs=jobs, queue=queue_config)


def build_collectors(job_config: IngestJobConfig):
    collector_conf = job_config.collector
    name = collector_conf.name
    rate_limit = collector_conf.rate_limit.per_minute
    kwargs = {
        "api_key": collector_conf.api_key,
        "base_url": collector_conf.base_url,
        "rate_limit_per_minute": rate_limit,
        "burst": collector_conf.rate_limit.burst,
        "params": collector_conf.params,
    }
    if name == "news" or name.startswith("news-"):
        return NewsCollector(**kwargs)
    if name == "social":
        return SocialCollector(**kwargs)
    if name == "onchain":
        return OnChainCollector(**kwargs)
    if name == "prices":
        return PriceCollector(**kwargs)
    raise ValueError(f"Unknown collector {name}")


def main() -> None:
    config = load_config_from_env()
    queue_backend = create_queue_backend(config.queue)

    jobs = []
    for job_conf in config.jobs:
        collector_instance = build_collectors(job_conf)
        jobs.append(
            IngestJobConfig(
                collector=collector_instance,  # type: ignore[arg-type]
                schedule=job_conf.schedule,
                target_language=job_conf.target_language,
                normalize=job_conf.normalize,
                deduplicate=job_conf.deduplicate,
                spam_filter=job_conf.spam_filter,
            )
        )

    runner = CollectorRunner(jobs, queue_backend)

    if os.getenv("PREFECT_DEPLOYMENT", "").lower() == "true":
        flow = create_prefect_flow(runner)
        flow()
    else:
        runner.run_once()


if __name__ == "__main__":
    main()
