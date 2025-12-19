# hisse
hesaplama ai

## Veri Toplama ve Ön İşleme Pipeline'ı

`pipelines/` altında haber, sosyal medya, zincir üstü ve fiyat verileri için ayrı kolektörler eklenmiştir. Her kolektör kendi API anahtarı ve hız limitiyle yapılandırılır; mesajlar Redis/Kafka ya da yerel kuyruk üzerinden ayrıştırılmış olarak ilerler.

### Çalıştırma

```bash
# ENV değişkenlerini doldurun
export NEWS_API_KEY=...
export SOCIAL_API_KEY=...
export ONCHAIN_API_KEY=...
export PRICES_API_KEY=...

# Kuyruk seçimi: local | redis | kafka
export QUEUE_BACKEND=redis
export REDIS_URL=redis://localhost:6379/0

# Prefect deployment olarak koşmak için
export PREFECT_DEPLOYMENT=true

python -m pipelines.run_ingest
```

- Cron: `pipelines/run_ingest.py` doğrudan çalıştırılarak cron job'ına eklenebilir. `ScheduleConfig.cron` alanları hazır değerler sağlar (ör. haber her 5 dakikada bir).
- Prefect/Airflow: Prefect kurulumu varsa `PREFECT_DEPLOYMENT=true` ile `CollectorRunner` bir Prefect flow olarak çalışır; Airflow DAG'lerinde aynı komut bir `PythonOperator` içinde çağrılabilir.

### Dil tespiti, çeviri ve normalizasyon

`pipelines/processing/text_processing.py` dil tespiti (`langdetect`), Marian MT tabanlı açık kaynak çeviri, metin normalizasyonu, deduplikasyon ve basit spam filtresi uygulayarak kuyrukta yalnızca temiz içerik kalmasını sağlar.
