# hisse
hesaplama ai

## Veri Toplama ve Ön İşleme Pipeline'ı

`pipelines/` altında haber, sosyal medya, zincir üstü ve fiyat verileri için ayrı kolektörler eklenmiştir. Her kolektör kendi API anahtarı ve hız limitiyle yapılandırılır; mesajlar Redis/Kafka ya da yerel kuyruk üzerinden ayrıştırılmış olarak ilerler.

### Çalıştırma

```bash
# ENV değişkenlerini doldurun
export NEWS_API_KEY=...
export NEWS_REGIONS=tr,us,global-tech  # varsayılan bölgeler; virgülle yeni preset ekleyin
export SOCIAL_API_KEY=...
export ONCHAIN_API_KEY=...
export PRICES_API_KEY=...
export WEB_CRAWL_URLS=https://example.com,https://example.com/blog  # API olmadan HTML tarama için başlangıç URL'leri
export WEB_CRAWL_MAX_PAGES=5
export WEB_CRAWL_SAME_DOMAIN_ONLY=true

# Kuyruk seçimi: local | redis | kafka
export QUEUE_BACKEND=redis
export REDIS_URL=redis://localhost:6379/0

# Prefect deployment olarak koşmak için
export PREFECT_DEPLOYMENT=true

python -m pipelines.run_ingest
```

- Cron: `pipelines/run_ingest.py` doğrudan çalıştırılarak cron job'ına eklenebilir. `ScheduleConfig.cron` alanları hazır değerler sağlar (ör. haber her 5 dakikada bir).
- Prefect/Airflow: Prefect kurulumu varsa `PREFECT_DEPLOYMENT=true` ile `CollectorRunner` bir Prefect flow olarak çalışır; Airflow DAG'lerinde aynı komut bir `PythonOperator` içinde çağrılabilir.
- Haber kaynakları: Varsayılan olarak NewsAPI `top-headlines` endpoint'i kullanılır ve `NEWS_REGIONS` ile Türkiye (`tr`), ABD (`us`) ve küresel teknoloji (`global-tech`) preset'leri beslenir. `NEWS_API_URL` değiştirerek farklı bir haber servisinin URL'ini girebilir, `NEWS_REGIONS` içine virgülle yeni preset anahtarları ekleyip kodda tanımlayabilirsiniz.
- Web tarayıcı: API vermek istemiyorsanız `WEB_CRAWL_URLS` ile başlangıç URL'leri sağlayarak basit bir HTML tarayıcısı çalıştırabilirsiniz. Varsayılan olarak aynı domain ile sınırlıdır (`WEB_CRAWL_SAME_DOMAIN_ONLY=true`) ve en fazla `WEB_CRAWL_MAX_PAGES` kadar sayfa çeker.

### Dil tespiti, çeviri ve normalizasyon

`pipelines/processing/text_processing.py` dil tespiti (`langdetect`), Marian MT tabanlı açık kaynak çeviri, metin normalizasyonu, deduplikasyon ve basit spam filtresi uygulayarak kuyrukta yalnızca temiz içerik kalmasını sağlar.

## Kurulum Notları

- **Tam Streamlit uygulaması** için Python 3.11 kullanın. TensorFlow 3.12 üzerinde wheel sağlamadığı için `pip install -r requirements.txt` 3.12+ ortamlarında TensorFlow'u otomatik atlar; bu durumda Streamlit uygulaması çalışmaz.
- **Sadece ingest pipeline** çalıştırmak istiyorsanız TensorFlow/Streamlit'e gerek yoktur. Hafif kurulum:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements-pipeline.txt
  python -m pipelines.run_ingest
  ```
