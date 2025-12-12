import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- Streamlit Sayfa Ayarları ---
st.set_page_config(
    page_title="LSTM Hisse Senedi Tahmin Uygulaması",
    layout="wide"
)

st.title("📈 LSTM ile Hisse Senedi Fiyatı Tahmini")
st.markdown(
    "Seçtiğiniz hisse senedinin geçmiş verilerini kullanarak bir **LSTM (Uzun-Kısa Süreli Bellek)** modeli eğitir ve fiyat tahminini görselleştirir."
)

# --- Kullanıcıdan Giriş Alma ---
st.sidebar.header("Ayarlar")
TICKER = st.sidebar.text_input("Hisse Senedi Sembolü (Örn: AAPL, THYAO)", "AAPL").upper()
LOOKBACK_DAYS = st.sidebar.slider("Girdi Olarak Kullanılacak Geçmiş Gün Sayısı (Zaman Adımı)", 30, 90, 60, 5)
EPOCHS = st.sidebar.slider("Eğitim Epoch Sayısı", 1, 10, 3, 1)

@st.cache_data
def load_data(ticker: str) -> pd.DataFrame | None:
    """Belirtilen sembol için yfinance'tan veriyi çeker ve sağlam hata kontrolü yapar."""
    if not ticker:
        st.warning("Lütfen bir hisse senedi sembolü girin.")
        return None

    # BIST sembolü için otomatik .IS uzantısı ekleme kontrolü
    if 4 <= len(ticker) <= 5 and ticker.isalpha() and ticker.isupper() and "." not in ticker:
        ticker = f"{ticker}.IS"
        st.info(f"Sembol BIST olarak algılandı. '{ticker}' olarak güncellendi.")

    try:
        # Daha stabil: period ile indir (son 2 yıl)
        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        if data is None or data.empty:
            st.error(f"'{ticker}' için veri bulunamadı (data.empty). Sembolü kontrol edin.")
            return None

        # MultiIndex kolonlar gelebilir (bazı ticker/ortamlarda)
        if isinstance(data.columns, pd.MultiIndex):
            # örn: ('Close', 'AAPL') gibi
            if "Close" in data.columns.get_level_values(0):
                data = data["Close"]  # Series veya DF dönebilir
            else:
                st.error("Çekilen veride 'Close' alanı yok (MultiIndex).")
                st.write("Kolonlar:", data.columns)
                return None

        # Series geldiyse DataFrame'e çevir
        if isinstance(data, pd.Series):
            data = data.to_frame("Close")

        # Close sütunu yoksa çık
        if "Close" not in data.columns:
            st.error(f"'{ticker}' için çekilen veride 'Close' sütunu bulunamadı.")
            st.write("Gelen kolonlar:", list(data.columns))
            return None

        data = data[["Close"]].copy()

        # Close komple NaN mı?
        if data["Close"].isna().all():
            st.error(
                "Veri çekildi ancak 'Close' değerleri tamamen NaN geldi. "
                "Sembol/ticker hatalı olabilir veya veri sağlayıcı geçici problem yaşıyor."
            )
            st.write("Ham veri (son 10 satır):")
            st.dataframe(data.tail(10))
            return None

        # Aradaki NaN satırlarını temizle
        data = data.dropna(subset=["Close"])

        if data.empty:
            st.error("NaN temizliği sonrası veri kalmadı.")
            return None

        return data

    except Exception as e:
        st.error(f"Veri çekme sırasında hata: {e}")
        return None


def train_and_predict(df: pd.DataFrame, lookback_days: int, epochs: int):
    """LSTM modelini eğitir ve tahminleri döndürür."""
    dataset = df[["Close"]].values  # garanti 2D

    # Yeterli veri kontrolü
    if dataset.shape[0] < lookback_days + 1:
        st.error(
            f"Tahmin için yeterli veri yok. En az {lookback_days + 1} gün gerekir, "
            f"ama {dataset.shape[0]} gün bulundu."
        )
        return None, None, None

    # %80 eğitim, %20 test ayrımı
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    # Normalizasyon
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]

    # LSTM için X ve Y oluşturma
    X_train, Y_train = [], []
    for i in range(lookback_days, len(train_data)):
        X_train.append(train_data[i - lookback_days:i, 0])
        Y_train.append(train_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    # 3D reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Eğitim
    with st.spinner(f"Model {epochs} epoch boyunca eğitiliyor..."):
        model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=0)
    st.success("Eğitim tamamlandı!")

    # Test seti
    test_data = scaled_data[training_data_len - lookback_days:, :]
    X_test = []
    Y_test = dataset[training_data_len:, :]  # orijinal ölçekte

    for i in range(lookback_days, len(test_data)):
        X_test.append(test_data[i - lookback_days:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

    # RMSE
    rmse = np.sqrt(np.mean((predictions - Y_test) ** 2))

    return predictions, rmse, training_data_len


# --- Uygulama Akışı ---
if st.sidebar.button("Analizi Başlat"):
    df = load_data(TICKER)

    if df is not None:
        st.subheader(f"📊 {TICKER} Hisse Senedi Verisi (Son 5 Gün)")
        st.dataframe(df.tail())

        predictions, rmse, training_data_len = train_and_predict(df, LOOKBACK_DAYS, EPOCHS)

        if predictions is None:
            st.stop()

        train = df[:training_data_len]
        valid = df[training_data_len:].copy()
        valid["Predictions"] = predictions

        st.subheader("Model Performansı")
        st.metric("Kök Ortalama Kare Hatası (RMSE)", f"{rmse:.2f}")
        st.info("RMSE düşükse tahminler genelde daha yakındır (tek başına yeterli metrik değildir).")

        st.subheader(f"{TICKER} Fiyat Tahmini Grafiği")
        fig = plt.figure(figsize=(16, 8))
        plt.title(f"{TICKER} Kapanış Fiyatı Tahmini (LSTM)")
        plt.xlabel("Tarih", fontsize=14)
        plt.ylabel("Kapanış Fiyatı", fontsize=14)
        plt.plot(train["Close"], label="Eğitim Verisi")
        plt.plot(valid["Close"], label="Gerçek Değerler")
        plt.plot(valid["Predictions"], label="Tahminler")
        plt.legend(loc="lower right")
        st.pyplot(fig)

        st.subheader("Gerçek vs. Tahmin Edilen Değerler (Son 10 Gün)")
        st.dataframe(valid.tail(10))

# --- Uygulama Talimatı ---
st.sidebar.markdown("---")
st.sidebar.markdown("**PROJE TAMAMLANDI!**")
st.sidebar.markdown("Bu kodu GitHub'a yükleyip Streamlit Cloud üzerinden yayınlayabilirsiniz.")
st.sidebar.markdown("Yayınlama adımları için bana **'yayınlama'** yazmanız yeterlidir.")
