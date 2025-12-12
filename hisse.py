import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --- Streamlit Sayfa Ayarları ---
st.set_page_config(page_title="LSTM Hisse Senedi Tahmin Uygulaması", layout="wide")

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
def load_data(ticker: str):
    """yfinance'tan veri çeker, BIST .IS ekler, MultiIndex/Close sorunlarını çözer."""
    if not ticker:
        st.warning("Lütfen bir hisse senedi sembolü girin.")
        return None

    # BIST için .IS otomatik ekle
    if 4 <= len(ticker) <= 5 and ticker.isalpha() and ticker.isupper() and "." not in ticker:
        ticker = f"{ticker}.IS"
        st.info(f"Sembol BIST olarak algılandı. '{ticker}' olarak güncellendi.")

    try:
        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            group_by="column",   # kritik: kolon bazlı düzenleme
            progress=False,
            threads=True
        )

        if data is None or data.empty:
            st.error(f"'{ticker}' için veri bulunamadı.")
            return None

        # MultiIndex kolonlar gelirse: hem level0 hem level1'de Close ara
        if isinstance(data.columns, pd.MultiIndex):

            # Örn: ('Close', 'THYAO.IS')
            if "Close" in data.columns.get_level_values(0):
                close_part = data["Close"]
                if isinstance(close_part, pd.Series):
                    data = close_part.to_frame("Close")
                else:
                    data = close_part.iloc[:, [0]]
                    data.columns = ["Close"]

            # Örn: ('THYAO.IS', 'Close')
            elif "Close" in data.columns.get_level_values(1):
                close_part = data.xs("Close", level=1, axis=1)
                if isinstance(close_part, pd.Series):
                    data = close_part.to_frame("Close")
                else:
                    data = close_part.iloc[:, [0]]
                    data.columns = ["Close"]

            else:
                st.error("MultiIndex geldi ama 'Close' bulunamadı.")
                st.write("Kolonlar:", data.columns)
                return None

        else:
            # Normal kolonlar
            if "Close" not in data.columns:
                st.error(f"'{ticker}' için çekilen veride 'Close' sütunu bulunamadı.")
                st.write("Gelen kolonlar:", list(data.columns))
                return None
            data = data[["Close"]].copy()

        # Close tamamen NaN mı?
        if data["Close"].isna().all():
            st.error(
                "Veri çekildi ancak 'Close' değerleri tamamen NaN geldi. "
                "Sembol/dönem hatalı olabilir veya yfinance veri sağlayıcısı sorun yaşıyor."
            )
            st.dataframe(data.tail(10))
            return None

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

    if dataset.shape[0] < lookback_days + 1:
        st.error(
            f"Tahmin için yeterli veri yok. En az {lookback_days + 1} gün gerekir, "
            f"ama {dataset.shape[0]} gün bulundu."
        )
        return None, None, None

    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:training_data_len, :]

    X_train, Y_train = [], []
    for i in range(lookback_days, len(train_data)):
        X_train.append(train_data[i - lookback_days:i, 0])
        Y_train.append(train_data[i, 0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    with st.spinner(f"Model {epochs} epoch boyunca eğitiliyor..."):
        model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=0)
    st.success("Eğitim tamamlandı!")

    test_data = scaled_data[training_data_len - lookback_days:, :]
    X_test = []
    Y_test = dataset[training_data_len:, :]  # orijinal ölçekte

    for i in range(lookback_days, len(test_data)):
        X_test.append(test_data[i - lookback_days:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)

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
