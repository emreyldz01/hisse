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
st.markdown("Seçtiğiniz hisse senedinin geçmiş verilerini kullanarak bir LSTM modeli eğitir ve fiyat tahminini görselleştirir.")

# --- Kullanıcıdan Giriş Alma ---
st.sidebar.header("Ayarlar")
TICKER = st.sidebar.text_input("Hisse Senedi Sembolü (Örn: AAPL, MSFT, GOOGL)", 'AAPL')
LOOKBACK_DAYS = st.sidebar.slider("Girdi Olarak Kullanılacak Gün Sayısı", 30, 90, 60, 5)
EPOCHS = st.sidebar.slider("Eğitim Epoch Sayısı", 1, 10, 3, 1)

# Veriyi önbelleğe alma (uygulama her yenilendiğinde tekrar indirmemek için)
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
        if data.empty:
            st.error(f"'{ticker}' sembolü için veri bulunamadı veya geçersiz.")
            return None
        return data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return None

# Model eğitimi ve tahmin fonksiyonu
def train_and_predict(df, lookback_days, epochs):
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8)) # %80 eğitim, %20 test

    # 1. Veri Ön İşleme (Normalizasyon)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    
    # Veri yapılandırma
    X_train, Y_train = [], []
    for i in range(lookback_days, len(train_data)):
        X_train.append(train_data[i-lookback_days:i, 0])
        Y_train.append(train_data[i, 0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 2. LSTM Modeli Oluşturma
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 3. Model Eğitimi
    with st.spinner(f"Model {epochs} epoch boyunca eğitiliyor... Bu biraz zaman alabilir."):
        model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=0)
    st.success("Eğitim Tamamlandı!")
    
    # 4. Tahmin
    test_data = scaled_data[training_data_len - lookback_days:, :]
    X_test = []
    Y_test = dataset[training_data_len:, :]
    for i in range(lookback_days, len(test_data)):
        X_test.append(test_data[i-lookback_days:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions) # Ölçeği geri çevirme

    # Performans Ölçümü (RMSE)
    rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
    
    return predictions, rmse, training_data_len

if st.sidebar.button("Analizi Başlat"):
    
    df = load_data(TICKER)
    
    if df is not None:
        st.subheader(f"📊 {TICKER} Hisse Senedi Verisi ve Tahmini")
        
        # Veri Yükleme Durumunu Gösterme
        st.dataframe(df.tail())
        
        # Model Eğitimi ve Tahmin
        predictions, rmse, training_data_len = train_and_predict(df, LOOKBACK_DAYS, EPOCHS)
        
        # Sonuçların Görselleştirilmesi
        train = df.filter(['Close'])[:training_data_len]
        valid = df.filter(['Close'])[training_data_len:]
        valid = valid.assign(Predictions=predictions)

        st.subheader("Model Performansı")
        st.metric("Kök Ortalama Kare Hatası (RMSE)", f"{rmse:.2f} USD")
        st.info("RMSE, tahminlerinizin gerçek değerlere ortalama olarak ne kadar yakın olduğunu gösterir. Düşük RMSE, daha iyi performanstır.")
        
        st.subheader(f"{TICKER} Fiyat Tahmini Grafiği")
        fig = plt.figure(figsize=(16, 8))
        plt.title(f'{TICKER} Kapanış Fiyatı Tahmini (LSTM)')
        plt.xlabel('Tarih', fontsize=18)
        plt.ylabel('Kapanış Fiyatı (USD)', fontsize=18)
        plt.plot(train['Close'], label='Eğitim Verisi', color='blue')
        plt.plot(valid['Close'], label='Gerçek Değerler', color='red')
        plt.plot(valid['Predictions'], label='Tahminler', color='green')
        plt.legend(loc='lower right')
        st.pyplot(fig)
        
        st.subheader("Gerçek vs. Tahmin Edilen Değerler (Son 10 Gün)")
        st.dataframe(valid.tail(10))