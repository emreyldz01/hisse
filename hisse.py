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
TICKER = st.sidebar.text_input("Hisse Senedi Sembolü (Örn: AAPL, MSFT, THYAO.IS)", 'AAPL').upper()
LOOKBACK_DAYS = st.sidebar.slider("Girdi Olarak Kullanılacak Geçmiş Gün Sayısı (Zaman Adımı)", 30, 90, 60, 5)
EPOCHS = st.sidebar.slider("Eğitim Epoch Sayısı", 1, 10, 3, 1)

# Veriyi önbelleğe alma (uygulama her yenilendiğinde tekrar indirmemek için)
@st.cache_data
def load_data(ticker):
    """Belirtilen sembol için yfinance'tan veriyi çeker ve hata kontrolü yapar."""
    if not ticker:
        st.warning("Lütfen bir hisse senedi sembolü girin.")
        return None
        
    try:
        data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
        
        if data.empty:
            st.error(f"'{ticker}' sembolü için **veritabanında hiç veri bulunamadı**.")
            return None
        
        # Sadece Kapanış sütununu alıyoruz
        data = data.filter(['Close'])
        
        if data['Close'].isnull().all():
            st.error(f"'{ticker}' sembolü için çekilen verilerde **Kapanış fiyatı eksik** veya tamamen boş.")
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Veri çekme hatası (Sembolü kontrol edin): {e}")
        return None

# Model eğitimi ve tahmin fonksiyonu
def train_and_predict(df, lookback_days, epochs):
    """LSTM modelini eğitir ve tahminleri döndürür."""
    dataset = df.values
    
    # Veri boyut kontrolü (önceki hatayı önler)
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1, 1)
        
    # Yeterli veri kontrolü
    if dataset.shape[0] < lookback_days + 1:
        st.error(f"Tahmin için yeterli veri yok. En az {lookback_days + 1} gün gereklidir, ancak {dataset.shape[0]} gün bulundu.")
        return None, None, None

    # %80 eğitim, %20 test
    training_data_len = int(np.ceil(len(dataset) * 0.8)) 

    # 1. Veri Ön İşleme (Normalizasyon)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:training_data_len, :]
    
    # Veri yapılandırma (LSTM için X ve Y oluşturma)
    X_train, Y_train = [], []
    for i in range(lookback_days, len(train_data)):
        X_train.append(train_data[i-lookback_days:i, 0])
        Y_train.append(train_data[i, 0])
    
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    
    # LSTM girdisi için 3D şekle dönüştürme: [örnek, zaman adımı, özellik]
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
        # verbose=0 ile eğitim çıktısını Streamlit'te göstermeyi kapatıyoruz
        model.fit(X_train, Y_train, batch_size=1, epochs=epochs, verbose=0)
    st.success("Eğitim Tamamlandı!")
    
    # 4. Tahmin
    # Test verisi için, son 'lookback_days' gününü de dahil etmeliyiz.
    test_data = scaled_data[training_data_len - lookback_days:, :]
    X_test = []
    # Y_test, gerçek (ölçeklendirilmemiş) değerlerdir
    Y_test = dataset[training_data_len:, :]
    
    for i in range(lookback_days, len(test_data)):
        X_test.append(test_data[i-lookback_days:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test, verbose=0)
    # Tahminleri orijinal ölçeğe geri çevirme
    predictions = scaler.inverse_transform(predictions) 

    # Performans Ölçümü (RMSE)
    rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
    
    return predictions, rmse, training_data_len

if st.sidebar.button("Analizi Başlat"):
    
    df = load_data(TICKER)
    
    if df is not None:
        
        # Veri setini gösterme
        st.subheader(f"📊 {TICKER} Hisse Senedi Verisi (Son 5 Gün)")
        st.dataframe(df.tail())
        
        # Model Eğitimi ve Tahmin
        predictions, rmse, training_data_len = train_and_predict(df, LOOKBACK_DAYS, EPOCHS)
        
        # Hata durumunda (None) çıkış yap
        if predictions is None:
            st.stop()
            
        # --- Sonuçların Görselleştirilmesi ---
        train = df.filter(['Close'])[:training_data_len]
        valid = df.filter(['Close'])[training_data_len:]
        
        # Tahminleri geçerli (test) veri setine ekleme
        valid = valid.assign(Predictions=predictions)

        st.subheader("Model Performansı")
        st.metric("Kök Ortalama Kare Hatası (RMSE)", f"{rmse:.2f} USD")
        st.info("RMSE, tahminlerinizin gerçek değerlere ortalama olarak ne kadar yakın olduğunu gösterir. Düşük RMSE, daha iyi performanstır.")
        
        st.subheader(f"{TICKER} Fiyat Tahmini Grafiği")
        
        # Matplotlib figürü oluşturma
        fig = plt.figure(figsize=(16, 8))
        plt.title(f'{TICKER} Kapanış Fiyatı Tahmini (LSTM)')
        plt.xlabel('Tarih', fontsize=18)
        plt.ylabel('Kapanış Fiyatı (USD)', fontsize=18)
        plt.plot(train['Close'], label='Eğitim Verisi', color='blue')
        plt.plot(valid['Close'], label='Gerçek Değerler', color='red')
        plt.plot(valid['Predictions'], label='Tahminler', color='green')
        plt.legend(loc='lower right')
        st.pyplot(fig) # Streamlit ile figürü gösterme
        
        st.subheader("Gerçek vs. Tahmin Edilen Değerler (Son 10 Gün)")
        st.dataframe(valid.tail(10))

# --- Uygulamayı Çalıştırma Talimatı ---
st.sidebar.markdown("---")
st.sidebar.markdown("**UYGULAMA TALİMATI**")
st.sidebar.code("streamlit run app.py")
st.sidebar.markdown("Başlatmadan önce tüm kütüphanelerin `requirements.txt` dosyasında olduğundan emin olun.")
