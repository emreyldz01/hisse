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
    "Seçtiğiniz hisse senedinin geçmiş verilerini kullanarak bir **LSTM (Uzun-Kısa Süreli Bellek)** modeli eğitir ve "
    "**AL/SAT indikatörleri** ile sinyal üretir."
)

# --- Sidebar Ayarları ---
st.sidebar.header("Ayarlar")
TICKER = st.sidebar.text_input("Hisse Senedi Sembolü (Örn: AAPL, THYAO)", "AAPL").upper()
LOOKBACK_DAYS = st.sidebar.slider("Girdi Olarak Kullanılacak Geçmiş Gün Sayısı (Zaman Adımı)", 30, 90, 60, 5)
EPOCHS = st.sidebar.slider("Eğitim Epoch Sayısı", 1, 10, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 AL/SAT İndikatörü")
SIGNAL_STRATEGY = st.sidebar.selectbox("Strateji Seç", ["SMA Kesişimi", "RSI", "MACD"])

SMA_FAST = st.sidebar.slider("SMA Kısa", 5, 50, 10)
SMA_SLOW = st.sidebar.slider("SMA Uzun", 20, 200, 50)

RSI_PERIOD = st.sidebar.slider("RSI Periyot", 7, 30, 14)
RSI_BUY = st.sidebar.slider("RSI AL Eşiği", 10, 40, 30)
RSI_SELL = st.sidebar.slider("RSI SAT Eşiği", 60, 90, 70)

SHOW_SIGNALS = st.sidebar.checkbox("Grafikte AL/SAT işaretlerini göster", True)


# --- Veri Çekme ---
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


# --- AL/SAT İndikatörleri ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_trade_signals(
    df: pd.DataFrame,
    strategy: str,
    sma_fast: int,
    sma_slow: int,
    rsi_period: int,
    rsi_buy: int,
    rsi_sell: int
) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["BuySignal"] = np.nan
    out["SellSignal"] = np.nan
    out["Signal"] = "BEKLE"

    if strategy == "SMA Kesişimi":
        out["SMA_FAST"] = close.rolling(sma_fast).mean()
        out["SMA_SLOW"] = close.rolling(sma_slow).mean()

        sig = (out["SMA_FAST"] > out["SMA_SLOW"]).astype(int)
        cross = sig.diff()

        buy_idx = cross[cross == 1].index
        sell_idx = cross[cross == -1].index

        out.loc[buy_idx, "BuySignal"] = out.loc[buy_idx, "Close"]
        out.loc[sell_idx, "SellSignal"] = out.loc[sell_idx, "Close"]

        if len(sig.dropna()) > 0:
            out.loc[out.index[-1], "Signal"] = "AL" if sig.iloc[-1] == 1 else "SAT"

    elif strategy == "RSI":
        out["RSI"] = compute_rsi(close, rsi_period)

        buy_idx = out.index[out["RSI"] < rsi_buy]
        sell_idx = out.index[out["RSI"] > rsi_sell]

        out.loc[buy_idx, "BuySignal"] = out.loc[buy_idx, "Close"]
        out.loc[sell_idx, "SellSignal"] = out.loc[sell_idx, "Close"]

        last_rsi = out["RSI"].iloc[-1]
        if pd.notna(last_rsi):
            if last_rsi < rsi_buy:
                out.loc[out.index[-1], "Signal"] = "AL"
            elif last_rsi > rsi_sell:
                out.loc[out.index[-1], "Signal"] = "SAT"
            else:
                out.loc[out.index[-1], "Signal"] = "BEKLE"

    elif strategy == "MACD":
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
        out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()

        sig = (out["MACD"] > out["MACD_SIGNAL"]).astype(int)
        cross = sig.diff()

        buy_idx = cross[cross == 1].index
        sell_idx = cross[cross == -1].index

        out.loc[buy_idx, "BuySignal"] = out.loc[buy_idx, "Close"]
        out.loc[sell_idx, "SellSignal"] = out.loc[sell_idx, "Close"]

        if len(sig.dropna()) > 0:
            out.loc[out.index[-1], "Signal"] = "AL" if sig.iloc[-1] == 1 else "SAT"

    return out


# --- LSTM Model ---
def train_and_predict(df: pd.DataFrame, lookback_days: int, epochs: int):
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
    Y_test = dataset[training_data_len:, :]

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

    if df is None:
        st.stop()

    st.subheader(f"📊 {TICKER} Hisse Senedi Verisi (Son 5 Gün)")
    st.dataframe(df.tail())

    # --- AL/SAT Sinyalleri ---
    df_sig = add_trade_signals(
        df,
        strategy=SIGNAL_STRATEGY,
        sma_fast=SMA_FAST,
        sma_slow=SMA_SLOW,
        rsi_period=RSI_PERIOD,
        rsi_buy=RSI_BUY,
        rsi_sell=RSI_SELL
    )

    st.subheader("🟢🔴 AL/SAT İndikatörü Sonucu")
    st.metric("Son Sinyal", df_sig["Signal"].iloc[-1])

    st.subheader("📉 İndikatör Grafiği")
    fig2 = plt.figure(figsize=(16, 6))
    plt.title(f"{TICKER} - {SIGNAL_STRATEGY}")
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat")
    plt.plot(df_sig["Close"], label="Close")

    if "SMA_FAST" in df_sig.columns:
        plt.plot(df_sig["SMA_FAST"], label=f"SMA {SMA_FAST}")
    if "SMA_SLOW" in df_sig.columns:
        plt.plot(df_sig["SMA_SLOW"], label=f"SMA {SMA_SLOW}")

    if SHOW_SIGNALS:
        buys = df_sig["BuySignal"].dropna()
        sells = df_sig["SellSignal"].dropna()
        plt.scatter(buys.index, buys.values, label="AL", marker="^")
        plt.scatter(sells.index, sells.values, label="SAT", marker="v")

    plt.legend()
    st.pyplot(fig2)

    st.subheader("📄 Sinyal Tablosu (Son 15 gün)")
    cols_to_show = [c for c in ["Close", "Signal", "SMA_FAST", "SMA_SLOW", "RSI", "MACD", "MACD_SIGNAL"] if c in df_sig.columns]
    st.dataframe(df_sig[cols_to_show].tail(15))

    st.markdown("---")

    # --- LSTM Eğitimi ve Tahmin ---
    predictions, rmse, training_data_len = train_and_predict(df, LOOKBACK_DAYS, EPOCHS)
    if predictions is None:
        st.stop()

    train = df[:training_data_len]
    valid = df[training_data_len:].copy()
    valid["Predictions"] = predictions

    st.subheader("📌 LSTM Model Performansı")
    st.metric("RMSE", f"{rmse:.2f}")

    st.subheader(f"📈 {TICKER} Fiyat Tahmini Grafiği (LSTM)")
    fig = plt.figure(figsize=(16, 8))
    plt.title(f"{TICKER} Kapanış Fiyatı Tahmini (LSTM)")
    plt.xlabel("Tarih", fontsize=14)
    plt.ylabel("Kapanış Fiyatı", fontsize=14)
    plt.plot(train["Close"], label="Eğitim Verisi")
    plt.plot(valid["Close"], label="Gerçek Değerler")
    plt.plot(valid["Predictions"], label="Tahminler")
    plt.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("📄 Gerçek vs. Tahmin Edilen Değerler (Son 10 Gün)")
    st.dataframe(valid.tail(10))

# --- Uygulama Talimatı ---
st.sidebar.markdown("---")
st.sidebar.markdown("**PROJE TAMAMLANDI!**")
st.sidebar.markdown("Bu kodu GitHub'a yükleyip Streamlit Cloud üzerinden yayınlayabilirsiniz.")
st.sidebar.markdown("Yayınlama adımları için bana **'yayınlama'** yazmanız yeterlidir.")
