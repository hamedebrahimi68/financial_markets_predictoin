import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from datetime import datetime, timedelta

def get_bitcoin_data():
    btc = yf.Ticker("BTC-USD")
    # داده‌های ساعتی برای تحلیل کوتاه‌مدت
    hourly_data = btc.history(period="2y", interval="1h")
    # داده‌های روزانه برای تحلیل بلندمدت
    daily_data = btc.history(period="2y", interval="1d")
    return hourly_data['Close'], daily_data['Close']

def calculate_technical_indicators(data):
    # محاسبه شاخص‌های تکنیکال
    df = pd.DataFrame(data)
    df['SMA_7'] = df.rolling(window=7).mean()
    df['SMA_30'] = df.rolling(window=30).mean()
    
    # محاسبه RSI
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df.rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df.rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df.rolling(window=20).std()
    
    return df.fillna(method='bfill')

def prepare_data(hourly_data, daily_data, short_term_steps=12, long_term_steps=30):
    # آماده‌سازی داده‌های کوتاه‌مدت
    hourly_df = calculate_technical_indicators(hourly_data)
    daily_df = calculate_technical_indicators(daily_data)
    
    scaler_hourly = MinMaxScaler()
    scaler_daily = MinMaxScaler()
    
    scaled_hourly = scaler_hourly.fit_transform(hourly_df)
    scaled_daily = scaler_daily.fit_transform(daily_df)
    
    X_short, X_long, y = [], [], []
    
    # ایجاد سری‌های زمانی برای آموزش
    for i in range(len(scaled_hourly) - short_term_steps):
        if i >= len(scaled_daily) - long_term_steps:
            break
            
        X_short.append(scaled_hourly[i:i+short_term_steps])
        X_long.append(scaled_daily[i:i+long_term_steps])
        y.append(scaled_hourly[i+short_term_steps, 0])  # فقط قیمت close
    
    return np.array(X_short), np.array(X_long), np.array(y), scaler_hourly, scaler_daily

def create_hybrid_model(short_term_steps, long_term_steps, n_features):
    # مدل برای داده‌های کوتاه‌مدت
    input_short = Input(shape=(short_term_steps, n_features))
    lstm_short = LSTM(50, return_sequences=True)(input_short)
    lstm_short = Dropout(0.2)(lstm_short)
    lstm_short = LSTM(50)(lstm_short)
    lstm_short = Dropout(0.2)(lstm_short)
    
    # مدل برای داده‌های بلندمدت
    input_long = Input(shape=(long_term_steps, n_features))
    lstm_long = LSTM(50, return_sequences=True)(input_long)
    lstm_long = Dropout(0.2)(lstm_long)
    lstm_long = LSTM(50)(lstm_long)
    lstm_long = Dropout(0.2)(lstm_long)
    
    # ترکیب دو مدل
    merged = Concatenate()([lstm_short, lstm_long])
    dense = Dense(50, activation='relu')(merged)
    output = Dense(1)(dense)
    
    model = Model(inputs=[input_short, input_long], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def predict_price(model, hourly_data, daily_data, scaler_hourly, scaler_daily, 
                 short_term_steps, long_term_steps):
    # آماده‌سازی داده‌های اخیر برای پیش‌بینی
    recent_hourly = calculate_technical_indicators(hourly_data[-short_term_steps:])
    recent_daily = calculate_technical_indicators(daily_data[-long_term_steps:])
    
    recent_hourly_scaled = scaler_hourly.transform(recent_hourly)
    recent_daily_scaled = scaler_daily.transform(recent_daily)
    
    # شکل‌دهی داده‌ها برای پیش‌بینی
    X_short_pred = recent_hourly_scaled.reshape(1, short_term_steps, -1)
    X_long_pred = recent_daily_scaled.reshape(1, long_term_steps, -1)
    
    # پیش‌بینی
    predicted_scaled = model.predict([X_short_pred, X_long_pred])
    predicted_price = scaler_hourly.inverse_transform(
        np.hstack([predicted_scaled, np.zeros((predicted_scaled.shape[0], recent_hourly.shape[1]-1))])
    )[:, 0]
    
    return predicted_price[0]

def main():
    print("دریافت داده‌های بیت‌کوین...")
    hourly_data, daily_data = get_bitcoin_data()
    
    # پارامترها
    SHORT_TERM_STEPS = 12  # 12 ساعت
    LONG_TERM_STEPS = 30   # 30 روز
    
    # آماده‌سازی داده‌ها
    X_short, X_long, y, scaler_hourly, scaler_daily = prepare_data(
        hourly_data, daily_data, SHORT_TERM_STEPS, LONG_TERM_STEPS
    )
    
    # تقسیم داده‌ها به train و test
    train_size = int(len(X_short) * 0.8)
    X_short_train, X_short_test = X_short[:train_size], X_short[train_size:]
    X_long_train, X_long_test = X_long[:train_size], X_long[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # ساخت و آموزش مدل
    print("آموزش مدل هیبریدی LSTM...")
    model = create_hybrid_model(SHORT_TERM_STEPS, LONG_TERM_STEPS, X_short.shape[2])
    model.fit(
        [X_short_train, X_long_train], 
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # پیش‌بینی قیمت
    current_price = hourly_data[-1]
    predicted_price = predict_price(
        model, hourly_data, daily_data, 
        scaler_hourly, scaler_daily,
        SHORT_TERM_STEPS, LONG_TERM_STEPS
    )
    
    print(f"\nقیمت فعلی بیت‌کوین: ${current_price:.2f}")
    print(f"قیمت پیش‌بینی شده برای 2 ساعت آینده: ${predicted_price:.2f}")
    print(f"درصد تغییر پیش‌بینی شده: {((predicted_price - current_price) / current_price * 100):.2f}%")
    
    # نمایش شاخص‌های تکنیکال اخیر
    recent_indicators = calculate_technical_indicators(hourly_data[-24:])
    print("\nشاخص‌های تکنیکال 24 ساعت اخیر:")
    print(f"RSI: {recent_indicators['RSI'].iloc[-1]:.2f}")
    print(f"SMA 7: ${recent_indicators['SMA_7'].iloc[-1]:.2f}")
    print(f"SMA 30: ${recent_indicators['SMA_30'].iloc[-1]:.2f}")

if __name__ == "__main__":
    main()