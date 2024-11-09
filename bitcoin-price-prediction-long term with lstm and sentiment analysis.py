import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from textblob import TextBlob
import tweepy
import newsapi
from datetime import datetime, timedelta
import time
import re

# تنظیمات API های مورد نیاز
TWITTER_API_KEY = "YOUR_TWITTER_API_KEY"
TWITTER_API_SECRET = "YOUR_TWITTER_API_SECRET"
TWITTER_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
TWITTER_ACCESS_TOKEN_SECRET = "YOUR_ACCESS_TOKEN_SECRET"
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"

class SentimentAnalyzer:
    def __init__(self):
        # تنظیم Twitter API
        auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
        
        # تنظیم News API
        self.newsapi = newsapi.NewsApiClient(api_key=NEWSAPI_KEY)
        
    def clean_text(self, text):
        # پاکسازی متن از لینک‌ها، ایموجی‌ها و کاراکترهای خاص
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_twitter_sentiment(self, query, count=100):
        sentiments = []
        try:
            tweets = self.twitter_api.search_tweets(q=query, lang="en", count=count)
            for tweet in tweets:
                clean_text = self.clean_text(tweet.text)
                sentiment = TextBlob(clean_text).sentiment.polarity
                sentiments.append(sentiment)
        except Exception as e:
            print(f"خطا در دریافت توییت‌ها: {str(e)}")
        return np.mean(sentiments) if sentiments else 0
    
    def get_news_sentiment(self, query, days_back=1):
        sentiments = []
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            news = self.newsapi.get_everything(q=query,
                                            from_param=from_date,
                                            language='en',
                                            sort_by='relevancy')
            
            for article in news['articles']:
                text = article['title'] + " " + (article['description'] or "")
                clean_text = self.clean_text(text)
                sentiment = TextBlob(clean_text).sentiment.polarity
                sentiments.append(sentiment)
        except Exception as e:
            print(f"خطا در دریافت اخبار: {str(e)}")
        return np.mean(sentiments) if sentiments else 0

def get_sentiment_data(time_points):
    analyzer = SentimentAnalyzer()
    sentiment_data = []
    
    for _ in range(len(time_points)):
        # تحلیل احساسات توییتر
        twitter_sentiment = analyzer.get_twitter_sentiment("bitcoin OR btc OR crypto")
        
        # تحلیل احساسات اخبار
        news_sentiment = analyzer.get_news_sentiment("bitcoin OR cryptocurrency")
        
        sentiment_data.append([twitter_sentiment, news_sentiment])
        
        # توقف کوتاه برای رعایت محدودیت‌های API
        time.sleep(1)
    
    return np.array(sentiment_data)

def create_hybrid_model(short_term_steps, long_term_steps, n_features, sentiment_features):
    # ورودی داده‌های کوتاه‌مدت
    input_short = Input(shape=(short_term_steps, n_features))
    lstm_short = LSTM(50, return_sequences=True)(input_short)
    lstm_short = Dropout(0.2)(lstm_short)
    lstm_short = LSTM(50)(lstm_short)
    lstm_short = Dropout(0.2)(lstm_short)
    
    # ورودی داده‌های بلندمدت
    input_long = Input(shape=(long_term_steps, n_features))
    lstm_long = LSTM(50, return_sequences=True)(input_long)
    lstm_long = Dropout(0.2)(lstm_long)
    lstm_long = LSTM(50)(lstm_long)
    lstm_long = Dropout(0.2)(lstm_long)
    
    # ورودی داده‌های احساسات
    input_sentiment = Input(shape=(sentiment_features,))
    sentiment_dense = Dense(20, activation='relu')(input_sentiment)
    
    # ترکیب همه ورودی‌ها
    merged = Concatenate()([lstm_short, lstm_long, sentiment_dense])
    dense = Dense(50, activation='relu')(merged)
    output = Dense(1)(dense)
    
    model = Model(inputs=[input_short, input_long, input_sentiment], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    return model

def prepare_data_with_sentiment(hourly_data, daily_data, sentiment_data, 
                              short_term_steps=12, long_term_steps=30):
    # آماده‌سازی داده‌های قیمت
    hourly_df = calculate_technical_indicators(hourly_data)
    daily_df = calculate_technical_indicators(daily_data)
    
    scaler_hourly = MinMaxScaler()
    scaler_daily = MinMaxScaler()
    scaler_sentiment = MinMaxScaler()
    
    scaled_hourly = scaler_hourly.fit_transform(hourly_df)
    scaled_daily = scaler_daily.fit_transform(daily_df)
    scaled_sentiment = scaler_sentiment.fit_transform(sentiment_data)
    
    X_short, X_long, X_sentiment, y = [], [], [], []
    
    for i in range(len(scaled_hourly) - short_term_steps):
        if i >= len(scaled_daily) - long_term_steps or i >= len(scaled_sentiment):
            break
            
        X_short.append(scaled_hourly[i:i+short_term_steps])
        X_long.append(scaled_daily[i:i+long_term_steps])
        X_sentiment.append(scaled_sentiment[i])
        y.append(scaled_hourly[i+short_term_steps, 0])
    
    return (np.array(X_short), np.array(X_long), np.array(X_sentiment), np.array(y),
            scaler_hourly, scaler_daily, scaler_sentiment)

def predict_with_sentiment(model, hourly_data, daily_data, sentiment_data,
                         scaler_hourly, scaler_daily, scaler_sentiment,
                         short_term_steps, long_term_steps):
    # آماده‌سازی داده‌های اخیر
    recent_hourly = calculate_technical_indicators(hourly_data[-short_term_steps:])
    recent_daily = calculate_technical_indicators(daily_data[-long_term_steps:])
    recent_sentiment = sentiment_data[-1:]
    
    recent_hourly_scaled = scaler_hourly.transform(recent_hourly)
    recent_daily_scaled = scaler_daily.transform(recent_daily)
    recent_sentiment_scaled = scaler_sentiment.transform(recent_sentiment)
    
    X_short_pred = recent_hourly_scaled.reshape(1, short_term_steps, -1)
    X_long_pred = recent_daily_scaled.reshape(1, long_term_steps, -1)
    X_sentiment_pred = recent_sentiment_scaled
    
    predicted_scaled = model.predict([X_short_pred, X_long_pred, X_sentiment_pred])
    predicted_price = scaler_hourly.inverse_transform(
        np.hstack([predicted_scaled, np.zeros((predicted_scaled.shape[0], recent_hourly.shape[1]-1))])
    )[:, 0]
    
    return predicted_price[0]

def main():
    print("دریافت داده‌های بیت‌کوین و تحلیل احساسات...")
    hourly_data, daily_data = get_bitcoin_data()
    
    # دریافت داده‌های احساسات
    sentiment_data = get_sentiment_data(hourly_data.index)
    
    SHORT_TERM_STEPS = 12
    LONG_TERM_STEPS = 30
    
    # آماده‌سازی داده‌ها با احساسات
    X_short, X_long, X_sentiment, y, scaler_hourly, scaler_daily, scaler_sentiment = \
        prepare_data_with_sentiment(hourly_data, daily_data, sentiment_data,
                                  SHORT_TERM_STEPS, LONG_TERM_STEPS)
    
    # تقسیم داده‌ها
    train_size = int(len(X_short) * 0.8)
    X_short_train = X_short[:train_size]
    X_long_train = X_long[:train_size]
    X_sentiment_train = X_sentiment[:train_size]
    y_train = y[:train_size]
    
    # ساخت و آموزش مدل
    print("آموزش مدل هیبریدی با تحلیل احساسات...")
    model = create_hybrid_model(SHORT_TERM_STEPS, LONG_TERM_STEPS,
                              X_short.shape[2], X_sentiment.shape[1])
    
    model.fit([X_short_train, X_long_train, X_sentiment_train],
              y_train,
              epochs=50,
              batch_size=32,
              validation_split=0.1,
              verbose=1)
    
    # پیش‌بینی با در نظر گرفتن احساسات
    current_price = hourly_data[-1]
    predicted_price = predict_with_sentiment(
        model, hourly_data, daily_data, sentiment_data,
        scaler_hourly, scaler_daily, scaler_sentiment,
        SHORT_TERM_STEPS, LONG_TERM_STEPS
    )
    
    # نمایش نتایج
    print(f"\nقیمت فعلی بیت‌کوین: ${current_price:.2f}")
    print(f"قیمت پیش‌بینی شده برای 2 ساعت آینده: ${predicted_price:.2f}")
    print(f"درصد تغییر پیش‌بینی شده: {((predicted_price - current_price) / current_price * 100):.2f}%")
    
    # نمایش تحلیل احساسات اخیر
    latest_sentiment = sentiment_data[-1]
    print("\nتحلیل احساسات اخیر:")
    print(f"احساسات توییتر: {latest_sentiment[0]:.2f} (-1 تا 1)")
    print(f"احساسات اخبار: {latest_sentiment[1]:.2f} (-1 تا 1)")

if __name__ == "__main__":
    main()