import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from ta.volatility import AverageTrueRange
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

NEWS_API_KEY = "07f95978137e4cd0ba2236bff4e304ad"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def fetch_financial_news(symbol="EURUSD", max_articles=3):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json().get("articles", [])
    except Exception as e:
        logging.error(f"Błąd pobierania newsów: {str(e)}")
        return []

def analyze_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.detach().numpy()[0]
    except Exception as e:
        logging.error(f"Błąd analizy sentymentu: {str(e)}")
        return [0.33, 0.33, 0.34]

def add_sentiment_features(df, symbol):
    try:
        articles = fetch_financial_news(symbol)
        if not articles:
            df["sentiment"] = 0.0
            return df

        sentiment_scores = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            scores = analyze_sentiment(text)
            sentiment_score = scores[2] - scores[0]
            sentiment_scores.append(sentiment_score)

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        df["sentiment"] = avg_sentiment
        return df
    except Exception as e:
        logging.error(f"Błąd dodawania sentymentu: {str(e)}")
        df["sentiment"] = 0.0
        return df

def test_trade_history(days_back=200):
    if not mt5.terminal_info():
        logging.error("MT5 nie jest zainicjalizowane.")
        return pd.DataFrame()

    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()

    history = mt5.history_orders_get(from_date, to_date)
    if history is None or len(history) == 0:
        logging.warning(f"Brak zleceń w historii od {from_date} do {to_date}.")
        return pd.DataFrame()

    deals = pd.DataFrame([{
        'time': order.time_setup,
        'type': order.type,
        'profit': 0,
        'volume': order.volume_current,
    } for order in history])

    deals['time'] = pd.to_datetime(deals['time'], unit='s')
    return deals

def fetch_historical_data(symbol, bars_m5=90000, bars_m15=90000):
    try:
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_m5)
        if rates_m5 is None or len(rates_m5) == 0:
            logging.error("Brak danych historycznych dla 5m.")
            df_m5 = pd.DataFrame()
        else:
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
            df_m5['close_smooth'] = df_m5['close'].rolling(window=3, min_periods=1).mean()
            atr_m5 = AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14)
            df_m5['atr'] = atr_m5.average_true_range()
            price_changes_m5 = df_m5['close'].diff().abs()
            threshold_m5 = 3 * df_m5['atr']
            df_m5 = df_m5[price_changes_m5 <= threshold_m5]
            df_m5 = add_sentiment_features(df_m5, symbol)

        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_m15)
        if rates_m15 is None or len(rates_m15) == 0:
            logging.error("Brak danych historycznych dla 15m.")
            df_m15 = pd.DataFrame()
        else:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
            df_m15['close_smooth'] = df_m15['close'].rolling(window=3, min_periods=1).mean()
            atr_m15 = AverageTrueRange(high=df_m15['high'], low=df_m15['low'], close=df_m15['close'], window=14)
            df_m15['atr'] = atr_m15.average_true_range()
            price_changes_m15 = df_m15['close'].diff().abs()
            threshold_m15 = 3 * df_m15['atr']
            df_m15 = df_m15[price_changes_m15 <= threshold_m15]
            df_m15 = add_sentiment_features(df_m15, symbol)

        return df_m5, df_m15
    except Exception as e:
        logging.exception("Problem z pobieraniem danych historycznych.")
        return pd.DataFrame(), pd.DataFrame()