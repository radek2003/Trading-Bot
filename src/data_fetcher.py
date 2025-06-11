import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import logging
from ta.volatility import AverageTrueRange
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import sys

NEWS_API_KEY = "07f95978137e4cd0ba2236bff4e304ad"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Cache for news and sentiment scores
news_cache = {}
last_news_fetch = {}
sentiment_cache = {}

# Sentiment mapping for user input
SENTIMENT_MAPPING = {
    "weak": -0.5,  # Negative sentiment
    "average": 0.0,  # Neutral sentiment
    "high": 0.5  # Positive sentiment
}


def get_user_sentiment(symbol):
    """Prompt the user to select a sentiment value for the symbol."""
    if not sys.stdin.isatty():  # Check if running in a non-interactive environment
        logging.warning(
            f"Non-interactive environment detected. Cannot prompt for sentiment for {symbol}. Defaulting to 0.0")
        return 0.0

    prompt = (
        f"Unable to fetch news for {symbol} and no previous sentiment available.\n"
        "Please select a sentiment value (weak, average, high): "
    )
    while True:
        try:
            user_input = input(prompt).strip().lower()
            if user_input in SENTIMENT_MAPPING:
                sentiment_value = SENTIMENT_MAPPING[user_input]
                logging.info(f"User selected sentiment '{user_input}' ({sentiment_value}) for {symbol}")
                return sentiment_value
            else:
                print(f"Invalid input. Please choose one of: weak, average, high")
        except KeyboardInterrupt:
            logging.warning(f"User interrupted input for {symbol}. Defaulting to 0.0")
            return 0.0
        except Exception as e:
            logging.error(f"Error getting user input for {symbol}: {str(e)}. Defaulting to 0.0")
            return 0.0


def fetch_financial_news(symbol="EURUSD", max_articles=3):
    """Fetch financial news with caching and rate limit handling."""
    global news_cache, last_news_fetch
    current_time = time.time()

    # Check cache
    if symbol in news_cache and symbol in last_news_fetch:
        if current_time - last_news_fetch[symbol] < 3600:  # Cache valid for 1 hour
            logging.debug(f"Using cached news for {symbol}")
            return news_cache[symbol]

    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&language=en&sortBy=publishedAt&pageSize={max_articles}&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        news_cache[symbol] = articles
        last_news_fetch[symbol] = current_time
        logging.info(f"Fetched news for {symbol}")
        return articles
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.warning(f"Rate limit hit for {symbol}. Skipping retries and using sentiment fallback.")
            news_cache[symbol] = []
            last_news_fetch[symbol] = current_time
            return []
        else:
            logging.error(f"Error fetching news for {symbol}: {str(e)}")
            news_cache[symbol] = []
            last_news_fetch[symbol] = current_time
            return []
    except Exception as e:
        logging.error(f"Error fetching news for {symbol}: {str(e)}")
        news_cache[symbol] = []
        last_news_fetch[symbol] = current_time
        return []


def analyze_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.detach().numpy()[0]
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {str(e)}")
        return [0.33, 0.33, 0.34]


def add_sentiment_features(df, symbol):
    global sentiment_cache
    try:
        articles = fetch_financial_news(symbol)
        if not articles:
            # If news fetching failed (e.g., rate limit), use the last sentiment score if available
            if symbol in sentiment_cache:
                logging.info(
                    f"Rate limit exceeded for {symbol}. Reusing last sentiment score: {sentiment_cache[symbol]}")
                df["sentiment"] = sentiment_cache[symbol]
            else:
                # Prompt user for sentiment input
                sentiment_value = get_user_sentiment(symbol)
                sentiment_cache[symbol] = sentiment_value  # Store user-selected sentiment
                df["sentiment"] = sentiment_value
            return df

        sentiment_scores = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            scores = analyze_sentiment(text)
            sentiment_score = scores[2] - scores[0]  # Positive - Negative
            sentiment_scores.append(sentiment_score)

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        # Store the new sentiment score in the cache
        sentiment_cache[symbol] = avg_sentiment
        logging.info(f"Computed new sentiment score for {symbol}: {avg_sentiment}")
        df["sentiment"] = avg_sentiment
        return df
    except Exception as e:
        logging.error(f"Error adding sentiment for {symbol}: {str(e)}")
        # On general failure, use the last sentiment score if available
        if symbol in sentiment_cache:
            logging.info(f"Error occurred for {symbol}. Reusing last sentiment score: {sentiment_cache[symbol]}")
            df["sentiment"] = sentiment_cache[symbol]
        else:
            # Prompt user for sentiment input
            sentiment_value = get_user_sentiment(symbol)
            sentiment_cache[symbol] = sentiment_value  # Store user-selected sentiment
            df["sentiment"] = sentiment_value
        return df


def test_trade_history(days_back=200):
    if not mt5.terminal_info():
        logging.error("MT5 is not initialized.")
        return pd.DataFrame()

    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()

    history = mt5.history_orders_get(from_date, to_date)
    if history is None or len(history) == 0:
        #logging.warning(f"No orders in history from {from_date} to {to_date}.")
        deals = pd.DataFrame([{
        'time': "No trades",
        'type': "No trades",
        'profit': "No trades",
        'volume': "No trades",
        }])

        return pd.DataFrame()#deals

    deals = pd.DataFrame([{
        'time': order.time_setup,
        'type': order.type,
        'profit': 0,
        'volume': order.volume_current,
    } for order in history])

    deals['time'] = pd.to_datetime(deals['time'], unit='s')
    return deals


def fetch_full_trade_history(days_back=200):
    if not mt5.initialize():
        raise RuntimeError("MT5 nie zostało zainicjalizowane.")

    from_date = datetime.now() - timedelta(days=days_back)
    to_date = datetime.now()

    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
        print("Brak transakcji w historii.")
        return pd.DataFrame()

    deals_df = pd.DataFrame([{
        'time': deal.time,
        'symbol': deal.symbol,
        'ticket': deal.ticket,
        'type': mt5.ORDER_TYPE_BUY if deal.type == 0 else "sell",
        'volume': deal.volume,
        'price': deal.price,
        'profit': deal.profit,
        'commission': deal.commission,
        'swap': deal.swap,
        'position_id': deal.position_id
    } for deal in deals])

    # Konwersja czasu
    deals_df['time'] = pd.to_datetime(deals_df['time'], unit='s')
    
    return deals_df

def fetch_historical_data(symbol, bars_m5=2000, bars_m15=2000):
    """    
    Adds :
    Sentiment - Sentiment score based on financial news articles
    ATR - ATR shows investors the average range prices swing for an investment over a specified period
    
    """
    try:
        rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_m5)
        if rates_m5 is None or len(rates_m5) == 0:
            logging.error(f"No historical data for 5m for {symbol}.")
            df_m5 = pd.DataFrame()
        else:
            df_m5 = pd.DataFrame(rates_m5)
            df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
            df_m5['close_smooth'] = df_m5['close'].rolling(window=3, min_periods=1).mean()
            atr_m5 = AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14)
            df_m5['atr'] = atr_m5.average_true_range()
            # fileting out the noise
            #price_changes_m5 = df_m5['close'].diff().abs()
            #threshold_m5 = 3 * df_m5['atr']
            #df_m5 = df_m5[price_changes_m5 <= threshold_m5]
            df_m5 = add_sentiment_features(df_m5, symbol)

        rates_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, bars_m15)
        if rates_m15 is None or len(rates_m15) == 0:
            logging.error(f"No historical data for 15m for {symbol}.")
            df_m15 = pd.DataFrame()
        else:
            df_m15 = pd.DataFrame(rates_m15)
            df_m15['time'] = pd.to_datetime(df_m15['time'], unit='s')
            df_m15['close_smooth'] = df_m15['close'].rolling(window=3, min_periods=1).mean()
            # fileting out the noise
            #atr_m15 = AverageTrueRange(high=df_m15['high'], low=df_m15['low'], close=df_m15['close'], window=14)
            #df_m15['atr'] = atr_m15.average_true_range()
            #price_changes_m15 = df_m15['close'].diff().abs()
            #threshold_m15 = 3 * df_m15['atr']
            #df_m15 = df_m15[price_changes_m15 <= threshold_m15]
            df_m15 = add_sentiment_features(df_m15, symbol)

        return df_m5, df_m15
    except Exception as e:
        logging.error(f"Problem fetching historical data for {symbol}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    pass
    #(test_trade_history(days_back=200))
    #mt5.initialize()
    #test_trade_history(days_back=200)
    # print(fetch_historical_data('EURUSD', bars_m5=250, bars_m15=250))