
import pandas as pd
import requests
import ta
from sklearn.preprocessing import MinMaxScaler

API_KEY = "D3H4W1PT57NKBUO0"
STOCK_SYMBOL = "TSLA"


class StockDataFetcher:
    """Handles fetching and preprocessing stock data."""

    @staticmethod
    def fetch_stock_data():
        """Fetches the latest stock data from Alpha Vantage."""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&outputsize=full&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" not in data:
            return None

        daily_data = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(daily_data, orient='index')
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    @staticmethod
    def preprocess_data(df):
        """Prepares stock data with indicators and scaling."""
        df['10_EMA'] = ta.trend.ema_indicator(df['close'], window=10)
        df['50_EMA'] = ta.trend.ema_indicator(df['close'], window=50)
        df['MACD'] = ta.trend.macd(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)

        df.dropna(inplace=True)

        # Feature selection
        features = ['10_EMA', '50_EMA', 'MACD', 'stoch']
        X = df[features]
        y = df['close'].shift(-1)

        df.dropna(inplace=True)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

        return X_scaled, y_scaled, scaler
