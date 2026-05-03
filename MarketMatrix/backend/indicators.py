import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Daily_Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))

    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    rolling_std = data["Close"].rolling(window=20).std()
    data["BB_upper"] = data["SMA_20"] + (2 * rolling_std)
    data["BB_lower"] = data["SMA_20"] - (2 * rolling_std)
    data["BB_width"] = data["BB_upper"] - data["BB_lower"]

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Volatility"] = data["Daily_Return"].rolling(window=20).std()
    data["Price_Range"] = data["High"] - data["Low"]
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Volume_Change"] = data["Volume"].pct_change()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data
