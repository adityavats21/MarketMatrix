import matplotlib.pyplot as plt
import seaborn as sns

from config import EDA_OUTPUT_DIR, ensure_dirs
from data_loader import load_sp500


def main():
    ensure_dirs()
    df = load_sp500()
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Close"], linewidth=2)
    plt.title("S&P 500 Closing Price History")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "01_sp500_closing_price.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["Daily_Return"], bins=80, kde=True, color="#2f80ed")
    plt.title("S&P 500 Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "02_sp500_daily_returns.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Close"], label="Close", linewidth=2)
    plt.plot(df["Date"], df["SMA_20"], label="SMA 20")
    plt.plot(df["Date"], df["SMA_50"], label="SMA 50")
    plt.plot(df["Date"], df["EMA_20"], label="EMA 20")
    plt.title("S&P 500 Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "03_sp500_moving_averages.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Close"], label="Close", linewidth=2)
    plt.plot(df["Date"], df["BB_upper"], label="Upper Band", alpha=0.8)
    plt.plot(df["Date"], df["BB_lower"], label="Lower Band", alpha=0.8)
    plt.fill_between(df["Date"], df["BB_lower"], df["BB_upper"], alpha=0.15)
    plt.title("S&P 500 Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "04_sp500_bollinger_bands.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df["Date"], df["RSI"], color="#9b51e0")
    plt.axhline(70, color="red", linestyle="--", linewidth=1)
    plt.axhline(30, color="green", linestyle="--", linewidth=1)
    plt.title("S&P 500 RSI")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "05_sp500_rsi.png", dpi=200)
    plt.close()

    corr_cols = ["Close", "Daily_Return", "RSI", "Volatility", "Momentum_5", "Momentum_10", "BB_width"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("S&P 500 Feature Correlation")
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR / "06_sp500_correlation_heatmap.png", dpi=200)
    plt.close()

    print(f"Saved S&P 500 EDA graphs to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
