import yfinance as yf
import pandas as pd

from config import END_DATE, SP500_CSV, SP500_TICKER, START_DATE, ensure_dirs
from indicators import add_technical_indicators


def main() -> None:
    ensure_dirs()
    raw = yf.download(SP500_TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)

    if raw.empty:
        raise RuntimeError("No S&P 500 data downloaded. Check internet connection or ticker.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    raw = raw[keep_cols]
    enriched = add_technical_indicators(raw)
    enriched.to_csv(SP500_CSV, index=False)

    print(f"Saved {len(enriched)} S&P 500 rows to {SP500_CSV}")
    print(f"Date range: {enriched['Date'].min().date()} to {enriched['Date'].max().date()}")


if __name__ == "__main__":
    main()
