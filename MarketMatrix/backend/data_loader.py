import pandas as pd

from config import FEATURE_COLS, SP500_CSV


def load_sp500() -> pd.DataFrame:
    if not SP500_CSV.exists():
        raise FileNotFoundError(
            f"{SP500_CSV} not found. Run `python download_sp500.py` first."
        )

    df = pd.read_csv(SP500_CSV, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(subset=FEATURE_COLS + ["Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
