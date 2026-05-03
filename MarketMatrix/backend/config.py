from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
ARIMA_OUTPUT_DIR = OUTPUTS_DIR / "arima_sp500"
LSTM_OUTPUT_DIR = OUTPUTS_DIR / "lstm_sp500"
COMPARISON_OUTPUT_DIR = OUTPUTS_DIR / "comparison_sp500"
LR_OUTPUT_DIR = OUTPUTS_DIR / "lr_sp500"

SP500_CSV = DATA_DIR / "SP500_data.csv"

SP500_TICKER = "^GSPC"
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

SEQ_LEN = 60
HORIZON = 1
TEST_RATIO = 0.20
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42

FEATURE_COLS = [
    "Close",
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI",
    "Volatility",
    "BB_upper",
    "BB_lower",
    "BB_width",
    "Momentum_5",
    "Momentum_10",
    "Daily_Return",
    "Volume_Change",
    "Price_Range",
]


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        MODELS_DIR,
        EDA_OUTPUT_DIR,
        LR_OUTPUT_DIR,
        ARIMA_OUTPUT_DIR,
        LSTM_OUTPUT_DIR,
        COMPARISON_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
