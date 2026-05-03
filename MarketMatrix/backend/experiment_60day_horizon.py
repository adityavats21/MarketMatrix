import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import FEATURE_COLS, MODELS_DIR, OUTPUTS_DIR, SEQ_LEN, TEST_RATIO, ensure_dirs
from data_loader import load_sp500
from metrics import regression_metrics, save_metrics


HORIZON = 60
EPOCHS = 50
BATCH_SIZE = 32
ORDER = (5, 1, 0)
EXP_DIR = OUTPUTS_DIR / "experiment_60day"


def build_window_indices(n_rows, seq_len=SEQ_LEN, horizon=HORIZON):
    X_idx, target_idx = [], []
    for i in range(seq_len, n_rows - horizon + 1):
        X_idx.append((i - seq_len, i))
        target_idx.append(i + horizon - 1)
    return X_idx, np.array(target_idx)


def save_prediction_plot(model_name, dates, actual, predicted):
    model_dir = EXP_DIR / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label="Actual", linewidth=2)
    plt.plot(dates, predicted, label=model_name, linewidth=2)
    plt.title(f"S&P 500 {model_name} 60-Day Ahead Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "actual_vs_predicted.png", dpi=200)
    plt.close()

    residuals = actual - predicted
    plt.figure(figsize=(14, 5))
    plt.plot(dates, residuals, color="crimson")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{model_name} 60-Day Residuals")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(model_dir / "residuals.png", dpi=200)
    plt.close()


def train_lr(df, split):
    features = df[FEATURE_COLS].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    dates = pd.to_datetime(df["Date"]).to_numpy()

    scaler = StandardScaler()
    scaler.fit(features[:split])
    scaled = scaler.transform(features)

    X_idx, target_idx = build_window_indices(len(df))
    X = np.array([scaled[start:end].reshape(-1) for start, end in X_idx])
    base_idx = np.array([end - 1 for _, end in X_idx])
    y = (close[target_idx] / close[base_idx]) - 1

    train_mask = target_idx < split
    test_mask = target_idx >= split

    model = LinearRegression()
    model.fit(X[train_mask], y[train_mask])
    pred_return = model.predict(X[test_mask])

    actual = close[target_idx[test_mask]]
    predicted = close[base_idx[test_mask]] * (1 + pred_return)
    metrics = regression_metrics(actual, predicted)

    model_dir = EXP_DIR / "linear_regression"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODELS_DIR / "lr_sp500_horizon60.pkl")
    pd.DataFrame(
        {
            "Date": dates[target_idx[test_mask]],
            "Actual": actual,
            "Predicted": predicted,
            "Predicted_Return_60D": pred_return,
        }
    ).to_csv(model_dir / "predictions.csv", index=False)
    save_metrics(metrics, model_dir / "metrics.json")
    save_prediction_plot("Linear Regression", dates[target_idx[test_mask]], actual, predicted)
    return metrics, model_dir / "predictions.csv"


def train_arima(df, split):
    close = df["Close"].astype(float).to_numpy()
    dates = pd.to_datetime(df["Date"]).to_numpy()
    actual = close[split:]
    pred_dates = dates[split:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(close[:split], order=ORDER).fit()
    predicted = fitted.forecast(steps=len(actual))

    metrics = regression_metrics(actual, predicted)
    model_dir = EXP_DIR / "arima"
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Date": pred_dates, "Actual": actual, "Predicted": predicted}
    ).to_csv(model_dir / "predictions.csv", index=False)
    save_metrics(metrics, model_dir / "metrics.json")
    save_prediction_plot("ARIMA", pred_dates, actual, predicted)
    return metrics, model_dir / "predictions.csv"


def build_lstm_model(n_features):
    model = Sequential(
        [
            Input(shape=(SEQ_LEN, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def train_lstm(df, split):
    features = df[FEATURE_COLS].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    dates = pd.to_datetime(df["Date"]).to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(features)
    scaled = scaler.transform(features)

    X_idx, target_idx = build_window_indices(len(df))
    X = np.array([scaled[start:end] for start, end in X_idx])
    base_idx = np.array([end - 1 for _, end in X_idx])
    y = (close[target_idx] / close[base_idx]) - 1

    train_mask = target_idx < split
    test_mask = target_idx >= split

    model = build_lstm_model(len(FEATURE_COLS))
    callbacks = [
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        X[train_mask],
        y[train_mask],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    pred_return = model.predict(X[test_mask], verbose=0).reshape(-1)
    actual = close[target_idx[test_mask]]
    predicted = close[base_idx[test_mask]] * (1 + (0.10 * pred_return))
    metrics = regression_metrics(actual, predicted)

    model_dir = EXP_DIR / "lstm"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_sp500_horizon60.keras")
    joblib.dump(scaler, MODELS_DIR / "lstm_sp500_horizon60_scaler.pkl")
    pd.DataFrame(
        {
            "Date": dates[target_idx[test_mask]],
            "Actual": actual,
            "Predicted": predicted,
            "Predicted_Return_60D": pred_return,
        }
    ).to_csv(model_dir / "predictions.csv", index=False)
    save_metrics(metrics, model_dir / "metrics.json")
    save_prediction_plot("LSTM", dates[target_idx[test_mask]], actual, predicted)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], color="#2f80ed")
    plt.title("LSTM 60-Day Horizon Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(model_dir / "training_loss.png", dpi=200)
    plt.close()
    return metrics, model_dir / "predictions.csv"


def save_comparison(metrics_by_model):
    metrics_df = pd.DataFrame(
        [{"Model": model_name, **metrics} for model_name, metrics in metrics_by_model.items()]
    )
    metrics_df.to_csv(EXP_DIR / "metrics_summary.csv", index=False)

    for metric in ["MAE", "RMSE", "MAPE"]:
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=metrics_df,
            x="Model",
            y=metric,
            hue="Model",
            palette=["#7b8794", "#2f80ed", "#eb5757"],
            legend=False,
        )
        plt.title(f"60-Day Horizon {metric} Comparison")
        plt.tight_layout()
        plt.savefig(EXP_DIR / f"{metric.lower()}_comparison.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=metrics_df,
        x="Model",
        y="R2",
        hue="Model",
        palette=["#7b8794", "#2f80ed", "#eb5757"],
        legend=False,
    )
    plt.title("60-Day Horizon R2 Comparison")
    plt.ylim(min(0, metrics_df["R2"].min() - 0.1), 1)
    plt.tight_layout()
    plt.savefig(EXP_DIR / "r2_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df.set_index("Model")[["MAE", "RMSE", "R2", "MAPE"]], annot=True, fmt=".3f", cmap="viridis")
    plt.title("60-Day Horizon Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(EXP_DIR / "metrics_heatmap.png", dpi=200)
    plt.close()

    return metrics_df


def save_model_overlay():
    lr = pd.read_csv(EXP_DIR / "linear_regression" / "predictions.csv", parse_dates=["Date"])
    arima = pd.read_csv(EXP_DIR / "arima" / "predictions.csv", parse_dates=["Date"])
    lstm = pd.read_csv(EXP_DIR / "lstm" / "predictions.csv", parse_dates=["Date"])

    merged = pd.merge(
        lr[["Date", "Actual", "Predicted"]].rename(columns={"Predicted": "Linear Regression"}),
        arima[["Date", "Predicted"]].rename(columns={"Predicted": "ARIMA"}),
        on="Date",
        how="inner",
    )
    merged = pd.merge(
        merged,
        lstm[["Date", "Predicted"]].rename(columns={"Predicted": "LSTM"}),
        on="Date",
        how="inner",
    )
    merged.to_csv(EXP_DIR / "aligned_model_predictions.csv", index=False)

    plt.figure(figsize=(14, 7))
    plt.plot(merged["Date"], merged["Actual"], label="Actual", linewidth=2.4, color="#2f80ed")
    plt.plot(
        merged["Date"],
        merged["Linear Regression"],
        label="LR",
        linewidth=1.8,
        color="#f2994a",
        alpha=0.85,
    )
    plt.plot(merged["Date"], merged["ARIMA"], label="ARIMA", linewidth=2, color="#27ae60")
    plt.plot(merged["Date"], merged["LSTM"], label="LSTM", linewidth=2.2, color="#d62728")
    plt.title("60-Day Horizon Model Comparison")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXP_DIR / "actual_vs_models_60day.png", dpi=220)
    plt.close()


def main():
    ensure_dirs()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df = load_sp500()
    split = int(len(df) * (1 - TEST_RATIO))

    print(f"Running 60-day horizon experiment on {len(df)} rows.")
    print("Training Linear Regression...")
    lr_metrics, _ = train_lr(df, split)
    print("Training ARIMA...")
    arima_metrics, _ = train_arima(df, split)
    print("Training LSTM...")
    lstm_metrics, _ = train_lstm(df, split)

    metrics_df = save_comparison(
        {
            "Linear Regression": lr_metrics,
            "ARIMA": arima_metrics,
            "LSTM": lstm_metrics,
        }
    )
    save_model_overlay()
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
