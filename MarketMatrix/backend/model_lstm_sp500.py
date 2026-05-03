import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURE_COLS,
    HORIZON,
    LSTM_OUTPUT_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    SEQ_LEN,
    TEST_RATIO,
    ensure_dirs,
)
from data_loader import load_sp500
from metrics import regression_metrics, save_metrics


def set_seed():
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def build_sequences(data, seq_len=SEQ_LEN, horizon=HORIZON):
    X, idx = [], []
    for i in range(seq_len, len(data) - horizon + 1):
        target_pos = i + horizon - 1
        X.append(data[i - seq_len : i, :])
        idx.append(target_pos)
    return np.array(X), np.array(idx)


def build_model(n_features):
    model = Sequential(
        [
            Input(shape=(SEQ_LEN, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def save_plots(dates_train, y_train_actual, train_pred_actual, dates_test, y_test_actual, test_pred_actual, history, metrics):
    LSTM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, y_test_actual, label="Actual", linewidth=2)
    plt.plot(dates_test, test_pred_actual, label="LSTM Predicted", linewidth=2)
    plt.title("S&P 500 LSTM Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_OUTPUT_DIR / "01_actual_vs_predicted.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(dates_train, y_train_actual, label="Train Actual", alpha=0.8)
    plt.plot(dates_train, train_pred_actual, label="Train Predicted", alpha=0.8)
    plt.plot(dates_test, y_test_actual, label="Test Actual", linewidth=2)
    plt.plot(dates_test, test_pred_actual, label="Test Predicted", linewidth=2)
    plt.title("S&P 500 LSTM Train and Test Fit")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_OUTPUT_DIR / "02_train_test_fit.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], color="#2f80ed")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.tight_layout()
    plt.savefig(LSTM_OUTPUT_DIR / "03_training_loss.png", dpi=200)
    plt.close()

    residuals = y_test_actual - test_pred_actual
    plt.figure(figsize=(14, 5))
    plt.plot(dates_test, residuals, color="crimson")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("LSTM Residuals")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(LSTM_OUTPUT_DIR / "04_residuals.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=["#2f80ed", "#56cc9d", "#f2c94c", "#9b51e0", "#eb5757"])
    plt.title("LSTM Metrics")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(LSTM_OUTPUT_DIR / "05_metrics.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    set_seed()
    df = load_sp500()
    dates = pd.to_datetime(df["Date"]).to_numpy()
    features = df[FEATURE_COLS].astype(float).to_numpy()
    close_values = df["Close"].astype(float).to_numpy()
    returns = df["Daily_Return"].astype(float).to_numpy()

    split = int(len(features) * (1 - TEST_RATIO))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features)
    scaled = scaler.transform(features)

    X, target_idx = build_sequences(scaled)
    y = returns[target_idx]
    train_mask = target_idx < split
    test_mask = target_idx >= split

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    idx_train, idx_test = target_idx[train_mask], target_idx[test_mask]

    print(f"LSTM train sequences: {X_train.shape}, test sequences: {X_test.shape}")
    model = build_model(n_features=len(FEATURE_COLS))
    callbacks = [
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    train_pred_return = model.predict(X_train, verbose=0).reshape(-1)
    test_pred_return = model.predict(X_test, verbose=0).reshape(-1)

    train_prev_close = close_values[idx_train - 1]
    test_prev_close = close_values[idx_test - 1]
    y_train_actual = close_values[idx_train]
    y_test_actual = close_values[idx_test]
    train_pred_actual = train_prev_close * (1 + train_pred_return)
    test_pred_actual = test_prev_close * (1 + test_pred_return)

    metrics = regression_metrics(y_test_actual, test_pred_actual)
    model.save(MODELS_DIR / "lstm_sp500.keras")
    joblib.dump(scaler, MODELS_DIR / "lstm_sp500_scaler.pkl")

    results = pd.DataFrame(
        {
            "Date": dates[idx_test],
            "Actual": y_test_actual,
            "Predicted": test_pred_actual,
            "Predicted_Return": test_pred_return,
        }
    )
    results.to_csv(LSTM_OUTPUT_DIR / "predictions.csv", index=False)
    save_metrics(metrics, LSTM_OUTPUT_DIR / "metrics.json")
    save_plots(
        dates[idx_train],
        y_train_actual,
        train_pred_actual,
        dates[idx_test],
        y_test_actual,
        test_pred_actual,
        history,
        metrics,
    )

    print("LSTM metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
