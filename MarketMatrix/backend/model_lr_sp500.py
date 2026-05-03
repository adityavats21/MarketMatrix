import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, MODELS_DIR, OUTPUTS_DIR, TEST_RATIO, ensure_dirs
from data_loader import load_sp500
from metrics import regression_metrics, save_metrics


LR_OUTPUT_DIR = OUTPUTS_DIR / "lr_sp500"


def build_lagged_features(data):
    X, idx = [], []
    for i in range(1, len(data)):
        X.append(data[i - 1, :])
        idx.append(i)
    return np.array(X), np.array(idx)


def save_plots(dates_test, actual, predicted, metrics):
    LR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, actual, label="Actual", linewidth=2)
    plt.plot(dates_test, predicted, label="Linear Regression", linewidth=2)
    plt.title("S&P 500 Linear Regression Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LR_OUTPUT_DIR / "01_actual_vs_predicted.png", dpi=200)
    plt.close()

    residuals = actual - predicted
    plt.figure(figsize=(14, 5))
    plt.plot(dates_test, residuals, color="crimson")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Linear Regression Residuals")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(LR_OUTPUT_DIR / "02_residuals.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        metrics.keys(),
        metrics.values(),
        color=["#2f80ed", "#56cc9d", "#f2c94c", "#9b51e0", "#eb5757"],
    )
    plt.title("Linear Regression Metrics")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(LR_OUTPUT_DIR / "03_metrics.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    LR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sp500()
    dates = pd.to_datetime(df["Date"]).to_numpy()
    features = df[FEATURE_COLS].astype(float).to_numpy()
    close_values = df["Close"].astype(float).to_numpy()
    returns = df["Daily_Return"].astype(float).to_numpy()

    split = int(len(features) * (1 - TEST_RATIO))
    scaler = StandardScaler()
    scaler.fit(features[:split])
    scaled = scaler.transform(features)

    X, target_idx = build_lagged_features(scaled)
    y = returns[target_idx]
    train_mask = target_idx < split
    test_mask = target_idx >= split

    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]
    idx_test = target_idx[test_mask]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_return = model.predict(X_test)
    actual = close_values[idx_test]
    prev_close = close_values[idx_test - 1]
    predicted = prev_close * (1 + pred_return)

    metrics = regression_metrics(actual, predicted)
    joblib.dump({"model": model, "scaler": scaler}, MODELS_DIR / "lr_sp500.pkl")

    results = pd.DataFrame(
        {
            "Date": dates[idx_test],
            "Actual": actual,
            "Predicted": predicted,
            "Predicted_Return": pred_return,
        }
    )
    results.to_csv(LR_OUTPUT_DIR / "predictions.csv", index=False)
    save_metrics(metrics, LR_OUTPUT_DIR / "metrics.json")
    save_plots(dates[idx_test], actual, predicted, metrics)

    print("Linear Regression metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
