import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from config import ARIMA_OUTPUT_DIR, MODELS_DIR, TEST_RATIO, ensure_dirs
from data_loader import load_sp500
from metrics import regression_metrics, save_metrics


ORDER = (5, 1, 0)


def walk_forward_arima(train_values, test_values, order=ORDER):
    history = list(train_values)
    predictions = []

    for step, actual in enumerate(test_values, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(history, order=order)
            fitted = model.fit()
        forecast = fitted.forecast(steps=1)[0]
        predictions.append(forecast)
        history.append(actual)

        if step % 50 == 0:
            print(f"ARIMA walk-forward progress: {step}/{len(test_values)}")

    return predictions


def save_plots(dates_test, actual, predicted, metrics):
    ARIMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, actual, label="Actual", linewidth=2)
    plt.plot(dates_test, predicted, label="ARIMA Forecast", linewidth=2)
    plt.title("S&P 500 ARIMA Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARIMA_OUTPUT_DIR / "01_actual_vs_predicted.png", dpi=200)
    plt.close()

    residuals = actual - predicted
    plt.figure(figsize=(14, 5))
    plt.plot(dates_test, residuals, color="crimson")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("ARIMA Residuals")
    plt.xlabel("Date")
    plt.ylabel("Error")
    plt.tight_layout()
    plt.savefig(ARIMA_OUTPUT_DIR / "02_residuals.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=["#2f80ed", "#56cc9d", "#f2c94c", "#9b51e0", "#eb5757"])
    plt.title("ARIMA Metrics")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(ARIMA_OUTPUT_DIR / "03_metrics.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    df = load_sp500()
    close = df["Close"].astype(float).to_numpy()
    dates = pd.to_datetime(df["Date"])

    split = int(len(close) * (1 - TEST_RATIO))
    train, test = close[:split], close[split:]
    dates_test = dates.iloc[split:]

    print(f"Training ARIMA{ORDER} on {len(train)} rows, testing on {len(test)} rows.")
    predictions = walk_forward_arima(train, test)
    metrics = regression_metrics(test, predictions)

    final_model = ARIMA(close, order=ORDER).fit()
    joblib.dump(final_model, MODELS_DIR / "arima_sp500.pkl")

    results = pd.DataFrame(
        {"Date": dates_test.values, "Actual": test, "Predicted": predictions}
    )
    results.to_csv(ARIMA_OUTPUT_DIR / "predictions.csv", index=False)
    save_metrics(metrics, ARIMA_OUTPUT_DIR / "metrics.json")
    save_plots(dates_test, test, predictions, metrics)

    print("ARIMA metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
