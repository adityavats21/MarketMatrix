import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BACKEND_DIR))

from config import (  # noqa: E402
    ARIMA_OUTPUT_DIR,
    COMPARISON_OUTPUT_DIR,
    FEATURE_COLS,
    LR_OUTPUT_DIR,
    LSTM_OUTPUT_DIR,
    OUTPUTS_DIR,
)
from data_loader import load_sp500  # noqa: E402
from metrics import load_metrics  # noqa: E402


app = Flask(__name__)
CORS(app)


def safe_png(directory: Path, name: str):
    filename = name if name.endswith(".png") else f"{name}.png"
    path = directory / filename
    if not path.exists():
        return jsonify({"error": f"Graph not found: {filename}"}), 404
    return send_file(path, mimetype="image/png")


def safe_json(path: Path):
    if not path.exists():
        return jsonify({"error": f"File not found: {path.name}. Run training first."}), 404
    return jsonify(load_metrics(path))


def safe_csv(path: Path):
    if not path.exists():
        return jsonify({"error": f"File not found: {path.name}. Run the required experiment first."}), 404
    return jsonify(pd.read_csv(path).to_dict(orient="records"))


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "project": "MarketMatrix",
            "dataset": "S&P 500 Index (^GSPC)",
            "models": ["Linear Regression", "ARIMA", "LSTM"],
        }
    )


@app.get("/api/dataset/summary")
def dataset_summary():
    try:
        df = load_sp500()
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(
        {
            "rows": int(len(df)),
            "start_date": str(pd.to_datetime(df["Date"]).min().date()),
            "end_date": str(pd.to_datetime(df["Date"]).max().date()),
            "features": FEATURE_COLS,
            "latest_close": float(df["Close"].iloc[-1]),
        }
    )


@app.get("/api/eda/graphs/<name>")
def eda_graph(name):
    return safe_png(BACKEND_DIR / "outputs" / "eda", name)


@app.get("/api/arima/metrics")
def arima_metrics():
    return safe_json(ARIMA_OUTPUT_DIR / "metrics.json")


@app.get("/api/lr/metrics")
def lr_metrics():
    return safe_json(LR_OUTPUT_DIR / "metrics.json")


@app.get("/api/lr/graphs/<name>")
def lr_graph(name):
    return safe_png(LR_OUTPUT_DIR, name)


@app.get("/api/arima/graphs/<name>")
def arima_graph(name):
    return safe_png(ARIMA_OUTPUT_DIR, name)


@app.get("/api/lstm/metrics")
def lstm_metrics():
    return safe_json(LSTM_OUTPUT_DIR / "metrics.json")


@app.get("/api/lstm/graphs/<name>")
def lstm_graph(name):
    return safe_png(LSTM_OUTPUT_DIR, name)


@app.get("/api/comparison/metrics")
def comparison_metrics():
    path = COMPARISON_OUTPUT_DIR / "metrics_summary.csv"
    if not path.exists():
        return jsonify({"error": "Comparison metrics not found. Run model_comparison_sp500.py first."}), 404
    return jsonify(pd.read_csv(path).to_dict(orient="records"))


@app.get("/api/comparison/graphs/<name>")
def comparison_graph(name):
    return safe_png(COMPARISON_OUTPUT_DIR, name)


@app.get("/api/experiment60/metrics")
def experiment60_metrics():
    path = OUTPUTS_DIR / "experiment_60day" / "metrics_summary.csv"
    return safe_csv(path)


@app.get("/api/experiment60/graphs/<path:name>")
def experiment60_graph(name):
    return safe_png(OUTPUTS_DIR / "experiment_60day", name)


@app.get("/api/experiment60/predictions/<model_name>")
def experiment60_predictions(model_name):
    normalized = model_name.lower()
    if normalized in {"lr", "linear-regression", "linear_regression"}:
        path = OUTPUTS_DIR / "experiment_60day" / "linear_regression" / "predictions.csv"
    elif normalized == "arima":
        path = OUTPUTS_DIR / "experiment_60day" / "arima" / "predictions.csv"
    elif normalized == "lstm":
        path = OUTPUTS_DIR / "experiment_60day" / "lstm" / "predictions.csv"
    else:
        return jsonify({"error": "model_name must be lr, arima, or lstm"}), 400
    return safe_csv(path)


@app.get("/api/predictions/<model_name>")
def predictions(model_name):
    if model_name.lower() == "arima":
        path = ARIMA_OUTPUT_DIR / "predictions.csv"
    elif model_name.lower() in {"lr", "linear-regression", "linear_regression"}:
        path = LR_OUTPUT_DIR / "predictions.csv"
    elif model_name.lower() == "lstm":
        path = LSTM_OUTPUT_DIR / "predictions.csv"
    else:
        return jsonify({"error": "model_name must be lr, arima, or lstm"}), 400

    if not path.exists():
        return jsonify({"error": f"{model_name} predictions not found. Run training first."}), 404
    return jsonify(pd.read_csv(path).to_dict(orient="records"))


@app.post("/api/predict/lr")
def predict_lr():
    return jsonify(
        {
            "error": "Linear Regression is included for test-set comparison. Future LR forecasting is disabled because future technical indicators are not known in this simple demo API.",
        }
    ), 400


@app.post("/api/predict/arima")
def predict_arima():
    payload = request.get_json(silent=True) or {}
    days = int(payload.get("days", 5))
    days = max(1, min(days, 30))

    predictions_path = ARIMA_OUTPUT_DIR / "predictions.csv"
    if not predictions_path.exists():
        return jsonify({"error": "ARIMA prediction file not found. Run model_arima_sp500.py first."}), 404

    df = load_sp500()
    predictions_df = pd.read_csv(predictions_path)
    latest_close = float(df["Close"].iloc[-1])
    recent_step = predictions_df["Predicted"].diff().dropna().tail(20).mean()
    if not np.isfinite(recent_step):
        recent_step = 0.0
    forecast = [latest_close + (recent_step * (i + 1)) for i in range(days)]
    return jsonify(
        {
            "model": "ARIMA",
            "days": days,
            "forecast": [float(x) for x in forecast],
            "note": "Lightweight deployment forecast based on saved ARIMA prediction trend.",
        }
    )


@app.post("/api/predict/lstm")
def predict_lstm():
    payload = request.get_json(silent=True) or {}
    days = int(payload.get("days", 1))
    days = max(1, min(days, 5))

    df = load_sp500()
    predictions_path = LSTM_OUTPUT_DIR / "predictions.csv"
    if not predictions_path.exists():
        return jsonify({"error": "LSTM prediction file not found. Run model_lstm_sp500.py first."}), 404

    predictions_df = pd.read_csv(predictions_path)
    if "Predicted_Return" in predictions_df.columns:
        pred_return = float(predictions_df["Predicted_Return"].tail(20).mean())
    else:
        pred_return = float(predictions_df["Predicted"].pct_change().tail(20).mean())
    if not np.isfinite(pred_return):
        pred_return = 0.0

    last_close = float(df["Close"].iloc[-1])

    forecasts = []
    for _ in range(days):
        pred_close = last_close * (1 + pred_return)
        forecasts.append(pred_close)
        last_close = pred_close

    return jsonify(
        {
            "model": "LSTM",
            "days": days,
            "forecast": forecasts,
            "note": "Lightweight deployment forecast based on saved LSTM prediction-return behavior.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
