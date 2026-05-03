import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ARIMA_OUTPUT_DIR, COMPARISON_OUTPUT_DIR, LR_OUTPUT_DIR, LSTM_OUTPUT_DIR, ensure_dirs
from metrics import load_metrics


def load_results():
    lr_metrics = load_metrics(LR_OUTPUT_DIR / "metrics.json")
    arima_metrics = load_metrics(ARIMA_OUTPUT_DIR / "metrics.json")
    lstm_metrics = load_metrics(LSTM_OUTPUT_DIR / "metrics.json")
    lr_pred = pd.read_csv(LR_OUTPUT_DIR / "predictions.csv", parse_dates=["Date"])
    arima_pred = pd.read_csv(ARIMA_OUTPUT_DIR / "predictions.csv", parse_dates=["Date"])
    lstm_pred = pd.read_csv(LSTM_OUTPUT_DIR / "predictions.csv", parse_dates=["Date"])
    return lr_metrics, arima_metrics, lstm_metrics, lr_pred, arima_pred, lstm_pred


def save_metric_bars(metrics_df):
    plot_metrics = ["MAE", "RMSE", "MAPE"]
    for metric in plot_metrics:
        plt.figure(figsize=(7, 5))
        sns.barplot(
            data=metrics_df,
            x="Model",
            y=metric,
            hue="Model",
            palette=["#7b8794", "#2f80ed", "#eb5757"],
            legend=False,
        )
        plt.title(f"{metric} Comparison")
        plt.tight_layout()
        plt.savefig(COMPARISON_OUTPUT_DIR / f"{metric.lower()}_comparison.png", dpi=200)
        plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=metrics_df,
        x="Model",
        y="R2",
        hue="Model",
        palette=["#7b8794", "#2f80ed", "#eb5757"],
        legend=False,
    )
    plt.title("R2 Score Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "r2_comparison.png", dpi=200)
    plt.close()


def save_prediction_overlay(lr_pred, arima_pred, lstm_pred):
    merged = pd.merge(
        arima_pred[["Date", "Actual", "Predicted"]],
        lstm_pred[["Date", "Predicted"]],
        on="Date",
        how="inner",
        suffixes=("_ARIMA", "_LSTM"),
    )
    merged.rename(columns={"Predicted_ARIMA": "ARIMA", "Predicted_LSTM": "LSTM"}, inplace=True)
    merged = pd.merge(
        merged,
        lr_pred[["Date", "Predicted"]].rename(columns={"Predicted": "Linear Regression"}),
        on="Date",
        how="inner",
    )
    merged.to_csv(COMPARISON_OUTPUT_DIR / "aligned_predictions.csv", index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(merged["Date"], merged["Actual"], label="Actual", linewidth=2)
    plt.plot(merged["Date"], merged["Linear Regression"], label="Linear Regression", linewidth=2)
    plt.plot(merged["Date"], merged["ARIMA"], label="ARIMA", linewidth=2)
    plt.plot(merged["Date"], merged["LSTM"], label="LSTM", linewidth=2)
    plt.title("S&P 500 Model Comparison")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "actual_vs_models.png", dpi=200)
    plt.close()


def save_heatmap(metrics_df):
    heatmap_df = metrics_df.set_index("Model")[["MAE", "RMSE", "R2", "MAPE"]]
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Model Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "metrics_heatmap.png", dpi=200)
    plt.close()


def save_radar(metrics_df):
    normalized = metrics_df.copy()
    for metric in ["MAE", "RMSE", "MAPE"]:
        max_val = normalized[metric].max()
        normalized[metric] = 1 - (normalized[metric] / max_val if max_val else 0)
    normalized["R2"] = normalized["R2"].clip(lower=0)

    labels = ["MAE Score", "RMSE Score", "R2", "MAPE Score"]
    value_cols = ["MAE", "RMSE", "R2", "MAPE"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for _, row in normalized.iterrows():
        values = [row[col] for col in value_cols]
        values += values[:1]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Normalized Model Strength Radar")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "radar_chart.png", dpi=200)
    plt.close()


def save_summary_table(metrics_df):
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.axis("off")
    table_data = metrics_df.round({"MAE": 3, "MSE": 3, "RMSE": 3, "R2": 4, "MAPE": 3})
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("LR vs ARIMA vs LSTM Summary")
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "summary_table.png", dpi=200)
    plt.close()


def main():
    ensure_dirs()
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_metrics, arima_metrics, lstm_metrics, lr_pred, arima_pred, lstm_pred = load_results()

    metrics_df = pd.DataFrame(
        [
            {"Model": "Linear Regression", **lr_metrics},
            {"Model": "ARIMA", **arima_metrics},
            {"Model": "LSTM", **lstm_metrics},
        ]
    )
    metrics_df.to_csv(COMPARISON_OUTPUT_DIR / "metrics_summary.csv", index=False)

    save_metric_bars(metrics_df)
    save_prediction_overlay(lr_pred, arima_pred, lstm_pred)
    save_heatmap(metrics_df)
    save_radar(metrics_df)
    save_summary_table(metrics_df)

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
