# MarketMatrix

Stock Market Data Analysis and Visualization System for B.Tech CSE SDP.

This fresh build uses only the S&P 500 index (`^GSPC`) for prediction models.

- Linear Regression: leakage-free baseline using flattened 60-day technical-feature windows
- ARIMA: univariate classical baseline using `Close`
- LSTM: multivariate 60-day sequence model using 14 technical features, predicting next-day return and reconstructing next close

## Step 1: Open the Project

1. Open Visual Studio Code.
2. Click `File > Open Folder`.
3. Select this folder:

```bash
/Users/adityavats/Documents/New project/MarketMatrix
```

4. Open the VS Code terminal with:

```bash
Control + `
```

5. Move into backend:

```bash
cd backend
```

## Step 2: Create Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

When the environment is active, your terminal should show `(venv)`.

## Step 3: Download S&P 500 Data

```bash
python download_sp500.py
```

This creates:

```text
backend/data/SP500_data.csv
```

The CSV includes OHLCV data plus technical indicators such as SMA, EMA, RSI, Bollinger Bands, momentum, volatility, and returns.

## Step 4: Generate EDA Graphs

```bash
python eda_sp500.py
```

Graphs are saved in:

```text
backend/outputs/eda/
```

## Step 5: Train Linear Regression

```bash
python model_lr_sp500.py
```

This baseline uses past 60-day technical-feature windows and predicts next-day return. It does not use same-day `High` or `Low` to predict same-day `Close`.

## Step 6: Train ARIMA

```bash
python model_arima_sp500.py
```

Outputs:

```text
backend/models/arima_sp500.pkl
backend/outputs/arima_sp500/metrics.json
backend/outputs/arima_sp500/predictions.csv
backend/outputs/arima_sp500/*.png
```

## Step 7: Train LSTM

```bash
python model_lstm_sp500.py
```

Important settings:

```python
SEQ_LEN = 60
HORIZON = 1
EPOCHS = 50
BATCH_SIZE = 32
```

Outputs:

```text
backend/models/lstm_sp500.keras
backend/models/lstm_sp500_scaler.pkl
backend/outputs/lstm_sp500/metrics.json
backend/outputs/lstm_sp500/predictions.csv
backend/outputs/lstm_sp500/*.png
```

## Step 8: Generate Model Comparison

```bash
python model_comparison_sp500.py
```

Outputs:

```text
backend/outputs/comparison_sp500/metrics_summary.csv
backend/outputs/comparison_sp500/actual_vs_models.png
backend/outputs/comparison_sp500/metrics_heatmap.png
backend/outputs/comparison_sp500/radar_chart.png
backend/outputs/comparison_sp500/summary_table.png
```

## Step 9: Run Flask API

```bash
python api/app.py
```

Open this URL in browser:

```text
http://127.0.0.1:5000/api/health
```

Useful endpoints:

```text
GET  /api/health
GET  /api/dataset/summary
GET  /api/eda/graphs/<graph_name>
GET  /api/lr/metrics
GET  /api/lr/graphs/<graph_name>
GET  /api/arima/metrics
GET  /api/arima/graphs/<graph_name>
GET  /api/lstm/metrics
GET  /api/lstm/graphs/<graph_name>
GET  /api/comparison/metrics
GET  /api/comparison/graphs/<graph_name>
GET  /api/predictions/lr
GET  /api/predictions/arima
GET  /api/predictions/lstm
POST /api/predict/arima
POST /api/predict/lstm
```

Example graph URL:

```text
http://127.0.0.1:5000/api/lstm/graphs/01_actual_vs_predicted
```

## Step 10: Open Frontend Dashboard

Keep the Flask API running, then open:

```text
/Users/adityavats/Documents/New project/MarketMatrix/frontend/index.html
```

The dashboard connects to:

```text
http://127.0.0.1:5000
```

Frontend sections:

```text
Predict
Comparison
Dashboard
About
```

The frontend opens directly on the Predict page. It includes model selection, horizon selection, prediction output, real-vs-predicted charts, next-day comparison, 60-day horizon comparison, EDA dashboard, and project/team details.

## Academic Explanation

The final LSTM setup is:

```text
Last 60 S&P 500 trading days + 14 technical features -> next-day return -> next close price
```

This is stronger than simple day-to-price regression because LSTM receives temporal structure and avoids absolute price-level extrapolation. ARIMA remains a fair statistical baseline because it also works on ordered time-series data, but it only sees the univariate `Close` series.

The main panel explanation:

> LSTM outperforms ARIMA when the problem is framed as sequence prediction and the dataset has learnable temporal structure. The S&P 500 index is smoother than individual stocks because it aggregates 500 companies, reducing company-specific noise.

## 60-Day Horizon Experiment

Run:

```bash
python experiment_60day_horizon.py
```

This experiment changes the task from:

```text
Past 60 days -> next trading day
```

to:

```text
Past 60 days -> price after 60 trading days
```

Results from the current run:

```text
Linear Regression
MAE  : 608,892,800,000+
RMSE : 2,372,609,000,000+
R2   : extremely negative

ARIMA
MAE  : 911.98
RMSE : 1113.92
R2   : -2.0009
MAPE : 17.31%

LSTM
MAE  : 301.90
RMSE : 348.07
R2   : 0.7070
MAPE : 6.03%
```

Panel explanation:

> For next-day prediction, all models perform similarly because index prices are highly continuous. But when the forecast horizon is increased to 60 trading days, uncertainty increases. Linear Regression fails because it cannot model long temporal behavior, ARIMA struggles because long recursive forecasts drift, while LSTM performs best because it learns sequence patterns from the 60-day window.

## Project Structure

```text
MarketMatrix/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── backend/
    ├── api/app.py
    ├── config.py
    ├── data_loader.py
    ├── download_sp500.py
    ├── eda_sp500.py
    ├── indicators.py
    ├── metrics.py
    ├── model_arima_sp500.py
    ├── model_lr_sp500.py
    ├── model_lstm_sp500.py
    ├── model_comparison_sp500.py
    ├── experiment_60day_horizon.py
    ├── requirements.txt
    ├── data/
    ├── models/
    └── outputs/
```
