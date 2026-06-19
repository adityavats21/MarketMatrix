# MarketMatrix 📈

A full-stack stock market forecasting system that compares LSTM, ARIMA, and Linear Regression models on S&P 500 data — with a live deployed dashboard.

🔗 **Live Demo:** [market-matrix-two.vercel.app](https://market-matrix-two.vercel.app)

---

## Key Results

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| LSTM | 301.90 | 348.07 | **0.707** | **6.03%** |
| ARIMA | 911.98 | 1113.92 | -2.00 | 17.31% |
| Linear Regression | — | — | Very negative | — |

> On a 60-day forecasting horizon, LSTM significantly outperforms classical models by learning temporal sequence patterns from 14 technical indicators.

---

## Features

- **3-model comparison** — LSTM vs ARIMA vs Linear Regression on the same dataset
- **14 technical indicators** — SMA, EMA, RSI, Bollinger Bands, momentum, volatility, returns
- **60-day horizon experiment** — tests long-range forecasting beyond next-day prediction
- **REST API** — Flask backend with 20+ endpoints for metrics, predictions, and graphs
- **Interactive dashboard** — deployed frontend with model selection, charts, and EDA visualizations

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML / Forecasting | Python, LSTM (Keras), ARIMA (statsmodels), scikit-learn |
| Backend | Flask, REST API |
| Frontend | HTML, CSS, JavaScript |
| Data | yfinance, pandas, numpy |
| Deployment | Vercel (frontend), Render (backend) |

---

## Architecture
MarketMatrix/

├── frontend/

│   ├── index.html

│   ├── styles.css

│   └── app.js

└── backend/

├── api/app.py               # Flask REST API

├── config.py

├── data_loader.py

├── download_sp500.py        # Fetches S&P 500 OHLCV data

├── eda_sp500.py             # EDA graph generation

├── indicators.py            # Technical indicator calculation

├── model_arima_sp500.py

├── model_lr_sp500.py

├── model_lstm_sp500.py

├── model_comparison_sp500.py

├── experiment_60day_horizon.py

├── requirements.txt

├── data/

├── models/

└── outputs/

---

## API Endpoints
GET  /api/health

GET  /api/dataset/summary

GET  /api/eda/graphs/<graph_name>

GET  /api/lr/metrics

GET  /api/arima/metrics

GET  /api/lstm/metrics

GET  /api/comparison/metrics

GET  /api/predictions/lr

GET  /api/predictions/arima

GET  /api/predictions/lstm

POST /api/predict/arima

POST /api/predict/lstm

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/adityavats21/MarketMatrix.git
cd MarketMatrix/MarketMatrix
```

### 2. Set up Python environment

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download S&P 500 data

```bash
python download_sp500.py
```

Generates `backend/data/SP500_data.csv` with OHLCV + technical indicators.

### 4. Train models

```bash
python model_lr_sp500.py
python model_arima_sp500.py
python model_lstm_sp500.py
python model_comparison_sp500.py
```

### 5. Run the API

```bash
python api/app.py
```

API available at `http://127.0.0.1:5000/api/health`

### 6. Open the dashboard

Open `frontend/index.html` in your browser. It connects to the local API at port 5000.

---

## Model Details

**LSTM**
Multivariate 60-day sequence model using 14 technical features, predicting next-day return and reconstructing next close price. Avoids absolute price-level extrapolation.

**ARIMA**
Univariate classical baseline on the `Close` series. Fair statistical comparison but limited to linear temporal patterns.

**Linear Regression**
Leakage-free baseline using flattened 60-day technical-feature windows. Fails at long horizons due to inability to model non-linear temporal dependencies.

---

## Author

**Aditya Vats**
[GitHub](https://github.com/adityavats21) · [LinkedIn](https://linkedin.com/in/adityavats21)
