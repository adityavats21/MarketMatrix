#!/usr/bin/env bash
set -e

python download_sp500.py
python eda_sp500.py
python model_lr_sp500.py
python model_arima_sp500.py
python model_lstm_sp500.py
python model_comparison_sp500.py
python experiment_60day_horizon.py

echo "MarketMatrix backend pipeline completed."
