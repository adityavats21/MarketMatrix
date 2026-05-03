# MarketMatrix Backend Notes

## Do Not Reintroduce These Mistakes

- Do not use AAPL/MSFT/GOOGL for model comparison.
- Do not add Linear Regression to the final comparison.
- Do not use same-day `High` or `Low` to predict same-day `Close`.
- Do not use `validation_split` and `EarlyStopping` for the current LSTM experiment.
- Do not use `ModelCheckpoint` on low-disk machines.
- Do not change `SEQ_LEN = 60` unless you are intentionally running a new experiment.

## Current Model Philosophy

ARIMA is the classical statistical baseline. LSTM is the deep-learning sequence model.

The final LSTM predicts next-day return from a 60-day multivariate window, then reconstructs next close using the previous close. This avoids the level-extrapolation failure seen when a neural model directly predicts absolute index price.

The 60-day horizon experiment is intentionally separate from the main next-day model. It demonstrates that longer forecast horizons are much harder:

- Linear Regression fails on the high-dimensional 60-day flattened input.
- Train-once ARIMA drifts over the long horizon.
- Conservative LSTM gives the best long-horizon result.

The project is educational, not a trading platform.
