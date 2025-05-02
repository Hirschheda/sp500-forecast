# S&P 500 Forecast

## Overview

This repository contains a machine learning pipeline to forecast the weekly closing price of the S&P 500 index using a Long Short-Term Memory (LSTM) neural network. The project leverages historical price data, technical indicators, and macro-economic features to predict future prices, providing a practical example of time-series forecasting with deep learning.

## Features

- Fetches S&P 500 data using `yfinance`.
- Computes technical indicators (RSI, EMA crossover, ATR, VWAP deviation, volume spikes) and macro features (yield spread, trend strength).
- Trains an LSTM model to predict the next week's closing price.
- Evaluates model performance with MAE, MSE, and R² metrics.
- Visualizes actual vs. predicted prices and feature importances.
- Includes a workaround for calculating feature importance with LSTM using permutation methods.

## Requirements

- Python 3.11 or later
- Required libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `matplotlib`

Install dependencies using:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
```
## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Hirschheda/sp500-forecast.git
   cd sp500-forecast
   ```
2.	Install the required dependencies as listed above.
3.	Ensure you have an internet connection to fetch data from Yahoo Finance.

OR

Run on Google Colab
## Customization

You can easily tweak each model’s hyperparameters by editing the corresponding notebook. Below are the key parameters and where to find them:

### LightGBM Regression  
_Notebook: `s&p500_prediction_LightGBM.ipynb`_  
In the “Train/Test Split & LightGBM Evaluation” cell, edit the `LGBMRegressor` instantiation:

```python
lgbm = LGBMRegressor(
    n_estimators=500,        # number of trees
    learning_rate=0.05,      # step size shrinkage
    num_leaves=31,           # maximum leaves per tree
    subsample=0.8,           # fraction of rows per tree
    colsample_bytree=0.8,    # fraction of features per tree
    random_state=1
)
```
- n_estimators: More trees can improve fit at the cost of speed.
- learning_rate: Lower values often generalize better.
- num_leaves: Controls tree complexity.
- subsample / colsample_bytree: Regularize by sampling rows/features.
### XGBoost Classification

_Notebook: `s&p500_prediction_XGBoost.ipynb`_  
In the “Train/Test Split & LightGBM Evaluation” cell, edit the `XGBClassifier` instantiation:
```python
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=1
)
```
- max_depth: Deeper trees capture more feature interactions.
### LSTM Regression

_Notebook: `s&p500_prediction_LSTM.ipynb`_  
In the “Train/Test Split & LightGBM Evaluation” cell, edit the `XGBClassifier` instantiation:
```python
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=1
)
```
- max_depth: Deeper trees capture more feature interactions.
### LSTM Regression  
_Notebook: `s&p500_prediction_LSTM.ipynb`_  

1. **Sequence length**  
```python
   window_size = 20  # number of past weeks used as input
```
2. **Network architecture & training**
```python
model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=[EarlyStopping(patience=10)],
    verbose=1
)
```
- LSTM units (64, 32): Number of memory cells per layer.
- Dropout: Fraction of inputs to drop (0.0–0.5).
- Epochs / batch_size: Adjust training length and stability.
- EarlyStopping patience: Epochs without improvement before stopping.
## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and commit them (git commit -m "Description of changes").
4. Push to the branch (git push origin feature-branch).
5. Open a pull request with a clear description of your changes.

Please ensure code follows PEP 8 style guidelines and includes appropriate comments.

## Acknowledgments
- Built with assistance from xAI's Grok 3 and ChatGPT o4-mini-high.
- Data sourced from Yahoo Finance via yfinance.
- Inspired by time-series forecasting techniques in financial modeling.
