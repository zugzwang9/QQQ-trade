# QQQ-trade

Algorithmic trading experiment using machine learning (Random Forest) to predict price movements on the QQQ (Nasdaq 100).

**Work in progress**

## How it Works

1. **`data.py`**
   * Grabs QQQ hourly data. 
   * Checks the move and volume between 15:30 and 16:00.
   * Looks at already open markets returns (DAX, Nikkei, S&P futures, VIX, TNX, RUT).
   * Dumps everything into `trading_data.csv`.

2. **`ml_training.py`**
   * Trains a RandomForest to predict if QQQ goes UP or DOWN between 16:00 and close.
   * Trade rules: Buy (`1`) if prob > 55%, Short (`-1`) if prob < 45%. Else skip (`0`).
   * Runs backtest to check win rate and profit factor.