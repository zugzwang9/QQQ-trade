# QQQ Trading Bot

This is an automated machine learning trading bot for QQQ (Nasdaq-100). It trades the US market session during Swedish afternoon and evening hours (16:00 to 22:00 Swedish time). At 16:01, it predicts how the market will go for the day, places a trade, and then sells everything before the market closes.

**Work in progress and currently in testing** Everything runs automatically on GitHub Actions and sends live signals and evening results to Discord.

## How it works

Every day at 16:01 Swedish time, the bot checks **11 market signals** and feeds them into a Random Forest model trained on 700 days of data.

To keep us safe and avoid bad trades, we use these rules:
* **The 56/44 Rule:** Buy (LONG) only if the model is >56% confident. Short (SHORT) only if confidence is <44%. Otherwise, do nothing (NO TRADE).
* **Volatility Filter:** If QQQ's first 30 minutes of movement is less than 60% of its normal average range (ATR), the trade is blocked because the market is too flat.
* **Risk Management:** We use a **0.25% trailing stop-loss**. If we are wrong, we exit with a tiny loss. If we are right, we let it run until 22:00. This makes our wins much bigger than our losses.

## The 11 signals the bot checks

* `qqq_open_move`: QQQ return from 15:30 to 16:00.
* `smh_move`: Semiconductor sector (SMH) return from 15:30 to 16:00.
* `vol_ratio`: Today's opening volume vs the 5-day average.
* `rsi_1600`: QQQ 14-hour RSI at 16:00.
* `atr_ratio`: Today's opening range vs the daily ATR.
* `vix_prev`: Yesterday's VIX close (market stress).
* `spy_move`: S&P 500 futures movement.
* `dax_move`: German market (DAX) movement today.
* `nikkei_move`: Japanese market (Nikkei) overnight change.
* `us10y_move`: US 10-year treasury yield movement.
* `rut_move`: Small-cap index (Russell 2000) movement.

## File structure

* `data.py`: Downloads historical data and saves it to `trading_data.csv`.
* `ml_training.py`: Trains the model and runs historical backtests.
* `live_bot.py`: Runs at 16:01 on GitHub Actions, predicts today's move, and posts the signal to Discord.
* `close_bot.py`: Runs at 22:05 on GitHub Actions, checks if we got stopped out or hit the close, updates our virtual $100 balance, and posts the daily report to Discord.
* `.github/workflows/run_bot.yml`: Handles the automation, scheduling, and saves our data and balance on GitHub daily.