import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DISCORD_URL = os.environ.get('DISCORD_WEBHOOK_URL')

def notify(msg):
    if DISCORD_URL:
        try:
            res = requests.post(DISCORD_URL, json={"content": msg})
            res.raise_for_status()
            print("Discord notification sent successfully.")
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")
    else:
        print(msg)

def run():
    print("Starting evening market check...")
    if not os.path.exists("today_trade.json"):
        print("No trade recorded today.")
        return
        
    with open("today_trade.json", "r") as f:
        trade = json.load(f)
        
    bal_file = "virtual_balance.json"
    if os.path.exists(bal_file):
        with open(bal_file, "r") as f:
            balance_data = json.load(f)
            balance = balance_data.get("balance", 100.0)
    else:
        balance = 100.0

    sig = trade["signal"]
    entry = float(trade["entry_price"])
    date_str = trade["date"]
    
    if sig == 0:
        msg = f"QQQ-report ({date_str})\n"
        msg += "Bot stayed out of the market today.\n"
        msg += f"Balance: {balance:.2f} USD"
        notify(msg)
        return

    today_data = yf.download('QQQ', start=datetime.now() - timedelta(days=5), interval='60m', progress=False)
    if isinstance(today_data.columns, pd.MultiIndex):
        today_data.columns = today_data.columns.get_level_values(0)
        
    today_data['Date'] = today_data.index.date
    day_candles = today_data[today_data['Date'].astype(str) == date_str]
    
    if len(day_candles) < 2:
        notify(f"Could not find sufficient price data for {date_str}.")
        return

    candles = day_candles.iloc[1:]
    close_val = float(day_candles.iloc[-1]['Close'])
    trail_pct = 0.25
    stopped = False
    ret = 0.0
    status_msg = ""

    if sig == 1:
        peak = entry
        stop = entry * (1 - trail_pct / 100)
        for _, candle in candles.iterrows():
            low_val = float(candle['Low'])
            high_val = float(candle['High'])
            if low_val <= stop:
                stopped = True
                ret = -trail_pct
                status_msg = f"Stopped out on trailing stop at {stop:.2f} USD."
                break
            if high_val > peak:
                peak = high_val
                stop = peak * (1 - trail_pct / 100)
        if not stopped:
            ret = ((close_val - entry) / entry) * 100
            status_msg = f"Closed at 22:00 at {close_val:.2f} USD ({ret:.2f}%)."

    elif sig == -1:
        floor = entry
        stop = entry * (1 + trail_pct / 100)
        for _, candle in candles.iterrows():
            high_val = float(candle['High'])
            low_val = float(candle['Low'])
            if high_val >= stop:
                stopped = True
                ret = -trail_pct
                status_msg = f"Stopped out on trailing stop at {stop:.2f} USD."
                break
            if low_val < floor:
                floor = low_val
                stop = floor * (1 + trail_pct / 100)
        if not stopped:
            ret = ((entry - close_val) / entry) * 100
            status_msg = f"Closed at 22:00 at {close_val:.2f} USD ({ret:.2f}%)."

    old_balance = balance
    balance = balance * (1 + ret / 100)
    diff = balance - old_balance
    diff_sign = "+" if diff >= 0 else ""
    
    with open(bal_file, "w") as f:
        json.dump({"balance": balance}, f)

    msg = f"QQQ-report ({date_str})\n"
    msg += f"Position: {'LONG' if sig == 1 else 'SHORT'} at {entry:.2f} USD\n"
    msg += f"Status: {status_msg}\n"
    msg += f"Result: {diff_sign}{diff:.2f} USD ({diff_sign}{ret:.2f}%)\n"
    msg += f"Balance: {balance:.2f} USD"
    notify(msg)

if __name__ == "__main__":
    run()