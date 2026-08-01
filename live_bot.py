import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

DISCORD_URL = os.environ.get('DISCORD_WEBHOOK_URL')

def notify(msg):
    if DISCORD_URL:
        try:
            res = requests.post(DISCORD_URL, json={"content": msg})
            res.raise_for_status()
            print("Discord message sent successfully.")
        except Exception as e:
            print(f"Failed to send Discord message: {e}")
    else:
        print("No DISCORD_WEBHOOK_URL found. Printing message locally:")
        print(msg)

def run():
    print("Starting prediction...")
    
    if not os.path.exists('trading_data.csv'):
        print("trading_data.csv missing. Running data.py...")
        from data import get_data
        df_new = get_data()
        df_new.to_csv('trading_data.csv')

    df = pd.read_csv('trading_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    df_hist = df[df.index.strftime('%Y-%m-%d') < today_str]
    if len(df_hist) == 0:
        df_hist = df

    df['Target'] = np.where(df['qqq_trade_move'] > 0, 1, 0)
    features = [
        'vix_prev', 'dax_move', 'spy_move', 'nikkei_move', 
        'us10y_move', 'rut_move', 'qqq_open_move', 'vol_ratio',
        'smh_move', 'rsi_1600', 'atr_ratio'
    ]
    
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42)
    model.fit(df[features], df['Target'])
    
    today = yf.download('QQQ', start=datetime.now() - timedelta(days=5), interval='60m', progress=False)
    if isinstance(today.columns, pd.MultiIndex):
        today.columns = today.columns.get_level_values(0)
        
    if today.empty:
        notify("Yahoo Finance returned empty data. Could not generate signal.")
        return

    diff = today['Close'].diff()
    up = diff.where(diff > 0, 0).rolling(14).mean()
    down = (-diff.where(diff < 0, 0)).rolling(14).mean()
    today['rsi'] = 100 - (100 / (1 + up / down))
    
    h_l = today['High'] - today['Low']
    h_pc = np.abs(today['High'] - today['Close'].shift(1))
    l_pc = np.abs(today['Low'] - today['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    today['atr'] = tr.rolling(14).mean()
    
    today['Date'] = today.index.date
    latest_date = today['Date'].iloc[-1]
    day_candles = today[today['Date'] == latest_date]
    
    if len(day_candles) == 0:
        notify(f"Warning: No market candles found for {latest_date}.")
        return

    open_1530 = float(day_candles.iloc[0]['Open'])
    price_1600 = float(day_candles.iloc[0]['Close'])
    vol_1600 = float(day_candles.iloc[0]['Volume'])
    range_1600 = float(day_candles.iloc[0]['High'] - day_candles.iloc[0]['Low'])
    
    last_rsi = float(day_candles.iloc[0]['rsi']) if not pd.isna(day_candles.iloc[0]['rsi']) else 50.0
    last_atr = float(day_candles.iloc[0]['atr']) if not pd.isna(day_candles.iloc[0]['atr']) else 1.0

    live_X = pd.DataFrame([{
        'vix_prev': float(df_hist['vix_prev'].iloc[-1]),
        'dax_move': float(df_hist['dax_move'].iloc[-1]),
        'spy_move': float(df_hist['spy_move'].iloc[-1]),
        'nikkei_move': float(df_hist['nikkei_move'].iloc[-1]),
        'us10y_move': float(df_hist['us10y_move'].iloc[-1]),
        'rut_move': float(df_hist['rut_move'].iloc[-1]),
        'qqq_open_move': float(((price_1600 - open_1530) / open_1530) * 100),
        'vol_ratio': float(vol_1600 / df_hist['vol_1530'].tail(5).mean()),
        'smh_move': float(df_hist['smh_move'].iloc[-1]),
        'rsi_1600': last_rsi,
        'atr_ratio': float(range_1600 / last_atr) if last_atr > 0 else 0.0
    }])
    
    prob = float(model.predict_proba(live_X)[0, 1] * 100)
    atr_val = float(live_X['atr_ratio'].iloc[0])
    open_move = float(live_X['qqq_open_move'].iloc[0])
    
    signal = 0
    msg = f"QQQ Trading Bot ({latest_date})\n"
    msg += f"Model Probability: {prob:.1f}%\n"
    msg += f"First 30m Return: {open_move:.2f}%\n"
    
    if atr_val < 0.60:
        msg += "Blocked by volatility filter. Action: NO TRADE"
    else:
        if prob > 56.0:
            msg += "Action: LONG QQQ"
            signal = 1
        elif prob < 44.0:
            msg += "Action: SHORT QQQ"
            signal = -1
        else:
            msg += "Action: NO TRADE"
            
    notify(msg)
    
    trade_info = {
        "date": str(latest_date),
        "signal": signal,
        "entry_price": float(price_1600)
    }
    with open("today_trade.json", "w") as f:
        json.dump(trade_info, f)

if __name__ == "__main__":
    run()