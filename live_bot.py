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
        requests.post(DISCORD_URL, json={"content": msg})
    else:
        print(msg)

def run():
    print("Starting prediction...")
    df = pd.read_csv('trading_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    
    df['Target'] = np.where(df['qqq_trade_move'] > 0, 1, 0)
    features = [
        'vix_prev', 'dax_move', 'spy_move', 'nikkei_move', 
        'us10y_move', 'rut_move', 'qqq_open_move', 'vol_ratio',
        'smh_move', 'rsi_1600', 'atr_ratio'
    ]
    
    #train model on all data
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42)
    model.fit(df[features], df['Target'])
    
    #fetch data
    today = yf.download('QQQ', start=datetime.now() - timedelta(days=4), interval='60m', progress=False)
    if isinstance(today.columns, pd.MultiIndex):
        today.columns = today.columns.get_level_values(0)
        
    diff = today['Close'].diff()
    up = diff.where(diff > 0, 0).rolling(14).mean()
    down = (-diff.where(diff < 0, 0)).rolling(14).mean()
    today['rsi'] = 100 - (100 / (1 + up/down))
    
    h_l = today['High'] - today['Low']
    h_pc = np.abs(today['High'] - today['Close'].shift(1))
    l_pc = np.abs(today['Low'] - today['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    today['atr'] = tr.rolling(14).mean()
    
    live_day = today.groupby(today.index.date).last().iloc[-1]
    day_candles = today[today.index.date == live_day.name]
    
    open_1530 = day_candles.iloc[0]['Open']
    price_1600 = day_candles.iloc[0]['Close']
    vol_1600 = day_candles.iloc[0]['Volume']
    range_1600 = day_candles.iloc[0]['High'] - day_candles.iloc[0]['Low']
    
    live_X = pd.DataFrame([{
        'vix_prev': df['vix_prev'].iloc[-1],
        'dax_move': df['dax_move'].iloc[-1],
        'spy_move': df['spy_move'].iloc[-1],
        'nikkei_move': df['nikkei_move'].iloc[-1],
        'us10y_move': df['us10y_move'].iloc[-1],
        'rut_move': df['rut_move'].iloc[-1],
        'qqq_open_move': ((price_1600 - open_1530) / open_1530) * 100,
        'vol_ratio': vol_1600 / df['vol_1530'].tail(5).mean(),
        'smh_move': df['smh_move'].iloc[-1],
        'rsi_1600': live_day['rsi'],
        'atr_ratio': range_1600 / live_day['atr']
    }])
    
    prob = model.predict_proba(live_X)[0, 1] * 100
    atr_val = live_X['atr_ratio'].iloc[0]
    
    signal = 0
    # message for discord
    msg = f"QQQ Trading Bot {live_day.name}\n"
    msg += f"Model Probability: {prob:.1f}%\n"
    msg += f"First 30m Return: {live_X['qqq_open_move'].iloc[0]:.2f}%\n"
    
    if atr_val < 0.60:
        msg += "Signal blocked by volatility. Action: NO TRADE"
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
        "date": str(live_day.name),
        "signal": signal,
        "entry_price": float(price_1600)
    }
    with open("today_trade.json", "w") as f:
        json.dump(trade_info, f)

if __name__ == "__main__":
    run()