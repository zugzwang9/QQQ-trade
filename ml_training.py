import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

try:
    df = pd.read_csv('trading_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
except FileNotFoundError:
    print("CSV missing. Run data.py.")
    exit()

df['Target'] = np.where(df['qqq_trade_move'] > 0, 1, 0)

features = [
    'vix_prev', 'dax_move', 'spy_move', 'nikkei_move', 
    'us10y_move', 'rut_move', 'qqq_open_move', 'vol_ratio',
    'smh_move', 'rsi_1600', 'atr_ratio'
]

X = df[features]
y = df['Target']

# split 70% train, 30% test
split_date = '2025-11-01'
X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]
y_train = y[X.index < split_date]
y_test = y[X.index >= split_date]

model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]

res = pd.DataFrame(index=X_test.index)
res['actual'] = y_test
res['prob_up'] = probs * 100
res['signal'] = 0 

for idx, row in X_test.iterrows():
    prob = res.loc[idx, 'prob_up']
    atr_val = row['atr_ratio']
    
    # Skip days with no market momentum, less than 60% of normal volatility
    if atr_val < 0.60:
        continue
        
    # confidence levels trading
    if prob > 56.0:
        res.loc[idx, 'signal'] = 1
    elif prob < 44.0:
        res.loc[idx, 'signal'] = -1

qqq_hour = yf.download('QQQ', start=X_test.index.min() - timedelta(days=5), end=X_test.index.max() + timedelta(days=5), interval='60m', progress=False)
if isinstance(qqq_hour.columns, pd.MultiIndex):
    qqq_hour.columns = qqq_hour.columns.get_level_values(0)
qqq_hour['Date'] = qqq_hour.index.date

trail_pct = 0.25
trade_returns = []

for idx, row in res.iterrows():
    sig = row['signal']
    if sig == 0:
        trade_returns.append(0.0)
        continue
        
    day_candles = qqq_hour[qqq_hour['Date'] == idx.date()]
    if len(day_candles) < 2:
        trade_returns.append(0.0)
        continue
        
    candles = day_candles.iloc[1:]
    entry = day_candles.iloc[0]['Close']
    stopped = False
    ret = 0.0
    
    if sig == 1:
        peak = entry
        stop = entry * (1 - trail_pct / 100)
        for _, candle in candles.iterrows():
            if candle['Low'] <= stop:
                stopped = True
                ret = ((stop - entry) / entry) * 100
                break
            if candle['High'] > peak:
                peak = candle['High']
                stop = peak * (1 - trail_pct / 100)
        if not stopped:
            ret = ((day_candles.iloc[-1]['Close'] - entry) / entry) * 100
            
    elif sig == -1:
        floor = entry
        stop = entry * (1 + trail_pct / 100)
        for _, candle in candles.iterrows():
            if candle['High'] >= stop:
                stopped = True
                ret = ((entry - stop) / entry) * 100
                break
            if candle['Low'] < floor:
                floor = candle['Low']
                stop = floor * (1 + trail_pct / 100)
        if not stopped:
            ret = ((entry - day_candles.iloc[-1]['Close']) / entry) * 100

    trade_returns.append(float(ret))

res['ret'] = trade_returns
res['actual_return'] = df.loc[X_test.index, 'qqq_trade_move']
res['balance'] = 100.0 * (1 + res['ret'] / 100).cumprod()

trades = res[res['signal'] != 0]
print(f"Test Days: {len(res)} | Trades Taken: {len(trades)}")

if len(trades) > 0:
    win_rate = (trades['ret'] > 0).mean() * 100
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Balance: {res['balance'].iloc[-1]:.2f} USD")
    
    wins = trades[trades['ret'] > 0]['ret']
    losses = trades[trades['ret'] < 0]['ret']
    if len(wins) > 0 and len(losses) > 0:
        pf = wins.sum() / abs(losses.sum())
        print(f"Profit Factor: {pf:.2f}")

# print last 8 days
print("\nLast 8 trading days:")
print(res[['actual', 'actual_return', 'prob_up', 'signal', 'balance']].tail(8))