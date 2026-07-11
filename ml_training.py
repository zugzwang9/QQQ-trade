import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

try:
    df = pd.read_csv('trading_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
except FileNotFoundError:
    print("Error: trading_data.csv not found. Run data.py first.")
    exit()

df['Target'] = np.where(df['QQQ_1600_to_2200_Return'] > 0, 1, 0)

features = [
    'VIX_Close_Yesterday', 'DAX_Return_Today', 'SP_Futures_Return_Today', 
    'Nikkei_Return_Today', 'TNX_Return_Today', 'RUT_Return_Today',
    'QQQ_First_30Min_Return', 'QQQ_First_30Min_Vol_Ratio'
]

x = df[features]
y = df['Target']

split = int(len(df) * 0.7)
x_train, x_test = x.iloc[:split], x.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

probs = model.predict_proba(x_test)[:, 1]

res = pd.DataFrame(index=x_test.index)
res['Actual_Outcome'] = y_test
res['Prob_Up'] = probs * 100
res['Signal'] = 0 

res.loc[res['Prob_Up'] > 55, 'Signal'] = 1
res.loc[res['Prob_Up'] < 45, 'Signal'] = -1

end_date = datetime.now()
start_date = end_date - timedelta(days=60)
qqq_hourly = yf.download('QQQ', start=start_date, end=end_date, interval='60m', progress=False)
if isinstance(qqq_hourly.columns, pd.MultiIndex):
    qqq_hourly.columns = qqq_hourly.columns.get_level_values(0)
qqq_hourly['Date'] = qqq_hourly.index.date

TRAILING_PCT = 0.5
res_returns = []

for date_idx, row in res.iterrows():
    signal = row['Signal']
    current_date = date_idx.date()
    
    if signal == 0:
        res_returns.append(0.0)
        continue
        
    day_candles = qqq_hourly[qqq_hourly['Date'] == current_date]
    
    if len(day_candles) < 2:
        res_returns.append(df.loc[date_idx, 'QQQ_1600_to_2200_Return'] if signal == 1 else -df.loc[date_idx, 'QQQ_1600_to_2200_Return'])
        continue
        
    trade_candles = day_candles.iloc[1:]
    entry_price = day_candles.iloc[0]['Close']
    
    stopped_out = False
    final_return = 0.0
    
    if signal == 1:
        highest_price = entry_price
        stop_loss = entry_price * (1 - TRAILING_PCT / 100)
        
        for _, candle in trade_candles.iterrows():
            if candle['Low'] <= stop_loss:
                stopped_out = True
                final_return = ((stop_loss - entry_price) / entry_price) * 100
                break

            if candle['High'] > highest_price:
                highest_price = candle['High']
                stop_loss = highest_price * (1 - TRAILING_PCT / 100)
                
        if not stopped_out:
            final_return = ((day_candles.iloc[-1]['Close'] - entry_price) / entry_price) * 100
            
    elif signal == -1:
        lowest_price = entry_price
        stop_loss = entry_price * (1 + TRAILING_PCT / 100)
        
        for _, candle in trade_candles.iterrows():

            if candle['High'] >= stop_loss:
                stopped_out = True
                final_return = ((entry_price - stop_loss) / entry_price) * 100
                break

            if candle['Low'] < lowest_price:
                lowest_price = candle['Low']
                stop_loss = lowest_price * (1 + TRAILING_PCT / 100)
                
        if not stopped_out:
            final_return = ((entry_price - day_candles.iloc[-1]['Close']) / entry_price) * 100

    res_returns.append(final_return)

res['Res_Return'] = res_returns
res['Actual_Return'] = df.loc[x_test.index, 'QQQ_1600_to_2200_Return']
#

capital = 100.0
res['Balance'] = capital * (1 + res['Res_Return'] / 100).cumprod()

trades = res[res['Signal'] != 0]
print(f"Test days: {len(res)}")
print(f"Trades taken: {len(trades)}")

if len(trades) > 0:
    correct = res['Res_Return'] > 0
    print(f"Trade Accuracy: {correct.mean() * 100:.2f}%")

print("\nRecent predictions:")
print(res[['Actual_Outcome', 'Actual_Return', 'Prob_Up', 'Signal', 'Balance']].tail(10))
print(f"\nFinal Balance: {res['Balance'].iloc[-1]:.2f} USD")

wins = res[res['Res_Return'] > 0]['Res_Return']
losses = res[res['Res_Return'] < 0]['Res_Return']

if len(wins) > 0 and len(losses) > 0:
    print(f"\nAvg Win: +{wins.mean():.2f}%")
    print(f"Avg Loss: {losses.mean():.2f}%")
    if abs(losses.sum()) > 0:
        pf = wins.sum() / abs(losses.sum())
        print(f"Profit Factor: {pf:.2f}")