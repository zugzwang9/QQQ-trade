import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    df = pd.read_csv('trading_data.csv', index_col=0)
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

# predict probability
probs = model.predict_proba(x_test)[:, 1]

#results
res = pd.DataFrame(index=x_test.index)
res['Actual_Outcome'] = y_test
res['Prob_Up'] = probs * 100
res['Signal'] = 0 

res.loc[res['Prob_Up'] > 55, 'Signal'] = 1
res.loc[res['Prob_Up'] < 45, 'Signal'] = -1

trades = res[res['Signal'] != 0]

print(f"Test days: {len(res)}")
print(f"Trades taken: {len(trades)}")

if len(trades) > 0:
    correct = ((trades['Signal'] == 1) & (trades['Actual_Outcome'] == 1)) | \
              ((trades['Signal'] == -1) & (trades['Actual_Outcome'] == 0))
    print(f"Trade Accuracy: {correct.mean() * 100:.2f}%")

# backtest performance
res['Actual_Return'] = df.loc[x_test.index, 'QQQ_1600_to_2200_Return']
res['Res_Return'] = res['Signal'] * res['Actual_Return']

capital = 100.0
res['Balance'] = capital * (1 + res['Res_Return'] / 100).cumprod()

print("\nRecent predictions:")
print(res[['Actual_Outcome', 'Actual_Return', 'Prob_Up', 'Signal', 'Balance']].tail(10))
print(f"\nFinal Balance: {res['Balance'].iloc[-1]:.2f} USD")

# risk reward
wins = res[res['Res_Return'] > 0]['Res_Return']
losses = res[res['Res_Return'] < 0]['Res_Return']

if len(wins) > 0 and len(losses) > 0:
    print(f"\nAvg Win: +{wins.mean():.2f}%")
    print(f"Avg Loss: {losses.mean():.2f}%")
    if abs(losses.sum()) > 0:
        pf = wins.sum() / abs(losses.sum())
        print(f"Profit Factor: {pf:.2f}")