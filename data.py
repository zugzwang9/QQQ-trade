import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def get_data(days=700):
    end = datetime.now()
    start = end - timedelta(days=days)

    print("Fetching QQQ data...")
    chunks = []
    curr_end = end
    
    #yfinance limit fix fetch in 30-day blocks
    while curr_end > start:
        curr_start = max(curr_end - timedelta(days=30), start)
        chunk = yf.download('QQQ', start=curr_start, end=curr_end, interval='60m', progress=False)
        if not chunk.empty:
            chunks.append(chunk)
        curr_end = curr_start
        time.sleep(0.2)

    df_raw = pd.concat(chunks).sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    #RSI
    diff = df_raw['Close'].diff()
    up = diff.where(diff > 0, 0).rolling(14).mean()
    down = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df_raw['rsi'] = 100 - (100 / (1 + up / down))
    
    #ATR
    h_l = df_raw['High'] - df_raw['Low']
    h_pc = np.abs(df_raw['High'] - df_raw['Close'].shift(1))
    l_pc = np.abs(df_raw['Low'] - df_raw['Close'].shift(1))
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    df_raw['atr'] = tr.rolling(14).mean()

    df_raw['Date'] = df_raw.index.date
    days_list = df_raw['Date'].unique()
    rows = []
    
    #daily price points
    for d in days_list:
        day = df_raw[df_raw['Date'] == d]
        if len(day) < 2:
            continue
            
        open_1530 = day.iloc[0]['Open']
        price_1600 = day.iloc[0]['Close'] 
        close_2200 = day.iloc[-1]['Close'] 
        vol_1530 = day.iloc[0]['Volume']
        
        rsi_1600 = day.iloc[0]['rsi']
        atr_1600 = day.iloc[0]['atr']
        range_1530 = day.iloc[0]['High'] - day.iloc[0]['Low']
        
        trade_window = day.iloc[1:]
        max_price = trade_window['High'].max()
        min_price = trade_window['Low'].min()
        
        rows.append({
            'Date': d,
            'open_1530': float(open_1530),
            'price_1600': float(price_1600),
            'close_2200': float(close_2200),
            'vol_1530': float(vol_1530),
            'max_after_1600': float(max_price),
            'min_after_1600': float(min_price),
            'rsi_1600': float(rsi_1600),
            'atr_1600': float(atr_1600),
            'range_1530': float(range_1530)
        })
            
    df = pd.DataFrame(rows).set_index('Date')
    df.index = pd.to_datetime(df.index)

    df['qqq_open_move'] = ((df['price_1600'] - df['open_1530']) / df['open_1530']) * 100
    df['qqq_trade_move'] = ((df['close_2200'] - df['price_1600']) / df['price_1600']) * 100
    df['vol_ratio'] = df['vol_1530'] / df['vol_1530'].rolling(5).mean()
    df['atr_ratio'] = df['range_1530'] / df['atr_1600']

    macros = {
        'spy': 'ES=F', 'dax': '^GDAXI', 'vix': '^VIX', 
        'nikkei': '^N225', 'us10y': '^TNX', 'rut': '^RUT', 'smh': 'SMH'
    }
    
    macro_df = pd.DataFrame(index=df.index)
    
    for name, ticker in macros.items():
        #SMH
        if name == 'smh':
            smh = yf.download('SMH', start=start, end=end, interval='60m', progress=False)
            if isinstance(smh.columns, pd.MultiIndex):
                smh.columns = smh.columns.get_level_values(0)
            smh['Date'] = smh.index.date
            
            smh_returns = {}
            for day_idx in smh['Date'].unique():
                day_smh = smh[smh['Date'] == day_idx]
                if not day_smh.empty:
                    ret = ((day_smh.iloc[0]['Close'] - day_smh.iloc[0]['Open']) / day_smh.iloc[0]['Open']) * 100
                    smh_returns[pd.to_datetime(day_idx)] = float(ret)
            macro_df['smh_move'] = pd.Series(smh_returns)
            continue

        raw = yf.download(ticker, start=start - timedelta(days=10), end=end, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.sort_index()
            
        if name == 'vix':
            macro_df['vix_prev'] = raw['Close'].shift(1)
        else:
            macro_df[f'{name}_move'] = ((raw['Open'] - raw['Close'].shift(1)) / raw['Close'].shift(1)) * 100

    df = df.join(macro_df).ffill().dropna()
    print(f"Data ready. Total days: {len(df)}")
    return df

if __name__ == "__main__":
    data = get_data()
    if not data.empty:
        data.to_csv("trading_data.csv")
        print("trading_data.csv saved.")