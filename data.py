import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_data(days_back=60):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    qqq_60m = yf.download('QQQ', start=start_date, end=end_date, interval='60m')
    
    if isinstance(qqq_60m.columns, pd.MultiIndex):
        qqq_60m.columns = qqq_60m.columns.get_level_values(0)
        
    qqq_60m['Time'] = qqq_60m.index.time
    qqq_60m['Date'] = qqq_60m.index.date
    
    days = qqq_60m['Date'].unique()
    daily_rows = []
    
    for d in days:
        day_data = qqq_60m[qqq_60m['Date'] == d]
        try:
            open_1530 = day_data.iloc[0]['Open']
            price_1600 = day_data.iloc[0]['Close'] 
            close_2200 = day_data.iloc[-1]['Close'] 
            vol_first_hour = day_data.iloc[0]['Volume']
            
            trading_period = day_data.iloc[1:]
            max_during_trade = trading_period['High'].max()
            min_during_trade = trading_period['Low'].min()
            
            daily_rows.append({
                'Date': d,
                'QQQ_Open_1530': float(open_1530),
                'QQQ_Price_1600': float(price_1600),
                'QQQ_Close_2200': float(close_2200),
                'QQQ_Vol_First_30Min': float(vol_first_hour),
                'QQQ_Max_After_1600': float(max_during_trade),
                'QQQ_Min_After_1600': float(min_during_trade)
            })
        except Exception:
            continue
            
    df = pd.DataFrame(daily_rows)
    if df.empty:
        return df
        
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    df['QQQ_First_30Min_Return'] = ((df['QQQ_Price_1600'] - df['QQQ_Open_1530']) / df['QQQ_Open_1530']) * 100
    df['QQQ_1600_to_2200_Return'] = ((df['QQQ_Close_2200'] - df['QQQ_Price_1600']) / df['QQQ_Price_1600']) * 100
    
    vol_sma5 = df['QQQ_Vol_First_30Min'].rolling(window=5).mean()
    df['QQQ_First_30Min_Vol_Ratio'] = df['QQQ_Vol_First_30Min'] / vol_sma5

    macro_tickers = {
        'SP_Futures': 'ES=F', 'DAX': '^GDAXI', 'VIX': '^VIX', 
        'Nikkei': '^N225', 'US10Y': '^TNX', 'Russell2000': '^RUT'
    }
    
    macro_df = pd.DataFrame(index=df.index)
    
    for name, ticker in macro_tickers.items():
        ticker_data = yf.download(ticker, start=start_date - timedelta(days=5), end=end_date)
        if isinstance(ticker_data.columns, pd.MultiIndex):
            ticker_data.columns = ticker_data.columns.get_level_values(0)
            
        if name == 'VIX':
            macro_df['VIX_Close_Yesterday'] = ticker_data['Close'].shift(1)
        elif name == 'DAX':
            macro_df['DAX_Return_Today'] = ((ticker_data['Open'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)) * 100
        elif name == 'SP_Futures':
            macro_df['SP_Futures_Return_Today'] = ((ticker_data['Open'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)) * 100
        elif name == 'Nikkei':
            macro_df['Nikkei_Return_Today'] = ((ticker_data['Close'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)) * 100
        elif name == 'US10Y':
            macro_df['TNX_Return_Today'] = ((ticker_data['Open'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)) * 100
        elif name == 'Russell2000':
            macro_df['RUT_Return_Today'] = ((ticker_data['Open'] - ticker_data['Close'].shift(1)) / ticker_data['Close'].shift(1)) * 100

    df = df.join(macro_df)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    print(f"Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    dataset = fetch_data(days_back=60)
    if not dataset.empty:
        dataset.to_csv("trading_data.csv")
        print("Data saved to trading_data.csv")