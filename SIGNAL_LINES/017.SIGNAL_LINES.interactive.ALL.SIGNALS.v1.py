import os
import sys
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import talib

def download_stock_data(ticker, period="5y"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_indicators(df):
    for span in [50, 100, 150, 200, 300]:
        df[f'EMA{span}'] = df['Close'].ewm(span=span, adjust=False).mean()

    upperband, middleband, lowerband = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['upperband'] = upperband
    df['middleband'] = middleband
    df['lowerband'] = lowerband

    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['STDDEV'] = talib.STDDEV(df['Close'], timeperiod=5, nbdev=1)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CMO'] = talib.CMO(df['Close'], timeperiod=14)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['STOCH_k'], df['STOCH_d'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    window = 14
    df['LR_Slope'] = talib.LINEARREG_SLOPE(df['Close'], timeperiod=window)

    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    df['ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)

    return df

def plot_data(df, ticker, title, start_date=None):
    if start_date:
        df = df[df.index >= start_date]
    else:
        start_date = df.index[-2*365]  # Default to last 2 years
        df = df[df.index >= start_date]

    fig = make_subplots(rows=17, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('Bollinger Bands', 'ADX', 'STDDEV', 'ATR', 'CMO', 'ROC', 'Stochastic',
                                        'Williams %R', 'LR Slope', 'RSI', 'MACD', 'OBV', 'SAR', 'ULTOSC', 'MFI',
                                        'CCI', 'ICHIMOKU'),
                        row_heights=[0.2] + [0.05] * 16)

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['upperband'], mode='lines', name='Upper Band', line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['middleband'], mode='lines', name='Middle Band', line=dict(color='white', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['lowerband'], mode='lines', name='Lower Band', line=dict(color='red', width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX', line=dict(color='lawngreen', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STDDEV'], mode='lines', name='STDDEV', line=dict(color='magenta', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(color='blue', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['CMO'], mode='lines', name='CMO', line=dict(color='red', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC'], mode='lines', name='ROC', line=dict(color='lime', width=1)), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['WILLR'], mode='lines', name='Williams %R', line=dict(color='cyan', width=1)), row=7, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_k'], mode='lines', name='STOCH_k', line=dict(color='red', width=1)), row=8, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCH_d'], mode='lines', name='STOCH_d', line=dict(color='blue', width=1)), row=8, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['LR_Slope'], mode='lines', name='LR Slope', line=dict(color='yellow', width=1)), row=9, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='orange', width=1)), row=10, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='purple', width=1)), row=11, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', name='OBV', line=dict(color='pink', width=1)), row=12, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SAR'], mode='lines', name='SAR', line=dict(color='gold', width=1)), row=13, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ULTOSC'], mode='lines', name='ULTOSC', line=dict(color='lightblue', width=1)), row=14, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], mode='lines', name='MFI', line=dict(color='aqua', width=1)), row=15, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], mode='lines', name='CCI', line=dict(color='violet', width=1)), row=16, col=1)

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', template='plotly_dark', height=3000, showlegend=False)

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_important_signals.png')
    
    fig.write_image(image_file, width=1920, height=1080, scale=2)  # Increase resolution

    fig.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <TICKER> [TIME_FRAME]")
        sys.exit(1)
    
    ticker = sys.argv[1]
    time_frame = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        data = download_stock_data(ticker, period="5y")
    except ValueError as e:
        print(e)
        sys.exit(1)

    data = calculate_indicators(data)
    
    if time_frame:
        plot_data(data, ticker, title=f"{ticker} Stock Data with Technical Indicators", start_date=datetime.strptime(time_frame, "%Y-%m-%d"))
    else:
        plot_data(data, ticker, title=f"{ticker} Stock Data with Technical Indicators")

if __name__ == '__main__':
    main()

