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
    # Existing indicators
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

    # New indicators
    df['DX'] = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(df['Close'])
    df['HT_SINE'] = talib.HT_SINE(df['Close'])[0]
    df['LINEARREG'] = talib.LINEARREG(df['Close'], timeperiod=14)
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['Close'], timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['Close'], timeperiod=14)
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['Close'], timeperiod=14)
    df['STOCHF_k'], df['STOCHF_d'] = talib.STOCHF(df['High'], df['Low'], df['Close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_k'], df['STOCHRSI_d'] = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TSF'] = talib.TSF(df['Close'], timeperiod=14)
    df['VAR'] = talib.VAR(df['Close'], timeperiod=5, nbdev=1)
    df['WCLPRICE'] = talib.WCLPRICE(df['High'], df['Low'], df['Close'])

    return df

def plot_data(df, ticker, title, start_date=None):
    if start_date:
        df = df[df.index >= start_date]
    else:
        start_date = df.index[-2*365]  # Default to last 2 years
        df = df[df.index >= start_date]

    fig = make_subplots(rows=9, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('TSF_and_WCLPRICE', 'VAR', 'DX', 'HT_PHASOR', 'HT_SINE', 'Linear Regression and Intercept',
                                        'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'STOCHF_and_STOCHRSI'))

    fig.add_trace(go.Scatter(x=df.index, y=df['TSF'], mode='lines', name='TSF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['WCLPRICE'], mode='lines', name='WCLPRICE'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VAR'], mode='lines', name='VAR'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['DX'], mode='lines', name='DX'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HT_PHASOR_inphase'], mode='lines', name='HT_PHASOR_inphase'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HT_PHASOR_quadrature'], mode='lines', name='HT_PHASOR_quadrature'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['HT_SINE'], mode='lines', name='HT_SINE'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG'], mode='lines', name='LINEARREG'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_INTERCEPT'], mode='lines', name='LINEARREG_INTERCEPT'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_ANGLE'], mode='lines', name='LINEARREG_ANGLE'), row=7, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['LINEARREG_SLOPE'], mode='lines', name='LINEARREG_SLOPE'), row=8, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCHF_k'], mode='lines', name='STOCHF_k'), row=9, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCHF_d'], mode='lines', name='STOCHF_d'), row=9, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCHRSI_k'], mode='lines', name='STOCHRSI_k'), row=9, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['STOCHRSI_d'], mode='lines', name='STOCHRSI_d'), row=9, col=1)

    fig.update_layout(title=title, template='plotly_dark', height=3000, showlegend=False)

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_favorite_signals.png')
    
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

