import os
import sys
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import plotly.subplots
from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator
from datetime import datetime, timedelta
#import imageio
import imageio.v2 as imageio


def download_stock_data(ticker, period="max"):
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

def add_technical_indicators(df):
    for span in [50, 100, 150, 200, 300]:
        df[f'MA{span}'] = df['Volume'].rolling(window=span).mean()

    macd = MACD(close=df['Volume'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    stoch = StochasticOscillator(high=df['Volume'], close=df['Volume'], low=df['Volume'])
    df['Stoch'] = stoch.stoch()
    df['Stoch_Signal'] = stoch.stoch_signal()

    rsi = RSIIndicator(close=df['Volume'])
    df['RSI'] = rsi.rsi()

    moving_average = df['Volume'].rolling(window=20).mean()
    moving_std_dev = df['Volume'].rolling(window=20).std()
    df['Bollinger_High'] = moving_average + (moving_std_dev * 2)
    df['Bollinger_Low'] = moving_average - (moving_std_dev * 2)

    return df

def plot_data(df, ticker, period_name, start_date):
    if df.empty or len(df) < 50:
        print(f"Not enough data for {period_name}. Skipping.")
        return None

    fig = plotly.subplots.make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("Volume Data", "MACD", "Volume with MA50", "Stochastic", "RSI"),
        row_heights=[0.6, 0.1, 0.1, 0.1, 0.1]
    )

    fig.add_trace(go.Scatter(x=df.index, y=df['Volume'], line=dict(color='blue', width=1), name='Volume'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_High'], line=dict(color='black', width=1), name='Bollinger High'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Low'], line=dict(color='black', width=1), name='Bollinger Low'), row=1, col=1)

    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='red', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='blue', width=2)), row=2, col=1)

    colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='orange', width=2), name='MA50'), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch'], line=dict(color='black', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_Signal'], line=dict(color='blue', width=1)), row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1)), row=5, col=1)

    fig.update_layout(height=1500, width=1200, showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="MACD", showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_yaxes(title_text="Stochastic", row=4, col=1)
    fig.update_yaxes(title_text="RSI", row=5, col=1)

    title = f'{ticker} Volume Analysis from {start_date.strftime("%Y-%m-%d")} to {datetime.today().strftime("%Y-%m-%d")}'
    fig.update_layout(title=title)

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_{period_name.replace(" ", "_").lower()}_volume.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)

    return image_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    try:
        data = download_stock_data(ticker)
    except ValueError as e:
        print(e)
        sys.exit(1)

    data = add_technical_indicators(data)
    
    date_ranges = {
        'Last 1 day': datetime.today() - timedelta(days=1),
        'Last 7 days': datetime.today() - timedelta(days=7),
        'Last 1 Month': datetime.today() - timedelta(days=30),
        'Last 3 Months': datetime.today() - timedelta(days=90),
        'Last 6 Months': datetime.today() - timedelta(days=180),
        'Last Year': datetime.today() - timedelta(days=365),
        'Last 2 Years': datetime.today() - timedelta(days=2*365),
        'Last 5 Years': datetime.today() - timedelta(days=5*365),
    }
    
    image_files = []
    for period_name, start_date in date_ranges.items():
        data_period = data[start_date:]
        image_file = plot_data(data_period, ticker, period_name, start_date)
        if image_file:
            image_files.append(image_file)
    
    if image_files:
        gif_path = f'images/{ticker}/{ticker}_volume.gif'
        with imageio.get_writer(gif_path, mode='I', duration=1, loop=0) as writer:
            for image_file in image_files:
                image = imageio.imread(image_file)
                writer.append_data(image)
        print(f"GIF created at {gif_path}")
    else:
        print("No images were created due to insufficient data.")

if __name__ == '__main__':
    main()
