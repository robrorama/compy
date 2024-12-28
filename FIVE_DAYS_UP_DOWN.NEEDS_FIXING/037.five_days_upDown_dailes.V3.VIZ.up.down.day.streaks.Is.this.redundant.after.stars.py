import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
import numpy as np

def download_data(ticker, period):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_consecutive_trends(df):
    df['Trend'] = np.sign(df['Close'].diff())
    df['Consecutive'] = df['Trend'].groupby((df['Trend'] != df['Trend'].shift()).cumsum()).cumcount() + 1
    df['Consecutive'] *= df['Trend']
    df['Pct_Change'] = df['Close'].pct_change() * 100
    return df

def plot_data_with_subplots(df, title, ticker):
    # Define row heights where the first two rows are half the size of the last two
    row_heights = [.1, .5, 1, 1]

    # Create subplots with 4 rows and custom row heights
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                         subplot_titles=(title, 'Market Data', 'Consecutive Days Up/Down', 'Daily Percentage Change'),
                         row_heights=row_heights)

    # Candlestick plot
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Data'), row=1, col=1)

    # Consecutive trends subplot (new subplot)
    positive_consecutive = df['Consecutive'].apply(lambda x: x if x > 0 else 0)
    negative_consecutive = df['Consecutive'].apply(lambda x: x if x < 0 else 0)
    fig.add_trace(go.Bar(x=df.index, y=positive_consecutive, marker_color='green', name='Consecutive Up'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=negative_consecutive, marker_color='red', name='Consecutive Down'), row=3, col=1)

    # Percentage change subplot
    fig.add_trace(go.Bar(x=df.index, y=df['Pct_Change'], marker_color=df['Pct_Change'].apply(lambda x: 'blue' if x > 0 else 'orange'), name='Percentage Change'), row=4, col=1)

    # Update layout
    # Adjust the height to accommodate the custom row heights
    fig.update_layout(height=1800, showlegend=False)

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_consecutive_days_and_daily_moves.png')
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"Plot saved as {image_file}")

    # Save a copy of the image to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_consecutive_days_and_daily_moves.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=2)
    print(f"Plot saved as {pngs_file}")

    # Show the figure
    fig.show()

# Example usage:
if len(sys.argv) < 2:
    print("Usage: python script.py <ticker_symbol> [period]")
    sys.exit(1) 

ticker = sys.argv[1]

# Default to 'max' if period not specified
if len(sys.argv) >= 3:
    period = sys.argv[2]
else:
    period = "max"
    print("No period specified, using 'max' as the default.")

ticker_data = download_data(ticker, period)
ticker_data = calculate_consecutive_trends(ticker_data)

# Plot data with subplots
plot_data_with_subplots(ticker_data, title=f"{ticker} Stock", ticker=ticker)

