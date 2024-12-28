import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys

def load_or_download_data(ticker, period='max'):
    today_str = datetime.today().strftime("%Y-%m-%d")
    ticker = ticker.lower()
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"Data for {ticker} loaded from disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def normalize_data(df):
    return (df["Close"] - df["Close"].mean()) / df["Close"].std()

def plot_comparison(ticker1, ticker2, period='1y'):
    df1 = load_or_download_data(ticker1, period)
    df2 = load_or_download_data(ticker2, period)
    
    norm_data1 = normalize_data(df1)
    norm_data2 = normalize_data(df2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=norm_data1.index, y=norm_data1, mode='lines', name=ticker1))
    fig.add_trace(go.Scatter(x=norm_data2.index, y=norm_data2, mode='lines', name=ticker2))

    fig.update_layout(
        title="Comparison of Commodity Prices",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        legend_title="Commodities",
        width=1200,
        height=800
    )

    # Save high-resolution image
    image_dir = "images/commodities"
    os.makedirs(image_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    image_file = os.path.join(image_dir, f'commodities_comparison_{today_str}.png')
    fig.write_image(image_file, scale=3)
    
    print(f"High-resolution image saved to {image_file}")

    fig.show()

def main():
    if len(sys.argv) != 4:
        print("Usage: python ip_interactive.compare2commodities.dropdownmenus.V2.py <ticker1> <ticker2> <period>")
        sys.exit(1)

    ticker1 = sys.argv[1].lower()
    ticker2 = sys.argv[2].lower()
    period = sys.argv[3]

    plot_comparison(ticker1, ticker2, period)

if __name__ == '__main__':
    main()

