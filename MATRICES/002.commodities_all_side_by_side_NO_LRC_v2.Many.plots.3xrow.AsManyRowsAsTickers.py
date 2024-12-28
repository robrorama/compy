import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.gridspec as gridspec
import sys

def download_data(ticker, period):
    today_str = datetime.today().strftime("%Y-%m-%d")
    ticker = ticker.lower()
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def normalize_data(df):
    return (df["Close"] - df["Close"].mean()) / df["Close"].std()

def plot_commodities(tickers, period="1y"):
    n_commodities = len(tickers)
    n_cols = 4
    n_rows = (n_commodities + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    for i, ticker in enumerate(tickers):
        name = ticker
        df = download_data(ticker, period)
        norm_data = normalize_data(df)

        ax = fig.add_subplot(gs[i])
        ax.plot(norm_data, label=name)
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()

    # Save high-resolution image
    image_dir = f"images/commodities"
    os.makedirs(image_dir, exist_ok=True)
    
    image_file = os.path.join(image_dir, 'commodities_normalized.png')
    fig.savefig(image_file, dpi=300)
    
    print(f"High-resolution image saved to {image_file}")

    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python ip_commodities_all_side_by_side.NO_LRC.v1.py <tickers_file>")
        sys.exit(1)

    # Read the CSV file into a DataFrame
    csv_file_path = sys.argv[1]
    tickers_df = pd.read_csv(csv_file_path)
    
    # Extract the list of ticker symbols and convert to lowercase
    tickers = tickers_df['Ticker'].str.lower().tolist()

    plot_commodities(tickers)

if __name__ == '__main__':
    main()

