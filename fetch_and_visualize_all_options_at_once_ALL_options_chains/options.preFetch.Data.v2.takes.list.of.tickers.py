import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import time

def get_all_expiry_dates(ticker):
    stock = yf.Ticker(ticker)
    return np.array(stock.options)  # Convert to NumPy array for easier handling

def download_options_data(ticker, date):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"options_data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/options.{date}.{ticker}.csv"

    # Check if the data already exists
    if os.path.exists(data_file):
        print(f"Options data for {ticker} on {date} already exists. Skipping download.")
        df = pd.read_csv(data_file)
    else:
        try:
            # Download options data
            stock = yf.Ticker(ticker)
            options = stock.option_chain(date)
            df = pd.concat([options.calls, options.puts])
            df.to_csv(data_file)
            print(f"Options data for {ticker} on {date} downloaded and saved to disk.")
        except Exception as e:
            print(f"Error downloading options data for {ticker} on {date}: {e}")
            return None  # Return None in case of an error

    return df

def download_all_options_data(ticker):
    expiry_dates = get_all_expiry_dates(ticker)
    all_data = {}
    for date in expiry_dates:
        data = download_options_data(ticker, date)
        if data is not None:
            all_data[date] = data
    return all_data

def download_options_data_for_tickers(tickers, delay):
    for ticker in tickers:
        try:
            ticker = ticker.strip().upper()
            print(f"Downloading options data for {ticker}...")
            download_all_options_data(ticker)
            time.sleep(delay)  # Introduce a delay between downloads
        except Exception as e:
            print(f"Error downloading options data for {ticker}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <tickers_file>")
        sys.exit(1)

    # Read the CSV file into a DataFrame
    csv_file_path = sys.argv[1]
    tickers_df = pd.read_csv(csv_file_path)
    
    # Extract the list of ticker symbols
    tickers = tickers_df['Ticker'].tolist()
    
    # Set the delay (in seconds) between downloads
    delay = 0.10
    
    # Download options data for all tickers
    download_options_data_for_tickers(tickers, delay)
    print("Options data for all tickers has been downloaded.")

if __name__ == "__main__":
    main()

