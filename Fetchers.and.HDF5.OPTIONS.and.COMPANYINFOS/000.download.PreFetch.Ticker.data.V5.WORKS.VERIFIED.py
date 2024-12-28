import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import os
import sys

def download_stock_data(tickers, delay):
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Iterate through each ticker
    for ticker in tickers:
        try:
            # Clean up the ticker symbol and convert to lowercase
            ticker = ticker.strip().lower()
            
            # Create directory for today's data for this ticker
            data_dir = f'data/{current_date}/{ticker}'
            os.makedirs(data_dir, exist_ok=True)
            
            # Define the file path for saving data
            file_path = f'{data_dir}/{ticker}.csv'
            
            # Check if the data already exists
            if os.path.exists(file_path):
                print(f'Data for {ticker} already exists. Skipping download.')
                continue
            
            # Download the data for the maximum available time frame
            data = yf.download(ticker, period="max")
            
            # Save the data to the local disk
            if not data.empty:
                data.to_csv(file_path)
                print(f'Data for {ticker} saved successfully.')
            else:
                print(f'No data found for {ticker}.')
            
            # Introduce a delay
            time.sleep(delay)
        except Exception as e:
            print(f'Error downloading data for {ticker}: {e}')

def main():
    if len(sys.argv) != 2:
        print("Usage: python download_stock_data.py <tickers_file>")
        sys.exit(1)
    
    # Read the CSV file into a DataFrame
    csv_file_path = sys.argv[1]
    tickers_df = pd.read_csv(csv_file_path)
    
    # Extract the list of ticker symbols and convert to lowercase
    tickers = tickers_df['Ticker'].str.lower().tolist()
    
    # Set the delay (in seconds) between downloads
    delay = 0.10
    
    # Download stock data
    download_stock_data(tickers, delay)

if __name__ == "__main__":
    main()

