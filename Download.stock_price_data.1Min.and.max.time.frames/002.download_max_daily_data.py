
import yfinance as yf
import pandas as pd
import sys
import os
from datetime import date

# Get the input file name from the command line argument
input_file = sys.argv[1]

# Read the list of ticker symbols from the input file
tickers = pd.read_csv(input_file)['Ticker']

# Initialize a counter for successful downloads
successful_downloads = 0
max_downloads = 10000

# Process each ticker symbol
for ticker in tickers:
    # Check if the maximum number of downloads has been reached
    if successful_downloads >= max_downloads:
        print("Reached the maximum number of downloads. Exiting...")
        break

    # Convert ticker to lowercase
    ticker = ticker.lower()

    # Create the directory path
    dir_path = f'data/{date.today()}/{ticker}'

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Construct the file path for company info
    info_file_path = f'{dir_path}/{ticker}.info.csv'
    
    # Construct the file path for daily data
    daily_data_file_path = f'{dir_path}/{ticker}.daily_data.csv'

    # Check if the info file already exists
    if not os.path.exists(info_file_path):
        try:
            # Get the ticker object
            ticker_obj = yf.Ticker(ticker)

            # Get the company info
            info = ticker_obj.info

            # Convert the info dictionary to a DataFrame
            data = pd.DataFrame(list(info.items()), columns=['Attribute', 'Value'])

            # Save the data to a CSV file
            data.to_csv(info_file_path, index=False)

            print(f"Saved company info for {ticker} to {info_file_path}")

        except Exception as e:
            print(f"Failed to download company info for {ticker}: {e}")

    # Check if the daily data file already exists
    if not os.path.exists(daily_data_file_path):
        try:
            # Download the maximum available daily historical data
            daily_data = ticker_obj.history(period="max", interval="1d")

            # Save the daily data to a CSV file
            daily_data.to_csv(daily_data_file_path)

            # Increment the counter
            successful_downloads += 1

            print(f"Saved daily data for {ticker} to {daily_data_file_path} (Download count: {successful_downloads})")

        except Exception as e:
            print(f"Failed to download daily data for {ticker}: {e}")

    else:
        print(f"Files for {ticker} already exist. Skipping...")

print("Script completed.")
