import yfinance as yf
import pandas as pd
import sys
import os
from datetime import date

# Get the input file name from the command line argument
input_file = sys.argv[1]

# Read the list of ticker symbols from the input file
tickers = pd.read_csv(input_file)['Ticker']

# Process each ticker symbol
for ticker in tickers:
    # Convert ticker to lowercase
    ticker = ticker.lower()

    # Create the directory path
    dir_path = f'data/{date.today()}/{ticker}'

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Construct the file path
    file_path = f'{dir_path}/{ticker}.info.csv'

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Get the ticker object
        ticker_obj = yf.Ticker(ticker)

        # Get the company info
        info = ticker_obj.info

        # Convert the info dictionary to a DataFrame
        data = pd.DataFrame(list(info.items()), columns=['Attribute', 'Value'])

        # Save the data to a CSV file
        data.to_csv(file_path, index=False)

        # Display the saved data
        print(f"Saved data for {ticker} to {file_path}")

    else:
        print(f"File {file_path} already exists. Skipping...")

