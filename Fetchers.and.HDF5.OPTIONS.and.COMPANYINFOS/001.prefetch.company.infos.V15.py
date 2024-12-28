import yfinance as yf
import pandas as pd
import sys
import os
from datetime import date
import time

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

    # Print the ticker name before the test
    print(f"Checking {ticker}...")

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Get the ticker object
        ticker_obj = yf.Ticker(ticker)

        # Get the company info
        info = ticker_obj.info

        # Create a Pandas DataFrame to store the data
        data = pd.DataFrame.from_dict(info, orient='index').reset_index()

        # Transpose the DataFrame
        data = data.transpose()

        # Save the data to a CSV file
        data.to_csv(file_path, index=False, header=False)

        # Read the CSV file back into a Pandas DataFrame, using the second row as the column headers
        data_read = pd.read_csv(file_path, header=1)

        # Display the data in a table
        print(data_read)

        # Display all the information in the DataFrame
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data_read)

        # Print the ticker name after the download
        print(f"Downloaded {ticker} successfully!")

        # Pause for half a second
        time.sleep(0.5)
    else:
        print(f"File {file_path} already exists. Skipping {ticker}...")

