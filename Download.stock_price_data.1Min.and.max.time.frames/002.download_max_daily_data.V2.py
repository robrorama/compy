import yfinance as yf
import pandas as pd
import sys
import os
from datetime import date

def main():
    # Ensure the script is called with the correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.csv>")
        sys.exit(1)
    
    # Get the input file name from the command line argument
    input_file = sys.argv[1]
    
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Read the list of ticker symbols from the input file
        tickers_df = pd.read_csv(input_file)
        
        # Ensure the 'Ticker' column exists
        if 'Ticker' not in tickers_df.columns:
            print("Error: The input CSV must contain a 'Ticker' column.")
            sys.exit(1)
        
        # Extract unique, non-null ticker symbols
        tickers = tickers_df['Ticker'].dropna().unique()
    except Exception as e:
        print(f"Error reading the input file: {e}")
        sys.exit(1)
    
    # Initialize a counter for successful downloads
    successful_downloads = 0
    max_downloads = 10000  # Adjust as needed
    
    # Process each ticker symbol
    for ticker in tickers:
        # Check if the maximum number of downloads has been reached
        if successful_downloads >= max_downloads:
            print("Reached the maximum number of downloads. Exiting...")
            break

        # Convert ticker to uppercase (yfinance typically uses uppercase tickers)
        ticker = str(ticker).upper().strip()

        # Create the directory path
        dir_path = os.path.join('data', str(date.today()), ticker)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Construct the file paths
        info_file_path = os.path.join(dir_path, f"{ticker}.info.csv")
        daily_data_file_path = os.path.join(dir_path, f"{ticker}.daily_data.csv")

        # Initialize the ticker object outside the conditional blocks
        ticker_obj = yf.Ticker(ticker)

        # Download and save company info if not already saved
        if not os.path.exists(info_file_path):
            try:
                # Get the company info
                info = ticker_obj.info

                # Check if info is not empty
                if not info:
                    print(f"No company info available for {ticker}.")
                else:
                    # Convert the info dictionary to a DataFrame
                    info_df = pd.DataFrame(list(info.items()), columns=['Attribute', 'Value'])

                    # Save the data to a CSV file
                    info_df.to_csv(info_file_path, index=False)

                    print(f"Saved company info for {ticker} to {info_file_path}")
            except Exception as e:
                print(f"Failed to download company info for {ticker}: {e}")

        # Download and save daily data if not already saved
        if not os.path.exists(daily_data_file_path):
            try:
                # Download the maximum available daily historical data
                daily_data = ticker_obj.history(period="max", interval="1d")

                # Check if daily data is not empty
                if daily_data.empty:
                    print(f"No historical daily data available for {ticker}.")
                else:
                    # Save the daily data to a CSV file
                    daily_data.to_csv(daily_data_file_path)

                    # Increment the counter
                    successful_downloads += 1

                    print(f"Saved daily data for {ticker} to {daily_data_file_path} (Download count: {successful_downloads})")
            except Exception as e:
                print(f"Failed to download daily data for {ticker}: {e}")
        else:
            print(f"Daily data for {ticker} already exists. Skipping...")

    print("Script completed.")

if __name__ == "__main__":
    main()
