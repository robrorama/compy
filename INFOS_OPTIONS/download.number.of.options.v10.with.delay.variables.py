import yfinance as yf
import csv
import time
import sys
from datetime import date, datetime

delay = 5
tickers_per_batch = 60  # Variable for tickers to process per batch

# Function to get the number of option expiration dates for a given ticker
def get_number_of_expirations(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        expiration_dates = ticker.options
        return len(expiration_dates)
    except Exception as e:
        print(f"Error retrieving data for {ticker_symbol}: {e}")
        return 0

# Get the input file name from the command line
if len(sys.argv) != 2:
    print("Usage: python script.py <tickers_file>")
    sys.exit(1)

input_file = sys.argv[1]

# Generate output file name with today's date
today = date.today()
output_file = f"merged.outputs.{today.strftime('%Y%m%d')}.csv"

# Read existing results from output CSV file (if it exists)
existing_results = set()
try:
    with open(output_file, mode='r') as outfile:
        reader = csv.reader(outfile)
        next(reader, None)  # Skip header row if it exists
        for row in reader:
            existing_results.add(row[0])
except FileNotFoundError:
    pass

# Prepare header for output if file is new
if not existing_results:
    output_header = ['Ticker', 'Existing Column 2', 'Number of Expirations']
else:
    output_header = None

# Process tickers and their data
results = []
with open(input_file, mode='r') as infile:
    reader = csv.reader(infile)
    for i, row in enumerate(reader):
        ticker = row[0]
        if ticker in existing_results:
            print(f"{ticker} already processed, skipping...")
            continue

        number_of_expirations = get_number_of_expirations(ticker)
        row.append(str(number_of_expirations))  # Append Number of Expirations
        print(f"{ticker}: {number_of_expirations} expiration dates")

        time.sleep(1.0)  # Sleep after each ticker download

        results.append(row)

        # Write results to output CSV file after every 'tickers_per_batch' tickers
        if (i + 1) % tickers_per_batch == 0:
            try:
                print("Results to be written:")
                print(results)

                with open(output_file, mode='a', newline='') as outfile:
                    writer = csv.writer(outfile)
                    if output_header and outfile.tell() == 0:  # Only write header if file is new
                        writer.writerow(output_header)
                    writer.writerows(results)
                print(f"Results successfully saved to {output_file}")
                results = []  # Clear the results list
            except Exception as e:
                print(f"Error writing to file: {e}")  # Log the error
                # Consider continuing processing instead of sys.exit(1)

            # Sleep for 10 seconds before continuing
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sleeping for {delay} seconds...")
            time.sleep(delay)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Sleep over. Continuing with the next ticker...")

# Write any remaining results to output CSV file
if results:
    try:
        print("Results to be written:")
        print(results)

        with open(output_file, mode='a', newline='') as outfile:
            writer = csv.writer(outfile)
            if output_header and outfile.tell() == 0:  # Only write header if file is new
                writer.writerow(output_header)
            writer.writerows(results)
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")  # Log the error 
        # Consider continuing processing instead of sys.exit(1)
