import yfinance as yf
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import sys

def download_options_data(ticker, date):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/options.{date}.{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        print(f"Options data for {ticker} found on disk.")
    except FileNotFoundError:
        try:
            # Download options data
            stock = yf.Ticker(ticker)
            options = stock.option_chain(date)
            df = pd.concat([options.calls, options.puts])
            df.to_csv(data_file)
            print(f"Options data for {ticker} downloaded and saved to disk.")
        except Exception as e:
            print(f"Error downloading options data for {ticker}: {e}")
            return None  # Return None in case of an error
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <ticker_symbol> [weeks_ahead]")
        sys.exit(1) 

    ticker = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            myWeeks = int(sys.argv[2])
        except ValueError:
            print("Invalid weeks_ahead value. Please provide an integer.")
            sys.exit(1)
    else:
        myWeeks = 4  # Default to 4 weeks ahead if not specified

    target_date = datetime.today() + timedelta(weeks=myWeeks)

    # Find the nearest options date and get current stock price
    stock = yf.Ticker(ticker)
    options_dates = stock.options
    nearest_date = min(options_dates, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target_date))

    # Download data
    options_data = download_options_data(ticker, nearest_date)
    if options_data is None:
        sys.exit(1)  # Exit if there was an error downloading data

    # Split the data into calls and puts
    calls_data = options_data[options_data['contractSymbol'].str.contains('C')]
    puts_data = options_data[options_data['contractSymbol'].str.contains('P')]

    # Calculate put/call ratio for each strike price
    combined_data = pd.merge(calls_data[['strike', 'openInterest']], puts_data[['strike', 'openInterest']], on='strike', suffixes=('_call', '_put'))
    combined_data['put_call_ratio'] = combined_data['openInterest_put'] / combined_data['openInterest_call']
    combined_data = combined_data.dropna()

    # Plotting Put/Call Ratio
    fig = px.bar(combined_data, x='strike', y='put_call_ratio', title='Put/Call Ratio by Strike Price')
    fig.update_layout(xaxis_title='Strike Price', yaxis_title='Put/Call Ratio', plot_bgcolor='black', paper_bgcolor='black', font_color='white')

    # Save high-resolution image
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    
    image_file = os.path.join(image_dir, f'{ticker}_put_call_ratio_{nearest_date}_spikes.png')
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {image_file}")

    # Save a copy of the image to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_put_call_ratio_{nearest_date}_spikes.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {pngs_file}")

    # Show the plot in the browser
    fig.show()

if __name__ == "__main__":
    main()

