import yfinance as yf
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import sys
import numpy as np

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

def plot_options_data(ticker, all_options_data):
    # Create a directory to save plots
    plot_dir = f"plots/{ticker}"
    os.makedirs(plot_dir, exist_ok=True)

    # Get current stock price
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="max")['Close'].iloc[-1]  # grabbing the last closing price

    # Prepare a combined DataFrame
    combined_data = pd.DataFrame()
    for i, (date, data) in enumerate(all_options_data.items()):
        data['expiry_date'] = pd.to_datetime(date)
        combined_data = pd.concat([combined_data, data])

    # Normalize the color index based on expiration dates
    min_date = combined_data['expiry_date'].min()
    max_date = combined_data['expiry_date'].max()
    combined_data['color_index'] = combined_data['expiry_date'].apply(lambda x: (x - min_date) / (max_date - min_date))

    # Define a color scale from blue to white to red
    color_scale = px.colors.diverging.RdBu_r

    # Plot all the data
    fig = px.scatter(
        combined_data,
        x="strike",
        y="openInterest",
        size="openInterest",
        color="color_index",
        color_continuous_scale=color_scale,
        opacity=0.8,
        title=f"{ticker} Options (Strike vs. Open Interest) - Current Price: ${current_price:.2f}",
        labels={"strike": "Strike Price", "openInterest": "Open Interest", "color_index": "Time to Expiration"},
    )

    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )

    # Save high-resolution images
    image_file = os.path.join(plot_dir, f'{ticker}_options_combined.png')
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {image_file}")

    # Save a copy of the image to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_options_combined.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {pngs_file}")

    # Show the combined plot in the browser
    fig.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <ticker_symbol>")
        sys.exit(1)

    ticker = sys.argv[1]

    # Download data for all expiration dates
    all_options_data = download_all_options_data(ticker)

    # Plot data for all expiration dates
    plot_options_data(ticker, all_options_data)

if __name__ == "__main__":
    main()

