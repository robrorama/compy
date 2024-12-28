import yfinance as yf
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import sys
import numpy as np

def get_next_expiry_date(ticker, weeks_ahead=1):
    stock = yf.Ticker(ticker)
    options_dates = np.array(stock.options)  # Convert to NumPy array for easier calculations
    today = datetime.today()

    # Calculate target date based on weeks ahead
    target_date = today + timedelta(weeks=weeks_ahead)

    # Find the index of the closest date
    idx = np.argmin(np.abs(pd.to_datetime(options_dates) - target_date))
    nearest_date = options_dates[idx]

    return nearest_date

def download_options_data(ticker, date):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"DATA/{today_str}/{ticker}"
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
            weeks_ahead = int(sys.argv[2])
        except ValueError:
            print("Invalid weeks_ahead value. Please provide an integer.")
            sys.exit(1)
    else:
        weeks_ahead = 1  # Default to 1 week ahead if not specified

    # Get the next options expiration date
    nearest_date = get_next_expiry_date(ticker, weeks_ahead)
    print("Next Options Expiration Date : ", nearest_date)

    # Get current stock price
    stock = yf.Ticker(ticker)
    current_price = stock.history(period="max")['Close'].iloc[-1]  # grabbing the last closing price

    # Download data and handle potential errors
    options_data = download_options_data(ticker, nearest_date)
    if options_data is None:
        sys.exit(1)  # Exit if there was an error downloading data

    # Split the data into calls and puts
    calls_data = options_data[options_data['contractSymbol'].str.contains('C')]
    puts_data = options_data[options_data['contractSymbol'].str.contains('P')]

    # Replace NaN values in 'volume' with a default size for both calls and puts
    default_size_calls = calls_data['volume'].dropna().min()
    default_size_puts = puts_data['volume'].dropna().min()
    calls_data['volume'].fillna(default_size_calls if pd.notna(default_size_calls) else 0, inplace=True)
    puts_data['volume'].fillna(default_size_puts if pd.notna(default_size_puts) else 0, inplace=True)

    # Custom function to create scatter plot
    def create_plot(data, color_scale, title):
        fig = px.scatter(data, x="strike", y="openInterest",
                         size="openInterest", color="openInterest",
                         opacity=0.8, color_continuous_scale=color_scale)

        fig.update_layout(title=f"{ticker} {title} (Strike vs. Open Interest) - Current Price: ${current_price:.2f}",
                          xaxis_title="Strike Price", yaxis_title="Open Interest",
                          xaxis_showgrid=False, yaxis_showgrid=False,
                          plot_bgcolor='black', paper_bgcolor='black',
                          font_color='white')
        return fig

    # Plotting Calls
    fig_calls = create_plot(calls_data, "Blues", "Call Options")
    
    # Plotting Puts
    fig_puts = create_plot(puts_data, "Reds", "Put Options")

    # Save high-resolution images
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    
    calls_image_file = os.path.join(image_dir, f'{ticker}_options_{nearest_date}_calls.png')
    puts_image_file = os.path.join(image_dir, f'{ticker}_options_{nearest_date}_puts.png')
    
    fig_calls.write_image(calls_image_file, width=1920, height=1080, scale=2)
    fig_puts.write_image(puts_image_file, width=1920, height=1080, scale=2)
    
    print(f"High-resolution images saved to {image_dir}")

    # Save a copy of the images to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    
    calls_pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_options_{nearest_date}_calls.png')
    puts_pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_options_{nearest_date}_puts.png')
    
    fig_calls.write_image(calls_pngs_file, width=1920, height=1080, scale=2)
    fig_puts.write_image(puts_pngs_file, width=1920, height=1080, scale=2)
    
    print(f"High-resolution images saved to {pngs_dir}")

    # Show the plots in the browser
    fig_calls.show()
    fig_puts.show()

if __name__ == "__main__":
    main()

