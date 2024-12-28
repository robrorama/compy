import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
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
    data_dir = f"options_data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/options.{date}.{ticker}.csv"

    # Check if the data already exists
    if os.path.exists(data_file):
        print(f"Options data for {ticker} already exists. Skipping download.")
        df = pd.read_csv(data_file)
    else:
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

    # Custom function to create subplots with watermarks and bars
    def create_subplot(fig, row, col, data, color, title, flip_y=False):
        # Add the scatter plot
        fig.add_trace(
            px.scatter(data, x="strike", y="openInterest", size="openInterest", color="openInterest", opacity=0.8, color_continuous_scale=color).data[0],
            row=row, col=col
        )

        # Add the bars with see-through opacity
        fig.add_trace(
            px.bar(data, x="strike", y="openInterest", opacity=0.3, color_discrete_sequence=[color[-1]]).data[0],
            row=row, col=col
        )

        # Add the watermark annotation (adjust x position based on column)
        fig.add_annotation(
            text=title,
            xref="paper", yref="paper",
            x=0.25 if col == 1 else 0.75,  # Adjust x position for left/right plot
            y=0.5,
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=80,
                color=px.colors.sequential.Blues[0] if color == px.colors.sequential.Blues else px.colors.sequential.Reds[0],
            ),
            opacity=0.2,
        )

        # Update axes and layout
        fig.update_xaxes(title_text="Strike Price", showgrid=False, row=row, col=col)

        if flip_y:
            fig.update_yaxes(autorange="reversed", title_text="Open Interest", showgrid=False, row=row, col=col)
        else:
            fig.update_yaxes(title_text="Open Interest", showgrid=False, row=row, col=col)

        fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white',
            title_text=f"{ticker} Options (Strike vs. Open Interest) - Current Price: ${current_price:.2f}",
            showlegend=False
        )

    # Create subplots figure
    fig = make_subplots(rows=1, cols=2)

    # Add plots with watermarks and bars
    create_subplot(fig, 1, 1, calls_data, px.colors.sequential.Blues, "Calls", flip_y=True)
    create_subplot(fig, 1, 2, puts_data, px.colors.sequential.Reds, "Puts", flip_y=True)

    # Save high-resolution images
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    
    image_file = os.path.join(image_dir, f'{ticker}_options_{nearest_date}_spikes.png')
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    
    print(f"High-resolution image saved to {image_file}")

    # Save a copy of the image to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_options_{nearest_date}_spikes.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {pngs_file}")

    # Show the combined plot in the browser
    fig.show()

if __name__ == "__main__":
    main()

