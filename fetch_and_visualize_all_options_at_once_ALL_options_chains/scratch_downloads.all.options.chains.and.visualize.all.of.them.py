import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
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

    for date, data in all_options_data.items():
        # Split the data into calls and puts
        calls_data = data[data['contractSymbol'].str.contains('C')]
        puts_data = data[data['contractSymbol'].str.contains('P')]

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
                title_text=f"{ticker} Options on {date} (Strike vs. Open Interest) - Current Price: ${current_price:.2f}",
                showlegend=False
            )

        # Create subplots figure
        fig = make_subplots(rows=1, cols=2)

        # Add plots with watermarks and bars
        create_subplot(fig, 1, 1, calls_data, px.colors.sequential.Blues, "Calls", flip_y=True)
        create_subplot(fig, 1, 2, puts_data, px.colors.sequential.Reds, "Puts", flip_y=True)

        # Save high-resolution images
        image_file = os.path.join(plot_dir, f'{ticker}_options_{date}_spikes.png')
        fig.write_image(image_file, width=1920, height=1080, scale=2)
        print(f"High-resolution image saved to {image_file}")

        # Save a copy of the image to the PNGS folder with ticker information and date
        pngs_dir = "PNGS"
        os.makedirs(pngs_dir, exist_ok=True)
        pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_options_{date}_spikes.png')
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

