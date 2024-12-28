import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import sys
import numpy as np

def get_next_expiry_date(ticker, weeks_ahead=1):
    stock = yf.Ticker(ticker)
    try:
        options_dates = pd.to_datetime(stock.options)
        today = datetime.today()
        target_date = today + timedelta(weeks=weeks_ahead)
        idx = np.argmin(np.abs(options_dates - target_date))
        nearest_date = options_dates[idx]
        return nearest_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error retrieving options dates for {ticker}: {e}")
        return None

def download_options_data(ticker, date):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = os.path.join("options_data", today_str, ticker)
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f"options.{date}.{ticker}.csv")

    if os.path.exists(data_file):
        print(f"Options data for {ticker} already exists. Skipping download.")
        return pd.read_csv(data_file)

    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(date)
        if options.calls.empty and options.puts.empty:
            print(f"No options data available for {ticker} on {date}.")
            return None
        df = pd.concat([options.calls, options.puts], ignore_index=True)
        df.to_csv(data_file, index=False)
        print(f"Options data for {ticker} downloaded and saved to {data_file}.")
        return df
    except Exception as e:
        print(f"Error downloading options data for {ticker}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <ticker_symbol> [weeks_ahead]")
        sys.exit(1)

    ticker = sys.argv[1]
    weeks_ahead = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    nearest_date = get_next_expiry_date(ticker, weeks_ahead)
    if not nearest_date:
        sys.exit(1)
    print(f"Next Options Expiration Date: {nearest_date}")

    stock = yf.Ticker(ticker)
    current_price = stock.history(period="max")['Close'].iloc[-1]

    options_data = download_options_data(ticker, nearest_date)
    if options_data is None:
        sys.exit(1)

    # Split and clean data
    calls_data = options_data[options_data['contractSymbol'].str.contains('C', na=False)].copy()
    puts_data = options_data[options_data['contractSymbol'].str.contains('P', na=False)].copy()

    # Fill NaN values explicitly
    calls_data.loc[:, 'openInterest'] = calls_data['openInterest'].fillna(0)
    puts_data.loc[:, 'openInterest'] = puts_data['openInterest'].fillna(0)

    # Calculate 1-sigma thresholds
    calls_mean = calls_data["openInterest"].mean()
    calls_std = calls_data["openInterest"].std()
    calls_threshold = calls_mean + calls_std

    puts_mean = puts_data["openInterest"].mean()
    puts_std = puts_data["openInterest"].std()
    puts_threshold = puts_mean + puts_std

    # Filter data based on 1-sigma rule
    calls_data_filtered = calls_data[calls_data["openInterest"] > calls_threshold]
    puts_data_filtered = puts_data[puts_data["openInterest"] > puts_threshold]

    # Create combined plot
    fig = go.Figure()

    # Add puts histogram rotated 90 degrees counterclockwise
    fig.add_trace(
        go.Bar(
            x=-puts_data_filtered["openInterest"],  # Negative values for left side
            y=puts_data_filtered["strike"],
            orientation="h",
            marker=dict(color="red"),
            name="Puts Open Interest",
            width=5,  # Thicker bars
        )
    )

    # Add calls histogram rotated 90 degrees clockwise
    fig.add_trace(
        go.Bar(
            x=calls_data_filtered["openInterest"],
            y=calls_data_filtered["strike"],
            orientation="h",
            marker=dict(color="green"),
            name="Calls Open Interest",
            width=5,  # Thicker bars
        )
    )

    # Add vertical sigma lines
    fig.add_trace(
        go.Scatter(
            x=[-puts_threshold, -puts_threshold],
            y=[puts_data_filtered["strike"].min(), puts_data_filtered["strike"].max()],
            mode="lines",
            line=dict(color="yellow", dash="dash"),
            name="Puts 1-Sigma Line",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[calls_threshold, calls_threshold],
            y=[calls_data_filtered["strike"].min(), calls_data_filtered["strike"].max()],
            mode="lines",
            line=dict(color="yellow", dash="dash"),
            name="Calls 1-Sigma Line",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Options Open Interest - Current Price: ${current_price:.2f}",
        xaxis_title="Open Interest",
        yaxis_title="Strike Price",
        yaxis=dict(
            autorange="reversed",
            title="Strike Price",
        ),
        xaxis=dict(
            title="Open Interest",
            zeroline=True,
        ),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(x=0.1, y=1),
    )

    # Save the plot
    image_dir = os.path.join("images", ticker)
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f"{ticker}_options_{nearest_date}_combined_histogram_with_sigma.png")
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {image_file}")

    # Show plot
    fig.show()

if __name__ == "__main__":
    main()

