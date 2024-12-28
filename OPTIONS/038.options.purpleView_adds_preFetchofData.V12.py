import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
import sys


def download_options_data(ticker, expiration_date):
    """
    Downloads options data for a specific ticker and expiration date.
    """
    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiration_date)
        if options.calls.empty and options.puts.empty:
            print(f"No options data available for {ticker} on {expiration_date}.")
            return None
        df = pd.concat([options.calls, options.puts], ignore_index=True)
        return df
    except Exception as e:
        print(f"Error downloading options data for {ticker} on {expiration_date}: {e}")
        return None


def plot_options_expiration(calls_data, puts_data, expiration_date, ticker, current_price):
    """
    Creates a plot for the given expiration date and saves it.
    """
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

    # Create plot
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
        title=f"{ticker} Options Open Interest ({expiration_date}) - Current Price: ${current_price:.2f}",
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
    image_file = os.path.join(image_dir, f"{ticker}_options_{expiration_date}_histogram.png")
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"High-resolution image saved to {image_file}")

    # Show the plot in the browser
    fig.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <ticker_symbol>")
        sys.exit(1)

    ticker = sys.argv[1]

    stock = yf.Ticker(ticker)
    options_dates = stock.options  # Get all available expiration dates
    current_price = stock.history(period="max")['Close'].iloc[-1]

    if not options_dates:
        print(f"No options expiration dates available for {ticker}.")
        sys.exit(1)

    for expiration_date in options_dates:
        print(f"Processing options for expiration date: {expiration_date}")

        # Download data for this expiration date
        options_data = download_options_data(ticker, expiration_date)
        if options_data is None:
            continue

        # Split and clean data
        calls_data = options_data[options_data['contractSymbol'].str.contains('C', na=False)].copy()
        puts_data = options_data[options_data['contractSymbol'].str.contains('P', na=False)].copy()

        # Fill NaN values explicitly
        calls_data.loc[:, 'openInterest'] = calls_data['openInterest'].fillna(0)
        puts_data.loc[:, 'openInterest'] = puts_data['openInterest'].fillna(0)

        # Plot the data for this expiration date
        plot_options_expiration(calls_data, puts_data, expiration_date, ticker, current_price)


if __name__ == "__main__":
    main()

