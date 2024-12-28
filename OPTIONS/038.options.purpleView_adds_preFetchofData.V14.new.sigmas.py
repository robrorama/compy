import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import os
import sys


def download_options_data(ticker, expiration_date):
    """
    Downloads options data for a specific ticker and expiration date if not already available locally.
    """
    data_dir = "options_data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}_{expiration_date}_options.csv")

    if os.path.exists(file_path):
        print(f"Using cached data for {ticker} on {expiration_date}.")
        return pd.read_csv(file_path)

    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain(expiration_date)
        if options.calls.empty and options.puts.empty:
            print(f"No options data available for {ticker} on {expiration_date}.")
            return None
        df = pd.concat([options.calls, options.puts], ignore_index=True)
        df.to_csv(file_path, index=False)
        print(f"Downloaded and cached data for {ticker} on {expiration_date}.")
        return df
    except Exception as e:
        print(f"Error downloading options data for {ticker} on {expiration_date}: {e}")
        return None

def plot_aggregated_options(calls_data, puts_data, ticker, current_price):
    """
    Creates a combined plot for all expiration dates, including 1, 2, 3, and 4-sigma thresholds.
    """
    # Calculate sigma thresholds for combined data
    calls_mean = calls_data["openInterest"].mean()
    calls_std = calls_data["openInterest"].std()
    calls_thresholds = [calls_mean + i * calls_std for i in range(1, 7)]  # 1, 2, 3, 4-sigma thresholds

    puts_mean = puts_data["openInterest"].mean()
    puts_std = puts_data["openInterest"].std()
    puts_thresholds = [puts_mean + i * puts_std for i in range(1, 7)]  # 1, 2, 3, 4-sigma thresholds

##########




##########


    sigma_level = 6  # Change this to 2, 3, 4, 5, or 6 to filter accordingly
    # Adjust filtering based on the desired sigma level
    calls_data_filtered = calls_data[calls_data["openInterest"] > calls_thresholds[sigma_level - 1]]
    puts_data_filtered = puts_data[puts_data["openInterest"] > puts_thresholds[sigma_level - 1]]
    # Filter data based on ±2-sigma threshold
    #calls_data_filtered = calls_data[calls_data["openInterest"] > calls_thresholds[1]]  # 2-sigma threshold
    #puts_data_filtered = puts_data[puts_data["openInterest"] > puts_thresholds[1]]  # 2-sigma threshold

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
            width=8,  # Thicker bars
        )
    )

    # Add sigma lines for puts
    for i, threshold in enumerate(puts_thresholds, start=1):
        fig.add_trace(
            go.Scatter(
                x=[-threshold, -threshold],
                y=[puts_data_filtered["strike"].min(), puts_data_filtered["strike"].max()],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Puts {i}-Sigma Line",
            )
        )

    # Add sigma lines for calls
    for i, threshold in enumerate(calls_thresholds, start=1):
        fig.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[calls_data_filtered["strike"].min(), calls_data_filtered["strike"].max()],
                mode="lines",
                line=dict(color="green", dash="dash"),
                name=f"Calls {i}-Sigma Line",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Options Open Interest (Aggregated) - Current Price: ${current_price:.2f}",
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
    image_file = os.path.join(image_dir, f"{ticker}_options_aggregated_histogram.png")
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

    # Initialize combined dataframes
    combined_calls = pd.DataFrame()
    combined_puts = pd.DataFrame()

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
        calls_data['openInterest'] = calls_data['openInterest'].fillna(0)
        puts_data['openInterest'] = puts_data['openInterest'].fillna(0)

        # Append to combined data
        combined_calls = pd.concat([combined_calls, calls_data], ignore_index=True)
        combined_puts = pd.concat([combined_puts, puts_data], ignore_index=True)

    # Plot the aggregated data
    plot_aggregated_options(combined_calls, combined_puts, ticker, current_price)


if __name__ == "__main__":
    main()

