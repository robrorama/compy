import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import sys
import os
from datetime import date, timedelta

def plot_ticker_minute_data_with_ma_fib(ticker, days):
    # Convert ticker to lowercase
    ticker = ticker.lower()

    # Define the directory path where the data is stored
    dir_path = f'data/{date.today()}/{ticker}'

    # Construct the file path for the minute-by-minute data
    file_path = f'{dir_path}/{ticker}.minute_data.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"No minute-by-minute data found for {ticker} at {file_path}")
        return

    # Load the minute-by-minute data from the CSV file
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Filter the data for the last `days` days
    end_date = data.index[-1]
    start_date = end_date - timedelta(days=days)
    filtered_data = data[start_date:end_date]

    if filtered_data.empty:
        print(f"No data available for the last {days} days for {ticker}.")
        return

    # Calculate Fibonacci retracement levels
    max_price = filtered_data['Close'].max()
    min_price = filtered_data['Close'].min()
    diff = max_price - min_price

    levels = {
        '0.0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50.0%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100.0%': min_price,
    }

    # Create a plot with candlestick chart, moving averages, volume, and Fibonacci retracement
    fig, axlist = mpf.plot(
        filtered_data,
        type='candle',
        mav=(9, 20, 50, 100, 150, 200, 300),
        volume=True,
        returnfig=True,
        title=f"{ticker.upper()} - {end_date.strftime('%Y-%m-%d')} (Last {days} Day(s))",
        style='yahoo',
    )

    ax = axlist[0]  # Access the main axis for the candlestick chart

    # Plot Fibonacci retracement lines
    for level in levels.values():
        ax.axhline(level, linestyle='--', alpha=0.5, color='gray')

    # Save the plot
    plot_path = f'{dir_path}/{ticker}_minute_data_with_ma_fib_last_{days}_days.png'
    fig.savefig(plot_path)
    plt.show()

    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    # Read the ticker and days from the command line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 plot_ticker_minute_data_with_ma_fib.py <ticker> <days>")
        sys.exit(1)

    ticker = sys.argv[1]
    days = int(sys.argv[2])

    if days < 1 or days > 5:
        print("Please specify a number between 1 and 5 for days.")
        sys.exit(1)

    plot_ticker_minute_data_with_ma_fib(ticker, days)
