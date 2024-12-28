import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import imageio.v2 as imageio
import sys

def download_data(ticker, period="max"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    ticker = ticker.lower()
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        if df.empty:
            print(f"Failed to download data for {ticker}.")
            return None
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_correlation(data, start_date=None, end_date=None):
    """Calculate correlation matrix for the given date range."""
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]
    
    close_prices = data.xs('Close', level=1, axis=1)
    return close_prices.corr()

def main():
    if len(sys.argv) != 2:
        print("Usage: python ip_new_correlation_matrix_v17_with_animated_gif.py <tickers_file>")
        sys.exit(1)

    # Read the CSV file into a DataFrame
    csv_file_path = sys.argv[1]
    tickers_df = pd.read_csv(csv_file_path)
    
    # Extract the list of ticker symbols and convert to lowercase
    tickers = tickers_df['Ticker'].str.lower().tolist()

    # Load data for all tickers
    all_data = {}
    for ticker in tickers:
        data = download_data(ticker)
        if data is not None:
            all_data[ticker] = data

    if not all_data:
        raise ValueError("No data loaded. Please ensure the data is available in the specified directories.")
    else:
        print(f"Data loaded for {len(all_data)} tickers.")

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)

    # Define date ranges for correlation matrices
    date_ranges = {
        'Last 10 Years': datetime.today() - timedelta(days=10*365),
        'Last 20 Years': datetime.today() - timedelta(days=20*365),
        'Last 30 Years': datetime.today() - timedelta(days=30*365),
        'Last 2 Years': datetime.today() - timedelta(days=2*365),
        'Last Year': datetime.today() - timedelta(days=365),
        'Last 6 Months': datetime.today() - timedelta(days=180),
        'Last Month': datetime.today() - timedelta(days=30),
        'Last Week': datetime.today() - timedelta(days=7),
        'Last Day': datetime.today() - timedelta(days=1),
    }

    # Directory to save the images
    png_dir = 'PNG'
    os.makedirs(png_dir, exist_ok=True)  # Ensure the PNG directory exists

    # Calculate and save correlation matrices
    image_files = []
    for period, start_date in date_ranges.items():
        if start_date:
            start_date = start_date.strftime('%Y-%m-%d')
        corr_matrix = calculate_correlation(combined_data, start_date=start_date)
        plt.figure(figsize=(16, 14))  # Adjusted figure size for larger graphs
        plt.matshow(corr_matrix, cmap='coolwarm_r', fignum=1)  # Reversed cmap
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8)  # Smaller font size
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index, fontsize=8)  # Smaller font size
        plt.colorbar(label='Correlation')
        plt.title(f'Correlation Matrix: {period}', fontsize=14)  # Adjusted title font size
        
        # Save the plot as a .png file with high resolution
        image_file = os.path.join(png_dir, f'correlation_matrix_{period.replace(" ", "_").lower()}.png')
        plt.savefig(image_file, dpi=300)
        plt.close()
        
        image_files.append(image_file)

    # Create animated GIFs
    gif_path_1s = os.path.join(png_dir, f'correlation_matrix_1s.gif')
    gif_path_5s = os.path.join(png_dir, f'correlation_matrix_5s.gif')

    # GIF with 1-second delay between images
    with imageio.get_writer(gif_path_1s, mode='I', duration=1, loop=0) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    # GIF with 5-second delay between images
    with imageio.get_writer(gif_path_5s, mode='I', duration=5.0, loop=0) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIFs created as {gif_path_1s} and {gif_path_5s}")
    print("Correlation matrices saved in the PNG folder.")

if __name__ == "__main__":
    main()

