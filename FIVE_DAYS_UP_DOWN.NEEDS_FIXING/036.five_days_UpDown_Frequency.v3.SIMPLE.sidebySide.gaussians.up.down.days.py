import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import numpy as np
import scipy.stats as stats
import sys

def download_data(ticker, period):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_consecutive_trends(df):
    df['Trend'] = np.sign(df['Close'].diff())
    df['Consecutive'] = df['Trend'].groupby((df['Trend'] != df['Trend'].shift()).cumsum()).cumcount() + 1
    df['Consecutive'] *= df['Trend']
    df['Pct_Change'] = df['Close'].pct_change() * 100
    return df

def plot_histograms_and_gaussian(df, title):
    # Separate the data into positive and negative consecutive days
    positive_consecutive = df['Consecutive'][df['Consecutive'] > 0]
    negative_consecutive = df['Consecutive'][df['Consecutive'] < 0]

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Consecutive Days Up', 'Consecutive Days Down'))

    # Histogram for positive consecutive days
    fig.add_trace(go.Histogram(x=positive_consecutive, name='Consecutive Up'), row=1, col=2)

    # Histogram for negative consecutive days
    fig.add_trace(go.Histogram(x=negative_consecutive, name='Consecutive Down'), row=1, col=1)

    # Gaussian Distribution for negative consecutive days
    if len(negative_consecutive) > 0:
        mean_neg = np.mean(negative_consecutive)
        std_dev_neg = np.std(negative_consecutive)
        x_neg = np.linspace(min(negative_consecutive), max(negative_consecutive), 100)
        gaussian_distribution_neg = stats.norm.pdf(x_neg, mean_neg, std_dev_neg)
        fig.add_trace(go.Scatter(x=x_neg, y=gaussian_distribution_neg, mode='lines', name='Gaussian Down'), row=1, col=1)

    # Gaussian Distribution for positive consecutive days
    if len(positive_consecutive) > 0:
        mean_pos = np.mean(positive_consecutive)
        std_dev_pos = np.std(positive_consecutive)
        x_pos = np.linspace(min(positive_consecutive), max(positive_consecutive), 100)
        gaussian_distribution_pos = stats.norm.pdf(x_pos, mean_pos, std_dev_pos)
        fig.add_trace(go.Scatter(x=x_pos, y=gaussian_distribution_pos, mode='lines', name='Gaussian Up'), row=1, col=2)

    # Update layout
    fig.update_layout(title=f'Consecutive Days and Gaussian Distribution for {title}', barmode='overlay')

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{title.split()[0]}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{title.split()[0]}_consecutive_days.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)

    # Save a copy of the image to the PNGS folder with ticker information and date
    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{title.split()[0]}_{datetime.today().strftime("%Y-%m-%d")}_consecutive_days.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=4)

    # Show the figure
    fig.show()

# Example usage
if len(sys.argv) < 2:
    print("Usage: python script.py <ticker_symbol> [period]")
    sys.exit(1) 

ticker = sys.argv[1]
period = "max"
if len(sys.argv) >= 3:
    period = sys.argv[2]
else:
    period = "max"
    print("No period specified, using 'max' as the default.")

ticker_data = download_data(ticker, period)
ticker_data = calculate_consecutive_trends(ticker_data)

# Plot histograms and Gaussian distributions
plot_histograms_and_gaussian(ticker_data, title=f"{ticker} Stock")

