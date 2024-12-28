import os
from scipy.stats import linregress
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import mplfinance as mpf
from datetime import datetime, timedelta
import imageio.v2 as imageio

# Function to download data
def download_data(ticker, period="max"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

# Function to calculate linear regression and standard deviation bands for the selected time frame
def add_linear_regression_and_std_bands(df, span=50):
    # Use the numeric representation of the date index to ensure smoothness
    df_numeric_index = pd.to_numeric(df.index)

    # Linear regression over the selected time frame
    slope, intercept, _, _, _ = linregress(df_numeric_index[-span:], df['Close'][-span:])
    df[f'Linear_Regression_{span}'] = intercept + slope * df_numeric_index

    # Calculate the deviation based on the slope of the linear regression line
    residuals_std = df['Close'][-span:].std()  # Standard deviation of the prices in the time frame

    # Create perfectly parallel standard deviation bands
    desired_values = [0.25 * i for i in range(1, 13)]  # 0.25 to 3 sigma
    for num_std in desired_values:
        df[f'Reg_High_{span}_{num_std}std'] = df[f'Linear_Regression_{span}'] + residuals_std * num_std
        df[f'Reg_Low_{span}_{num_std}std'] = df[f'Linear_Regression_{span}'] - residuals_std * num_std

    return df

# Function to add exponential moving averages (EMAs)
def add_ema(df, periods=[20, 50, 100, 200]):
    for period in periods:
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

# Plotting function using mplfinance for candlesticks, volume, and linear regression with standard deviation bands
def plot_mplfinance_v7(df, ticker, period_name, span):
    ap = []

    # Add EMAs
    ema_colors = ['cyan', 'blue', 'green', 'red']
    for ema_period, color in zip([20, 50, 100, 200], ema_colors):
        ap.append(mpf.make_addplot(df[f'EMA_{ema_period}'], color=color))

    # Add Linear Regression and standard deviation bands
    ap.append(mpf.make_addplot(df[f'Linear_Regression_{span}'], color='orange'))
    desired_values = [0.25 * i for i in range(1, 13)]  # 0.25 to 3 sigma
    for num_std in desired_values:
        ap.append(mpf.make_addplot(df[f'Reg_High_{span}_{num_std}std'], color='gray', linestyle='dotted'))
        ap.append(mpf.make_addplot(df[f'Reg_Low_{span}_{num_std}std'], color='gray', linestyle='dotted'))

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_{period_name.replace(" ", "_").lower()}.png')

    # Set y-axis limits to focus more on the candlesticks
    price_min = df['Low'].min()
    price_max = df['High'].max()
    y_limits = (price_min - (price_max - price_min) * 0.05, price_max + (price_max - price_min) * 0.05)

    mpf.plot(df, type='candle', style='charles', title=f'{ticker} Candlesticks with EMAs and {span}-Day Linear Regression ({period_name})',
             ylabel='Price', addplot=ap, volume=True, figsize=(15, 10), ylim=y_limits, savefig=image_file)

    return image_file

# Interactive Plotly plotting function for only 6 months
def plot_plotly_v9(df, ticker, span):
    # Define six months ago
    six_months_ago = datetime.today() - timedelta(days=180)
    df = df[df.index >= six_months_ago]

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Data'))

    # Add Linear Regression and standard deviation bands
    fig.add_trace(go.Scatter(x=df.index, y=df[f'Linear_Regression_{span}'], line=dict(color='orange', width=2), name=f'Linear Regression {span}'))
    desired_values = [0.25 * i for i in range(1, 13)]  # Up to 3 sigma
    for num_std in desired_values:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_High_{span}_{num_std}std'], line=dict(color='gray', width=1, dash='dot'), name=f'Reg High {num_std}σ'))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_Low_{span}_{num_std}std'], line=dict(color='gray', width=1, dash='dot'), name=f'Reg Low {num_std}σ'))

    # EMAs
    ema_colors = ['cyan', 'blue', 'green', 'red']
    for ema_period, color in zip([20, 50, 100, 200], ema_colors):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{ema_period}'], line=dict(color=color, width=2), name=f'EMA {ema_period}'))

    # Set y-axis limits to focus more on the candlesticks
    price_min = df['Low'].min()
    price_max = df['High'].max()
    fig.update_layout(
        yaxis=dict(range=[price_min - (price_max - price_min) * 0.05, price_max + (price_max - price_min) * 0.05]),
        title=f'{ticker} (Last 6 Months) with EMAs and {span}-Day Linear Regression',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )

    fig.show()

# Main function to process the data and create both types of plots
def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <TICKER> <SPAN>")
        sys.exit(1)

    ticker = sys.argv[1]
    span = int(sys.argv[2])

    # Validate the span
    valid_spans = [9, 20, 50, 100, 200, 300]
    if span not in valid_spans:
        print(f"Invalid span: {span}. Valid options are {valid_spans}.")
        sys.exit(1)

    # Download data
    df = download_data(ticker)

    # Add Linear Regression and standard deviation bands for the specified span
    df = add_linear_regression_and_std_bands(df, span=span)
    
    # Add EMAs
    df = add_ema(df)

    # Define date ranges
    date_ranges = {
        'Last 1 day': datetime.today() - timedelta(days=1),
        'Last 7 days': datetime.today() - timedelta(days=7),
        'Last 1 Month': datetime.today() - timedelta(days=30),
        'Last 3 Months': datetime.today() - timedelta(days=90),
        'Last 6 Months': datetime.today() - timedelta(days=180),
        'Last Year': datetime.today() - timedelta(days=365),
        'Last 2 Years': datetime.today() - timedelta(days=2*365),
        'Last 5 Years': datetime.today() - timedelta(days=5*365),
    }

    # Generate mplfinance plots for each time range
    image_files = []
    for period_name, start_date in date_ranges.items():
        data_period = df[start_date:]
        if len(data_period) < 50:  # Ensure there's enough data for plotting
            print(f"Not enough data for {period_name}. Skipping.")
            continue
        image_file = plot_mplfinance_v7(data_period, ticker, period_name, span)
        image_files.append(image_file)

    # Display the interactive plot using Plotly
    plot_plotly_v9(df, ticker, span)

if __name__ == '__main__':
    main()

