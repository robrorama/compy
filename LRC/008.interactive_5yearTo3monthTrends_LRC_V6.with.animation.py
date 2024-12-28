import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from scipy.stats import linregress
from datetime import datetime, timedelta
import imageio.v2 as imageio

def download_stock_data(ticker, period="max"):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_moving_averages(df):
    """Calculate moving averages and add them to the dataframe."""
    for ema_period in [9, 50, 100, 200]:
        df[f'EMA_{ema_period}'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

def calculate_linear_regression_channels(df, period, use_log=True):
    """Calculate linear regression channels for a given period and add them to the dataframe."""
    if len(df) < period:
        return df
    
    if use_log:
        df['Value'] = np.log(df['Close'])
    else:
        df['Value'] = df['Close']
        
    df['Index'] = np.arange(len(df))
    
    slope, intercept, _, _, _ = linregress(df['Index'][-period:], df['Value'][-period:])
    df['RegValue'] = slope * df['Index'] + intercept
    df['Residual'] = df['Value'] - df['RegValue']
    std_dev = df['Residual'][-period:].std()
    
    deviations = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    for dev in deviations:
        df[f'Upper{dev}_{period}'] = (df['RegValue'] + dev * std_dev).apply(np.exp if use_log else lambda x: x)
        df[f'Lower{dev}_{period}'] = (df['RegValue'] - dev * std_dev).apply(np.exp if use_log else lambda x: x)
    
    return df

def plot_data_with_plotly(data, ticker, period_name, use_log=True):
    """Plot data with linear regression channels, candlesticks, and EMAs using Plotly."""
    
    # Calculate linear regression channels and moving averages
    data = calculate_moving_averages(data)
    data = calculate_linear_regression_channels(data, 144, use_log)
    data = calculate_linear_regression_channels(data, 50, use_log)
    
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks'
    ))

    # Add linear regression channels
    for period in [50, 144]:
        if len(data) < period:
            continue
        deviations = [1, 2]
        for dev in deviations:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[f'Upper{dev}_{period}'],
                mode='lines', line=dict(color='lightblue' if dev % 1 else 'blue', dash='dot' if dev % 1 else 'solid'),
                name=f'Upper {dev}σ (Period {period})'
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data[f'Lower{dev}_{period}'],
                mode='lines', line=dict(color='lightpink' if dev % 1 else 'red', dash='dot' if dev % 1 else 'solid'),
                name=f'Lower {dev}σ (Period {period})'
            ))

    # Add EMAs
    for ema_period, color in zip([9, 50, 100, 200], ['cyan', 'blue', 'green', 'red']):
        fig.add_trace(go.Scatter(
            x=data.index, y=data[f'EMA_{ema_period}'],
            mode='lines', line=dict(color=color),
            name=f'EMA {ema_period}'
        ))

    # Add layout and show figure
    fig.update_layout(
        title=f'{ticker} Linear Regression and EMAs ({period_name})',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_{period_name.replace(" ", "_").lower()}.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()

    return image_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    data = download_stock_data(ticker)
    
    # Define date ranges for plotting
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
    
    image_files = []
    for period_name, start_date in date_ranges.items():
        data_period = data[start_date:]
        if len(data_period) < 50:  # Ensure there is enough data for the shortest period
            print(f"Not enough data for {period_name}. Skipping.")
            continue
        image_file = plot_data_with_plotly(data_period, ticker, period_name)
        image_files.append(image_file)

    if not image_files:
        print("No valid images were created. Exiting.")
        return
    
    # Create animated GIF
    gif_path = f'images/{ticker}/{ticker}_analysis.gif'
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I', duration=1, loop=0) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)
    
    print(f"GIF created at {gif_path}")

if __name__ == '__main__':
    main()

