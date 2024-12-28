import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta
import yfinance as yf

def download_data(ticker):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def filter_data(df, period):
    """Filter the dataframe to only include data from the specified period."""
    end_date = datetime.today()
    if period == '1y':
        start_date = end_date - timedelta(days=365)
    elif period == '2y':
        start_date = end_date - timedelta(days=365*2)
    elif period == '5y':
        start_date = end_date - timedelta(days=365*5)
    elif period == '10y':
        start_date = end_date - timedelta(days=365*10)
    elif period == '6mo':
        start_date = end_date - timedelta(days=30*6)
    elif period == '3mo':
        start_date = end_date - timedelta(days=30*3)
    elif period == '1mo':
        start_date = end_date - timedelta(days=30)
    elif period == '7mo':
        start_date = end_date - timedelta(days=30*7)
    else:
        # Default to 1 year if an unknown period is specified
        start_date = end_date - timedelta(days=365)
    
    return df[df.index >= start_date]

def calculate_moving_averages(df):
    """Calculate moving averages and add them to the dataframe."""
    for ema_period in [9, 50, 100, 200]:
        df[f'EMA_{ema_period}'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    return df

def calculate_linear_regression_channels(df, period, use_log=True):
    """Calculate linear regression channels for a given period and add them to the dataframe."""
    if len(df) < period:
        print(f"Not enough data to calculate linear regression channels for period {period}")
        return df

    # Ensure we have only positive close values for logarithm
    df = df[df['Close'] > 0].copy()
    
    if df.empty:
        print(f"No valid data after filtering for positive values in period {period}")
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
    
    deviations = [1, 2]
    for dev in deviations:
        upper_key = f'Upper{dev}_{period}'
        lower_key = f'Lower{dev}_{period}'
        
        df[upper_key] = (df['RegValue'] + dev * std_dev).apply(np.exp if use_log else lambda x: x)
        df[lower_key] = (df['RegValue'] - dev * std_dev).apply(np.exp if use_log else lambda x: x)

    return df

def load_data(ticker):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} loaded from disk.")
    except FileNotFoundError:
        print(f"Data file for {ticker} not found.")
        return None
    return df

def plot_commodities(commodities, period='1y', use_log=True):
    n_commodities = len(commodities)
    n_cols = 4
    n_rows = (n_commodities + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), constrained_layout=True)

    for i, (name, ticker) in enumerate(commodities.items()):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        df = load_data(ticker)
        if df is None or df.empty:
            print(f"No data available for {ticker}")
            continue

        # Filter data to the specified period
        df = filter_data(df, period)

        df = calculate_moving_averages(df)
        df = calculate_linear_regression_channels(df, 144, use_log)
        df = calculate_linear_regression_channels(df, 50, use_log)

        ax.plot(df['Close'], label='Close')
        for period_length in [50, 144]:
            deviations = [1, 2]
            for dev in deviations:
                upper_key = f'Upper{dev}_{period_length}'
                lower_key = f'Lower{dev}_{period_length}'
                if upper_key in df.columns:
                    ax.plot(df[upper_key], linestyle='--' if dev % 1 != 0 else '-', color='lightblue')
                if lower_key in df.columns:
                    ax.plot(df[lower_key], linestyle='--' if dev % 1 != 0 else '-', color='lightpink')
        
        for ema_period, color in zip([9, 50, 100, 200], ['cyan', 'blue', 'green', 'red']):
            ax.plot(df[f'EMA_{ema_period}'], label=f'EMA_{ema_period}', color=color)

        ax.set_title(name)
        ax.legend()
        
        # Dynamically adjust x-axis and y-axis limits
        ax.autoscale(enable=True, axis='both', tight=True)

    # Save high-resolution image
    image_dir = f"images/commodities"
    os.makedirs(image_dir, exist_ok=True)
    
    image_file = os.path.join(image_dir, 'commodities_normalized_with_lr_channels.png')
    fig.savefig(image_file, dpi=300)
    
    print(f"High-resolution image saved to {image_file}")

    plt.show()

def main(period='1y'):
    commodities = {
        "Light Sweet Crude Oil": "CL=F",
        "Brent Crude Oil": "BZ=F",
        "Natural Gas": "NG=F",
        "Heating Oil": "HO=F",
        "RBOB Gasoline": "RB=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Platinum": "PL=F",
        "Palladium": "PA=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F",
        "Soybeans": "ZS=F",
        "Soybean Oil": "ZL=F",
        "Soybean Meal": "ZM=F",
        "Sugar": "SB=F",
        "Coffee": "KC=F",
        "Cocoa": "CC=F",
        "Cotton": "CT=F",
        "Orange Juice": "OJ=F",
        "Live Cattle": "LE=F",
        "Lean Hogs": "HE=F",
        "Feeder Cattle": "GF=F"
    }

    for name, ticker in commodities.items():
        download_data(ticker)

    plot_commodities(commodities, period)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        period = sys.argv[1]
    else:
        period = '1y'
    main(period)

