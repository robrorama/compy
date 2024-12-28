import os
import sys
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import linregress
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Function to download data and save to CSV
def download_data(ticker, period="1y"):  # Default period set to '1y' for last year
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is of datetime type
        df.set_index('Date', inplace=True)  # Set 'Date' as the index
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)  # Download data for the last year
        if df.empty:
            print(f"Failed to download data for {ticker}.")
            sys.exit(1)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

# Function to add linear regression bands to the DataFrame
def add_linear_regression_bands(df):
    df_numeric_index = pd.to_numeric(df.index)
    slope, intercept, _, _, _ = linregress(df_numeric_index, df['Close'])
    df['Linear_Reg'] = intercept + slope * df_numeric_index
    df['Residuals'] = df['Close'] - df['Linear_Reg']
    residuals_std = df['Residuals'].std()

    desired_values = [.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, num_std in enumerate(desired_values):
        df[f'Reg_High_{i+1}std'] = df['Linear_Reg'] + residuals_std * num_std
        df[f'Reg_Low_{i+1}std'] = df['Linear_Reg'] - residuals_std * num_std
    return df

# Function to add EMAs to the DataFrame
def add_ema(df, time_periods):
    for time_period in time_periods:
        df[f'EMA_{time_period}'] = df['Close'].ewm(span=time_period, adjust=False).mean()
    return df

# Function to plot the data with technical indicators and save as PNG
def plot_data(df, ticker):
    fig = go.Figure()

    desired_values = [.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Data'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Reg'], line=dict(color='blue', width=2), name='Linear Regression'))

    colors = ['grey', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'blue']
    for i, num_std in enumerate(desired_values):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_High_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg High {num_std} std'))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_Low_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg Low {num_std} std'))

    ema_colors = ['purple', 'orange', 'green', 'red', 'blue']
    for i, time_period in enumerate([20, 50, 100, 200, 300]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{time_period}'], line=dict(color=ema_colors[i], width=2), name=f'{time_period}-day EMA'))

    fig.update_layout(title=f"{ticker} Linear Regression Channel, CandleSticks, and EMAs", xaxis_title='Date', yaxis_title='Price', height=800, width=1200)

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_trend_line_analysis.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()

# Main function to run the analysis
def main():
    ticker = sys.argv[1]  # Get the ticker from command line argument
    period = sys.argv[2] if len(sys.argv) > 2 else "1y"  # Get the period from command line argument or use default "1y"
    df = download_data(ticker, period)
    df = add_linear_regression_bands(df)
    df = add_ema(df, [20, 50, 100, 200, 300])
    plot_data(df, ticker)

if __name__ == '__main__':
    main()

