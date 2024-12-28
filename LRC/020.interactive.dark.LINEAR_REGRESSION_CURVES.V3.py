import os
import sys
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import imageio.v2 as imageio
import numpy as np
from scipy.stats import linregress

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
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}.")
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_linear_regression_curves(df, spans):
    for span in spans:
        df, curve_color = add_linear_regression_curve(df, window=span, curve_color=f'Linear_Regression_{span}')
    return df

def add_linear_regression_curve(df, window=20, curve_color='cyan'):
    def linear_reg_slope_intercept(y_values):
        x_values = np.arange(len(y_values))
        slope, intercept, _, _, _ = linregress(x_values, y_values)
        return slope, intercept

    rolling_slope_intercept = df['Close'].rolling(window=window).apply(
        lambda x: linear_reg_slope_intercept(x)[0] * (window - 1) + linear_reg_slope_intercept(x)[1], raw=False)

    df[f'Linear_Regression_{window}'] = rolling_slope_intercept
    return df, curve_color

def plot_data(df, ticker, period_name, start_date):
    if df.empty or len(df) < 50:
        print(f"Not enough data for {period_name}. Skipping.")
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
    for span in [12, 20, 50, 100, 150]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Linear_Regression_{span}'], mode='lines', name=f'Linear Regression {span}'))

    fig.update_layout(
        title=f'{ticker} Closing Price and Linear Regression Curves from {start_date.strftime("%Y-%m-%d")} to {datetime.today().strftime("%Y-%m-%d")}', 
        xaxis_title='Date', 
        yaxis_title='Price', 
        template='plotly_dark'
    )

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_{period_name.replace(" ", "_").lower()}_lrc.png')
    
    fig.write_image(image_file, width=1920, height=1080, scale=4)

    fig.show()  # Display the interactive plot

    return image_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1]
    try:
        data = download_stock_data(ticker)
    except ValueError as e:
        print(e)
        sys.exit(1)

    data = calculate_linear_regression_curves(data, spans=[12, 20, 50, 100, 150])
    
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
        image_file = plot_data(data_period, ticker, period_name, start_date)
        if image_file:
            image_files.append(image_file)
    
    if image_files:
        gif_path = f'images/{ticker}/{ticker}_lrc.gif'
        with imageio.get_writer(gif_path, mode='I', duration=1, loop=0) as writer:
            for image_file in image_files:
                image = imageio.imread(image_file)
                writer.append_data(image)
        print(f"GIF created at {gif_path}")
    else:
        print("No images were created due to insufficient data.")

if __name__ == '__main__':
    main()

