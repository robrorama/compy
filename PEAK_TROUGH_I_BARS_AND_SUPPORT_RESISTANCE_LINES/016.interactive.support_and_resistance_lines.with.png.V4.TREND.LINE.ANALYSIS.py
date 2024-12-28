import os
import sys
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
from plotly.subplots import make_subplots

def download_data(ticker, period="max"):
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
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def plot_data_with_trend_lines(df, ticker):
    # Create candlestick trace
    trace = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlesticks'
    )

    # Set up the Plotly figure
    fig = go.Figure(trace)

    # Get the last 30 days of data
    last_30_days = df.loc[df.index[-30:]]

    # Find the highest resistance level in the last 30 days and add a trend line
    highest_resistance_date = last_30_days['Close'].idxmax()
    highest_resistance_value = last_30_days['Close'].max()

    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]],
                             y=[highest_resistance_value, highest_resistance_value],
                             mode='lines', line=dict(color='red'), name='Resistance'))

    # Find the lowest support level in the last 30 days and add a trend line
    lowest_support_date = last_30_days['Close'].idxmin()
    lowest_support_value = last_30_days['Close'].min()

    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]],
                             y=[lowest_support_value, lowest_support_value],
                             mode='lines', line=dict(color='green'), name='Support'))

    # Add layout and show figure
    fig.update_layout(title=f"{ticker} Trend Line Analysis", xaxis_title='Date', yaxis_title='Price')
    
    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_trend_line_analysis.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")
    
    # Show the plot
    fig.show()

def main():
    ticker = sys.argv[1]  # Get the ticker from command line argument
    period = sys.argv[2] if len(sys.argv) > 2 else "max"  # Get the period from command line argument or use default "max"
    df = download_data(ticker, period)
    plot_data_with_trend_lines(df, ticker)

if __name__ == '__main__':
    main()

