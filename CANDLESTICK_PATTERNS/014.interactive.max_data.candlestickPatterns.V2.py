import os
import sys
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import talib
from datetime import datetime

# Function to download data and save to CSV
def download_data(ticker, period):
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
        df = yf.download(ticker, period=period)
        if df.empty:
            print(f"Failed to download data for {ticker}.")
            sys.exit(1)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

# Function to add selected candlestick patterns
def add_selected_candlestick_patterns(df):
    # Selected patterns
    selected_patterns = ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                         'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                         'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
                         'CDLHARAMI']

    for pattern in selected_patterns:
        pattern_function = getattr(talib, pattern)
        df[pattern] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])

    return df

# Function to plot the data as candlestick charts using Plotly
def plot_data(df, title):
    if df.empty:
        print(f"No data to plot for {title}.")
        return

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

    # Highlight the selected patterns on the chart
    selected_patterns = ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                         'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                         'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
                         'CDLHARAMI']

    for pattern in selected_patterns:
        pattern_indices = df[df[pattern] != 0].index
        fig.add_trace(go.Scatter(
            x=pattern_indices,
            y=df.loc[pattern_indices, 'Close'],
            mode='markers',
            name=pattern[3:]
        ))

    # Add layout
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')

    # Show the plot
    fig.show()

# Main function to run the analysis
def main():
    ticker = sys.argv[1]  # Get the ticker from command line argument
    period = sys.argv[2] if len(sys.argv) > 2 else "max"  # Get the period from command line argument or use default "max"
    df = download_data(ticker, period)
    df = add_selected_candlestick_patterns(df)
    plot_data(df, title=f"{ticker} Candlestick Patterns Analysis")

if __name__ == '__main__':
    main()

