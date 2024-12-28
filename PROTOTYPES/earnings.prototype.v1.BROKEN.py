import os
import sys
import argparse
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import warnings
import numpy as np

# Suppress UserWarnings (optional)
warnings.filterwarnings('ignore', category=UserWarning)

def fetch_stock_data(ticker, start_date, end_date, save_path):
    """
    Downloads stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_path (str): File path to save the CSV data.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    """
    try:
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No data found for ticker {ticker}. Please check the ticker symbol and date range.")
            sys.exit(1)
        # Reset index to have 'Date' as a column
        data.reset_index(inplace=True)
        data.to_csv(save_path, index=False)
        print(f"Data for {ticker} saved to {save_path}.")
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        sys.exit(1)

def load_stock_data(ticker, data_dir, start_date, end_date):
    """
    Loads stock data from local disk if available, otherwise downloads it.

    Args:
        ticker (str): Stock ticker symbol.
        data_dir (str): Directory to store/load data.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing the stock data.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{ticker}.csv")

    if os.path.isfile(file_path):
        try:
            print(f"Loading existing data for {ticker} from {file_path}...")
            data = pd.read_csv(file_path, parse_dates=['Date'])
            # Filter data for the desired date range
            mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
            data_filtered = data.loc[mask]
            if data_filtered.empty:
                print(f"No data in the specified date range for {ticker}. Downloading missing data.")
                # Download data and append
                data_new = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data_new.empty:
                    print(f"No additional data found for ticker {ticker}.")
                    sys.exit(1)
                data_new.reset_index(inplace=True)
                data = pd.concat([data, data_new], ignore_index=True)
                data.drop_duplicates(subset='Date', keep='last', inplace=True)
                data.to_csv(file_path, index=False)
                print(f"Data for {ticker} updated and saved to {file_path}.")
            else:
                print(f"Data for {ticker} loaded successfully.")
            return data_filtered
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            sys.exit(1)
    else:
        # Download data and save
        data = fetch_stock_data(ticker, start_date, end_date, file_path)
        return data

def calculate_indicators(data):
    """
    Calculates technical indicators and adds them to the DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing stock data.

    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    data.set_index('Date', inplace=True)
    data['20DMA'] = data['Close'].rolling(window=20).mean()
    data['50DMA'] = data['Close'].rolling(window=50).mean()
    data['9DMA'] = data['Close'].rolling(window=9).mean()
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper_1std'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 1)
    data['BB_Upper_2std'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower_1std'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 1)
    data['BB_Lower_2std'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
    return data

def plot_stock_data(data, ticker):
    """
    Plots stock data with technical indicators using Plotly.

    Args:
        data (pd.DataFrame): DataFrame containing stock data with indicators.
        ticker (str): Stock ticker symbol.
    """
    # Fetch earnings dates
    stock = yf.Ticker(ticker)
    try:
        earnings_dates = stock.get_earnings_dates()
        # Correctly filter earnings dates within the data's date range
        earnings_dates = earnings_dates[
            (earnings_dates.index >= pd.to_datetime(data.index.min())) &
            (earnings_dates.index <= pd.to_datetime(data.index.max()))
        ]

    except Exception as e:
        print(f"Error fetching earnings dates: {e}")
        earnings_dates = pd.DataFrame()

    # Create the candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks'
    )

    # Add moving averages
    ma_20 = go.Scatter(
        x=data.index,
        y=data['20DMA'],
        mode='lines',
        name='20DMA',
        line=dict(color='blue')
    )
    ma_50 = go.Scatter(
        x=data.index,
        y=data['50DMA'],
        mode='lines',
        name='50DMA',
        line=dict(color='red')
    )
    ma_9 = go.Scatter(
        x=data.index,
        y=data['9DMA'],
        mode='lines',
        name='9DMA',
        line=dict(color='purple')
    )

    # Add Bollinger Bands
    bb_middle = go.Scatter(
        x=data.index,
        y=data['BB_Middle'],
        mode='lines',
        name='BB_Middle',
        line=dict(color='gray', dash='dash')
    )
    bb_upper_1std = go.Scatter(
        x=data.index,
        y=data['BB_Upper_1std'],
        mode='lines',
        name='BB_Upper_1std',
        line=dict(color='green', dash='dot')
    )
    bb_upper_2std = go.Scatter(
        x=data.index,
        y=data['BB_Upper_2std'],
        mode='lines',
        name='BB_Upper_2std',
        line=dict(color='lightgreen', dash='dot')
    )
    bb_lower_1std = go.Scatter(
        x=data.index,
        y=data['BB_Lower_1std'],
        mode='lines',
        name='BB_Lower_1std',
        line=dict(color='orange', dash='dot')
    )
    bb_lower_2std = go.Scatter(
        x=data.index,
        y=data['BB_Lower_2std'],
        mode='lines',
        name='BB_Lower_2std',
        line=dict(color='pink', dash='dot')
    )

    # Add dots for open and close prices
    open_dots = go.Scatter(
        x=data.index,
        y=data['Open'],
        mode='markers',
        name='Open',
        marker=dict(color='cyan', size=4)
    )
    close_dots = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='markers',
        name='Close',
        marker=dict(color='white', size=4)
    )

    # Add dots for high and low prices
    high_dots = go.Scatter(
        x=data.index,
        y=data['High'],
        mode='markers',
        name='High',
        marker=dict(color='green', size=4)
    )
    low_dots = go.Scatter(
        x=data.index,
        y=data['Low'],
        mode='markers',
        name='Low',
        marker=dict(color='yellow', size=4)
    )

    # Add dots for midpoint prices
    midpoint_dots = go.Scatter(
        x=data.index,
        y=(data['High'] + data['Low']) / 2,
        mode='markers',
        name='Midpoint',
        marker=dict(color='orange', size=4)
    )

    # Conditional Dots for High/Low beyond 2std Bollinger Bands
    high_beyond_2std_upper = go.Scatter(
        x=data.index[data['High'] > data['BB_Upper_2std']],
        y=data['High'][data['High'] > data['BB_Upper_2std']],
        mode='markers',
        name='High > 2std Upper',
        marker=dict(color='lawngreen', size=8, line=dict(color='lawngreen', width=2))
    )
    low_below_2std_lower = go.Scatter(
        x=data.index[data['Low'] < data['BB_Lower_2std']],
        y=data['Low'][data['Low'] < data['BB_Lower_2std']],
        mode='markers',
        name='Low < 2std Lower',
        marker=dict(color='red', size=8, line=dict(color='red', width=2))
    )

    # Fill the area between 20DMA and 50DMA
    fill_between = []
    for i in range(1, len(data)):
        if not pd.isna(data['20DMA'].iloc[i]) and not pd.isna(data['50DMA'].iloc[i]):
            if data['20DMA'].iloc[i] > data['50DMA'].iloc[i]:
                fill_between.append(go.Scatter(
                    x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                    y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                    fill='toself',
                    fillcolor='rgba(0, 255, 0, 0.2)',
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            elif data['20DMA'].iloc[i] < data['50DMA'].iloc[i]:
                fill_between.append(go.Scatter(
                    x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                    y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Horizontal Lines for Highest/Lowest points beyond 2std Bollinger Bands (last 6 months)
    six_months_ago = data.index[-1] - pd.DateOffset(months=6)
    last_six_months_data = data[data.index > six_months_ago]

    highest_above_2std_upper = last_six_months_data['High'][last_six_months_data['High'] > last_six_months_data['BB_Upper_2std']].max()
    lowest_below_2std_lower = last_six_months_data['Low'][last_six_months_data['Low'] < last_six_months_data['BB_Lower_2std']].min()

    highest_line = None
    if not pd.isna(highest_above_2std_upper):
        highest_line = go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[highest_above_2std_upper, highest_above_2std_upper],
            mode='lines',
            name='Highest Above 2std Upper (Last 6M)',
            line=dict(color='lawngreen', dash='dash', width=1)
        )

    lowest_line = None
    if not pd.isna(lowest_below_2std_lower):
        lowest_line = go.Scatter(
            x=[data.index[0], data.index[-1]],
            y=[lowest_below_2std_lower, lowest_below_2std_lower],
            mode='lines',
            name='Lowest Below 2std Lower (Last 6M)',
            line=dict(color='red', dash='dash', width=1)
        )

    # Add earnings date markers
    earnings_markers = []
    if not earnings_dates.empty:
        for index, row in earnings_dates.iterrows():
            date = index
            if date in data.index:
                earnings_markers.append(go.Scatter(
                    x=[date],
                    y=[data.loc[date, 'High'] * 1.05],  # Place marker slightly above the 'High' price
                    mode='markers',
                    name='Earnings Date',
                    marker=dict(symbol='diamond-wide', color='magenta', size=10, line=dict(color='white', width=1)),
                    text=[f"Earnings: {row['EPS Estimate']}"],
                    hoverinfo='text+x+y'
                ))

    # Combine all traces
    traces = [
        candlestick, ma_20, ma_50, ma_9, bb_middle, bb_upper_1std, bb_upper_2std, bb_lower_1std, bb_lower_2std,
        open_dots, close_dots, high_dots, low_dots, midpoint_dots,
        high_beyond_2std_upper, low_below_2std_lower
    ] + fill_between + earnings_markers

    if highest_line:
        traces.append(highest_line)
    if lowest_line:
        traces.append(lowest_line)

    # Layout configuration
    layout = go.Layout(
        title=f'{ticker} Stock Price and Moving Averages',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price ($)'),
        template='plotly_dark',
        legend=dict(x=0, y=1.0),
        hovermode='x'
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Update layout for legend and appearance
    fig.update_layout(legend_title_text='Indicators', showlegend=True)

    # Show plot
    fig.show()

def main():
    """
    Main function to execute the script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fetch and plot stock data with technical indicators.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--start_date', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument
