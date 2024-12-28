import os
import sys
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import talib
from datetime import datetime, timedelta

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
def plot_data_as_candlesticks(df, ticker):
    #last 90 days only
    if df.empty:
        print(f"No data to plot for {ticker}.")
        return

    # Get the last 90 days of data
    last_90_days = df.index[-1] - timedelta(days=90)
    last_90_days_data = df.loc[last_90_days:]

    if last_90_days_data.empty:
        print(f"No data available for the last 90 days for {ticker}.")
        return

    # Create candlestick trace
    trace = go.Candlestick(
        x=last_90_days_data.index,
        open=last_90_days_data['Open'],
        high=last_90_days_data['High'],
        low=last_90_days_data['Low'],
        close=last_90_days_data['Close'],
        name='Candlesticks'
    )

    # Set up the Plotly figure
    fig = go.Figure(trace)

    # Find the highest peak in the last 90 days and add a trend line
    highest_peak_last_90_days_date = last_90_days_data['Close'].idxmax()
    highest_peak_last_90_days_value = last_90_days_data['Close'].max()

    fig.add_trace(go.Scatter(x=[highest_peak_last_90_days_date, last_90_days_data.index[-1]],
                             y=[highest_peak_last_90_days_value, last_90_days_data['Close'][-1]],
                             mode='lines', line=dict(color='red'), name='Trend Line - Peaks'))

    # Find the lowest trough in the last 90 days and add a trend line
    lowest_point_last_90_days_date = last_90_days_data['Close'].idxmin()
    lowest_point_last_90_days_value = last_90_days_data['Close'].min()

    fig.add_trace(go.Scatter(x=[lowest_point_last_90_days_date, last_90_days_data.index[-1]],
                             y=[lowest_point_last_90_days_value, last_90_days_data['Close'][-1]],
                             mode='lines', line=dict(color='green'), name='Trend Line - Troughs'))

    # Highlight the selected patterns on the chart
    for pattern in ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                    'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                    'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
                    'CDLHARAMI']:
        fig.add_trace(go.Scatter(
            x=last_90_days_data.index,
            y=last_90_days_data[pattern],
            mode='markers',
            name=pattern[3:]
        ))

    # Set the y-axis range to start from 0
    fig.update_yaxes(range=[0, max(last_90_days_data['High'])])

    # Add layout and save figure
    fig.update_layout(title=f"{ticker} Trend Line and Candlestick Patterns Analysis (Last 90 Days)", xaxis_title='Date', yaxis_title='Price')

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_trend_line_candlestick_patterns_analysis_last_90_days.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()


def plot_data_as_candlesticksONLY_LAST_30_days(df, ticker):
    if df.empty:
        print(f"No data to plot for {ticker}.")
        return

    # Get the last 30 days of data
    last_30_days = df.index[-1] - timedelta(days=30)
    last_30_days_data = df.loc[last_30_days:]

    if last_30_days_data.empty:
        print(f"No data available for the last 30 days for {ticker}.")
        return

    # Create candlestick trace
    trace = go.Candlestick(
        x=last_30_days_data.index,
        open=last_30_days_data['Open'],
        high=last_30_days_data['High'],
        low=last_30_days_data['Low'],
        close=last_30_days_data['Close'],
        name='Candlesticks'
    )

    # Set up the Plotly figure
    fig = go.Figure(trace)

    # Find the highest peak in the last 30 days and add a trend line
    highest_peak_last_30_days_date = last_30_days_data['Close'].idxmax()
    highest_peak_last_30_days_value = last_30_days_data['Close'].max()

    fig.add_trace(go.Scatter(x=[highest_peak_last_30_days_date, last_30_days_data.index[-1]],
                             y=[highest_peak_last_30_days_value, last_30_days_data['Close'][-1]],
                             mode='lines', line=dict(color='red'), name='Trend Line - Peaks'))

    # Find the lowest trough in the last 30 days and add a trend line
    lowest_point_last_30_days_date = last_30_days_data['Close'].idxmin()
    lowest_point_last_30_days_value = last_30_days_data['Close'].min()

    fig.add_trace(go.Scatter(x=[lowest_point_last_30_days_date, last_30_days_data.index[-1]],
                             y=[lowest_point_last_30_days_value, last_30_days_data['Close'][-1]],
                             mode='lines', line=dict(color='green'), name='Trend Line - Troughs'))

    # Highlight the selected patterns on the chart
    for pattern in ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                    'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                    'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
                    'CDLHARAMI']:
        fig.add_trace(go.Scatter(
            x=last_30_days_data.index,
            y=last_30_days_data[pattern],
            mode='markers',
            name=pattern[3:]
        ))

    # Add layout and save figure
    fig.update_layout(title=f"{ticker} Trend Line and Candlestick Patterns Analysis (Last 30 Days)", xaxis_title='Date', yaxis_title='Price')

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_trend_line_candlestick_patterns_analysis_last_30_days.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()

def plot_data_as_candlesticksOriginal_plotted_as_muchData_asDownloaded(df, ticker):
    if df.empty:
        print(f"No data to plot for {ticker}.")
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

    # Get the last six months of data
    six_months_ago = df.index[-1] - timedelta(days=180)
    last_six_months_data = df.loc[six_months_ago:]

    if last_six_months_data.empty:
        print(f"No data available for the last six months for {ticker}.")
        return

    # Find the highest peak in the last six months and add a trend line
    highest_peak_last_six_months_date = last_six_months_data['Close'].idxmax()
    highest_peak_last_six_months_value = last_six_months_data['Close'].max()

    fig.add_trace(go.Scatter(x=[highest_peak_last_six_months_date, df.index[-1]],
                             y=[highest_peak_last_six_months_value, df['Close'][-1]],
                             mode='lines', line=dict(color='red'), name='Trend Line - Peaks'))

    # Get the last year of data
    one_year_ago = df.index[-1] - timedelta(days=365)
    last_year_data = df.loc[one_year_ago:]

    if last_year_data.empty:
        print(f"No data available for the last year for {ticker}.")
        return

    # Find the lowest trough in the last year and add a trend line
    lowest_point_last_year_date = last_year_data['Close'].idxmin()
    lowest_point_last_year_value = last_year_data['Close'].min()

    fig.add_trace(go.Scatter(x=[lowest_point_last_year_date, df.index[-1]],
                             y=[lowest_point_last_year_value, df['Close'][-1]],
                             mode='lines', line=dict(color='green'), name='Trend Line - Troughs'))

    # Highlight the selected patterns on the chart
    for pattern in ['CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLENGULFING',
                    'CDLPIERCING', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
                    'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
                    'CDLHARAMI']:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[pattern],
            mode='markers',
            name=pattern[3:]
        ))

    # Add layout and save figure
    fig.update_layout(title=f"{ticker} Trend Line and Candlestick Patterns Analysis", xaxis_title='Date', yaxis_title='Price')

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_trend_line_candlestick_patterns_analysis.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()

# Main function to run the analysis
def main():
    ticker = sys.argv[1]  # Get the ticker from command line argument
    period = sys.argv[2] if len(sys.argv) > 2 else "max"  # Get the period from command line argument or use default "max"
    df = download_data(ticker, period)
    df = add_selected_candlestick_patterns(df)
    plot_data_as_candlesticks(df, ticker)

if __name__ == '__main__':
    main()

