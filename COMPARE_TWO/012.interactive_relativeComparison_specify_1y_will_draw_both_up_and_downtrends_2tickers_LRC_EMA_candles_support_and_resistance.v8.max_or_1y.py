import os
import sys
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from scipy.stats import linregress
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# Function to load or download data and save to CSV
def load_or_download_data(ticker, period='max'):
    today_str = datetime.today().strftime("%Y-%m-%d")
    ticker = ticker.lower()
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' column is of datetime type
        df.set_index('Date', inplace=True)  # Set 'Date' as the index
        print(f"Data for {ticker} loaded from disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        if df.empty:
            print(f"Failed to download data for {ticker}.")
            sys.exit(1)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

# Function to filter the data based on the specified period
def filter_data_by_period(df, period):
    if period == 'max':
        return df
    else:
        end_date = df.index.max()
        if period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '3mo':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)  # Default to 1 year if period is not recognized
        
        return df.loc[start_date:end_date]

# Function to add linear regression bands to the DataFrame
def add_linear_regression_bands(df):
    df_numeric_index = pd.to_numeric(df.index)
    slope, intercept, _, _, _ = linregress(df_numeric_index, df['Relative_Close'])
    df['Linear_Reg'] = intercept + slope * df_numeric_index
    df['Residuals'] = df['Relative_Close'] - df['Linear_Reg']
    residuals_std = df['Residuals'].std()

    desired_values = [.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    for i, num_std in enumerate(desired_values):
        df[f'Reg_High_{i+1}std'] = df['Linear_Reg'] + residuals_std * num_std
        df[f'Reg_Low_{i+1}std'] = df['Linear_Reg'] - residuals_std * num_std
    return df

# Function to add EMAs to the DataFrame
def add_ema(df, time_periods):
    for time_period in time_periods:
        df[f'EMA_{time_period}'] = df['Relative_Close'].ewm(span=time_period, adjust=False).mean()
    return df

# Function to add support and resistance lines
def add_support_resistance(df):
    last_30_days = df.loc[df.index[-30:]]

    # Find the highest resistance level in the last 30 days
    highest_resistance_value = last_30_days['Relative_Close'].max()

    # Find the lowest support level in the last 30 days
    lowest_support_value = last_30_days['Relative_Close'].min()

    return highest_resistance_value, lowest_support_value

# Function to add peak and trough trend lines
def add_peak_trough_trend_lines(df):
    # Get the last six months of data
    six_months_ago = df.index[-1] - timedelta(days=180)
    last_six_months_data = df.loc[six_months_ago:]

    # Find the highest peak in the last six months
    highest_peak_last_six_months_date = last_six_months_data['Relative_Close'].idxmax()
    highest_peak_last_six_months_value = last_six_months_data['Relative_Close'].max()

    # Get the last year of data
    one_year_ago = df.index[-1] - timedelta(days=365)
    last_year_data = df.loc[one_year_ago:]

    # Find the lowest trough in the last year
    lowest_point_last_year_date = last_year_data['Relative_Close'].idxmin()
    lowest_point_last_year_value = last_year_data['Relative_Close'].min()

    return (highest_peak_last_six_months_date, highest_peak_last_six_months_value), (lowest_point_last_year_date, lowest_point_last_year_value)

# Function to plot the data with technical indicators and save as PNG
def plot_data(df, ticker, second_ticker):
    fig = go.Figure()

    desired_values = [.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    fig.add_trace(go.Scatter(x=df.index, y=df['Relative_Close'], mode='lines', name='Relative Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Linear_Reg'], line=dict(color='blue', width=2), name='Linear Regression'))

    colors = ['grey', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'blue']
    for i, num_std in enumerate(desired_values):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_High_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg High {num_std} std'))
        fig.add_trace(go.Scatter(x=df.index, y=df[f'Reg_Low_{i+1}std'], line=dict(color=colors[i], width=1, dash='dot'), name=f'Reg Low {num_std} std'))

    ema_colors = ['purple', 'orange', 'green', 'red', 'blue']
    for i, time_period in enumerate([20, 50, 100, 200, 300]):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'EMA_{time_period}'], line=dict(color=ema_colors[i], width=2), name=f'{time_period}-day EMA'))

    # Add support and resistance lines
    highest_resistance_value, lowest_support_value = add_support_resistance(df)
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[highest_resistance_value, highest_resistance_value], mode='lines', line=dict(color='red'), name='Resistance'))
    fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[lowest_support_value, lowest_support_value], mode='lines', line=dict(color='green'), name='Support'))

    # Add peak and trough trend lines
    (highest_peak_date, highest_peak_value), (lowest_trough_date, lowest_trough_value) = add_peak_trough_trend_lines(df)
    fig.add_trace(go.Scatter(x=[highest_peak_date, df.index[-1]], y=[highest_peak_value, df['Relative_Close'][-1]], mode='lines', line=dict(color='green', dash='dash'), name='Trend Line - Peaks'))
    fig.add_trace(go.Scatter(x=[lowest_trough_date, df.index[-1]], y=[lowest_trough_value, df['Relative_Close'][-1]], mode='lines', line=dict(color='red', dash='dash'), name='Trend Line - Troughs'))

    fig.update_layout(
        title=f"{ticker} / {second_ticker} Relative Price Analysis",
        xaxis_title='Date',
        yaxis_title='Relative Price',
        height=800,
        width=1200,
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # Save the plot as a PNG file with high resolution
    image_dir = f"images/{ticker}_{second_ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_vs_{second_ticker}_relative_analysis.png')
    fig.write_image(image_file, width=1920, height=1080, scale=4)
    print(f"Plot saved as {image_file}")

    # Show the plot
    fig.show()

# Main function to run the analysis
def main():
    if len(sys.argv) != 4:
        print("Usage: python ip_compare2tickersSupportResistPeakTroughs_LRC_EMA_candlesticks.v5.py <ticker1> <ticker2> <period>")
        sys.exit(1)

    ticker = sys.argv[1].lower()  # Get the first ticker from command line argument and convert to lowercase
    second_ticker = sys.argv[2].lower()  # Get the second ticker from command line argument and convert to lowercase
    period = sys.argv[3]  # Get the period from command line argument

    df, second_df = load_or_download_data(ticker, period), load_or_download_data(second_ticker, period)

    # Filter data by specified period
    df = filter_data_by_period(df, period)
    second_df = filter_data_by_period(second_df, period)

    # Align the data by index and calculate relative price
    df = df.loc[df.index.isin(second_df.index)]
    second_df = second_df.loc[second_df.index.isin(df.index)]
    df['Relative_Close'] = df['Close'] / second_df['Close']

    df = add_linear_regression_bands(df)
    df = add_ema(df, [20, 50, 100, 200, 300])
    
    plot_data(df, ticker, second_ticker)

if __name__ == '__main__':
    main()

