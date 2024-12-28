import os
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import linregress

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

# Function to plot the data with technical indicators and show on matplotlib
def plot_data(df, ticker, second_ticker):
    plt.figure(figsize=(12, 8))

    desired_values = [.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 3, 4]
    plt.plot(df.index, df['Relative_Close'], label='Relative Price', linestyle='-', color='black')

    # Adding dashed lines for the regression high and low bands
    colors = ['grey', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'blue']
    for i, num_std in enumerate(desired_values):
        plt.plot(df.index, df[f'Reg_High_{i+1}std'], linestyle=':', color=colors[i], label=f'Reg High {num_std} std')
        plt.plot(df.index, df[f'Reg_Low_{i+1}std'], linestyle=':', color=colors[i], label=f'Reg Low {num_std} std')

    # Adding EMAs to the plot
    ema_colors = ['purple', 'orange', 'green', 'red', 'blue']
    for i, time_period in enumerate([20, 50, 100, 200, 300]):
        plt.plot(df.index, df[f'EMA_{time_period}'], label=f'{time_period}-day EMA', color=ema_colors[i])

    # Add support and resistance lines
    highest_resistance_value, lowest_support_value = add_support_resistance(df)
    plt.axhline(y=highest_resistance_value, color='r', label='Resistance', linestyle='-')
    plt.axhline(y=lowest_support_value, color='g', label='Support', linestyle='-')

    # Add peak and trough trend lines
    (highest_peak_date, highest_peak_value), (lowest_trough_date, lowest_trough_value) = add_peak_trough_trend_lines(df)
    plt.plot([highest_peak_date, df.index[-1]], [highest_peak_value, df['Relative_Close'][-1]], linestyle='--', color='green', label='Trend Line - Peaks')
    plt.plot([lowest_trough_date, df.index[-1]], [lowest_trough_value, df['Relative_Close'][-1]], linestyle='--', color='red', label='Trend Line - Troughs')

    # Add interactive click handling
    fig = plt.gcf()
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the plot
    plt.title(f"{ticker} / {second_ticker} Relative Price Analysis")
    plt.xlabel('Date')
    plt.ylabel('Relative Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Initialize variables to store the clicked points
clicked_points = []

# Function to handle mouse clicks
def on_click(event):
    global clicked_points

    if event.inaxes:  # Check if the click is within the axes
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        print(f"Point clicked: ({x}, {y})")

        # If two points are clicked, draw the line
        if len(clicked_points) == 2:
            x0, y0 = clicked_points[0]
            x1, y1 = clicked_points[1]

            # Plot the dashed line
            plt.plot([x0, x1], [y0, y1], color='blue', linestyle='--', label='User Line')

            # Show price information on the line
            print(f"Price at point 1: {y0} (Date: {x0})")
            print(f"Price at point 2: {y1} (Date: {x1})")

            # Redraw the plot with the line added
            plt.legend()
            plt.draw()

# Main function to execute the analysis
def main():
    if len(sys.argv) != 4:
        print("Usage: python ip_compare2tickersSupportResistPeakTroughs_LRC_EMA_candlesticks.v5.py <ticker1> <ticker2> <period>")
        sys.exit(1)

    ticker = sys.argv[1].lower()  # Get the first ticker from command line argument and convert to lowercase
    second_ticker = sys.argv[2].lower()  # Get the second ticker from command line argument and convert to lowercase
    period = sys.argv[3]  # Get the period from command line argument

    df, second_df = load_or_download_data(ticker, period), load_or_download_data(second_ticker, period)

    # Create combined folder for the tickers
    combined_dir = f"data/{datetime.today().strftime('%Y-%m-%d')}/{ticker}_{second_ticker}"
    os.makedirs(combined_dir, exist_ok=True)

    # Align the data by index and calculate relative price
    df = df.loc[df.index.isin(second_df.index)]
    second_df = second_df.loc[second_df.index.isin(df.index)]
    df['Relative_Close'] = df['Close'] / second_df['Close']

    df = add_linear_regression_bands(df)
    df = add_ema(df, [20, 50, 100, 200, 300])
    
    plot_data(df, ticker, second_ticker)

if __name__ == '__main__':
    main()

