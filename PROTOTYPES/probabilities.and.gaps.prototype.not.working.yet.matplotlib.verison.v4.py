import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import numpy as np

def download_data(ticker, save_path):
    """
    Downloads historical data for the specified ticker using yfinance and saves it as a CSV file.

    Args:
        ticker (str): The ticker symbol to download data for.
        save_path (str): The file path to save the downloaded CSV data.

    Returns:
        pd.DataFrame: The downloaded historical data.
    """
    try:
        print(f"Downloading data for ticker '{ticker}' from yfinance...")
        data = yf.download(ticker, start="2000-01-01", progress=False)
        if data.empty:
            print(f"Error: No data found for ticker '{ticker}'. Please check the ticker symbol.")
            sys.exit(1)
        data.to_csv(save_path)
        print(f"Data for '{ticker}' saved to '{save_path}'.")
        return data
    except Exception as e:
        print(f"Error downloading data for ticker '{ticker}': {e}")
        sys.exit(1)

def load_data(ticker, data_dir):
    """
    Loads historical data for the specified ticker. Downloads the data if not found locally.

    Args:
        ticker (str): The ticker symbol to load data for.
        data_dir (str): The base directory where data is stored.

    Returns:
        pd.DataFrame: The historical data for the ticker.
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    ticker_dir = os.path.join(data_dir, today_str)
    os.makedirs(ticker_dir, exist_ok=True)
    file_path = os.path.join(ticker_dir, f"{ticker}.csv")

    if os.path.isfile(file_path):
        print(f"Loading existing data for '{ticker}' from '{file_path}'.")
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            return data
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
            sys.exit(1)
    else:
        # Download data and save
        data = download_data(ticker, file_path)
        return data

def preprocess_data(df):
    """
    Preprocesses the DataFrame by computing necessary columns.

    Args:
        df (pd.DataFrame): The raw historical data.

    Returns:
        pd.DataFrame: The preprocessed data with additional columns.
    """
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Compute daily percentage change
    df['Pct_Change'] = df['Adj Close'].pct_change()

    # Determine Gap_Direction
    # Assuming 'Gap' refers to a gap up or gap down from the previous day's close to today's open
    # 1 for Gap Up, -1 for Gap Down, 0 for no gap
    df['Gap_Direction'] = 0
    df['Gap_Direction'] = np.where(df['Open'] > df['Close'].shift(1), 1, 
                                   np.where(df['Open'] < df['Close'].shift(1), -1, 0))

    # Drop rows with NaN in 'Pct_Change' or 'Gap_Direction'
    df.dropna(subset=['Pct_Change', 'Gap_Direction'], inplace=True)

    return df

def probability_after_gap(df, days_ahead=5):
    """
    Computes the probabilities of price movements after a gap event.

    Args:
        df (pd.DataFrame): The preprocessed historical data.
        days_ahead (int): Number of days ahead to analyze.

    Returns:
        dict: Dictionary containing probability metrics.
    """
    # Initialize counters
    gap_up_higher = 0
    gap_up_lower = 0
    gap_down_higher = 0
    gap_down_lower = 0
    total_gap_up = 0
    total_gap_down = 0

    # Iterate through the DataFrame
    for idx in range(len(df) - days_ahead):
        current_gap = df['Gap_Direction'].iloc[idx]
        future_return = df['Pct_Change'].iloc[idx + days_ahead]

        if current_gap == 1:
            total_gap_up += 1
            if future_return > 0:
                gap_up_higher += 1
            elif future_return < 0:
                gap_up_lower += 1
        elif current_gap == -1:
            total_gap_down += 1
            if future_return > 0:
                gap_down_higher += 1
            elif future_return < 0:
                gap_down_lower += 1

    # Compute probabilities
    gap_up_higher_prob = gap_up_higher / total_gap_up if total_gap_up > 0 else 0
    gap_up_lower_prob = gap_up_lower / total_gap_up if total_gap_up > 0 else 0
    gap_down_higher_prob = gap_down_higher / total_gap_down if total_gap_down > 0 else 0
    gap_down_lower_prob = gap_down_lower / total_gap_down if total_gap_down > 0 else 0

    return {
        'gap_up_higher_prob': gap_up_higher_prob,
        'gap_up_lower_prob': gap_up_lower_prob,
        'gap_down_higher_prob': gap_down_higher_prob,
        'gap_down_lower_prob': gap_down_lower_prob
    }

def visualize_ticker(ticker, df, days_ahead=5):
    """
    Visualizes various statistics for a given ticker.

    Args:
        ticker (str): Ticker symbol (e.g., 'SPY').
        df (pd.DataFrame): Preprocessed historical data for the ticker.
        days_ahead (int): Number of days ahead to analyze in probability_after_gap.
    """
    # 1. Plot Adjusted Close Price and Highlight Gap Days
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Adj Close'], label='Adj Close', color='blue', linewidth=1)

    # Highlight gap up days (Gap_Direction = 1) in green
    gap_up_dates = df.index[df['Gap_Direction'] == 1]
    plt.scatter(gap_up_dates, df.loc[gap_up_dates, 'Adj Close'], color='green', marker='^', label='Gap Up', alpha=0.7)

    # Highlight gap down days (Gap_Direction = -1) in red
    gap_down_dates = df.index[df['Gap_Direction'] == -1]
    plt.scatter(gap_down_dates, df.loc[gap_down_dates, 'Adj Close'], color='red', marker='v', label='Gap Down', alpha=0.7)

    plt.title(f'{ticker}: Adjusted Close Price with Gap Events Highlighted', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Adjusted Close Price ($)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 2. Distribution of Daily Returns (Pct_Change)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Pct_Change'].dropna(), kde=True, color='purple', bins=50)
    plt.title(f'{ticker}: Distribution of Daily % Changes', fontsize=16)
    plt.xlabel('Daily % Change', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3. Comparing Probabilities as a Bar Chart
    results = probability_after_gap(df, days_ahead=days_ahead)

    # Create a small dataframe for plotting these probabilities
    prob_df = pd.DataFrame({
        'Condition': ['Gap Up Higher', 'Gap Up Lower', 'Gap Down Higher', 'Gap Down Lower'],
        'Probability': [
            results['gap_up_higher_prob'], 
            results['gap_up_lower_prob'], 
            results['gap_down_higher_prob'], 
            results['gap_down_lower_prob']
        ]
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Condition', y='Probability', data=prob_df, palette='coolwarm')
    plt.title(f'{ticker}: Probability {days_ahead} Days After Gap', fontsize=16)
    plt.ylim(0, 1)  # Probabilities range from 0 to 1
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    # Annotate bars with probability values
    for index, row in prob_df.iterrows():
        plt.text(index, row['Probability'] + 0.01, f"{row['Probability']:.2f}", ha='center', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the visualization script.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize stock data with gap events using yfinance data.")
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., SPY)')
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory to store/download data')
    parser.add_argument('--days_ahead', type=int, default=5, help='Number of days ahead to analyze after a gap event')

    args = parser.parse_args()

    ticker = args.ticker.upper()
    data_dir = args.data_dir
    days_ahead = args.days_ahead

    # Load or download data
    raw_data = load_data(ticker, data_dir)

    # Preprocess data
    df = preprocess_data(raw_data)

    if df.empty:
        print(f"Error: No valid data available for ticker '{ticker}' after preprocessing.")
        sys.exit(1)

    # Visualize data
    visualize_ticker(ticker, df, days_ahead=days_ahead)

if __name__ == "__main__":
    main()
